"""Async real-time chunking (RTC) client for a Piper arm served by a Training RTC policy.

Unlike main.py (synchronous: the arm waits while the policy infers), this client keeps
executing the tail of the current action chunk while the next chunk is being inferred.
Each request carries the actions that will be executed during the inference latency as a
hard prefix, so consecutive chunks connect without a pause or jump at the boundary.

Requires a checkpoint trained with a *_rtc config; the served policy must expose
training_rtc metadata. See docs/training_rtc.md ("推理接口") for the wire format.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import logging
import math
import operator
import pathlib
import sys
import time

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from main import CameraVisualizer
from main import GripperHoldClose
from main import PiperArm
from main import RealSenseCamera
from main import build_policy_observation
from main import clip_joint_limits
from main import format_action_chunk_debug_summary
from main import gripper_action_to_sdk_units
from main import gripper_sdk_units_to_meters
from main import joint_radians_to_sdk_units
from main import joint_sdk_units_to_radians
from main import validate_action_chunk
from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Async RTC Piper control client arguments."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    can_name: str = "can0"
    prompt: str = 'Pick up the book labeled "Wang Guowei Selected Works" and place it in the black grid.'

    head_camera_serial: str = "339322074804"
    wrist_camera_serial: str = "346522074547"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_timeout_ms: int = 1000
    camera_warmup_frames: int = 5
    show_cameras: bool = False
    camera_preview_window: str = "Piper cameras (RTC)"
    camera_preview_scale: float = 1.5

    max_timesteps: int = 2000
    control_hz: float = 30.0
    move_speed_percent: int = 30
    enable_timeout_s: float = 30.0

    # Prefix length in control ticks committed to each request. Must be <= the policy's
    # trained max_delay. None = measure warm round trips plus one tick of margin.
    rtc_delay: int | None = None
    # Measured requests after the first (cold compilation) inference.
    warmup_requests: int = 3
    # Maximum wait for a single inference, including cold compilation.
    inference_timeout_s: float = 30.0
    # Stop after repeated missed deadlines instead of indefinitely holding/retrying.
    max_consecutive_misses: int = 3

    gripper_open_mm: float = 70.0
    gripper_closed_mm: float = 0.0
    gripper_threshold_mm: float = 35.0
    binarize_gripper: bool = True
    gripper_effort: int = 2000  # scale: 0-5000
    gripper_hold_close: bool = False
    gripper_close_trigger_mm: float = 25.0
    gripper_release_trigger_mm: float = 45.0
    gripper_hold_mm: float = 0.0

    log_actions: bool = False


@dataclasses.dataclass
class _InflightRequest:
    request_id: int
    delay: int
    fire_step: int
    prefix: np.ndarray
    started_at: float
    future: concurrent.futures.Future


def _validate_args(args: Args) -> None:
    if not math.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be finite and > 0.")
    if args.warmup_requests < 1 or args.max_consecutive_misses < 1:
        raise ValueError("warmup_requests and max_consecutive_misses must be >= 1.")
    if not math.isfinite(args.inference_timeout_s) or args.inference_timeout_s <= 0:
        raise ValueError("inference_timeout_s must be finite and > 0.")
    if args.max_timesteps < 1:
        raise ValueError("max_timesteps must be >= 1.")
    if args.rtc_delay is not None and args.rtc_delay < 1:
        raise ValueError("rtc_delay must be >= 1; a zero delay cannot cover inference latency and stalls the queue.")
    if not 1 <= args.move_speed_percent <= 100:
        raise ValueError("move_speed_percent must be between 1 and 100.")
    if args.enable_timeout_s <= 0:
        raise ValueError("enable_timeout_s must be > 0.")
    if args.gripper_hold_close and args.gripper_release_trigger_mm <= args.gripper_close_trigger_mm:
        raise ValueError("gripper_release_trigger_mm must be greater than gripper_close_trigger_mm.")
    if args.binarize_gripper and not min(
        args.gripper_open_mm, args.gripper_closed_mm
    ) <= args.gripper_threshold_mm <= max(args.gripper_open_mm, args.gripper_closed_mm):
        raise ValueError("gripper_threshold_mm must be between gripper_closed_mm and gripper_open_mm.")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be an integer.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _rtc_server_settings(metadata: dict) -> tuple[int, int]:
    """Return (max_delay, action_horizon); refuse non-RTC policies."""
    if not isinstance(metadata, dict):
        raise ValueError("The policy server metadata must be a dictionary.")
    if metadata.get("training_rtc") is not True:
        raise ValueError(
            "The served policy is not a Training RTC policy (missing training_rtc metadata). "
            "Serve a checkpoint trained with a *_rtc config."
        )
    if metadata.get("wire_action_space") != "absolute":
        raise ValueError("RTC policy must report wire_action_space='absolute'.")
    horizon = _integer(metadata.get("action_horizon"), "action_horizon")
    max_delay = _integer(metadata.get("max_delay"), "max_delay")
    # A zero-delay model can be used by the synchronous client, but it cannot
    # hide any inference latency and therefore cannot support this async loop.
    if max_delay < 1:
        raise ValueError("RTC async control requires a policy max_delay >= 1.")
    if max_delay >= horizon:
        raise ValueError(f"RTC policy max_delay={max_delay} must be smaller than action_horizon={horizon}.")
    if "raw_action_dim" in metadata:
        raw_action_dim = _integer(metadata["raw_action_dim"], "raw_action_dim")
        if raw_action_dim != 7:
            raise ValueError("RTC Piper control requires raw_action_dim=7.")
    return max_delay, horizon


def _validate_rtc_response(response: object, *, action_horizon: int) -> np.ndarray:
    if not isinstance(response, dict):
        raise RuntimeError(f"Policy response must be a dictionary, got {type(response).__name__}.")
    try:
        actions = np.asarray(response["actions"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Policy response lacks a valid actions field.") from exc
    return validate_action_chunk(actions, expected_action_horizon=action_horizon)


def _validate_rtc_reply(response: dict, request: _InflightRequest, *, action_horizon: int) -> np.ndarray:
    actions = _validate_rtc_response(response, action_horizon=action_horizon)
    reply = response.get("rtc")
    if not isinstance(reply, dict):
        raise RuntimeError("RTC policy response lacks its rtc acknowledgement.")
    expected = {
        "request_id": request.request_id,
        "start_index": request.fire_step,
        "delay": request.delay,
        "action_space": "absolute",
    }
    for key, value in expected.items():
        if reply.get(key) != value:
            raise RuntimeError(f"RTC response {key}={reply.get(key)!r} does not match request {value!r}.")
    if not np.array_equal(actions[: request.delay], request.prefix):
        raise RuntimeError("RTC response changed the committed prefix.")
    return actions


def _canonicalize_action(action: np.ndarray, args: Args) -> np.ndarray:
    """Return the exact absolute command represented by Piper's SDK request."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"Policy action must be finite with shape (7,), got {action.shape}.")
    joints = clip_joint_limits(action[:6])
    joints = joint_sdk_units_to_radians(joint_radians_to_sdk_units(joints))
    gripper_sdk = gripper_action_to_sdk_units(
        float(action[-1]),
        open_mm=args.gripper_open_mm,
        closed_mm=args.gripper_closed_mm,
        threshold_mm=args.gripper_threshold_mm,
        binarize_gripper=args.binarize_gripper,
    )
    return np.concatenate([joints, np.asarray([gripper_sdk_units_to_meters(gripper_sdk)], dtype=np.float32)]).astype(
        np.float32
    )


def _canonicalize_prefix(actions: np.ndarray, args: Args, gripper_hold: GripperHoldClose) -> np.ndarray:
    """Preview execution transforms without mutating the live gripper latch."""
    preview_hold = dataclasses.replace(gripper_hold)
    return np.stack(
        [_canonicalize_action(preview_hold.apply(action), args) for action in np.asarray(actions)],
        axis=0,
    )


def run(args: Args) -> None:
    _validate_args(args)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: executor.shutdown(wait=True, cancel_futures=True))
        policy_client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
        # Closing the websocket first unblocks a request that is still waiting
        # when the control loop exits, before executor shutdown joins its worker.
        stack.callback(policy_client.close)
        server_metadata = policy_client.get_server_metadata()
        logger.info("Server metadata: %s", server_metadata)
        max_delay, action_horizon = _rtc_server_settings(server_metadata)
        if args.rtc_delay is not None and args.rtc_delay > max_delay:
            raise ValueError(f"rtc_delay={args.rtc_delay} exceeds the policy max_delay={max_delay}.")
        head_camera = RealSenseCamera(
            args.head_camera_serial,
            name="head",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            timeout_ms=args.camera_timeout_ms,
            warmup_frames=args.camera_warmup_frames,
        )
        stack.callback(head_camera.close)
        wrist_camera = RealSenseCamera(
            args.wrist_camera_serial,
            name="right wrist",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            timeout_ms=args.camera_timeout_ms,
            warmup_frames=args.camera_warmup_frames,
        )
        stack.callback(wrist_camera.close)
        visualizer = CameraVisualizer(
            enabled=args.show_cameras,
            window_name=args.camera_preview_window,
            scale=args.camera_preview_scale,
        )
        stack.callback(visualizer.close)
        arm = PiperArm(
            args.can_name,
            move_speed_percent=args.move_speed_percent,
            enable_timeout_s=args.enable_timeout_s,
            gripper_open_mm=args.gripper_open_mm,
            gripper_closed_mm=args.gripper_closed_mm,
            gripper_threshold_mm=args.gripper_threshold_mm,
            gripper_effort=args.gripper_effort,
            binarize_gripper=args.binarize_gripper,
        )
        stack.callback(arm.close)

        logger.warning("Piper is enabled. Warming up inference before sending actions.")
        warm_round_trips = []
        for warmup in range(args.warmup_requests + 1):
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            state = arm.read_state()
            request_data = build_policy_observation(head_rgb, wrist_rgb, state, args.prompt)
            t0 = time.monotonic()
            response = executor.submit(policy_client.infer, request_data).result(timeout=args.inference_timeout_s)
            round_trip_s = time.monotonic() - t0
            chunk = _validate_rtc_response(response, action_horizon=action_horizon)
            logger.info("Warmup %d round trip: %.1f ms.", warmup, round_trip_s * 1000.0)
            # The first request includes JAX compilation; it doesn't estimate steady latency.
            if warmup:
                warm_round_trips.append(round_trip_s)

        if args.rtc_delay is not None:
            delay = args.rtc_delay
        else:
            needed = math.ceil(max(warm_round_trips) * args.control_hz) + 1
            delay = min(max(needed, 1), max_delay)
            if needed > max_delay:
                logger.warning(
                    "Inference needs ~%d control ticks but max_delay=%d; queue underruns are likely. "
                    "Lower --control-hz or speed up inference.",
                    needed,
                    max_delay,
                )
        logger.info(
            "Using rtc_delay=%d ticks (%.0f ms at %.1f Hz).", delay, delay / args.control_hz * 1000.0, args.control_hz
        )

        completed = 0
        inflight: _InflightRequest | None = None
        next_request_id = 1
        underruns = 0
        consecutive_misses = 0
        last_action: np.ndarray | None = None
        control_period = 1.0 / args.control_hz
        gripper_hold = GripperHoldClose(
            enabled=args.gripper_hold_close,
            close_trigger_mm=args.gripper_close_trigger_mm,
            release_trigger_mm=args.gripper_release_trigger_mm,
            hold_mm=args.gripper_hold_mm,
        )

        for step in range(args.max_timesteps):
            tick_start = time.monotonic()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()

            if inflight is not None and time.monotonic() - inflight.started_at > args.inference_timeout_s:
                raise TimeoutError(f"Policy request {inflight.request_id} exceeded inference_timeout_s.")

            # Keep executing the original commands/latch transitions even if the
            # reply arrives early. Only install after the whole prefix is sent.
            if inflight is not None and inflight.future.done() and step >= inflight.fire_step + inflight.delay:
                try:
                    response = inflight.future.result()
                except Exception as exc:
                    raise RuntimeError(f"Policy request {inflight.request_id} failed.") from exc
                new_chunk = _validate_rtc_reply(response, inflight, action_horizon=action_horizon)
                executed_prefix = step - inflight.fire_step
                if executed_prefix > inflight.delay:
                    # The prefix has expired and the arm has already held its
                    # last target. The continuation was conditioned on an
                    # older state, so using it would replay stale commands.
                    logger.warning(
                        "Discarding stale RTC response %d after %d ticks (committed delay=%d).",
                        inflight.request_id,
                        executed_prefix,
                        inflight.delay,
                    )
                    consecutive_misses += 1
                    if consecutive_misses >= args.max_consecutive_misses:
                        raise RuntimeError(
                            "Repeated RTC deadlines missed. Increase rtc_delay within trained support "
                            "or reduce inference latency before restarting."
                        )
                    assert last_action is not None
                    chunk = np.repeat(last_action[None], delay, axis=0)
                    completed = 0
                else:
                    consecutive_misses = 0
                    chunk = new_chunk
                    completed = executed_prefix
                    timing = response.get("policy_timing") or {}
                    logger.info(
                        "Handoff request %d: delay=%d, executed_prefix=%d, infer_ms=%.1f.",
                        inflight.request_id,
                        inflight.delay,
                        executed_prefix,
                        float(timing.get("infer_ms", float("nan"))),
                    )
                    if args.log_actions:
                        logger.info("Action chunk: %s", format_action_chunk_debug_summary(state, chunk))
                inflight = None

            # Launch the next inference early enough that the chunk tail covers its latency.
            if inflight is None and step < args.max_timesteps - 1:
                remaining = chunk.shape[0] - completed
                if remaining <= delay:
                    fire_delay = min(delay, remaining)
                    request_data = build_policy_observation(head_rgb, wrist_rgb, state, args.prompt)
                    committed_prefix = _canonicalize_prefix(
                        chunk[completed : completed + fire_delay], args, gripper_hold
                    )
                    request_data["rtc"] = {
                        "prefix": committed_prefix,
                        "delay": fire_delay,
                        "start_index": step,
                        "request_id": next_request_id,
                    }
                    started_at = time.monotonic()
                    future = executor.submit(policy_client.infer, request_data)
                    inflight = _InflightRequest(next_request_id, fire_delay, step, committed_prefix, started_at, future)
                    next_request_id += 1

            if completed >= chunk.shape[0]:
                # Response has not arrived within the committed prefix; hold the last target.
                underruns += 1
                if (underruns - 1) % max(1, round(args.control_hz)) == 0:
                    logger.warning(
                        "Action queue underrun (%d ticks); holding the last target. "
                        "Inference is slower than rtc_delay=%d ticks covers.",
                        underruns,
                        delay,
                    )
                if last_action is not None:
                    arm.send_action(last_action)
            else:
                # PiperArm applies the SDK conversion once. The request prefix
                # previews that conversion; don't binarize the command twice.
                action = gripper_hold.apply(chunk[completed])
                completed += 1
                last_action = action
                arm.send_action(action)

            elapsed = time.monotonic() - tick_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

        if underruns:
            logger.warning("Finished with %d underrun ticks in total.", underruns)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        run(tyro.cli(Args))
    except KeyboardInterrupt:
        logger.info("Piper RTC control stopped by user.")


if __name__ == "__main__":
    main()
