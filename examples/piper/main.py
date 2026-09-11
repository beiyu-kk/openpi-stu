"""Run a trained OpenPI policy on a Piper arm with two RealSense cameras."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import logging
import math
import signal
import time
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

logger = logging.getLogger(__name__)

PIPER_CONTROL_FREQUENCY = 30.0
JOINT_LIMITS_RAD = np.array(
    [
        [-2.6179, 2.6179],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.09439, 2.09439],
    ],
    dtype=np.float32,
)


@dataclasses.dataclass
class Args:
    """Piper control client arguments."""

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
    camera_preview_window: str = "Piper cameras"
    camera_preview_scale: float = 1.5

    max_timesteps: int = 2000
    open_loop_horizon: int = 30
    control_hz: float = PIPER_CONTROL_FREQUENCY
    move_speed_percent: int = 30
    enable_timeout_s: float = 30.0

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


def joint_radians_to_sdk_units(joints_rad: np.ndarray) -> list[int]:
    """Convert six joint angles from radians to Piper SDK 0.001 degree units."""
    joints = np.asarray(joints_rad, dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"Expected six joint angles, got shape {joints.shape}.")
    if not np.all(np.isfinite(joints)):
        raise ValueError("Joint targets contain non-finite values.")
    return [round(math.degrees(float(joint)) * 1000.0) for joint in joints]


def joint_sdk_units_to_radians(joints_sdk: list[int] | np.ndarray) -> np.ndarray:
    """Convert six Piper SDK 0.001 degree joint readings to radians."""
    joints = np.asarray(joints_sdk, dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"Expected six joint readings, got shape {joints.shape}.")
    return np.deg2rad(joints * 1e-3).astype(np.float32)


def gripper_action_to_sdk_units(
    gripper_action: float,
    *,
    open_mm: float,
    closed_mm: float,
    threshold_mm: float,
    binarize_gripper: bool,
) -> int:
    """Convert a model gripper target in meters to Piper SDK 0.001 mm units."""
    if not math.isfinite(gripper_action):
        raise ValueError("Gripper target is not finite.")
    action_mm = gripper_action * 1000.0
    if binarize_gripper:
        target_mm = closed_mm if action_mm <= threshold_mm else open_mm
    else:
        target_mm = float(np.clip(action_mm, min(closed_mm, open_mm), max(closed_mm, open_mm)))
    return round(target_mm * 1000.0)


def gripper_sdk_units_to_meters(gripper_sdk: int | float) -> float:
    """Convert Piper gripper feedback from 0.001 mm to meters."""
    return float(gripper_sdk) * 1e-6


def format_action_debug_summary(state: np.ndarray, action: np.ndarray) -> str:
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if state.shape != (7,) or action.shape != (7,):
        raise ValueError(f"Expected state and action shape (7,), got {state.shape} and {action.shape}.")
    return (
        f"state_joints_rad={np.array2string(state[:6], precision=3, suppress_small=True)} "
        f"target_joints_rad={np.array2string(action[:6], precision=3, suppress_small=True)} "
        f"delta_joints_rad={np.array2string(action[:6] - state[:6], precision=3, suppress_small=True)} "
        f"state_gripper_mm={state[-1] * 1000.0:.1f} "
        f"target_gripper_mm={action[-1] * 1000.0:.1f}"
    )


def format_action_chunk_debug_summary(state: np.ndarray, action_chunk: np.ndarray) -> str:
    state = np.asarray(state, dtype=np.float32)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if state.shape != (7,) or action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
        raise ValueError(
            f"Expected state shape (7,) and action chunk shape (N, 7), got {state.shape} and {action_chunk.shape}."
        )
    joint_deltas = action_chunk[:, :6] - state[:6]
    gripper_mm = action_chunk[:, -1] * 1000.0
    return (
        f"chunk_joint_max_abs_delta_rad={np.max(np.abs(joint_deltas)):.4f} "
        f"first_delta_joints_rad={np.array2string(joint_deltas[0], precision=3, suppress_small=True)} "
        f"last_delta_joints_rad={np.array2string(joint_deltas[-1], precision=3, suppress_small=True)} "
        f"chunk_gripper_mm_range=[{np.min(gripper_mm):.1f}, {np.max(gripper_mm):.1f}]"
    )


def get_policy_action_horizon(metadata: dict[str, Any]) -> int | None:
    """Read the policy action horizon from server metadata when available."""
    for container in (metadata, metadata.get("model") if isinstance(metadata.get("model"), dict) else None):
        if not container:
            continue
        value = container.get("action_horizon")
        if value is None:
            continue
        horizon = int(value)
        if horizon < 1:
            raise ValueError(f"Invalid policy action_horizon {horizon}; expected >= 1.")
        return horizon
    return None


def validate_action_chunk(actions: np.ndarray, *, expected_action_horizon: int | None) -> np.ndarray:
    """Validate a Piper action chunk while supporting different policy horizons."""
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Expected action chunk shape (N, 7), got {actions.shape}.")
    if actions.shape[0] < 1:
        raise RuntimeError("Expected the action chunk to contain at least one action.")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy returned non-finite actions.")
    if expected_action_horizon is not None and actions.shape[0] != expected_action_horizon:
        raise RuntimeError(f"Expected action chunk shape ({expected_action_horizon}, 7), got {actions.shape}.")
    return actions


@dataclasses.dataclass
class GripperHoldClose:
    """Latch the gripper closed until the policy explicitly releases it."""

    enabled: bool
    close_trigger_mm: float
    release_trigger_mm: float
    hold_mm: float
    _holding: bool = False

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if not self.enabled:
            return action

        gripper_mm = float(action[-1]) * 1000.0
        if self._holding:
            if gripper_mm >= self.release_trigger_mm:
                self._holding = False
                logger.info("Released gripper hold-close at policy target %.1f mm.", gripper_mm)
            else:
                action[-1] = min(float(action[-1]), self.hold_mm * 1e-3)
                return action
        elif gripper_mm <= self.close_trigger_mm:
            self._holding = True
            logger.info("Activated gripper hold-close at policy target %.1f mm.", gripper_mm)
            action[-1] = min(float(action[-1]), self.hold_mm * 1e-3)

        return action


def build_policy_observation(
    head_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Send original-resolution images; the policy resizes them using its training config."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected Piper state shape (7,), got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Piper state contains non-finite values.")
    return {
        "observation/top_image": _as_uint8_rgb(head_rgb),
        "observation/right_wrist_image": _as_uint8_rgb(wrist_rgb),
        "observation/state": state,
        "prompt": prompt,
    }


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = image_tools.convert_to_uint8(np.asarray(image))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {image.shape}.")
    return image


def clip_joint_limits(joints_rad: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_rad, dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"Expected six joint targets, got shape {joints.shape}.")
    if not np.all(np.isfinite(joints)):
        raise ValueError("Joint targets contain non-finite values.")
    return np.clip(joints, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


@contextlib.contextmanager
def prevent_keyboard_interrupt() -> Iterator[None]:
    """Delay Ctrl-C until an in-flight policy request has completed."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        *,
        name: str,
        width: int,
        height: int,
        fps: int,
        timeout_ms: int,
        warmup_frames: int,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._name = name
        self._timeout_ms = int(timeout_ms)
        self._pipeline = rs.pipeline()
        self._started = False
        config = rs.config()
        config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self._pipeline.start(config)
        self._started = True
        try:
            for _ in range(max(0, int(warmup_frames))):
                self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
        except Exception:
            self.close()
            raise

    def read_rgb(self) -> np.ndarray:
        try:
            frame_set = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
        except RuntimeError as exc:
            raise RuntimeError(f"Timed out waiting for a frame from the {self._name} camera.") from exc
        color = frame_set.get_color_frame()
        if not color:
            raise RuntimeError(f"Failed to read a color frame from the {self._name} camera.")
        bgr = np.asanyarray(color.get_data())
        return bgr[..., ::-1].copy()

    def close(self) -> None:
        if self._started:
            self._pipeline.stop()
            self._started = False


class CameraVisualizer:
    def __init__(self, *, enabled: bool, window_name: str, scale: float) -> None:
        self._enabled = bool(enabled)
        self._window_name = window_name
        self._scale = float(scale)
        self._cv2 = None
        self._window_created = False
        self._last_window_size: tuple[int, int] | None = None

        if self._enabled:
            if self._scale <= 0:
                raise ValueError("camera_preview_scale must be > 0.")
            import cv2

            self._cv2 = cv2
            self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
            self._window_created = True

    def show(self, head_rgb: np.ndarray, wrist_rgb: np.ndarray) -> bool:
        if not self._enabled:
            return True
        assert self._cv2 is not None
        head = _as_uint8_rgb(head_rgb)
        wrist = _as_uint8_rgb(wrist_rgb)
        if head.shape[:2] != wrist.shape[:2]:
            wrist = image_tools.resize_with_pad(wrist, head.shape[0], head.shape[1])
        preview_rgb = np.concatenate([head, wrist], axis=1)
        if self._scale != 1.0:
            width = max(1, round(preview_rgb.shape[1] * self._scale))
            height = max(1, round(preview_rgb.shape[0] * self._scale))
            preview_rgb = self._cv2.resize(preview_rgb, (width, height))
        window_size = (preview_rgb.shape[1], preview_rgb.shape[0])
        if window_size != self._last_window_size:
            self._cv2.resizeWindow(self._window_name, *window_size)
            self._last_window_size = window_size
        self._cv2.imshow(self._window_name, preview_rgb[..., ::-1])
        return (self._cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self._window_created:
            assert self._cv2 is not None
            self._cv2.destroyWindow(self._window_name)
            self._window_created = False


class PiperArm:
    def __init__(
        self,
        can_name: str,
        *,
        move_speed_percent: int,
        enable_timeout_s: float,
        gripper_open_mm: float,
        gripper_closed_mm: float,
        gripper_threshold_mm: float,
        gripper_effort: int,
        binarize_gripper: bool,
    ) -> None:
        from piper_sdk import C_PiperInterface_V2

        self._move_speed_percent = move_speed_percent
        self._gripper_open_mm = gripper_open_mm
        self._gripper_closed_mm = gripper_closed_mm
        self._gripper_threshold_mm = gripper_threshold_mm
        self._binarize_gripper = binarize_gripper
        self._gripper_effort = gripper_effort
        self._piper = C_PiperInterface_V2(can_name)

        try:
            self._piper.ConnectPort()
            logger.info("Enabling Piper arm on %s.", can_name)
            deadline = time.monotonic() + enable_timeout_s
            while not self._piper.EnablePiper():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out enabling Piper arm on {can_name} after {enable_timeout_s:.1f}s.")
                time.sleep(0.01)
            self._piper.ModeCtrl(0x01, 0x01, move_speed_percent, 0x00)
        except Exception:
            self.close()
            raise

    def read_state(self) -> np.ndarray:
        joint_state = self._piper.GetArmJointMsgs().joint_state
        joints_sdk = [
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        ]
        joints_rad = joint_sdk_units_to_radians(joints_sdk)
        gripper_m = gripper_sdk_units_to_meters(self._piper.GetArmGripperMsgs().gripper_state.grippers_angle)
        state = np.concatenate([joints_rad, np.array([gripper_m], dtype=np.float32)]).astype(np.float32)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Piper returned a non-finite state.")
        return state

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected one Piper action with shape (7,), got {action.shape}.")
        joints_sdk = joint_radians_to_sdk_units(clip_joint_limits(action[:6]))
        gripper_sdk = gripper_action_to_sdk_units(
            float(action[-1]),
            open_mm=self._gripper_open_mm,
            closed_mm=self._gripper_closed_mm,
            threshold_mm=self._gripper_threshold_mm,
            binarize_gripper=self._binarize_gripper,
        )
        self._piper.ModeCtrl(0x01, 0x01, self._move_speed_percent, 0x00)
        self._piper.JointCtrl(*joints_sdk)
        self._piper.GripperCtrl(gripper_sdk, self._gripper_effort, 0x01, 0)

    def close(self) -> None:
        if hasattr(self._piper, "DisconnectPort"):
            self._piper.DisconnectPort()


def _validate_args(args: Args) -> None:
    if args.open_loop_horizon < 1:
        raise ValueError("open_loop_horizon must be >= 1.")
    if args.control_hz <= 0:
        raise ValueError("control_hz must be > 0.")
    if args.max_timesteps < 1:
        raise ValueError("max_timesteps must be >= 1.")
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


def run(args: Args) -> None:
    _validate_args(args)

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
    server_metadata = policy_client.get_server_metadata()
    logger.info("Server metadata: %s", server_metadata)
    policy_action_horizon = get_policy_action_horizon(server_metadata)
    if policy_action_horizon is not None and args.open_loop_horizon > policy_action_horizon:
        raise ValueError(
            f"open_loop_horizon={args.open_loop_horizon} exceeds policy action_horizon={policy_action_horizon}."
        )

    with contextlib.ExitStack() as stack:
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

        logger.warning("Piper is enabled and live control is starting.")
        actions_from_chunk_completed = 0
        pred_action_chunk: np.ndarray | None = None
        control_period = 1.0 / args.control_hz
        gripper_hold = GripperHoldClose(
            enabled=args.gripper_hold_close,
            close_trigger_mm=args.gripper_close_trigger_mm,
            release_trigger_mm=args.gripper_release_trigger_mm,
            hold_mm=args.gripper_hold_mm,
        )

        for step in range(args.max_timesteps):
            start = time.monotonic()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()

            if pred_action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon:
                request_data = build_policy_observation(
                    head_rgb,
                    wrist_rgb,
                    state,
                    args.prompt,
                )
                with prevent_keyboard_interrupt():
                    response = policy_client.infer(request_data)
                pred_action_chunk = validate_action_chunk(
                    np.asarray(response["actions"], dtype=np.float32),
                    expected_action_horizon=policy_action_horizon,
                )
                if args.open_loop_horizon > pred_action_chunk.shape[0]:
                    raise ValueError(
                        f"open_loop_horizon={args.open_loop_horizon} exceeds returned action chunk length "
                        f"{pred_action_chunk.shape[0]}."
                    )
                if policy_action_horizon is None:
                    policy_action_horizon = int(pred_action_chunk.shape[0])
                    logger.info("Inferred policy action_horizon=%d.", policy_action_horizon)
                actions_from_chunk_completed = 0
                logger.info("Received action chunk at step %d with shape %s.", step, pred_action_chunk.shape)
                if args.log_actions:
                    logger.info("Action chunk: %s", format_action_chunk_debug_summary(state, pred_action_chunk))

            action = gripper_hold.apply(pred_action_chunk[actions_from_chunk_completed])
            actions_from_chunk_completed += 1
            if args.log_actions and step % args.open_loop_horizon == 0:
                logger.info("Sending action at step %d: %s", step, format_action_debug_summary(state, action))
            arm.send_action(action)

            elapsed = time.monotonic() - start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        run(tyro.cli(Args))
    except KeyboardInterrupt:
        logger.info("Piper control stopped by user.")


if __name__ == "__main__":
    main()
