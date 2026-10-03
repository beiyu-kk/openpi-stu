# ruff: noqa: SLF001
import copy
import dataclasses
import threading
import types

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import pytest
import websockets.exceptions
import websockets.sync.server

from examples.piper import rtc_main as rtc

METADATA = {
    "training_rtc": True,
    "wire_action_space": "absolute",
    "raw_action_dim": 7,
    "max_delay": 3,
    "action_horizon": 6,
}


def actions():
    result = np.zeros((6, 7), np.float32)
    result[:, 0] = np.arange(6) * 0.1
    result[:, -1] = [0.06, 0.06, 0.06, 0.02, 0.04, 0.05]
    return result


@pytest.fixture
def simulation(monkeypatch):
    clock = types.SimpleNamespace(now=0.0)
    events = []
    settings = types.SimpleNamespace(latency=0.01, corrupt=None, cold_latency=1.0)
    sent = []
    requests = []
    args = rtc.Args(control_hz=10, rtc_delay=3, warmup_requests=1, max_timesteps=10, gripper_hold_close=True)

    def sleep(duration):
        clock.now += duration

    monkeypatch.setattr(rtc, "time", types.SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))

    class Future:
        def __init__(self, response, delay):
            self.response = response
            self.ready_at = clock.now + delay

        def done(self):
            return clock.now + 1e-9 >= self.ready_at

        def result(self, timeout=None):
            if timeout is not None:
                if self.ready_at - clock.now > timeout:
                    raise TimeoutError("simulated receive timeout")
                clock.now = max(clock.now, self.ready_at)
            assert self.done()
            return self.response

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, obs):
            response = fn(obs)
            delay = settings.latency if "rtc" in obs else settings.cold_latency if len(requests) == 1 else 0.02
            return Future(response, delay)

        def shutdown(self, **kwargs):
            events.append("executor closed")

    class Client:
        def __init__(self, *a, **k):
            pass

        def get_server_metadata(self):
            return METADATA.copy()

        def infer(self, obs):
            requests.append(copy.deepcopy(obs))
            chunk = actions()
            if "rtc" not in obs:
                return {"actions": chunk}
            req = obs["rtc"]
            chunk[: req["delay"]] = req["prefix"]
            chunk[req["delay"] :, 0] += 1
            result = {
                "actions": chunk,
                "rtc": {key: req[key] for key in ("delay", "start_index", "request_id")},
            }
            result["rtc"]["action_space"] = "absolute"
            if settings.corrupt:
                settings.corrupt(result)
            return result

        def close(self):
            events.append("client closed")

    class Camera:
        def __init__(self, *a, **k):
            events.append("camera opened")

        def read_rgb(self):
            return np.zeros((4, 4, 3), np.uint8)

        def close(self):
            events.append("camera closed")

    class Arm:
        def __init__(self, *a, **k):
            pass

        def read_state(self):
            return np.zeros(7, np.float32)

        def send_action(self, action):
            sent.append(rtc._canonicalize_action(action, args))

        def close(self):
            events.append("arm closed")

    monkeypatch.setattr(rtc.concurrent.futures, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(rtc.websocket_client_policy, "WebsocketClientPolicy", Client)
    monkeypatch.setattr(rtc, "RealSenseCamera", Camera)
    monkeypatch.setattr(rtc, "PiperArm", Arm)
    return types.SimpleNamespace(args=args, sent=sent, requests=requests, events=events, settings=settings)


@pytest.mark.parametrize("latency", [0.01, 0.3])
def test_early_and_on_time_replies_preserve_committed_commands(simulation, latency):
    s = simulation
    s.settings.latency = latency
    rtc.run(s.args)
    req = next(obs["rtc"] for obs in s.requests if "rtc" in obs)
    start = req["start_index"]
    np.testing.assert_array_equal(s.sent[start : start + req["delay"]], req["prefix"])
    np.testing.assert_allclose(np.asarray(s.sent)[3:6, -1], [0, 0, 0.07])
    assert s.sent[6][0] > 1  # New suffix, without replaying the prefix.
    assert s.events.index("client closed") < s.events.index("executor closed")


def test_late_reply_is_discarded_and_retry_conditions_on_hold(simulation):
    s = simulation
    s.settings.latency = 0.41
    s.args.max_timesteps = 10
    rtc.run(s.args)
    requests = [obs["rtc"] for obs in s.requests if "rtc" in obs]
    assert len(requests) == 2
    np.testing.assert_array_equal(requests[1]["prefix"], np.repeat(s.sent[5][None], 3, axis=0))
    assert all(a[0] < 1 for a in s.sent)


def test_repeated_late_replies_stop_instead_of_retrying_forever(simulation):
    s = simulation
    s.args.max_timesteps = 50
    s.args.max_consecutive_misses = 2
    s.settings.latency = 0.41
    with pytest.raises(RuntimeError, match="Repeated RTC deadlines"):
        rtc.run(s.args)
    assert "arm closed" in s.events
    assert "client closed" in s.events


@pytest.mark.parametrize("corruption", ["prefix", "start_index", "delay", "request_id", "action_space", "nonfinite"])
def test_bad_responses_are_rejected_before_handoff(simulation, corruption):
    s = simulation

    def corrupt(response):
        if corruption == "prefix":
            response["actions"][0, 0] += 0.1
        elif corruption == "nonfinite":
            response["actions"][-1, 0] = np.nan
        else:
            response["rtc"].pop(corruption)

    s.settings.corrupt = corrupt
    with pytest.raises(RuntimeError):
        rtc.run(s.args)
    assert len(s.sent) == 6


def test_warmup_timeout_closes_connection_and_arm(simulation):
    s = simulation
    s.args.inference_timeout_s = 0.1
    with pytest.raises(TimeoutError):
        rtc.run(s.args)
    assert not s.sent
    assert s.events.index("client closed") < s.events.index("executor closed")


def test_auto_delay_excludes_cold_compile_time(simulation):
    s = simulation
    s.args.rtc_delay = None
    rtc.run(s.args)
    req = next(obs["rtc"] for obs in s.requests if "rtc" in obs)
    assert req["delay"] == 2  # ceil(20 ms * 10 Hz) + one margin tick; cold request was 1 s.


@pytest.mark.parametrize("value", [0, -1, 6, 2.5, True, None])
def test_invalid_delay_metadata(value):
    with pytest.raises(ValueError, match="max_delay"):
        rtc._rtc_server_settings({**METADATA, "max_delay": value})


def test_prefix_preview_does_not_advance_live_gripper_latch():
    args = rtc.Args(gripper_hold_close=True)
    hold = rtc.GripperHoldClose(enabled=True, close_trigger_mm=25, release_trigger_mm=45, hold_mm=0)
    before = dataclasses.asdict(hold)
    prefix = rtc._canonicalize_prefix(actions()[3:], args, hold)
    assert dataclasses.asdict(hold) == before
    np.testing.assert_allclose(prefix[:, -1], [0, 0, 0.07])


def test_close_unblocks_real_websocket_inference():
    import concurrent.futures

    received = threading.Event()

    def handler(ws):
        ws.send(msgpack_numpy.packb(METADATA))
        ws.recv(timeout=2)
        received.set()
        # Deliberately don't send a response. Closing the client must release
        # the worker even though the server has not completed inference.
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            ws.recv(timeout=2)

    with websockets.sync.server.serve(handler, "127.0.0.1", 0) as server:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", server.socket.getsockname()[1])
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            pending = executor.submit(client.infer, {"state": np.zeros(7)})
            assert received.wait(timeout=2)
            client.close()
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                pending.result(timeout=2)
        finally:
            client.close()
            executor.shutdown(wait=True)
            server.shutdown()
            server_thread.join(timeout=2)
