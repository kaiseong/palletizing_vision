"""Software-only integration gates for the staged picking entrypoint.

All camera, controller, robot, stream, and command collaborators are fakes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import pytest

from parcel_pose_common import operation_authority

import box_picking


@dataclass
class _Trace:
    events: list[str] = field(default_factory=list)
    constructions: Counter[str] = field(default_factory=Counter)
    commands: Counter[str] = field(default_factory=Counter)

    def add(self, event: str) -> None:
        self.events.append(event)


class _FakeAutoGrabError(RuntimeError):
    pass


def _patch_authority_trace(
    monkeypatch: pytest.MonkeyPatch,
    trace: _Trace,
) -> None:
    real_authorize = operation_authority.authorize_operation

    def traced_authorize(request: Any, **kwargs: Any) -> Any:
        trace.add("authorize")
        return real_authorize(request, **kwargs)

    monkeypatch.setattr(operation_authority, "authorize_operation", traced_authorize)
    if hasattr(box_picking, "authorize_operation"):
        monkeypatch.setattr(box_picking, "authorize_operation", traced_authorize)


def _patch_config_and_calibration(
    monkeypatch: pytest.MonkeyPatch,
    trace: _Trace,
) -> None:
    def load_json(_path: Path) -> dict[str, object]:
        trace.add("config_loaded")
        return {}

    def load_calibration(_path: Path) -> SimpleNamespace:
        trace.add("calibration_loaded")
        return SimpleNamespace(absolute_base_validated=True)

    monkeypatch.setattr("parcel_pose_common.calibration.load_json", load_json)
    monkeypatch.setattr(
        "parcel_pose_common.calibration.load_calibration",
        load_calibration,
    )
    monkeypatch.setattr(
        "parcel_pose_picking.cli._estimator_config",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "parcel_pose_picking.cli._recording_context",
        lambda *_args: {"source": "phase2-integration"},
    )


def _entrypoint_args(*, orientation: str = "horizontal") -> SimpleNamespace:
    return SimpleNamespace(
        orientation=orientation,
        config=Path("unused-picking-config.json"),
        calibration=Path("unused-picking-calibration.json"),
        robot_address="never-connected.test:50051",
        robot_power="none",
        fullscreen=False,
        headless=True,
        log_jsonl=None,
        max_frames=4,
        output_mp4=None,
        warmup_frames=0,
        window_name="phase2-integration",
    )


def _install_staged_fakes(
    monkeypatch: pytest.MonkeyPatch,
    trace: _Trace,
    *,
    handoff_ready: bool = True,
    stop_failure: str | None = None,
    close_failure: str | None = None,
) -> None:
    evidence = object()
    fake_frame = object()
    fake_base_pose = object()

    class FakeAutoGrabConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeAutoGrabRuntime:
        def __init__(
            self,
            _config: FakeAutoGrabConfig,
            *,
            execute: bool,
        ) -> None:
            assert execute is True
            trace.constructions["runtime"] += 1
            trace.constructions["controller"] += 1
            trace.add("initialize")

        def start(self) -> None:
            trace.constructions["robot_connection"] += 1
            trace.constructions["ready_posture"] += 1
            trace.constructions["stream"] += 1
            trace.constructions["sequencer"] += 1
            trace.add("ready")

        def update(
            self,
            base_pose: object,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            assert base_pose is fake_base_pose
            assert pose_timestamp_s <= now_s
            trace.add("decide_xy_yaw")
            return handoff_ready

        def stop_alignment_for_grasp(self) -> object:
            trace.add("grasp_alignment_stop_requested")
            if stop_failure is not None:
                raise _FakeAutoGrabError(stop_failure)
            trace.add("grasp_alignment_stopped_and_released")
            return evidence

        def execute_grasp(self, received: object) -> None:
            assert received is evidence
            trace.commands["grasp_and_lift"] += 1
            trace.add("grasp_and_lift_completed")

        def close(self) -> None:
            trace.add("robot_close_requested")
            if close_failure is not None:
                raise _FakeAutoGrabError(close_failure)
            trace.add("robot_session_closed")

    auto_grab_module = ModuleType("parcel_pose_picking.auto_grab")
    auto_grab_module.AutoGrabConfig = FakeAutoGrabConfig
    auto_grab_module.AutoGrabError = _FakeAutoGrabError
    auto_grab_module.AutoGrabRuntime = FakeAutoGrabRuntime
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.auto_grab", auto_grab_module)

    class FakeAlignmentSession:
        def __init__(self) -> None:
            self.processed_frames = 0
            self.handoff_ready = False
            self.user_cancelled = False

        def __enter__(self) -> "FakeAlignmentSession":
            trace.constructions["acquisition"] += 1
            trace.add("acquisition_ready")
            return self

        def __exit__(self, *_exc_info: Any) -> None:
            trace.add("acquisition_closed")

        def has_frame_budget(self) -> bool:
            return self.processed_frames < 4

        def acquire_frame(self) -> object:
            trace.add("acquire_frame")
            return fake_frame

        def perceive_frame(self, frame: object) -> SimpleNamespace:
            assert frame is fake_frame
            trace.add("perceive_frame")
            return SimpleNamespace(
                base_pose=fake_base_pose,
                pose_result=SimpleNamespace(timestamp_s=1.0),
            )

        def record_frame(
            self,
            observation: SimpleNamespace,
            *,
            handoff_ready: bool,
        ) -> None:
            assert observation.base_pose is fake_base_pose
            self.processed_frames += 1
            self.handoff_ready = handoff_ready
            trace.add("record_frame")
            if handoff_ready:
                trace.add("handoff_ready")
            elif not self.has_frame_budget():
                trace.add("no_handoff")

        def cancel(self) -> None:
            self.user_cancelled = True
            self.handoff_ready = False
            trace.add("cancel")

        def outcome(self) -> SimpleNamespace:
            return SimpleNamespace(
                processed_frames=self.processed_frames,
                handoff_ready=self.handoff_ready,
                user_cancelled=self.user_cancelled,
            )

        def watch(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("box_picking must own the frame loop, not call session.watch")

        def watch_for_alignment(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("box_picking must not call watch_for_alignment")

    def resolve_live_view_plan(**_kwargs: Any) -> SimpleNamespace:
        trace.add("preflight")
        return SimpleNamespace(
            handoff_ready=False,
            log_stream=None,
            processed_frames=0,
            user_cancelled=False,
            video_writer=None,
            window_created=False,
        )

    def open_alignment_session(**kwargs: Any) -> FakeAlignmentSession:
        assert kwargs["metadata_context"] == {"source": "phase2-integration"}
        assert kwargs["max_frames"] == 4
        trace.add("acquisition_open_requested")
        return FakeAlignmentSession()

    realtime_module = ModuleType("parcel_pose_picking.realtime")
    realtime_module.LiveViewUnavailableError = _FakeAutoGrabError
    realtime_module.resolve_live_view_plan = resolve_live_view_plan
    realtime_module.open_alignment_session = open_alignment_session

    def legacy_watch_forbidden(**_kwargs: Any) -> None:
        pytest.fail("staged entrypoint must not delegate lifecycle to watch_and_grab")

    realtime_module.watch_and_grab = legacy_watch_forbidden
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.realtime", realtime_module)


def test_vertical_live_refuses_before_every_construction_or_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _Trace()
    _patch_authority_trace(monkeypatch, trace)
    _patch_config_and_calibration(monkeypatch, trace)
    _install_staged_fakes(monkeypatch, trace)

    result = box_picking.main(["--orientation", "vertical", "--headless"])

    assert result == 2
    assert trace.events.count("authorize") == 1
    assert "initialize" not in trace.events
    assert trace.constructions == Counter()
    assert trace.commands == Counter()
    assert capsys.readouterr().err.strip() == (
        "vertical pick live refused; missing fields: "
        "perception_validation, ready_pose, grasp_pose"
    )


def test_authorized_horizontal_pick_closes_acquisition_before_grasp_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace()
    _patch_authority_trace(monkeypatch, trace)
    _patch_config_and_calibration(monkeypatch, trace)
    _install_staged_fakes(monkeypatch, trace)

    assert box_picking.pick_box(_entrypoint_args()) == 0

    semantic_names = {
        "authorize",
        "initialize",
        "acquisition_ready",
        "ready",
        "acquire_frame",
        "perceive_frame",
        "decide_xy_yaw",
        "record_frame",
        "handoff_ready",
        "acquisition_closed",
        "grasp_alignment_stopped_and_released",
        "grasp_and_lift_completed",
        "robot_session_closed",
    }
    semantic_events = [event for event in trace.events if event in semantic_names]
    assert semantic_events == [
        "authorize",
        "initialize",
        "acquisition_ready",
        "ready",
        "acquire_frame",
        "perceive_frame",
        "decide_xy_yaw",
        "record_frame",
        "handoff_ready",
        "acquisition_closed",
        "grasp_alignment_stopped_and_released",
        "grasp_and_lift_completed",
        "robot_session_closed",
    ]
    assert trace.constructions == Counter(
        {
            "runtime": 1,
            "controller": 1,
            "acquisition": 1,
            "robot_connection": 1,
            "ready_posture": 1,
            "stream": 1,
            "sequencer": 1,
        }
    )
    assert trace.commands == Counter({"grasp_and_lift": 1})


def test_failed_stopped_and_released_stage_blocks_grasp_and_still_tears_down(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _Trace()
    _patch_authority_trace(monkeypatch, trace)
    _patch_config_and_calibration(monkeypatch, trace)
    _install_staged_fakes(
        monkeypatch,
        trace,
        stop_failure="cannot release the stopped mobility stream",
    )

    assert box_picking.pick_box(_entrypoint_args()) == 2

    assert trace.events.index("acquisition_closed") < trace.events.index(
        "grasp_alignment_stop_requested"
    )
    assert "grasp_alignment_stopped_and_released" not in trace.events
    assert "grasp_and_lift_completed" not in trace.events
    assert trace.commands["grasp_and_lift"] == 0
    assert trace.events.count("robot_session_closed") == 1
    assert capsys.readouterr().err.strip() == (
        "cannot release the stopped mobility stream"
    )


def test_no_handoff_close_failure_returns_two_without_grasp_or_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _Trace()
    _patch_authority_trace(monkeypatch, trace)
    _patch_config_and_calibration(monkeypatch, trace)
    _install_staged_fakes(
        monkeypatch,
        trace,
        handoff_ready=False,
        close_failure="robot teardown failed after no handoff",
    )

    assert box_picking.pick_box(_entrypoint_args()) == 2

    assert trace.events[-3:] == [
        "no_handoff",
        "acquisition_closed",
        "robot_close_requested",
    ]
    assert "grasp_alignment_stop_requested" not in trace.events
    assert "grasp_alignment_stopped_and_released" not in trace.events
    assert "grasp_and_lift_completed" not in trace.events
    assert trace.commands["grasp_and_lift"] == 0
    captured = capsys.readouterr()
    assert captured.err.strip() == "robot teardown failed after no handoff"
    assert "Traceback" not in captured.err


def test_entrypoint_source_owns_picking_loop_and_forbids_hidden_runners() -> None:
    import ast
    import inspect
    import textwrap

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(box_picking._run_authorized_horizontal_pick))
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert len([node for node in ast.walk(tree) if isinstance(node, ast.While)]) == 1
    assert {
        "has_frame_budget",
        "acquire_frame",
        "perceive_frame",
        "update",
        "record_frame",
    } <= called_attributes
    assert len([node for node in ast.walk(tree) if isinstance(node, ast.Break)]) >= 2
    assert {
        "watch",
        "watch_for_alignment",
        "watch_and_grab",
    }.isdisjoint(called_attributes)
