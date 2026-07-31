"""Software-only contracts for the Phase-4 placing orchestration seam.

These tests replace every camera, robot, controller stream, and placement
command collaborator with a fake.  They specify only the public stage order
and the evidence passed between stages; the lower services remain responsible
for SDK messages and lifecycle side effects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import numpy as np
import pytest

import box_pallet
from parcel_pose_common.models import PoseResult
from parcel_pose_placing import pallet_perception, pallet_perception_adapter
from parcel_pose_placing.pallet_place import PlacementRequest


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "placing_config.json"


@dataclass
class _Trace:
    events: list[str] = field(default_factory=list)
    constructions: Counter[str] = field(default_factory=Counter)
    commands: Counter[str] = field(default_factory=Counter)

    def add(self, event: str) -> None:
        self.events.append(event)


def _install_public_stage_fakes(
    monkeypatch: pytest.MonkeyPatch,
    trace: _Trace,
    *,
    execute: bool = True,
) -> tuple[object, object, object]:
    """Install the intended staged runtime without importing any hardware SDK."""

    descent_plan = SimpleNamespace(valid=True)
    alignment_evidence = object()
    place_evidence = object()
    retreat_evidence = object()

    class FakePlacementLifecycleRuntime:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args
            assert kwargs.get("controller") is not None
            assert callable(kwargs.get("prepare"))
            assert callable(kwargs.get("release_alignment"))
            self._prepare = kwargs["prepare"]
            self._release_alignment = kwargs["release_alignment"]
            trace.constructions["lifecycle"] += 1
            trace.constructions["controller"] += 1

        def start(self) -> None:
            self._prepare()

        def stop_alignment_for_place(self) -> object:
            assert self._release_alignment() is True
            trace.add("place_alignment_stopped_and_released")
            return alignment_evidence

        def execute_place(
            self,
            received: object,
            *,
            descent_plan: object,
            await_release_authorization: Any,
        ) -> object:
            assert received is alignment_evidence
            assert descent_plan is globals_descent_plan
            assert callable(await_release_authorization)
            trace.commands["place"] += 1
            trace.add("place")
            assert await_release_authorization() is True
            trace.add("place_acknowledged_and_released")
            return place_evidence

        def execute_retreat(self, received: object) -> object:
            assert received is place_evidence
            trace.commands["retreat"] += 1
            trace.add("retreat_completed")
            return retreat_evidence

        def close(self) -> None:
            trace.add("teardown")

    # Avoid a same-name closure assignment in execute_place's signature.
    globals_descent_plan = descent_plan

    lifecycle_module = ModuleType("parcel_pose_placing.placement_lifecycle")
    lifecycle_module.PlacementLifecycleRuntime = FakePlacementLifecycleRuntime
    monkeypatch.setitem(
        sys.modules,
        "parcel_pose_placing.placement_lifecycle",
        lifecycle_module,
    )

    plan = SimpleNamespace(selected_slot=1)
    stack = object()
    state = object()

    def resolve_live_plan(**kwargs: Any) -> object:
        assert kwargs["slot"] == 1
        assert kwargs["execute"] is execute
        trace.add("initialize")
        return plan

    def assemble_live_stack(**kwargs: Any) -> object:
        assert kwargs["selected_slot"] == 1
        trace.constructions["acquisition"] += 1
        trace.constructions["sequencer"] += 1
        return stack

    def initial_run_state(**kwargs: Any) -> object:
        assert kwargs == {"plan": plan, "root_config": {}}
        return state

    class FakePlacingSession:
        def __init__(self, controller: object) -> None:
            self.controller = controller
            self.user_cancelled = False
            self._phase = "alignment"
            self._finished_frames = 0
            self._current_frame: object | None = None

        def __enter__(self) -> "FakePlacingSession":
            trace.add("acquisition_ready")
            return self

        def __exit__(self, *_exc_info: Any) -> None:
            trace.add("acquisition_closed")

        def prepare(self) -> None:
            if not execute:
                trace.add("perception_prepare")
                return
            trace.constructions["robot_connection"] += 1
            trace.constructions["ready_posture"] += 1
            trace.constructions["stream"] += 1
            trace.add("ready")

        def release_alignment(self) -> bool:
            return True

        def open_acquisition(self) -> None:
            trace.add("acquisition_opened")

        def has_frame_budget(self) -> bool:
            return self._finished_frames < 2

        def acquire_frame(self) -> object:
            self._current_frame = object()
            trace.add(f"{self._phase}_acquire")
            return self._current_frame

        def perceive_frame(self, frame: object) -> SimpleNamespace:
            assert frame is self._current_frame
            trace.add(f"{self._phase}_perceive")
            return SimpleNamespace(frame=frame, phase=self._phase)

        def decide_base_motion(self, perceived: SimpleNamespace) -> object:
            assert perceived.phase == self._phase
            trace.add(f"{self._phase}_decide_xy_yaw")
            return alignment_evidence

        def advance_placement(
            self,
            perceived: SimpleNamespace,
            base_motion: object,
        ) -> SimpleNamespace:
            assert perceived.phase == self._phase
            assert base_motion is alignment_evidence
            trace.add(f"{self._phase}_advance")
            if self._phase == "alignment":
                placement = SimpleNamespace(
                    descent_plan=descent_plan,
                    faulted=False,
                    reason="",
                    release_authorized=False,
                    request=PlacementRequest.LOWER_CARTESIAN_PLANNED,
                )
            else:
                placement = SimpleNamespace(
                    descent_plan=None,
                    faulted=False,
                    reason="",
                    release_authorized=True,
                    request=PlacementRequest.SPREAD_RELEASE,
                )
            return SimpleNamespace(placement_output=placement)

        def record_frame(
            self,
            perceived: SimpleNamespace,
            base_motion: object,
            placement_step: SimpleNamespace,
        ) -> None:
            assert perceived.phase == self._phase
            assert base_motion is alignment_evidence
            assert placement_step.placement_output is not None
            trace.add(f"{self._phase}_record")

        def finish_frame(self) -> None:
            self._finished_frames += 1
            trace.add(f"{self._phase}_finish")

        def accept_descent_plan(self, received: object) -> None:
            assert received is descent_plan
            trace.add("descent_plan_accepted")

        def begin_release_observation(self) -> None:
            self._phase = "release"
            trace.add("ack_release")

        def handle_interrupt(self) -> None:
            self.user_cancelled = True
            trace.add("interrupt_handled")

        def align(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("box_pallet must own alignment, not call session.align")

        def await_release_authorization(self) -> None:
            pytest.fail("box_pallet must own the release-observation loop")

    def open_placing_session(**kwargs: Any) -> FakePlacingSession:
        assert kwargs["plan"] is plan
        assert kwargs["stack"] is stack
        assert kwargs["state"] is state
        if execute:
            assert kwargs["controller"] is not None
        else:
            assert kwargs["controller"] is None
        trace.add("acquisition_open_requested")
        return FakePlacingSession(kwargs["controller"])

    def legacy_monolith_forbidden(**_kwargs: Any) -> None:
        pytest.fail("box_pallet must not delegate Phase-4 flow to align_and_place")

    runtime_module = ModuleType("parcel_pose_placing.pallet_runtime")
    runtime_module.resolve_live_plan = resolve_live_plan
    runtime_module.assemble_live_stack = assemble_live_stack
    runtime_module.initial_run_state = initial_run_state
    runtime_module.open_placing_session = open_placing_session
    runtime_module.align_and_place = legacy_monolith_forbidden
    monkeypatch.setitem(
        sys.modules,
        "parcel_pose_placing.pallet_runtime",
        runtime_module,
    )
    # A dotted import may reuse the cached package attribute even after
    # sys.modules changes. Patch both caches to make test order irrelevant.
    import parcel_pose_placing

    monkeypatch.setattr(
        parcel_pose_placing,
        "pallet_runtime",
        runtime_module,
        raising=False,
    )
    return alignment_evidence, place_evidence, retreat_evidence


def test_entrypoint_declares_the_requested_placing_stage_order() -> None:
    assert box_pallet.PLACING_STAGE_ORDER == (
        "preflight",
        "authorize",
        "initialize",
        "ready",
        "alignment_acquire",
        "alignment_perceive",
        "alignment_decide_x_y_yaw",
        "alignment_advance_placement",
        "alignment_record",
        "alignment_loop_exit",
        "stop_alignment",
        "place",
        "release_acquire",
        "release_perceive",
        "release_decide_x_y_yaw",
        "release_advance_placement",
        "release_record",
        "release_authorized",
        "retreat",
        "teardown",
    )


def test_slot1_public_orchestration_preserves_the_semantic_success_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace()
    _install_public_stage_fakes(monkeypatch, trace)

    real_selected_slot = box_pallet._selected_slot

    def traced_preflight(root_config: object, slot: int | None) -> int:
        trace.add("preflight")
        return real_selected_slot(root_config, slot)

    class Allowed:
        def require_authorized(self) -> "Allowed":
            return self

    def authorize(slot: int, mode: object) -> Allowed:
        assert slot == 1
        assert str(getattr(mode, "value", mode)) == "live"
        trace.add("authorize")
        return Allowed()

    monkeypatch.setattr(box_pallet, "_selected_slot", traced_preflight)
    monkeypatch.setattr(box_pallet, "authorize_slot_operation", authorize)
    controller = SimpleNamespace(config=SimpleNamespace(selected_slot=1))

    assert (
        box_pallet.place_box(
            {},
            execute=True,
            auto_place_slot1=True,
            ensure_slot1_ready=True,
            slot=1,
            headless=True,
            controller=controller,
        )
        == 0
    )

    semantic = {
        "preflight",
        "authorize",
        "initialize",
        "acquisition_ready",
        "ready",
        "acquisition_opened",
        "alignment_acquire",
        "alignment_perceive",
        "alignment_decide_xy_yaw",
        "alignment_advance",
        "alignment_record",
        "alignment_finish",
        "descent_plan_accepted",
        "place_alignment_stopped_and_released",
        "place",
        "ack_release",
        "release_acquire",
        "release_perceive",
        "release_decide_xy_yaw",
        "release_advance",
        "release_record",
        "release_finish",
        "place_acknowledged_and_released",
        "retreat_completed",
        "acquisition_closed",
        "teardown",
    }
    assert [event for event in trace.events if event in semantic] == [
        "preflight",
        "authorize",
        "initialize",
        "acquisition_ready",
        "ready",
        "acquisition_opened",
        "alignment_acquire",
        "alignment_perceive",
        "alignment_decide_xy_yaw",
        "alignment_advance",
        "alignment_record",
        "alignment_finish",
        "descent_plan_accepted",
        "place_alignment_stopped_and_released",
        "place",
        "ack_release",
        "release_acquire",
        "release_perceive",
        "release_decide_xy_yaw",
        "release_advance",
        "release_record",
        "release_finish",
        "place_acknowledged_and_released",
        "retreat_completed",
        "acquisition_closed",
        "teardown",
    ]
    assert trace.constructions == Counter(
        {
            "lifecycle": 1,
            "controller": 1,
            "robot_connection": 1,
            "ready_posture": 1,
            "stream": 1,
            "acquisition": 1,
            "sequencer": 1,
        }
    )
    assert trace.commands == Counter({"place": 1, "retreat": 1})


def test_one_frame_runs_one_estimator_one_pose_facade_and_exposes_pose_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    scene = object()
    expected_pose = PoseResult(
        x_m=0.70,
        y_m=0.02,
        yaw_rad=0.10,
        valid=True,
        reason="",
        timestamp_s=10.0,
        diagnostics={"slot": 1},
    )

    class Estimator:
        def estimate(self, *args: Any, **kwargs: Any) -> object:
            calls.append(("estimate", (args, kwargs)))
            return scene

    def pose_facade(
        received: object,
        *,
        slot: int,
        frame_id: int | None = None,
    ) -> PoseResult:
        assert received is scene
        assert slot == 1
        assert frame_id == 7
        calls.append(("pallet_pose_result", received))
        return expected_pose

    # Support either a module-global import or a deliberately local adapter
    # lookup while keeping the observed call count shared.
    monkeypatch.setattr(
        pallet_perception_adapter,
        "pallet_pose_result",
        pose_facade,
    )
    monkeypatch.setattr(
        pallet_perception,
        "pallet_pose_result",
        pose_facade,
        raising=False,
    )
    monkeypatch.setattr(
        "parcel_pose_placing.pallet_runtime._live_result_fresh",
        lambda *_args: False,
    )

    frame = SimpleNamespace(
        raw_depth_z16=np.zeros((2, 2), dtype=np.uint16),
        raw_color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        color_on_depth_bgr=None,
        depth_frame_number=7,
    )
    contract = SimpleNamespace(depth_scale_m=0.001, depth_intrinsics=object())
    held = SimpleNamespace(
        center_base_xyz_m=(0.8, 0.0, 0.7),
        yaw_base_rad=0.0,
        source="phase4-test",
    )

    observed = pallet_perception.observe_pallet_frame(
        frame,
        root_config={},
        contract=contract,
        estimator=Estimator(),
        controller=None,
        calibration_status="test",
        configured_T_base_depth=np.eye(4, dtype=np.float64),
        configured_held_proxy=held,
        capture_monotonic_s=10.0,
        accepted_scene_sequence=0,
        maximum_box_height_m=0.164,
        box_bottom_uncertainty_m=0.015,
        slot=1,
    )

    assert observed.pose_result is expected_pose
    assert [name for name, _payload in calls] == [
        "estimate",
        "pallet_pose_result",
    ]


def test_decision_primitive_passes_pose_result_to_exactly_one_xy_yaw_decision() -> None:
    """A rich scene may feed safety gates, but pose conversion stays singular."""

    import ast
    import inspect
    import textwrap

    from parcel_pose_placing import pallet_runtime
    from parcel_pose_placing.placing_session import PlacingSession

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(PlacingSession.decide_base_motion))
    )
    decision_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_decide_base_motion"
    ]
    assert len(decision_calls) == 1
    pose_keywords = [
        keyword.value
        for keyword in decision_calls[0].keywords
        if keyword.arg == "pose_result"
    ]
    assert len(pose_keywords) == 1
    pose_value = pose_keywords[0]
    assert isinstance(pose_value, ast.Attribute) and pose_value.attr == "pose_result"
    assert isinstance(pose_value.value, ast.Name) and pose_value.value.id == "observed"

    decision_tree = ast.parse(
        textwrap.dedent(inspect.getsource(pallet_runtime._decide_base_motion))
    )
    pose_fields = {
        node.attr
        for node in ast.walk(decision_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pose_result"
    }
    assert {"x_m", "y_m", "yaw_rad"} <= pose_fields


def test_entrypoint_owns_both_placing_loops_and_lower_advance_never_dispatches() -> None:
    import ast
    import inspect
    import textwrap

    from parcel_pose_placing.placing_session import PlacingSession

    tree = ast.parse(textwrap.dedent(inspect.getsource(box_pallet._run_placing_flow)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_attributes = [
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    ]

    assert len([node for node in ast.walk(tree) if isinstance(node, ast.While)]) == 2
    for primitive in (
        "acquire_frame",
        "perceive_frame",
        "decide_base_motion",
        "advance_placement",
        "record_frame",
        "finish_frame",
    ):
        assert called_attributes.count(primitive) == 2
    assert {
        "open_acquisition",
        "accept_descent_plan",
        "stop_alignment_for_place",
        "begin_release_observation",
        "execute_place",
        "execute_retreat",
    } <= set(called_attributes)
    assert {"align", "await_release_authorization"}.isdisjoint(called_attributes)
    assert "align_and_place" not in called_attributes

    advance_tree = ast.parse(
        textwrap.dedent(inspect.getsource(PlacingSession.advance_placement))
    )
    placement_calls = [
        node
        for node in ast.walk(advance_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_advance_placement"
    ]
    assert len(placement_calls) == 1
    dispatch_values = [
        keyword.value
        for keyword in placement_calls[0].keywords
        if keyword.arg == "dispatch_manipulation"
    ]
    assert len(dispatch_values) == 1
    assert isinstance(dispatch_values[0], ast.Constant)
    assert dispatch_values[0].value is False


def test_real_session_factory_and_context_have_zero_controller_or_camera_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory/enter/exit prepare no hardware; only explicit stages may do so."""

    from parcel_pose_placing import pallet_runtime, placing_session

    effects: list[str] = []

    class Controller:
        def connect(self) -> None:
            effects.append("controller.connect")

        def bootstrap_loaded_slot1_ready(self, **_kwargs: Any) -> None:
            effects.append("controller.bootstrap_ready")

        def send_ready_transition_once(self, *_args: Any, **_kwargs: Any) -> None:
            effects.append("controller.ready_command")

        def start_combined_stream(self) -> None:
            effects.append("controller.stream")

        def close(self, **_kwargs: Any) -> None:
            effects.append("controller.close")

    class CameraConstructionTrap:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            effects.append("camera.construct")
            raise AssertionError(
                "camera construction belongs to align, not session open"
            )

    monkeypatch.setattr(placing_session, "RealSenseAdapter", CameraConstructionTrap)
    controller = Controller()
    state = SimpleNamespace(
        T_base_depth=object(),
        accepted_scene_sequence=0,
        box_bottom_uncertainty_m=0.015,
        calibration_status="test",
        containment=None,
        frame_count=0,
        frame_gate=object(),
        held_proxy=object(),
        last_placement_output=None,
        last_placement_runtime_diagnostics=None,
        maximum_box_height_m=0.164,
        placement_alignment_ready_since_s=None,
        placement_lowering_started=False,
        placement_release_started=False,
        scene_window=[],
        shutdown_pending=False,
    )
    plan = SimpleNamespace(selected_slot=1, stream_config=object())

    session = pallet_runtime.open_placing_session(
        controller=controller,
        auto_place_slot1=True,
        ensure_slot1_ready=True,
        execute=True,
        plan=plan,
        state=state,
        stack=object(),
        headless=True,
        log_jsonl=None,
        max_frames=None,
        output_mp4=None,
        robot_address="never-connect.test:50051",
        robot_power="none",
        root_config={},
        window_name="phase4-real-session",
    )

    assert session.controller is controller
    assert effects == []
    with session as entered:
        assert entered is session
        assert entered.controller is controller
        assert effects == []
    assert effects == []


def test_injected_controller_slot_mismatch_refuses_before_connect_or_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from parcel_pose_placing import pallet_runtime

    root_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    effects: list[str] = []

    class MismatchedController:
        config = SimpleNamespace(selected_slot=5)
        is_connected = False

        def connect(self) -> None:
            effects.append("connect")

        def bootstrap_loaded_slot1_ready(self, **_kwargs: Any) -> None:
            effects.append("bootstrap_ready")

        def start_combined_stream(self) -> None:
            effects.append("stream")

        def close(self, **_kwargs: Any) -> None:
            effects.append("close")

    def forbidden_initialize(**_kwargs: Any) -> None:
        effects.append("assemble_live_stack")
        pytest.fail("slot mismatch reached runtime construction")

    monkeypatch.setattr(
        pallet_runtime,
        "assemble_live_stack",
        forbidden_initialize,
    )

    with pytest.raises(
        ValueError,
        match=(
            "injected controller selected_slot 5 does not match requested slot 1"
        ),
    ):
        box_pallet.place_box(
            root_config,
            execute=True,
            auto_place_slot1=True,
            ensure_slot1_ready=True,
            slot=1,
            headless=True,
            controller=MismatchedController(),
        )

    assert effects == []


def test_perception_only_uses_entrypoint_loop_without_lifecycle_or_monolith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace()
    _install_public_stage_fakes(monkeypatch, trace, execute=False)

    class Allowed:
        def require_authorized(self) -> "Allowed":
            return self

    monkeypatch.setattr(
        box_pallet,
        "authorize_slot_operation",
        lambda _slot, _mode: Allowed(),
    )

    assert (
        box_pallet.place_box(
            {},
            execute=False,
            auto_place_slot1=False,
            ensure_slot1_ready=False,
            slot=1,
            headless=True,
            controller=None,
        )
        == 0
    )

    assert trace.constructions == Counter({"acquisition": 1, "sequencer": 1})
    assert trace.commands == Counter()
    assert trace.events.count("alignment_acquire") == 2
    assert trace.events.count("alignment_perceive") == 2
    assert trace.events.count("alignment_decide_xy_yaw") == 2
    assert trace.events.count("alignment_advance") == 2
    assert trace.events.count("alignment_record") == 2
    assert "place" not in trace.events
    assert "teardown" not in trace.events
