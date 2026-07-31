"""Software-only tests for the prepared picking acquisition session."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from parcel_pose_common.models import PoseResult
from parcel_pose_picking import realtime
from parcel_pose_picking.evaluation import BasePoseDiagnostic


def _base_pose_payload() -> dict[str, Any]:
    return BasePoseDiagnostic(
        # x/y deliberately differ from the PoseResult below. The motion
        # decision must consume the narrow facade values, not this payload.
        box_center_xyz_m=(9.0, 8.0, 0.200),
        top_center_xyz_m=(9.0, 8.0, 0.320),
        yaw_mod_180_deg=0.0,
        yaw_signed_deg=-90.0,
        canonical_reference_deg=90,
        canonical_residual_deg=0.0,
        registration="validated",
    ).to_dict()


def test_prepared_session_calls_one_facade_and_one_decision_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    facade_calls: list[dict[str, Any]] = []
    decisions: list[BasePoseDiagnostic | None] = []
    calibration = SimpleNamespace(
        T_base_from_depth=np.eye(4, dtype=np.float64),
        diagnostics={},
    )
    plan = realtime.resolve_live_view_plan(
        calibration=calibration,
        fullscreen=False,
        headless=True,
        log_jsonl=None,
        max_frames=2,
        output_mp4=None,
        warmup_frames=0,
        window_name="prepared-session",
    )
    intrinsics = object()
    metadata = SimpleNamespace(
        depth_profile=SimpleNamespace(intrinsics=intrinsics),
        depth_scale_m=0.001,
    )

    class FakeCamera:
        def __init__(self, stream_config: Any) -> None:
            assert stream_config is plan.stream_config
            self.frame_number = 0

        def __enter__(self) -> FakeCamera:
            trace.append("camera_open")
            return self

        def __exit__(self, *exc_info: Any) -> None:
            trace.append("camera_closed")

        def session_metadata(self, **context: Any) -> Any:
            assert context == {"run": "phase2"}
            trace.append("profile_validated")
            return metadata

        def capture(self) -> Any:
            self.frame_number += 1
            trace.append(f"capture_{self.frame_number}")
            return SimpleNamespace(
                raw_color_bgr=np.full(
                    (1, 1, 3), self.frame_number, dtype=np.uint8
                ),
                raw_depth_z16=np.full(
                    (1, 1), self.frame_number, dtype=np.uint16
                ),
                depth_timestamp_ms=10.0 * self.frame_number,
                depth_frame_number=self.frame_number,
            )

    class FakeEstimator:
        last_evidence = None

        def __init__(
            self,
            selected_intrinsics: Any,
            selected_calibration: Any,
            estimator_config: Any,
        ) -> None:
            assert selected_intrinsics is intrinsics
            assert selected_calibration is calibration
            assert estimator_config == "estimator-config"
            trace.append("estimator_constructed")

        def estimate(self, *args: Any, **kwargs: Any) -> object:
            pytest.fail("the realtime loop must estimate only through the facade")

    def fake_perceive_box_pose(
        rgb: Any,
        depth: Any,
        selected_intrinsics: Any,
        selected_calibration: Any,
        **kwargs: Any,
    ) -> PoseResult:
        frame_number = int(kwargs["frame_id"])
        trace.append(f"perceive_{frame_number}")
        assert selected_intrinsics is intrinsics
        assert selected_calibration is calibration
        assert isinstance(kwargs["estimator"], FakeEstimator)
        assert kwargs["depth_scale"] == 0.001
        assert kwargs["sensor_timestamp_ms"] == 10.0 * frame_number
        assert rgb.shape == (1, 1, 3)
        assert depth.shape == (1, 1)
        facade_calls.append(dict(kwargs))
        if frame_number == 2:
            return PoseResult(
                x_m=None,
                y_m=None,
                yaw_rad=None,
                valid=False,
                reason="no_box_pixels",
                timestamp_s=kwargs["timestamp_s"],
                diagnostics={"frame_id": frame_number},
            )
        return PoseResult(
            x_m=0.740,
            y_m=-0.020,
            yaw_rad=math.pi / 2.0,
            valid=True,
            reason="",
            timestamp_s=kwargs["timestamp_s"],
            diagnostics={"base_pose": _base_pose_payload()},
        )

    class FakeAutomation:
        def start(self) -> None:
            trace.append("automation_started")

        def update(
            self,
            base_pose: BasePoseDiagnostic | None,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            trace.append(f"decision_{len(decisions) + 1}")
            assert pose_timestamp_s <= now_s
            decisions.append(base_pose)
            return False

        def handoff(self) -> None:
            pytest.fail("alignment session must not execute a grasp handoff")

        def close(self) -> None:
            trace.append("automation_closed")

    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(realtime, "perceive_box_pose", fake_perceive_box_pose)

    automation = FakeAutomation()
    with realtime.open_alignment_session(
        handoff_ready=plan.handoff_ready,
        log_stream=plan.log_stream,
        processed_frames=plan.processed_frames,
        user_cancelled=plan.user_cancelled,
        video_writer=plan.video_writer,
        window_created=plan.window_created,
        plan=plan,
        calibration=calibration,
        estimator_config="estimator-config",
        fullscreen=False,
        headless=True,
        max_frames=2,
        metadata_context={"run": "phase2"},
        window_name="prepared-session",
    ) as session:
        # Entering the context proves camera/profile/estimator readiness before
        # the caller starts any robot-owned service.
        assert trace == [
            "camera_open",
            "profile_validated",
            "estimator_constructed",
        ]
        automation.start()
        outcome = session.watch(automation)

    automation.close()

    assert outcome.processed_frames == 2
    assert outcome.handoff_ready is False
    assert outcome.user_cancelled is False
    assert len(facade_calls) == 2
    assert len(decisions) == 2
    assert decisions[0] is not None
    assert decisions[0].box_center_xyz_m == (0.740, -0.020, 0.200)
    assert decisions[0].yaw_mod_180_deg == pytest.approx(90.0)
    assert decisions[1] is None
    assert trace == [
        "camera_open",
        "profile_validated",
        "estimator_constructed",
        "automation_started",
        "capture_1",
        "perceive_1",
        "decision_1",
        "capture_2",
        "perceive_2",
        "decision_2",
        "camera_closed",
        "automation_closed",
    ]


def test_preparation_failure_never_enters_robot_owned_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    calibration = SimpleNamespace(
        T_base_from_depth=np.eye(4, dtype=np.float64),
        diagnostics={"camera_profile": {"serial": "expected"}},
    )
    plan = realtime.resolve_live_view_plan(
        calibration=calibration,
        fullscreen=False,
        headless=True,
        log_jsonl=None,
        max_frames=1,
        output_mp4=None,
        warmup_frames=0,
        window_name="prepared-session",
    )

    class MismatchedCamera:
        def __init__(self, stream_config: Any) -> None:
            assert stream_config is plan.stream_config

        def __enter__(self) -> MismatchedCamera:
            trace.append("camera_open")
            return self

        def __exit__(self, *exc_info: Any) -> None:
            trace.append("camera_closed")

        def session_metadata(self, **context: Any) -> Any:
            return SimpleNamespace(camera_serial="wrong")

    monkeypatch.setattr(realtime, "RealSenseAdapter", MismatchedCamera)

    with pytest.raises(
        realtime.LiveViewUnavailableError,
        match="calibration camera serial mismatch",
    ):
        with realtime.open_alignment_session(
            handoff_ready=plan.handoff_ready,
            log_stream=plan.log_stream,
            processed_frames=plan.processed_frames,
            user_cancelled=plan.user_cancelled,
            video_writer=plan.video_writer,
            window_created=plan.window_created,
            plan=plan,
            calibration=calibration,
            estimator_config=object(),
            fullscreen=False,
            headless=True,
            max_frames=1,
            metadata_context={},
            window_name="prepared-session",
        ):
            trace.append("robot_start")

    assert trace == ["camera_open", "camera_closed"]
