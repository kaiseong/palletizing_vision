from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from parcel_pose.evaluation import BasePoseDiagnostic, base_pose_from_estimate
from parcel_pose.models import (
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    EstimatorConfig,
    Plane,
    PoseEstimate,
)
from parcel_pose.realtime import (
    LiveViewUnavailableError,
    draw_live_overlay,
    live_overlay_lines,
    run_live_view,
)
from tests.test_recording import make_frame, make_metadata


def _base_pose() -> BasePoseDiagnostic:
    return BasePoseDiagnostic(
        box_center_xyz_m=(0.7123, -0.0456, 0.8249),
        top_center_xyz_m=(0.7123, -0.0456, 0.8999),
        yaw_mod_180_deg=168.4,
        yaw_signed_deg=-11.6,
        canonical_reference_deg=0,
        canonical_residual_deg=-11.6,
        registration="nominal_unverified",
    )


def _calibration() -> Calibration:
    return Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=Plane(normal=[0.0, 0.0, 1.0], d=0.75, frame="depth"),
        T_base_from_head=np.eye(4),
        T_head_from_depth=np.eye(4),
    )


def _camera_bound_calibration(serial: str) -> Calibration:
    calibration = _calibration()
    return Calibration(
        state=calibration.state,
        table_plane=calibration.table_plane,
        T_base_from_head=calibration.T_base_from_head,
        T_head_from_depth=calibration.T_head_from_depth,
        diagnostics={"camera_profile": {"serial": serial}},
    )


def _estimate() -> PoseEstimate:
    return PoseEstimate(
        frame="table_plane",
        center_depth_m=(0.0, 0.0, 0.90),
        yaw_rad=0.0,
        yaw_mod_180_deg=0.0,
        observability={"yaw": "constrained"},
        diagnostics={"nominal_unverified_base": {"yaw_rad": 0.0}},
        geometry_valid=True,
        full_pose_valid=True,
        calibration_state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        base_registration="nominal_unverified",
    )


def test_live_text_contains_only_requested_fields() -> None:
    lines = live_overlay_lines(_base_pose(), 12.34)

    assert lines == (
        "x=+0.712 m   y=-0.046 m   z=+0.825 m",
        "yaw=-11.6 deg   latency=12.3 ms",
    )
    joined = " ".join(lines).lower()
    assert all(f"{field}=" in joined for field in ("x", "y", "z", "yaw", "latency"))
    for forbidden in (
        "frame",
        "status",
        "registration",
        "confidence",
        "canonical",
        "reason",
        "valid",
        "abstain",
        "no gt",
    ):
        assert forbidden not in joined


def test_live_text_uses_placeholders_when_base_pose_is_unavailable() -> None:
    assert live_overlay_lines(None, 8.0) == (
        "x=-- m   y=-- m   z=-- m",
        "yaw=-- deg   latency=8.0 ms",
    )


def test_live_text_uses_the_same_corrected_y_as_mobile_control() -> None:
    calibration = Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=Plane(normal=[0.0, 0.0, 1.0], d=0.75, frame="depth"),
        T_base_from_head=np.eye(4),
        T_head_from_depth=np.eye(4),
        base_translation_correction_m=(0.0, 0.05, 0.0),
    )
    estimate = _estimate()
    estimate = PoseEstimate(
        frame=estimate.frame,
        center_depth_m=(0.740, -0.05, 0.90),
        yaw_rad=estimate.yaw_rad,
        yaw_mod_180_deg=estimate.yaw_mod_180_deg,
        observability=estimate.observability,
        diagnostics=estimate.diagnostics,
        geometry_valid=estimate.geometry_valid,
        full_pose_valid=estimate.full_pose_valid,
        calibration_state=estimate.calibration_state,
        base_registration=estimate.base_registration,
    )

    base_pose = base_pose_from_estimate(estimate, calibration)

    assert base_pose is not None
    assert base_pose.box_center_xyz_m[1] == pytest.approx(0.0)
    assert "x=+0.740 m   y=+0.000 m" in live_overlay_lines(base_pose, 5.0)[0]


class _OverlayCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.polygons: list[tuple[np.ndarray, bool]] = []
        self.text: list[str] = []

    def addWeighted(self, first, alpha, second, beta, gamma, destination) -> None:
        destination[...] = np.rint(alpha * first + beta * second + gamma).astype(
            np.uint8
        )

    def putText(
        self, image, text, origin, font, scale, color, thickness, line_type
    ) -> None:
        self.text.append(text)

    def polylines(self, image, polygons, closed, color, thickness, line_type) -> None:
        self.polygons.append((polygons[0].copy(), bool(closed)))


def test_live_overlay_draws_one_closed_top_rectangle_and_no_other_annotations() -> None:
    cv2 = _OverlayCv2()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    intrinsics = CameraIntrinsics(
        width=320,
        height=240,
        fx=300.0,
        fy=300.0,
        cx=159.5,
        cy=119.5,
    )
    projection = SimpleNamespace(
        plane=Plane(normal=[0.0, 0.0, 1.0], d=1.0, frame="depth"),
        origin_3d_m=np.array([0.0, 0.0, 1.0]),
        basis_u_3d=np.array([1.0, 0.0, 0.0]),
        basis_v_3d=np.array([0.0, 1.0, 0.0]),
    )
    evidence = SimpleNamespace(
        projection=projection,
        rectangle=SimpleNamespace(
            corners_xy_m=((-0.2, -0.125), (0.2, -0.125), (0.2, 0.125), (-0.2, 0.125))
        ),
    )

    output = draw_live_overlay(
        image,
        _base_pose(),
        evidence=evidence,
        color_from_depth=np.eye(4),
        color_intrinsics=intrinsics,
        estimator_latency_ms=5.0,
        cv2_module=cv2,
    )

    assert output.shape == image.shape
    assert len(cv2.polygons) == 1
    polygon, closed = cv2.polygons[0]
    assert closed
    np.testing.assert_array_equal(
        polygon.reshape(-1, 2),
        [[100, 82], [220, 82], [220, 157], [100, 157]],
    )
    assert set(cv2.text) == set(live_overlay_lines(_base_pose(), 5.0))


class _WindowCv2:
    WINDOW_NORMAL = 0
    WND_PROP_FULLSCREEN = 1
    WINDOW_FULLSCREEN = 2

    def __init__(self, key: int) -> None:
        self.key = key
        self.named: list[str] = []
        self.images: list[np.ndarray] = []
        self.destroyed: list[str] = []

    def currentUIFramework(self) -> str:
        return "FAKE"

    def namedWindow(self, name: str, mode: int) -> None:
        self.named.append(name)

    def setWindowProperty(self, name: str, prop: int, value: int) -> None:
        pass

    def imshow(self, name: str, image: np.ndarray) -> None:
        self.images.append(image.copy())

    def waitKey(self, delay_ms: int) -> int:
        return self.key

    def destroyWindow(self, name: str) -> None:
        self.destroyed.append(name)


def test_live_loop_uses_raw_rgb_disables_alignment_and_cleans_up(
    monkeypatch,
) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    frame = make_frame(aligned=False)
    cv2 = _WindowCv2(ord("q"))
    camera_instances = []
    overlay_inputs = []

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            self.stream_config = stream_config
            self.stopped = False
            camera_instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.stopped = True

        def session_metadata(self, **context):
            return metadata

        def capture(self):
            return frame

    class FakeEstimator:
        def __init__(self, intrinsics, calibration, config) -> None:
            self.last_evidence = None

        def estimate(self, depth, **kwargs):
            return _estimate()

    def fake_overlay(image, base_pose, **kwargs):
        overlay_inputs.append((image, base_pose, kwargs))
        return image.copy()

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(realtime, "draw_live_overlay", fake_overlay)

    displayed = run_live_view(
        _calibration(),
        EstimatorConfig(),
        {},
        warmup_frames=7,
    )

    assert displayed == 1
    assert len(camera_instances) == 1
    assert camera_instances[0].stream_config.warmup_frames == 7
    assert camera_instances[0].stream_config.align_color_to_depth is False
    assert camera_instances[0].stopped
    assert overlay_inputs[0][0] is frame.raw_color_bgr
    assert overlay_inputs[0][1] is not None
    assert overlay_inputs[0][2]["estimator_latency_ms"] >= 0.0
    assert len(cv2.images) == 1
    assert cv2.destroyed == ["RB-Y1 Parcel Pose"]


def test_live_view_rejects_headless_opencv_before_opening_camera(
    monkeypatch,
) -> None:
    import parcel_pose.realtime as realtime

    cv2 = _WindowCv2(-1)
    cv2.currentUIFramework = lambda: ""
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)

    with pytest.raises(LiveViewUnavailableError, match="no GUI backend"):
        run_live_view(_calibration(), EstimatorConfig(), {}, max_frames=1)


def test_live_view_rejects_a_camera_that_does_not_match_calibration(
    monkeypatch,
) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    cv2 = _WindowCv2(-1)

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def session_metadata(self, **context):
            return metadata

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)

    with pytest.raises(LiveViewUnavailableError, match="camera serial mismatch"):
        run_live_view(
            _camera_bound_calibration("another-D435"),
            EstimatorConfig(),
            {},
            max_frames=1,
        )


def test_automation_handoff_runs_after_camera_exit_and_closes_once(
    monkeypatch,
) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    frame = make_frame(aligned=False)
    cv2 = _WindowCv2(-1)
    events: list[str] = []

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            pass

        def __enter__(self):
            events.append("camera_enter")
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append("camera_exit")

        def session_metadata(self, **context):
            return metadata

        def capture(self):
            return frame

    class FakeEstimator:
        def __init__(self, intrinsics, calibration, config) -> None:
            self.last_evidence = None

        def estimate(self, depth, **kwargs):
            return _estimate()

    class FakeAutomation:
        def start(self) -> None:
            events.append("automation_start")

        def update(
            self,
            base_pose,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            assert base_pose is not None
            assert np.isfinite(pose_timestamp_s)
            assert np.isfinite(now_s)
            assert pose_timestamp_s <= now_s
            events.append("automation_update")
            return True

        def handoff(self) -> None:
            events.append("handoff")

        def close(self) -> None:
            events.append("automation_close")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(
        realtime,
        "draw_live_overlay",
        lambda image, base_pose, **kwargs: image.copy(),
    )

    assert run_live_view(
        _calibration(),
        EstimatorConfig(),
        {},
        automation=FakeAutomation(),
    ) == 1
    assert events == [
        "camera_enter",
        "automation_start",
        "automation_update",
        "camera_exit",
        "handoff",
        "automation_close",
    ]


def test_user_quit_cancels_automation_without_grasp(monkeypatch) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    frame = make_frame(aligned=False)
    cv2 = _WindowCv2(ord("q"))
    events: list[str] = []

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append("camera_exit")

        def session_metadata(self, **context):
            return metadata

        def capture(self):
            return frame

    class FakeEstimator:
        def __init__(self, intrinsics, calibration, config) -> None:
            self.last_evidence = None

        def estimate(self, depth, **kwargs):
            return _estimate()

    class FakeAutomation:
        def start(self) -> None:
            events.append("start")

        def update(
            self,
            base_pose,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            assert pose_timestamp_s <= now_s
            events.append("update_ready")
            return True

        def handoff(self) -> None:
            events.append("handoff")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(
        realtime,
        "draw_live_overlay",
        lambda image, base_pose, **kwargs: image.copy(),
    )

    assert run_live_view(
        _calibration(),
        EstimatorConfig(),
        {},
        automation=FakeAutomation(),
    ) == 1
    assert events == ["start", "update_ready", "camera_exit", "close"]


def test_live_view_preserves_pre_estimation_pose_timestamp(monkeypatch) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    frame = make_frame(aligned=False)
    cv2 = _WindowCv2(ord("q"))
    observed_times: list[tuple[float, float]] = []
    monotonic_values = iter((10.0, 10.401))

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def session_metadata(self, **context):
            return metadata

        def capture(self):
            return frame

    class FakeEstimator:
        def __init__(self, intrinsics, calibration, config) -> None:
            self.last_evidence = None

        def estimate(self, depth, **kwargs):
            return _estimate()

    class FakeAutomation:
        def start(self) -> None:
            pass

        def update(
            self,
            base_pose,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            observed_times.append((pose_timestamp_s, now_s))
            return False

        def handoff(self) -> None:  # pragma: no cover - q prevents handoff
            raise AssertionError("handoff must not run")

        def close(self) -> None:
            pass

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(realtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        realtime,
        "draw_live_overlay",
        lambda image, base_pose, **kwargs: image.copy(),
    )

    assert run_live_view(
        _calibration(),
        EstimatorConfig(),
        {},
        automation=FakeAutomation(),
    ) == 1
    assert observed_times == [(10.0, 10.401)]


def test_keyboard_interrupt_during_handoff_propagates_after_cleanup(monkeypatch) -> None:
    import parcel_pose.realtime as realtime

    metadata = make_metadata()
    frame = make_frame(aligned=False)
    cv2 = _WindowCv2(-1)
    events: list[str] = []

    class FakeCamera:
        def __init__(self, stream_config) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append("camera_exit")

        def session_metadata(self, **context):
            return metadata

        def capture(self):
            return frame

    class FakeEstimator:
        def __init__(self, intrinsics, calibration, config) -> None:
            self.last_evidence = None

        def estimate(self, depth, **kwargs):
            return _estimate()

    class FakeAutomation:
        def start(self) -> None:
            pass

        def update(
            self,
            base_pose,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            return True

        def handoff(self) -> None:
            events.append("handoff")
            raise KeyboardInterrupt

        def close(self) -> None:
            events.append("close")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(realtime, "_cv2", lambda: cv2)
    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(
        realtime,
        "draw_live_overlay",
        lambda image, base_pose, **kwargs: image.copy(),
    )

    with pytest.raises(KeyboardInterrupt):
        run_live_view(
            _calibration(),
            EstimatorConfig(),
            {},
            automation=FakeAutomation(),
        )
    assert events == ["camera_exit", "handoff", "close"]
