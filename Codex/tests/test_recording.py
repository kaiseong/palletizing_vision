import json

import numpy as np
import pytest

from parcel_pose.models import BoxDimensionPrior, BoxModel, CameraIntrinsics
from parcel_pose.recording import SessionReader, SessionWriter, write_session
from parcel_pose.session import (
    FactoryExtrinsics,
    RecordedFrame,
    SessionMetadata,
    SessionValidationError,
    StreamProfile,
)


def make_metadata(*, width=4, height=3, include_dimension_prior=False):
    depth_intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fps=30,
        fx=320.1,
        fy=321.2,
        cx=1.7,
        cy=1.3,
        distortion_model="brown_conrady",
        # D400 raw depth is rectified and reports zero coefficients.  Keep
        # non-zero distortion coverage on the independent RGB profile below.
        coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    color_intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fps=30,
        fx=330.1,
        fy=331.2,
        cx=1.8,
        cy=1.4,
        distortion_model="inverse_brown_conrady",
        coeffs=(-0.1, 0.01, 0.0, 0.0, 0.0),
    )
    depth_to_color = FactoryExtrinsics(
        target_stream="color",
        source_stream="depth",
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation_m=(0.015, 0.0, 0.0),
    )
    color_to_depth = FactoryExtrinsics(
        target_stream="depth",
        source_stream="color",
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation_m=(-0.015, 0.0, 0.0),
    )
    prior = (
        BoxDimensionPrior(
            samples_m=(
                (0.400, 0.253, 0.160),
                (0.395, 0.252, 0.164),
                (0.401, 0.256, 0.156),
            ),
            source="manual_test_measurements",
        )
        if include_dimension_prior
        else None
    )
    box_model = (
        BoxModel(
            long_m=prior.representative_m[0],
            short_m=prior.representative_m[1],
            height_m=prior.representative_m[2],
            model_id="test_population_median",
        )
        if prior is not None
        else BoxModel()
    )
    return SessionMetadata(
        camera_serial="D435-test-001",
        camera_firmware="5.16.test",
        usb_type="3.2",
        depth_scale_m=0.001,
        depth_profile=StreamProfile("depth", "z16", depth_intrinsics),
        color_profile=StreamProfile("color", "bgr8", color_intrinsics),
        depth_to_color=depth_to_color,
        color_to_depth=color_to_depth,
        capture_options={
            "exposure": 8500.0,
            "gain": 16.0,
            "emitter_enabled": 1.0,
            "laser_power": 150.0,
            "visual_preset": 3.0,
        },
        robot_state={
            "head_joints": [0.0, 0.0],
            "torso_joints": [0.0] * 6,
            "base_state": {"stationary": True},
            "T_base_from_head": np.eye(4).tolist(),
        },
        nominal_transform={
            "target_frame": "link_head_2",
            "source_frame": "d435_color_optical_frame",
            "translation_m": [0.049, -0.0115, 0.057],
            "euler_zyx_deg": [-90.0, 0.0, -90.0],
            "euler_input_order": ["roll", "pitch", "yaw"],
            "rotation_formula": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        },
        table={
            "plane": {"normal": [0.0, -1.0, 0.0], "d": -0.7, "frame": "depth"},
            "config_schema_version": 1,
        },
        box_model=box_model,
        box_dimension_prior=prior,
        annotation={
            "session_id": "test-session",
            "yaw_family": "0",
            "crop_sides": ["right"],
            "lighting": "lab",
            "box_moved_between_bursts": True,
        },
    )


def make_frame(index=0, *, width=4, height=3, aligned=True):
    depth = np.arange(width * height, dtype=np.uint16).reshape(height, width) + index
    color = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3) + index
    return RecordedFrame(
        raw_depth_z16=depth,
        raw_color_bgr=color,
        color_on_depth_bgr=np.flip(color, axis=1).copy() if aligned else None,
        depth_timestamp_ms=1000.0 + index * 33.3,
        color_timestamp_ms=1000.5 + index * 33.3,
        depth_frame_number=10 + index,
        color_frame_number=20 + index,
        hardware_timestamp_ms=900.0 + index * 33.3,
        system_timestamp_ns=123456789 + index,
        frame_metadata={"depth_timestamp_domain": "hardware_clock"},
    )


def test_session_metadata_exact_json_roundtrip():
    original = make_metadata()
    restored = SessionMetadata.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()


def test_session_metadata_preserves_dimension_prior_and_representative():
    original = make_metadata(include_dimension_prior=True)

    payload = json.loads(json.dumps(original.to_dict()))
    restored = SessionMetadata.from_dict(payload)

    assert restored.box_dimension_prior == original.box_dimension_prior
    assert restored.box_model == original.box_model
    assert payload["box_dimension_prior_m"]["sample_count"] == 3


def test_session_metadata_wraps_malformed_dimension_summary() -> None:
    payload = make_metadata(include_dimension_prior=True).to_dict()
    del payload["box_dimension_prior_m"]["mean"]["height"]

    with pytest.raises(SessionValidationError, match="invalid session metadata"):
        SessionMetadata.from_dict(payload)


def test_raw_arrays_and_metadata_roundtrip(tmp_path):
    metadata = make_metadata()
    frames = [make_frame(0), make_frame(1, aligned=False)]
    write_session(tmp_path / "session", metadata, frames)
    reader = SessionReader(tmp_path / "session")
    assert reader.complete
    assert reader.metadata.to_dict() == metadata.to_dict()
    restored = list(reader)
    assert len(restored) == 2
    for expected, actual in zip(frames, restored, strict=True):
        assert actual.raw_depth_z16.dtype == np.uint16
        assert actual.raw_color_bgr.dtype == np.uint8
        np.testing.assert_array_equal(actual.raw_depth_z16, expected.raw_depth_z16)
        np.testing.assert_array_equal(actual.raw_color_bgr, expected.raw_color_bgr)
        if expected.color_on_depth_bgr is None:
            assert actual.color_on_depth_bgr is None
        else:
            np.testing.assert_array_equal(actual.color_on_depth_bgr, expected.color_on_depth_bgr)
        assert actual.depth_timestamp_ms == expected.depth_timestamp_ms
        assert actual.frame_metadata == expected.frame_metadata
    np.testing.assert_allclose(restored[0].depth_m(metadata.depth_scale_m), frames[0].raw_depth_z16 * 0.001)


def test_raw_streams_are_authoritative_when_aligned_is_absent(tmp_path):
    with SessionWriter(tmp_path / "session", make_metadata()) as writer:
        writer.add_frame(make_frame(aligned=False))
    [frame] = SessionReader(tmp_path / "session")
    assert frame.color_on_depth_bgr is None
    assert frame.raw_depth_z16.size > 0
    assert frame.raw_color_bgr.size > 0


def test_bad_or_missing_schema_field_is_actionable():
    payload = make_metadata().to_dict()
    del payload["factory_extrinsics"]["depth_to_color"]
    with pytest.raises(SessionValidationError, match="depth_to_color"):
        SessionMetadata.from_dict(payload)


def test_checksum_corruption_is_detected(tmp_path):
    write_session(tmp_path / "session", make_metadata(), [make_frame()])
    frame_path = tmp_path / "session" / "frames" / "000000.npz"
    frame_path.write_bytes(frame_path.read_bytes() + b"corruption")
    with pytest.raises(SessionValidationError, match="checksum mismatch"):
        list(SessionReader(tmp_path / "session"))


def test_overwrite_refuses_non_session_directory(tmp_path):
    destination = tmp_path / "important_project"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        SessionWriter(destination, make_metadata(), overwrite=True)

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_overwrite_replaces_only_an_existing_recording_session(tmp_path):
    destination = tmp_path / "session"
    write_session(destination, make_metadata(), [make_frame(0)])

    with SessionWriter(destination, make_metadata(), overwrite=True) as writer:
        writer.add_frame(make_frame(1))

    [restored] = SessionReader(destination)
    assert restored.depth_frame_number == 11


def test_manifest_frame_path_cannot_escape_session(tmp_path):
    destination = tmp_path / "session"
    write_session(destination, make_metadata(), [make_frame(0)])
    manifest_path = destination / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["frames"][0]["file"] = "../../outside.npz"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SessionValidationError, match="inside the session"):
        list(SessionReader(destination))
