from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import math
from pathlib import Path

import pytest

from parcel_pose.benchmark import (
    _environment_fingerprint,
    _line_angle_error_deg,
    _signature_digest,
    _validate_output_path,
    compare_reports,
)


def _public_pose(index: int) -> dict[str, object]:
    reasons = ["absolute_base_transform_unvalidated"]
    return {
        "timestamp_ms": 1000.0 + index,
        "frame_id": 100 + index,
        "frame": "table_plane",
        "box_model_m": {"long": 0.4, "short": 0.25, "height": 0.15},
        "center_plane_xy_m": [0.1, -0.1],
        "center_depth_m": [0.7, 0.0, 0.905],
        "yaw_rad": math.radians(89.98),
        "yaw_mod_180_deg": 89.98,
        "canonical_reference_deg": 90,
        "canonical_residual_deg": -0.02,
        "classification_margin_deg": 44.98,
        "long_axis_plane_xy": [0.0, 1.0],
        "short_axis_plane_xy": [-1.0, 0.0],
        "observability": {"yaw": "constrained", "reference": "constrained"},
        "center_feasible_set": {"center_short": [-0.01, 0.01]},
        "calibration": {
            "state": "plane_calibrated_partial",
            "base_registration": "nominal_unverified",
            "base_registration_valid": False,
            "absolute_base_validated": False,
        },
        "confidence": {
            "geometry_valid": True,
            "full_pose_valid": True,
            "absolute_base_pose_valid": False,
            "per_field": {
                "center_long": 0.90,
                "center_short": 0.85,
                "yaw": 0.90,
                "reference": 0.90,
            },
            "reasons": reasons,
        },
    }


def _frame(index: int) -> dict[str, object]:
    return {
        "frame_index": index,
        "frame_id": 100 + index,
        "timestamp_ms": 1000.0 + index,
        "full_pose_valid": True,
        "absolute_valid": False,
        "base_registration": "nominal_unverified",
        "canonical_reference_deg": 90,
        "observability": {"yaw": "constrained", "reference": "constrained"},
        "reasons": ["absolute_base_transform_unvalidated"],
        "yaw_confidence": 0.9,
        "public_pose": _public_pose(index),
        "box_center_base_xyz_m": [0.7, 0.0, 0.83],
        "top_center_base_xyz_m": [0.7, 0.0, 0.905],
        "yaw_base_signed_deg": 89.98,
    }


def _report() -> dict[str, object]:
    environment = {"machine": "test", "cpu_affinity": [0]}
    frames = [_frame(0), _frame(1)]
    signature = _signature_digest(frames)
    return {
        "schema_version": 1,
        "kind": "parcel_pose_estimator_benchmark",
        "fixture": {
            "manifest_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "config_sha256": "c" * 64,
        },
        "settings": {
            "warmup_passes": 1,
            "repeats": 5,
            "estimator_lifecycle": "one instance",
            "timed_scope": "estimate only",
            "depth_preloaded": True,
            "gc_disabled_during_measured_pass": True,
            "opencv_opencl_enabled": False,
        },
        "environment": environment,
        "environment_fingerprint": _environment_fingerprint(environment),
        "source_provenance": {"package_python_sha256": "d" * 64},
        "behavior": {
            "frame_count": 2,
            "full_pose_valid_frames": 2,
            "abstained_frames": 0,
            "absolute_base_pose_frames": 0,
            "signature_sha256": signature,
            "repeat_signature_sha256": [signature] * 5,
            "repeat_outputs_identical": True,
            "frames": frames,
        },
        "latency": {
            "frame_count": 2,
            "repeat_count": 5,
            "per_frame_median_ms": {"p50": 100.0, "p95": 120.0},
            "per_pass": {
                "estimator_total_ms": [222.0] * 5,
                "wall_total_ms": [223.0] * 5,
                "p95_ms": [120.0] * 5,
                "throughput_fps": [9.0] * 5,
                "median_throughput_fps": 9.0,
                "p95_mad_ratio": 0.0,
            },
        },
        "stability": {
            "outputs_identical": True,
            "p95_mad_ratio_within_3_percent": True,
        },
    }


def _refresh_behavior_signature(report: dict[str, object]) -> None:
    behavior = report["behavior"]
    frames = behavior["frames"]
    signature = _signature_digest(frames)
    behavior["signature_sha256"] = signature
    behavior["repeat_signature_sha256"] = [signature] * report["settings"][
        "repeats"
    ]


def test_line_angle_error_wraps_at_180_degrees() -> None:
    assert _line_angle_error_deg(89.98, -89.98) == pytest.approx(0.04)


def test_compare_reports_passes_equivalent_faster_candidate() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["latency"]["per_frame_median_ms"] = {"p50": 90.0, "p95": 105.0}
    candidate["latency"]["per_pass"]["median_throughput_fps"] = 10.0
    candidate["latency"]["per_pass"]["throughput_fps"] = [10.0] * 5
    candidate["behavior"]["frames"][0]["box_center_base_xyz_m"][0] += 0.00005
    candidate["behavior"]["frames"][0]["yaw_base_signed_deg"] = -89.98
    _refresh_behavior_signature(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "pass"
    assert comparison["behavior_pass"] is True
    assert comparison["speed_pass"] is True


def test_compare_reports_fails_validity_or_speed_regression() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["behavior"]["frames"][1]["full_pose_valid"] = False
    candidate["behavior"]["frames"][1]["public_pose"]["confidence"][
        "full_pose_valid"
    ] = False
    candidate["behavior"]["full_pose_valid_frames"] = 1
    candidate["behavior"]["abstained_frames"] = 1
    _refresh_behavior_signature(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "fail"
    assert comparison["behavior_pass"] is False
    assert comparison["speed_pass"] is False
    assert "frame_1:full_pose_valid" in comparison["structural_errors"]


def test_compare_reports_is_inconclusive_across_environments() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["environment"] = {"machine": "different", "cpu_affinity": [0]}
    candidate["environment_fingerprint"] = _environment_fingerprint(
        candidate["environment"]
    )

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert comparison["comparable"] is False


def test_compare_reports_rejects_malformed_fixture_hashes() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    del candidate["fixture"]["config_sha256"]

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert "candidate:fixture.config_sha256" in comparison["report_validation_errors"]


def test_compare_reports_catches_omitted_public_safety_fields() -> None:
    baseline = _report()
    mutations = (
        lambda frame: frame["public_pose"]["confidence"].update(
            {"geometry_valid": False}
        ),
        lambda frame: frame["public_pose"]["calibration"].update(
            {"base_registration_valid": True}
        ),
        lambda frame: frame["public_pose"]["confidence"]["per_field"].update(
            {"center_long": 0.70}
        ),
        lambda frame: frame["public_pose"]["center_feasible_set"].update(
            {"center_short": [-0.01, 0.012]}
        ),
    )

    for mutate in mutations:
        candidate = deepcopy(baseline)
        candidate["latency"]["per_frame_median_ms"] = {
            "p50": 90.0,
            "p95": 105.0,
        }
        candidate["latency"]["per_pass"]["median_throughput_fps"] = 10.0
        candidate["latency"]["per_pass"]["throughput_fps"] = [10.0] * 5
        mutate(candidate["behavior"]["frames"][0])
        _refresh_behavior_signature(candidate)

        comparison = compare_reports(baseline, candidate)

        assert comparison["status"] in {"fail", "inconclusive"}
        assert comparison["behavior_pass"] is False


def test_compare_reports_rejects_internally_inconsistent_report() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["behavior"]["full_pose_valid_frames"] = 1

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert (
        "candidate:behavior.full_pose_valid_frames"
        in comparison["report_validation_errors"]
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda report: report["latency"]["per_pass"].update(
            {
                "p95_ms": [10.0, 20.0, 30.0, 40.0, 50.0],
                "p95_mad_ratio": 99.0,
            }
        ),
        lambda report: report["latency"]["per_pass"].update(
            {"median_throughput_fps": 99.0}
        ),
    ),
)
def test_compare_reports_recomputes_timing_integrity(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    mutate(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert "candidate:latency" in comparison["report_validation_errors"]


def test_compare_reports_recomputes_timing_stability_boolean() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    p95_values = [100.0, 110.0, 120.0, 130.0, 140.0]
    candidate["latency"]["per_pass"]["p95_ms"] = p95_values
    candidate["latency"]["per_pass"]["p95_mad_ratio"] = 10.0 / 120.0

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert (
        "candidate:stability.p95_mad_ratio_consistency"
        in comparison["report_validation_errors"]
    )


def test_compare_reports_rejects_invalid_timing_scope() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["settings"]["timed_scope"] = None

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert "candidate:settings.timed_scope" in comparison["report_validation_errors"]


def test_compare_reports_catches_frame_yaw_confidence_change() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["behavior"]["frames"][0]["yaw_confidence"] = 0.5
    candidate["behavior"]["frames"][0]["public_pose"]["confidence"][
        "per_field"
    ]["yaw"] = 0.5
    _refresh_behavior_signature(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "fail"
    assert comparison["behavior_pass"] is False


def test_compare_reports_catches_unknown_future_public_field() -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    baseline["behavior"]["frames"][0]["public_pose"]["future_safety_state"] = "safe"
    candidate["behavior"]["frames"][0]["public_pose"]["future_safety_state"] = "unsafe"
    _refresh_behavior_signature(baseline)
    _refresh_behavior_signature(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "fail"
    assert "frame_0:public_pose.future_safety_state" in comparison[
        "structural_errors"
    ]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda pose: pose.update({"center_feasible_set": "bad"}),
        lambda pose: pose.update({"center_feasible_set": {"center_short": None}}),
        lambda pose: pose.update({"center_plane_xy_m": [0.0, "bad"]}),
        lambda pose: pose.update({"yaw_rad": "bad"}),
        lambda pose: pose["confidence"]["per_field"].update({"yaw": "bad"}),
    ),
)
def test_compare_reports_rejects_malformed_public_numeric_fields(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    baseline = _report()
    candidate = deepcopy(baseline)
    mutate(candidate["behavior"]["frames"][0]["public_pose"])
    _refresh_behavior_signature(candidate)

    comparison = compare_reports(baseline, candidate)

    assert comparison["status"] == "inconclusive"
    assert any(
        error.startswith("candidate:behavior.frames[0].public_pose")
        for error in comparison["report_validation_errors"]
    )


def test_benchmark_output_cannot_alias_inputs(tmp_path: Path) -> None:
    source = tmp_path / "calibration.json"

    with pytest.raises(ValueError, match="distinct from every input"):
        _validate_output_path(source, [source])

    with pytest.raises(ValueError, match=".json suffix"):
        _validate_output_path(tmp_path / "candidate.txt", [source])
