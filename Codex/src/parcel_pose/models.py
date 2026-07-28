"""Typed physical and estimator contracts.

All rigid transforms use active column-vector, target-from-source semantics.
Lengths are metres and internal angles are radians unless a field name ends in
``_deg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tuple_of_floats(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(_finite_float(v, name) for v in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


def _matrix4(value: Any | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    result = matrix.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"
    coeffs: tuple[float, ...] = ()
    fps: int = 30

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsic width and height must be positive")
        if self.fps <= 0:
            raise ValueError("intrinsic fps must be positive")
        for name in ("fx", "fy"):
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "cx", _finite_float(self.cx, "cx"))
        object.__setattr__(self, "cy", _finite_float(self.cy, "cy"))
        object.__setattr__(self, "coeffs", tuple(float(v) for v in self.coeffs))

    @property
    def ppx(self) -> float:
        """RealSense-compatible alias for the principal point x coordinate."""

        return self.cx

    @property
    def ppy(self) -> float:
        """RealSense-compatible alias for the principal point y coordinate."""

        return self.cy

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_model": self.distortion_model,
            "coeffs": list(self.coeffs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraIntrinsics":
        try:
            return cls(
                width=int(value["width"]),
                height=int(value["height"]),
                fps=int(value.get("fps", 30)),
                fx=float(value["fx"]),
                fy=float(value["fy"]),
                cx=float(value.get("cx", value.get("ppx"))),
                cy=float(value.get("cy", value.get("ppy"))),
                distortion_model=str(value.get("distortion_model", "none")),
                coeffs=tuple(float(v) for v in value.get("coeffs", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid camera intrinsics: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Plane:
    """Unit-normal plane represented as ``normal dot point = d``."""

    normal: FloatArray
    d: float
    frame: str = "depth"

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("plane normal must be a finite length-3 vector")
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("plane normal must be non-zero")
        d = _finite_float(self.d, "plane d") / norm
        normalized = (normal / norm).copy()
        normalized.setflags(write=False)
        object.__setattr__(self, "normal", normalized)
        object.__setattr__(self, "d", d)

    def signed_distance(self, points: Any) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("points must end in a length-3 coordinate")
        return values @ self.normal - self.d

    def point_on_plane(self) -> FloatArray:
        return self.normal * self.d

    def to_dict(self) -> dict[str, Any]:
        return {"normal": self.normal.tolist(), "d": self.d, "frame": self.frame}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Plane":
        try:
            return cls(
                normal=np.asarray(value["normal"], dtype=np.float64),
                d=float(value["d"]),
                frame=str(value.get("frame", "depth")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid plane: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BoxModel:
    long_m: float = 0.400
    short_m: float = 0.253
    height_m: float = 0.160
    model_id: str = "parcel_measured_prior_400x253x160_v1"

    def __post_init__(self) -> None:
        for name in ("long_m", "short_m", "height_m"):
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.long_m <= self.short_m:
            raise ValueError("long_m must be greater than short_m")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "long": self.long_m,
            "short": self.short_m,
            "height": self.height_m,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoxModel":
        try:
            dimension_keys = {
                "long_m",
                "long",
                "short_m",
                "short",
                "height_m",
                "height",
            }
            dimensions_explicit = any(key in value for key in dimension_keys)
            long_m = float(value.get("long_m", value.get("long", 0.400)))
            short_m = float(value.get("short_m", value.get("short", 0.253)))
            height_m = float(value.get("height_m", value.get("height", 0.160)))
            if "model_id" in value:
                model_id = str(value["model_id"])
            elif not dimensions_explicit:
                model_id = "parcel_measured_prior_400x253x160_v1"
            else:
                dimensions_mm = (1000.0 * long_m, 1000.0 * short_m, 1000.0 * height_m)
                dimension_token = "x".join(f"{dimension:g}" for dimension in dimensions_mm)
                model_id = f"parcel_configured_{dimension_token}"
            return cls(
                long_m=long_m,
                short_m=short_m,
                height_m=height_m,
                model_id=model_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid box model: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BoxDimensionPrior:
    """Measured population evidence for one folded parcel family.

    Samples are stored instead of only derived summary values so the physical
    evidence remains auditable.  The estimator uses the component-wise median
    as its fixed metric model; it does not freely resize a cropped rectangle
    from frame to frame.
    """

    samples_m: tuple[tuple[float, float, float], ...]
    source: str
    strategy: str = "componentwise_population_median"

    def __post_init__(self) -> None:
        normalized: list[tuple[float, float, float]] = []
        for index, sample in enumerate(self.samples_m):
            values = _tuple_of_floats(sample, 3, f"samples_m[{index}]")
            long_m, short_m, height_m = values
            if not long_m > short_m > 0.0 or height_m <= 0.0:
                raise ValueError(
                    "each dimension sample must satisfy long_m > short_m > 0 "
                    "and height_m > 0"
                )
            # Values above two metres almost always mean millimetres were
            # accidentally supplied to a metre-valued API.
            if max(values) > 2.0:
                raise ValueError("dimension samples must be expressed in metres")
            normalized.append(values)
        if not normalized:
            raise ValueError("box dimension prior requires at least one sample")
        source = str(self.source).strip()
        strategy = str(self.strategy).strip()
        if not source:
            raise ValueError("box dimension prior source cannot be empty")
        if strategy != "componentwise_population_median":
            raise ValueError(
                "box dimension prior strategy must be "
                "'componentwise_population_median'"
            )
        object.__setattr__(self, "samples_m", tuple(normalized))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "strategy", strategy)

    @property
    def sample_count(self) -> int:
        return len(self.samples_m)

    def _array(self) -> NDArray[np.float64]:
        return np.asarray(self.samples_m, dtype=np.float64)

    @property
    def representative_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.median(self._array(), axis=0))

    @property
    def mean_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.mean(self._array(), axis=0))

    @property
    def sample_std_m(self) -> tuple[float, float, float]:
        if self.sample_count == 1:
            return (0.0, 0.0, 0.0)
        return tuple(float(value) for value in np.std(self._array(), axis=0, ddof=1))

    @property
    def observed_min_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.min(self._array(), axis=0))

    @property
    def observed_max_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.max(self._array(), axis=0))

    @staticmethod
    def _named(values: Sequence[float]) -> dict[str, float]:
        return dict(zip(("long", "short", "height"), (float(v) for v in values), strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "strategy": self.strategy,
            "sample_count": self.sample_count,
            "representative": self._named(self.representative_m),
            "mean": self._named(self.mean_m),
            "sample_std": self._named(self.sample_std_m),
            "observed_min": self._named(self.observed_min_m),
            "observed_max": self._named(self.observed_max_m),
            "samples": [self._named(sample) for sample in self.samples_m],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoxDimensionPrior":
        def sample_tuple(sample: Any, index: int) -> tuple[float, float, float]:
            if isinstance(sample, Mapping):
                try:
                    return (
                        float(sample["long"]),
                        float(sample["short"]),
                        float(sample["height"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid dimension sample {index}: {exc}") from exc
            try:
                return tuple(float(item) for item in sample)  # type: ignore[arg-type, return-value]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid dimension sample {index}: {exc}") from exc

        try:
            raw_samples = value["samples"]
            if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
                raise ValueError("samples must be an array")
            prior = cls(
                samples_m=tuple(
                    sample_tuple(sample, index)
                    for index, sample in enumerate(raw_samples)
                ),
                source=str(value["source"]),
                strategy=str(
                    value.get("strategy", "componentwise_population_median")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("invalid dimension sample"):
                raise
            raise ValueError(f"invalid box dimension prior: {exc}") from exc

        # Derived fields are optional on input.  When present (for example in
        # a recording manifest), verify them so stale summaries cannot silently
        # disagree with the preserved raw measurements.
        expected = prior.to_dict()
        for name in (
            "sample_count",
            "representative",
            "mean",
            "sample_std",
            "observed_min",
            "observed_max",
        ):
            if name not in value:
                continue
            if name == "sample_count":
                if int(value[name]) != expected[name]:
                    raise ValueError("box dimension prior sample_count disagrees with samples")
                continue
            actual_mapping = value[name]
            if not isinstance(actual_mapping, Mapping):
                raise ValueError(f"box dimension prior {name} must be an object")
            try:
                actual = np.asarray(
                    [actual_mapping[key] for key in ("long", "short", "height")],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"box dimension prior {name} must contain numeric "
                    "long, short, and height fields"
                ) from exc
            reference = np.asarray(
                [expected[name][key] for key in ("long", "short", "height")],
                dtype=np.float64,
            )
            if not np.allclose(actual, reference, rtol=0.0, atol=1e-12):
                raise ValueError(f"box dimension prior {name} disagrees with samples")
        return prior


class CalibrationState(str, Enum):
    NOMINAL = "nominal"
    PLANE_CALIBRATED_PARTIAL = "plane_calibrated_partial"
    BASE_VALIDATED = "base_validated"


class ObservabilityState(str, Enum):
    BOTH_EDGES = "both_edges"
    ONE_EDGE_INFERRED = "one_edge_inferred"
    UNDERCONSTRAINED = "underconstrained"
    CONSTRAINED = "constrained"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Calibration:
    state: CalibrationState = CalibrationState.NOMINAL
    table_plane: Plane | None = None
    T_base_from_head: FloatArray | None = None
    T_head_from_color: FloatArray | None = None
    E_color_from_depth: FloatArray | None = None
    T_head_from_depth: FloatArray | None = None
    base_frame: str = "base"
    head_frame: str = "link_head_2"
    color_frame: str = "d435_color_optical_frame"
    depth_frame: str = "d435_depth_optical_frame"
    notes: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", CalibrationState(self.state))
        for name in (
            "T_base_from_head",
            "T_head_from_color",
            "E_color_from_depth",
            "T_head_from_depth",
        ):
            object.__setattr__(self, name, _matrix4(getattr(self, name), name))
        object.__setattr__(self, "notes", tuple(str(v) for v in self.notes))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def T_base_from_depth(self) -> FloatArray | None:
        if self.T_base_from_head is None:
            return None
        if self.T_head_from_depth is not None:
            result = self.T_base_from_head @ self.T_head_from_depth
        elif self.T_head_from_color is not None and self.E_color_from_depth is not None:
            result = self.T_base_from_head @ self.T_head_from_color @ self.E_color_from_depth
        else:
            return None
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    @property
    def has_base_transform_chain(self) -> bool:
        return self.T_base_from_depth is not None

    @property
    def absolute_base_validated(self) -> bool:
        return self.state is CalibrationState.BASE_VALIDATED and self.has_base_transform_chain

    def to_dict(self) -> dict[str, Any]:
        def matrix(value: FloatArray | None) -> list[list[float]] | None:
            return None if value is None else value.tolist()

        return {
            "state": self.state.value,
            "table_plane": None if self.table_plane is None else self.table_plane.to_dict(),
            "T_base_from_head": matrix(self.T_base_from_head),
            "T_head_from_color": matrix(self.T_head_from_color),
            "E_color_from_depth": matrix(self.E_color_from_depth),
            "T_head_from_depth": matrix(self.T_head_from_depth),
            "frames": {
                "base": self.base_frame,
                "head": self.head_frame,
                "color": self.color_frame,
                "depth": self.depth_frame,
            },
            "notes": list(self.notes),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Calibration":
        frames = value.get("frames", {})
        if not isinstance(frames, Mapping):
            raise ValueError("calibration frames must be a mapping")
        plane_value = value.get("table_plane")
        return cls(
            state=CalibrationState(value.get("state", CalibrationState.NOMINAL.value)),
            table_plane=None if plane_value is None else Plane.from_dict(plane_value),
            T_base_from_head=value.get("T_base_from_head"),
            T_head_from_color=value.get("T_head_from_color"),
            E_color_from_depth=value.get("E_color_from_depth"),
            T_head_from_depth=value.get("T_head_from_depth"),
            base_frame=str(frames.get("base", "base")),
            head_frame=str(frames.get("head", "link_head_2")),
            color_frame=str(frames.get("color", "d435_color_optical_frame")),
            depth_frame=str(frames.get("depth", "d435_depth_optical_frame")),
            notes=tuple(str(v) for v in value.get("notes", ())),
            diagnostics=dict(value.get("diagnostics", {})),
        )


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    box_model: BoxModel = field(default_factory=BoxModel)
    box_dimension_prior: BoxDimensionPrior | None = None
    min_depth_m: float = 0.20
    max_depth_m: float = 2.00
    top_plane_tolerance_m: float = 0.020
    rectangle_containment_tolerance_m: float = 0.010
    border_margin_px: int = 3
    edge_band_m: float = 0.015
    min_points: int = 120
    max_points: int = 6_000
    coarse_angle_step_deg: float = 2.0
    fine_angle_step_deg: float = 0.2
    center_search_step_m: float = 0.005
    outlier_quantile: float = 0.95
    min_candidate_margin: float = 0.03
    min_side_span: float = 0.35
    random_seed: int = 0
    workspace_xy_m: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_depth_m",
            "max_depth_m",
            "top_plane_tolerance_m",
            "rectangle_containment_tolerance_m",
            "edge_band_m",
            "coarse_angle_step_deg",
            "fine_angle_step_deg",
            "center_search_step_m",
            "outlier_quantile",
            "min_candidate_margin",
            "min_side_span",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
        if (
            self.top_plane_tolerance_m <= 0.0
            or self.rectangle_containment_tolerance_m <= 0.0
            or self.edge_band_m <= 0.0
        ):
            raise ValueError(
                "plane tolerance, rectangle containment tolerance, and edge band "
                "must be positive"
            )
        if self.min_points < 1 or self.max_points < self.min_points:
            raise ValueError("point limits must satisfy 1 <= min_points <= max_points")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px cannot be negative")
        if not 0.0 < self.outlier_quantile <= 1.0:
            raise ValueError("outlier_quantile must be in (0, 1]")
        if not 0.0 <= self.min_side_span <= 1.0:
            raise ValueError("min_side_span must be in [0, 1]")
        if self.workspace_xy_m is not None:
            workspace = _tuple_of_floats(self.workspace_xy_m, 4, "workspace_xy_m")
            if workspace[0] >= workspace[1] or workspace[2] >= workspace[3]:
                raise ValueError("workspace must be (xmin, xmax, ymin, ymax)")
            object.__setattr__(self, "workspace_xy_m", workspace)
        if self.box_dimension_prior is not None:
            if not isinstance(self.box_dimension_prior, BoxDimensionPrior):
                raise ValueError("box_dimension_prior must be a BoxDimensionPrior")
            representative = np.asarray(
                self.box_dimension_prior.representative_m,
                dtype=np.float64,
            )
            configured = np.asarray(
                (
                    self.box_model.long_m,
                    self.box_model.short_m,
                    self.box_model.height_m,
                ),
                dtype=np.float64,
            )
            if not np.allclose(configured, representative, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "box_model must equal the component-wise median of "
                    "box_dimension_prior samples"
                )
            maximum_height_delta = float(
                np.max(
                    np.abs(
                        np.asarray(
                            (
                                self.box_dimension_prior.observed_min_m[2],
                                self.box_dimension_prior.observed_max_m[2],
                            )
                        )
                        - self.box_model.height_m
                    )
                )
            )
            if self.top_plane_tolerance_m < maximum_height_delta:
                raise ValueError(
                    "top_plane_tolerance_m does not cover the measured height range"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EstimatorConfig":
        fields = dict(value)
        if "box_model" in fields and not isinstance(fields["box_model"], BoxModel):
            fields["box_model"] = BoxModel.from_dict(fields["box_model"])
        if "box_dimension_prior" in fields and not isinstance(
            fields["box_dimension_prior"], BoxDimensionPrior
        ):
            fields["box_dimension_prior"] = BoxDimensionPrior.from_dict(
                fields["box_dimension_prior"]
            )
        if "workspace_xy_m" in fields and fields["workspace_xy_m"] is not None:
            fields["workspace_xy_m"] = tuple(fields["workspace_xy_m"])
        return cls(**fields)

    @classmethod
    def from_root_config(cls, value: Mapping[str, Any]) -> "EstimatorConfig":
        """Build one estimator config from the repository-level JSON schema."""

        fields = dict(value.get("estimator", {}))
        fields["box_model"] = BoxModel.from_dict(value.get("box_model_m", {}))
        prior = value.get("box_dimension_prior_m")
        if prior is not None:
            if not isinstance(prior, Mapping):
                raise ValueError("box_dimension_prior_m must be an object")
            fields["box_dimension_prior"] = BoxDimensionPrior.from_dict(prior)
        return cls.from_dict(fields)


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """A perception result with independent geometry/calibration validity."""

    timestamp_ms: float = 0.0
    frame_id: int = 0
    frame: str = "depth"
    box_model: BoxModel = field(default_factory=BoxModel)
    center_plane_xy_m: tuple[float, float] | None = None
    center_depth_m: tuple[float, float, float] | None = None
    center_base_xy_m: tuple[float, float] | None = None
    top_center_base_xyz_m: tuple[float, float, float] | None = None
    box_center_base_xyz_m: tuple[float, float, float] | None = None
    yaw_rad: float | None = None
    yaw_mod_180_deg: float | None = None
    canonical_reference_deg: int | None = None
    canonical_residual_deg: float | None = None
    classification_margin_deg: float | None = None
    long_axis_plane_xy: tuple[float, float] | None = None
    short_axis_plane_xy: tuple[float, float] | None = None
    long_axis_base_xy: tuple[float, float] | None = None
    short_axis_base_xy: tuple[float, float] | None = None
    observability: Mapping[str, str] = field(default_factory=dict)
    feasible_set: Mapping[str, Any] | None = None
    per_field_confidence: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    calibration_state: CalibrationState = CalibrationState.NOMINAL
    base_registration: str = "unavailable"
    geometry_valid: bool = False
    full_pose_valid: bool = False
    base_registration_valid: bool = False
    absolute_valid: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", _finite_float(self.timestamp_ms, "timestamp_ms"))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "calibration_state", CalibrationState(self.calibration_state))
        for name, length in (
            ("center_plane_xy_m", 2),
            ("center_depth_m", 3),
            ("center_base_xy_m", 2),
            ("top_center_base_xyz_m", 3),
            ("box_center_base_xyz_m", 3),
            ("long_axis_plane_xy", 2),
            ("short_axis_plane_xy", 2),
            ("long_axis_base_xy", 2),
            ("short_axis_base_xy", 2),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _tuple_of_floats(value, length, name))
        for name in (
            "yaw_rad",
            "yaw_mod_180_deg",
            "canonical_residual_deg",
            "classification_margin_deg",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        if self.canonical_reference_deg not in (None, 0, 90):
            raise ValueError("canonical_reference_deg must be 0, 90, or None")
        if self.absolute_valid and not (
            self.geometry_valid
            and self.full_pose_valid
            and self.base_registration_valid
            and self.calibration_state is CalibrationState.BASE_VALIDATED
        ):
            raise ValueError(
                "absolute_valid requires a full geometry result and base_validated registration"
            )
        object.__setattr__(self, "observability", dict(self.observability))
        object.__setattr__(
            self,
            "per_field_confidence",
            {str(k): float(v) for k, v in self.per_field_confidence.items()},
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(str(v) for v in self.reasons)))

    @property
    def absolute_base_pose_valid(self) -> bool:
        return self.absolute_valid


__all__ = [
    "BoxDimensionPrior",
    "BoxModel",
    "Calibration",
    "CalibrationState",
    "CameraIntrinsics",
    "EstimatorConfig",
    "ObservabilityState",
    "Plane",
    "PoseEstimate",
]
