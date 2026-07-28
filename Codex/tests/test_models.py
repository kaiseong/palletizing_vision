from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from parcel_pose.models import BoxDimensionPrior, BoxModel, EstimatorConfig


SAMPLES = (
    (0.400, 0.253, 0.160),
    (0.395, 0.252, 0.164),
    (0.395, 0.254, 0.164),
    (0.400, 0.256, 0.161),
    (0.401, 0.252, 0.159),
    (0.401, 0.255, 0.156),
    (0.399, 0.253, 0.160),
    (0.400, 0.253, 0.159),
)


def _prior() -> BoxDimensionPrior:
    return BoxDimensionPrior(
        samples_m=SAMPLES,
        source="manual_tape_measurements_2026-07-28",
    )


def test_measured_dimension_prior_preserves_samples_and_exact_summary() -> None:
    prior = _prior()

    assert prior.sample_count == 8
    np.testing.assert_allclose(prior.representative_m, [0.400, 0.253, 0.160])
    np.testing.assert_allclose(prior.mean_m, [0.398875, 0.2535, 0.160375])
    np.testing.assert_allclose(prior.observed_min_m, [0.395, 0.252, 0.156])
    np.testing.assert_allclose(prior.observed_max_m, [0.401, 0.256, 0.164])
    np.testing.assert_allclose(
        prior.sample_std_m,
        [0.002474873734, 0.001414213562, 0.002669269563],
        atol=1e-12,
    )
    assert BoxDimensionPrior.from_dict(prior.to_dict()) == prior


def test_root_config_uses_measured_median_and_prior_provenance() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_rby1_nominal.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    config = EstimatorConfig.from_root_config(payload)

    assert config.box_model == BoxModel()
    assert config.box_dimension_prior == _prior()
    assert config.rectangle_containment_tolerance_m == pytest.approx(0.010)


def test_custom_dimensions_without_model_id_do_not_claim_measured_prior() -> None:
    model = BoxModel.from_dict(
        {"long": 0.400, "short": 0.250, "height": 0.150}
    )

    assert model.model_id == "parcel_configured_400x250x150"
    assert model.model_id != BoxModel().model_id


def test_explicit_median_dimensions_without_provenance_use_configured_id() -> None:
    model = BoxModel.from_dict(
        {"long": 0.400, "short": 0.253, "height": 0.160}
    )

    assert model.model_id == "parcel_configured_400x253x160"
    assert BoxModel.from_dict({}) == BoxModel()


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "manual", "samples": []},
        {"source": "manual", "samples": [[400.0, 253.0, 160.0]]},
        {"source": "manual", "samples": [[0.250, 0.400, 0.160]]},
        {"source": "manual", "samples": [[0.400, 0.253, -0.160]]},
    ],
)
def test_dimension_prior_rejects_invalid_or_wrong_unit_samples(payload) -> None:
    with pytest.raises(ValueError):
        BoxDimensionPrior.from_dict(payload)


def test_dimension_prior_rejects_stale_derived_summary() -> None:
    payload = _prior().to_dict()
    payload["representative"]["height"] = 0.150

    with pytest.raises(ValueError, match="representative disagrees"):
        BoxDimensionPrior.from_dict(payload)


@pytest.mark.parametrize(
    "malformed_summary",
    [
        {"long": 0.400, "short": 0.253},
        {"long": 0.400, "short": "not-a-number", "height": 0.160},
    ],
)
def test_dimension_prior_normalizes_malformed_summary_to_value_error(
    malformed_summary,
) -> None:
    payload = _prior().to_dict()
    payload["mean"] = malformed_summary

    with pytest.raises(ValueError, match="must contain numeric"):
        BoxDimensionPrior.from_dict(payload)


def test_estimator_config_rejects_model_that_disagrees_with_measurements() -> None:
    with pytest.raises(ValueError, match="component-wise median"):
        EstimatorConfig(
            box_model=BoxModel(short_m=0.250, height_m=0.150, model_id="legacy"),
            box_dimension_prior=_prior(),
        )


def test_estimator_config_requires_height_gate_to_cover_measured_population() -> None:
    with pytest.raises(ValueError, match="measured height range"):
        EstimatorConfig(
            box_dimension_prior=_prior(),
            top_plane_tolerance_m=0.003,
        )
