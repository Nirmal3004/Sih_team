"""Service, inference and extreme-event tests (SIH26078, Phase 4)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from extreme_weather.downscaling.checkpoint import DownscalingCheckpoint, save_checkpoint
from extreme_weather.downscaling.dataset import (
    SplitIndices,
    build_downscaling_dataset,
    load_coarse_dataset,
)
from extreme_weather.downscaling.inference import (
    ai_downscale,
    evaluate_extreme_event,
    predict_ai_field,
)
from extreme_weather.downscaling.model import ModelConfig
from extreme_weather.downscaling.service import (
    SUPPORTED_VARIABLES,
    DownscalingDataError,
    DownscalingModelUnavailableError,
    DownscalingResolutionError,
    DownscalingService,
    DownscalingVariableError,
)
from extreme_weather.downscaling.training import TrainingConfig, train

from .downscaling_fixtures import FINE_RESOLUTION, make_axis, make_coarse_field

COARSE_LAT = make_axis(10.0, 9, 0.125)
COARSE_LON = make_axis(74.0, 9, 0.125)
FINE_LAT = make_axis(10.0, 21, 0.05)
FINE_LON = make_axis(74.0, 21, 0.05)


def _replace_split(dataset, **kwargs):
    """Return a copy of ``dataset`` with a different split."""
    return type(dataset)(**{**dataset.__dict__, "split": SplitIndices(**kwargs)})


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def trained_model(tiny_downscaling_dataset):
    """A trained model on the tiny fixture."""
    result = train(
        tiny_downscaling_dataset,
        ModelConfig(hidden_channels=8, num_blocks=2),
        TrainingConfig(epochs=10, seed=5),
    )
    return tiny_downscaling_dataset, result


@pytest.fixture
def model_file(tmp_path, trained_model):
    ds, result = trained_model
    checkpoint = DownscalingCheckpoint(
        model_config=result.model.config,
        normalization=ds.normalization,
        training={"seed": 5, "epochs": 10, "best_validation_loss": 0.5, "best_epoch": 2},
        model_state={n: t.detach().clone() for n, t in result.model.state_dict().items()},
        input_resolution_deg=ds.input_resolution_deg,
        output_resolution_deg=ds.output_resolution_deg,
        input_shape=ds.input_shape,
        output_shape=ds.output_shape,
        variable=ds.variable,
    )
    return save_checkpoint(checkpoint, tmp_path / "best.pt")


@pytest.fixture
def service(model_file):
    """A service with a real checkpoint."""
    return DownscalingService(checkpoint_path=model_file)


# --- baselines --------------------------------------------------------------


def test_nearest_and_cubic_run_at_the_target_resolution():
    from extreme_weather.downscaling import downscale

    data = xr.DataArray(
        np.linspace(1.0, 5.0, 81).reshape(9, 9),
        dims=("latitude", "longitude"),
        coords={"latitude": COARSE_LAT, "longitude": COARSE_LON},
    )
    for method in ("nearest", "cubic"):
        assert downscale(data, FINE_RESOLUTION, method=method).shape == (21, 21)


def test_ai_is_registered_alongside_the_baselines():
    from extreme_weather.downscaling import available_methods

    methods = available_methods()
    assert {"nearest", "cubic", "ai"} <= set(methods)


def test_ai_method_fails_clearly_without_a_checkpoint(tmp_path, monkeypatch):
    from extreme_weather.downscaling import downscale
    from extreme_weather.downscaling.baseline import DownscalingError

    monkeypatch.setattr(
        "extreme_weather.downscaling.checkpoint.checkpoint_path",
        lambda *a, **k: tmp_path / "absent.pt",
    )
    data = xr.DataArray(
        np.zeros((9, 9)),
        dims=("latitude", "longitude"),
        coords={"latitude": COARSE_LAT, "longitude": COARSE_LON},
    )
    with pytest.raises(DownscalingError, match="needs a trained checkpoint"):
        downscale(data, FINE_RESOLUTION, method="ai")


# --- AI inference -----------------------------------------------------------


def test_ai_downscale_returns_the_target_grid(service, tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    data = xr.DataArray(
        ds.coarse[0],
        dims=("latitude", "longitude"),
        coords={"latitude": COARSE_LAT, "longitude": COARSE_LON},
    )
    checkpoint = service.checkpoint()
    field = ai_downscale(
        checkpoint.build_model(), data, FINE_RESOLUTION, checkpoint.normalization
    )
    assert field.shape == ds.output_shape
    assert (field >= 0.0).all()


def test_ai_downscale_rejects_a_non_2d_field(service):
    from extreme_weather.downscaling.model import DownscalingError

    checkpoint = service.checkpoint()
    data = xr.DataArray(
        np.zeros((2, 9, 9)),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": COARSE_LAT, "longitude": COARSE_LON},
    )
    with pytest.raises(DownscalingError, match="2-D"):
        ai_downscale(checkpoint.build_model(), data, FINE_RESOLUTION, checkpoint.normalization)


def test_ai_inference_is_deterministic(service, tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    checkpoint = service.checkpoint()
    model = checkpoint.build_model()
    first = predict_ai_field(model, ds.features[0], checkpoint.normalization, ds.output_shape)
    second = predict_ai_field(model, ds.features[0], checkpoint.normalization, ds.output_shape)
    assert np.array_equal(first, second)
    assert (first >= 0.0).all()


# --- service: status and info ----------------------------------------------


def test_status_reports_a_missing_model_as_unavailable(tmp_path):
    status = DownscalingService(checkpoint_path=tmp_path / "absent.pt").status()
    assert status["available"] is False
    assert status["model_version"] is None
    assert "train_downscaling.py" in status["message"]


def test_status_reports_a_trained_model(service):
    status = service.status()
    assert status["available"] is True
    assert status["model_version"]
    assert status["input_resolution_deg"] == pytest.approx(0.125)
    assert status["output_resolution_deg"] == pytest.approx(FINE_RESOLUTION, abs=1e-6)
    assert status["scale_factor"] == pytest.approx(2.5, abs=1e-6)
    assert status["variable"] == "precipitation"
    assert "pytorch" in status["backend"]


def test_has_model_matches_the_status(service, tmp_path):
    assert service.has_model() is True
    assert DownscalingService(checkpoint_path=tmp_path / "absent.pt").has_model() is False


def test_model_info_describes_the_architecture(service):
    info = service.model_info()
    assert info["residual_learning"] is True
    assert "ReLU" in info["output_constraint"]
    assert len(info["input_features"]) == 10
    assert "residual" in info["architecture"].lower()


def test_model_info_works_without_a_checkpoint(tmp_path):
    info = DownscalingService(checkpoint_path=tmp_path / "absent.pt").model_info()
    assert info["available"] is False
    assert info["input_features"]


# --- service: validation ---------------------------------------------------


def test_unsupported_variable_is_rejected():
    with pytest.raises(DownscalingVariableError, match="not supported"):
        DownscalingService.validate_variable("pressure")
    assert DownscalingService.validate_variable("precipitation") == "precipitation"
    assert SUPPORTED_VARIABLES == ("precipitation",)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_resolution_is_rejected(value):
    with pytest.raises(DownscalingResolutionError):
        DownscalingService.validate_resolution(value)


def test_non_numeric_resolution_is_rejected():
    with pytest.raises(DownscalingResolutionError, match="must be a number"):
        DownscalingService.validate_resolution("coarse")


def test_missing_dataset_is_reported(tmp_path, model_file):
    service = DownscalingService(checkpoint_path=model_file, dataset_path=tmp_path / "absent.nc")
    with pytest.raises(DownscalingDataError, match="No dataset"):
        service.load_coarse()


def test_unknown_method_is_rejected(service, tiny_downscaling_dataset):
    with pytest.raises(DownscalingDataError, match="Unknown method"):
        service.run_field(
            "magic", tiny_downscaling_dataset.coarse[0], COARSE_LAT, COARSE_LON, FINE_RESOLUTION
        )


def test_ai_without_features_is_reported(service, tiny_downscaling_dataset):
    """Features are optional: the service builds them when not supplied."""
    field, result = service.run_field(
        "ai", tiny_downscaling_dataset.coarse[0], COARSE_LAT, COARSE_LON, FINE_RESOLUTION
    )
    assert field.shape == tiny_downscaling_dataset.output_shape
    assert result.model_version is not None


# --- single-frame runs ------------------------------------------------------


def test_run_field_reports_metadata(service, tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    for method in ("nearest", "cubic"):
        field, result = service.run_field(
            method, ds.coarse[0], COARSE_LAT, COARSE_LON, FINE_RESOLUTION
        )
        assert field.shape == ds.output_shape
        assert result.method == method
        assert result.input_resolution_deg == pytest.approx(0.125)
        assert result.output_resolution_deg == pytest.approx(FINE_RESOLUTION, abs=1e-6)
        assert result.model_version is None
        assert result.statistics["max"] >= result.statistics["min"]
    assert "not a forecast skill score" in result.notice


def test_run_field_ai_uses_the_model(service, tiny_downscaling_dataset):
    _, result = service.run_field(
        "ai",
        tiny_downscaling_dataset.coarse[0],
        COARSE_LAT,
        COARSE_LON,
        FINE_RESOLUTION,
        features=tiny_downscaling_dataset.features[0],
    )
    assert result.model_version is not None
    assert "operational forecast" in result.notice


# --- comparison -------------------------------------------------------------


def test_compare_scores_every_method_on_the_same_samples(service, tiny_downscaling_dataset):
    report = service.compare(tiny_downscaling_dataset, methods=("nearest", "cubic", "ai"))
    assert [row.method for row in report.metrics] == ["nearest", "cubic", "ai"]
    assert report.test_sample_count == len(tiny_downscaling_dataset.split.test)
    for row in report.metrics:
        assert row.error.mae >= 0.0
        assert row.error.n_points > 0
    assert len(report.extreme_event) == 3


def test_compare_can_run_baselines_without_ai(service, tiny_downscaling_dataset):
    report = service.compare(tiny_downscaling_dataset, methods=("nearest", "cubic"))
    assert [row.method for row in report.metrics] == ["nearest", "cubic"]


def test_compare_evaluates_the_event_on_a_test_frame(service, tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    report = service.compare(ds, methods=("nearest",))
    event_frame = max(ds.split.test, key=lambda index: float(ds.references[index].max()))
    assert report.extreme_event[0].reference_peak == pytest.approx(
        float(ds.references[event_frame].max())
    )


def test_compare_notice_disclaims_real_world_skill(service, tiny_downscaling_dataset):
    report = service.compare(tiny_downscaling_dataset, methods=("nearest",))
    assert "SYNTHETIC" in report.notice
    assert "does not imply higher accuracy" in report.notice


def test_compare_serialises(service, tiny_downscaling_dataset):
    payload = service.compare(tiny_downscaling_dataset, methods=("nearest", "ai")).as_dict()
    assert json.loads(json.dumps(payload, default=float))["variable"] == "precipitation"


def test_compare_rejects_an_empty_test_block(service, tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    empty = _replace_split(ds, train=ds.split.train, validation=[], test=[])
    with pytest.raises(DownscalingDataError, match="no test samples"):
        service.compare(empty, methods=("nearest",))


# --- extreme event ----------------------------------------------------------


def test_extreme_event_metrics_of_a_perfect_field():
    rng = np.random.default_rng(0)
    reference = rng.random((21, 21)) * 10.0
    result = evaluate_extreme_event("m", reference, reference, FINE_LAT, FINE_LON)
    assert result.peak_ratio == pytest.approx(1.0)
    assert result.centroid_distance_km == pytest.approx(0.0, abs=1e-9)
    assert result.area_ratio == pytest.approx(1.0)
    assert result.structure_correlation == pytest.approx(1.0)


def test_extreme_event_detects_a_peak_shift():
    reference = np.zeros((21, 21))
    reference[10, 10] = 50.0
    shifted = np.zeros((21, 21))
    shifted[10, 13] = 50.0
    result = evaluate_extreme_event("m", shifted, reference, FINE_LAT, FINE_LON)
    assert result.centroid_distance_km > 0.0
    assert result.peak_value == pytest.approx(50.0)


def test_extreme_event_reports_overestimation():
    reference = np.zeros((21, 21))
    reference[10, 10] = 10.0
    inflated = reference.copy()
    inflated[10, 10] = 20.0
    result = evaluate_extreme_event("m", inflated, reference, FINE_LAT, FINE_LON)
    assert result.peak_overestimation == pytest.approx(10.0)
    assert result.peak_ratio == pytest.approx(2.0)
    assert result.peak_absolute_error == pytest.approx(10.0)


def test_extreme_event_wet_area_is_in_square_km():
    """0.05 deg is about 5.5 km, so 21x21 cells span roughly 115 x 115 km."""
    reference = np.ones((21, 21))
    result = evaluate_extreme_event("m", reference, reference, FINE_LAT, FINE_LON)
    assert 10_000.0 < result.wet_area_km2 < 20_000.0
    assert result.area_ratio == pytest.approx(1.0)


def test_extreme_event_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate_extreme_event("m", np.zeros((5, 5)), np.zeros((6, 6)), FINE_LAT, FINE_LON)


def test_extreme_event_reports_a_measured_centroid_not_a_claim():
    reference = np.zeros((21, 21))
    reference[10, 10] = 10.0
    result = evaluate_extreme_event("m", reference, reference, FINE_LAT, FINE_LON)
    assert "centroid_distance_km" in result.as_dict()
    assert result.reference_centroid_lat == pytest.approx(FINE_LAT[10])


# --- dataset loading --------------------------------------------------------


def test_load_coarse_dataset_rejects_a_file_without_precipitation(tmp_path):
    path = tmp_path / "other.nc"
    xr.Dataset(
        {"temperature": (("latitude", "longitude"), np.zeros((4, 4)))},
        coords={"latitude": make_axis(10.0, 4, 0.125), "longitude": make_axis(74.0, 4, 0.125)},
    ).to_netcdf(path)
    with pytest.raises(KeyError, match="precipitation"):
        load_coarse_dataset(path)


def test_dataset_rejects_too_few_frames():
    field, latitude, longitude, timestamps = make_coarse_field(time_steps=2, size=5)
    with pytest.raises(ValueError, match="At least 3 time steps"):
        build_downscaling_dataset(field, latitude, longitude, timestamps, FINE_RESOLUTION)


