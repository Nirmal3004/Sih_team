"""Feature and reference tests (SIH26078, Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.downscaling.dataset import bilinear_baseline
from extreme_weather.downscaling.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureConfig,
    build_feature_stack,
    feature_metadata,
)
from extreme_weather.downscaling.reference import (
    REFERENCE_DISCLAIMER,
    ReferenceConfig,
    block_average,
    detect_resolution,
    generate_reference,
)

from .downscaling_fixtures import COARSE_RESOLUTION, FINE_RESOLUTION, make_axis, make_coarse_field

# --- resolution detection --------------------------------------------------


def test_detect_resolution_reads_the_axis():
    assert detect_resolution(make_axis(10.0, 49, 0.125)) == pytest.approx(0.125)


def test_detect_resolution_uses_the_median_not_the_first_step():
    """A truncated final interval must not change the reported spacing."""
    axis = make_axis(10.0, 49, 0.125)
    axis[-1] = axis[-2] + 0.05
    assert detect_resolution(axis) == pytest.approx(0.125)


def test_detect_resolution_rejects_a_single_point():
    with pytest.raises(ValueError, match="at least 2 points"):
        detect_resolution(np.array([10.0]))


def test_detect_resolution_rejects_non_monotonic_coordinates():
    with pytest.raises(ValueError, match="monotonic"):
        detect_resolution(np.array([10.0, 10.2, 10.1]))


def test_project_grid_resolutions_are_detected_not_hardcoded(ds_coarse_field):
    field, latitude, longitude, _ = ds_coarse_field
    assert detect_resolution(latitude) == pytest.approx(COARSE_RESOLUTION)
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert reference.output_resolution_deg == pytest.approx(FINE_RESOLUTION, abs=1e-6)
    assert reference.input_resolution_deg == pytest.approx(COARSE_RESOLUTION)


def test_grid_dimensions_come_from_the_data(ds_coarse_field):
    field, latitude, longitude, _ = ds_coarse_field
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert reference.input_shape == field[0].shape
    assert reference.output_shape == (21, 21)  # 1 deg / 0.05 deg
    assert reference.scale_factor == pytest.approx(2.5, abs=1e-6)


# --- features ---------------------------------------------------------------


def test_feature_stack_has_the_documented_dimension():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    stack = build_feature_stack(field[0], latitude, longitude)
    assert stack.shape == (FEATURE_COUNT, 5, 5)
    assert len(FEATURE_NAMES) == FEATURE_COUNT


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_features_are_finite_everywhere():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    assert np.all(np.isfinite(build_feature_stack(field[0], latitude, longitude)))


def test_coarse_channel_holds_the_scaled_precipitation():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    stack = build_feature_stack(field[0], latitude, longitude)
    index = FEATURE_NAMES.index("coarse_precipitation")
    assert stack[index] == pytest.approx(field[0] / 100.0, abs=1e-6)


def test_coordinate_channels_are_normalised():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    stack = build_feature_stack(field[0], latitude, longitude)
    for name in ("latitude_scaled", "longitude_scaled"):
        channel = stack[FEATURE_NAMES.index(name)]
        assert channel.min() >= -1.0 and channel.max() <= 1.0


def test_neighbour_channels_use_edge_replication_not_wraparound():
    """A value on the west edge must not reappear from the east edge."""
    field = np.zeros((1, 5, 5))
    field[0, 2, 0] = 10.0
    stack = build_feature_stack(
        field[0], make_axis(10.0, 5, 0.125), make_axis(74.0, 5, 0.125)
    )
    east = FEATURE_NAMES.index("east_neighbour")
    west = FEATURE_NAMES.index("west_neighbour")
    # The wet cell sits at column 0. Its east neighbour is column 1 (dry), not a
    # wrap-around of column 4 ...
    assert stack[east][2, 0] == pytest.approx(0.0)
    # ... and its west neighbour is edge-replicated back onto itself.
    assert stack[west][2, 0] == pytest.approx(10.0 / 100.0, abs=1e-6)
    # Column 1 has the wet cell as its west neighbour and dry ground to the east.
    assert stack[west][2, 1] == pytest.approx(10.0 / 100.0, abs=1e-6)
    assert stack[east][2, 1] == pytest.approx(0.0)


def test_gradient_channels_encode_the_local_trend():
    """``longitude_gradient`` is ``east - west``; it flips sign across a cell."""
    field = np.zeros((1, 5, 5))
    field[0, 2, 2] = 10.0
    stack = build_feature_stack(
        field[0], make_axis(10.0, 5, 0.125), make_axis(74.0, 5, 0.125)
    )
    zonal = FEATURE_NAMES.index("longitude_gradient")
    meridional = FEATURE_NAMES.index("latitude_gradient")
    # Row 2 is [0, 0, 10, 0, 0]; gradient[j] = field[j+1] - field[j-1].
    assert stack[zonal][2, 1] > 0.0  # rising flank, looking east at the cell
    assert stack[zonal][2, 3] < 0.0  # falling flank
    # gradient[i] = field[i-1] - field[i+1] along latitude.
    assert stack[meridional][3, 2] > 0.0
    assert stack[meridional][1, 2] < 0.0
    # Far from the cell both gradients vanish.
    assert stack[zonal][0, 0] == pytest.approx(0.0)


def test_features_never_read_the_high_resolution_target():
    """Inputs depend only on the coarse field and the coordinates."""
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    first = build_feature_stack(field[0], latitude, longitude)
    second = build_feature_stack(field[0] * 0.0, latitude, longitude)
    assert first.shape == second.shape
    assert not np.allclose(first, second)


def test_feature_config_rejects_a_non_positive_divisor():
    with pytest.raises(ValueError, match="precip_scale"):
        FeatureConfig(precip_scale=0.0)


def test_feature_metadata_lists_the_contract():
    metadata = feature_metadata()
    assert metadata["feature_count"] == FEATURE_COUNT
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert "never used as an input" in metadata["note"]


# --- reference --------------------------------------------------------------


def test_reference_conserves_the_coarse_area_average():
    """The defining property: the coarse field is the area average of the reference."""
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=9)
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert reference.metadata["conservation_error_max"] < 1e-9


def test_reference_shape_and_resolution():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=9)
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert reference.field.shape == (21, 21)
    assert reference.output_resolution_deg == pytest.approx(FINE_RESOLUTION, abs=1e-6)


def test_reference_has_fine_structure_the_coarse_field_cannot_express():
    """There must be something to learn, otherwise the task is vacuous."""
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=9)
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert reference.field.std() > field[0].std()
    residual = reference.field - bilinear_baseline(field[0], reference.output_shape)
    assert np.abs(residual).max() > 1e-3


def test_zero_structure_strength_gives_a_smooth_field():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=9)
    reference = generate_reference(
        field[0],
        latitude,
        longitude,
        FINE_RESOLUTION,
        config=ReferenceConfig(structure_strength=0.0),
    )
    residual = reference.field - bilinear_baseline(field[0], reference.output_shape)
    assert np.abs(residual).max() < 0.5


def test_reference_is_deterministic():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=9)
    first = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    second = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert np.array_equal(first.field, second.field)


def test_reference_carries_the_synthetic_disclaimer():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    reference = generate_reference(field[0], latitude, longitude, FINE_RESOLUTION)
    assert "SYNTHETIC" in reference.metadata["reference_disclaimer"]
    assert "NOT observational truth" in reference.metadata["reference_disclaimer"]
    assert "NOT observational truth" in REFERENCE_DISCLAIMER
    assert "SYNTHETIC" in reference.as_dataarray().attrs["reference_disclaimer"]


def test_reference_config_rejects_invalid_settings():
    with pytest.raises(ValueError, match="structure_strength"):
        ReferenceConfig(structure_strength=-0.1)
    with pytest.raises(ValueError, match="structure_sigma_fine"):
        ReferenceConfig(structure_sigma_fine=-1.0)


def test_reference_rejects_inconsistent_inputs():
    field, latitude, longitude, _ = make_coarse_field(time_steps=1, size=5)
    with pytest.raises(ValueError, match="latitude axis"):
        generate_reference(field[0], latitude[:-1], longitude, FINE_RESOLUTION)
    with pytest.raises(ValueError, match="target_resolution"):
        generate_reference(field[0], latitude, longitude, 0.0)
    with pytest.raises(ValueError, match="2-D"):
        generate_reference(field, latitude, longitude, FINE_RESOLUTION)


def test_block_average_recovers_a_constant_field():
    assert block_average(np.full((21, 21), 7.0), (9, 9)) == pytest.approx(7.0)


def test_block_average_is_area_weighted_not_index_blocked():
    """With a non-integer ratio (21 fine / 9 coarse), index blocking would differ.

    A single wet fine cell contributes only part of its value to the coarse cell
    it overlaps, and nothing to the others - the area-weighted definition, which
    is what makes the reference conserve the coarse mean.
    """
    field = np.zeros((21, 21))
    field[10, 10] = 21.0
    averaged = block_average(field, (9, 9))
    assert averaged.max() < 21.0
    assert (averaged > 0).sum() == 1  # it lands in exactly one coarse cell
    # The weighted value is the fine value times the overlapping area fraction.
    assert averaged.max() > 0.0
