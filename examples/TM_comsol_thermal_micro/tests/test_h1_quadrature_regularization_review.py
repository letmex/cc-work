from pathlib import Path
import inspect
import math
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import thermal_quadrature  # noqa: E402
import train_heat_only  # noqa: E402


DTYPE = torch.float64


def _simple_triangle():
    nodes_mm = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=DTYPE)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return nodes_mm, elements


def test_h1_no_training_lift_baseline_has_near_zero_residual_and_correction():
    result = train_heat_only.run_h1_no_training_lift_baseline(nx=4, ny=4, dtype=DTYPE)

    assert result["case_id"] == "H1_no_training_lift"
    assert result["max_abs_temperature_error_K"] < 1.0e-10
    assert result["l2_temperature_error_K"] < 1.0e-10
    assert result["bottom_boundary_max_abs_error_K"] < 1.0e-12
    assert result["top_boundary_max_abs_error_K"] < 1.0e-12
    assert result["max_abs_strong_residual_W_per_m3"] < 1.0e-4
    assert result["max_abs_correction_K"] < 1.0e-12
    assert result["l2_correction_K"] < 1.0e-12
    assert result["max_correction_gradient_K_per_m"] < 1.0e-6
    assert result["normalized_residual_max"] < 1.0e-12


def test_triangle_3point_quadrature_weights_sum_to_total_area():
    nodes_mm, elements = _simple_triangle()

    points, weights = thermal_quadrature.triangle_quadrature_points_mm(
        nodes_mm,
        elements,
        rule="triangle_3point",
    )
    area = thermal_quadrature.triangle_areas_m2(nodes_mm, elements)

    assert points.shape == (3, 2)
    assert weights.shape == (3,)
    assert torch.allclose(torch.sum(weights), torch.sum(area), atol=1.0e-18)
    assert torch.all(weights > 0.0)


def test_triangle_3point_quadrature_points_lie_inside_simple_triangle():
    nodes_mm, elements = _simple_triangle()

    points, _weights = thermal_quadrature.triangle_quadrature_points_mm(
        nodes_mm,
        elements,
        rule="triangle_3point",
    )

    x = points[:, 0]
    y = points[:, 1]
    assert torch.all(x >= 0.0)
    assert torch.all(y >= 0.0)
    assert torch.all(x + y <= 2.0)


def test_centroid_and_triangle_3point_have_same_constant_density_mean():
    nodes_mm, elements = _simple_triangle()
    centroid_points, centroid_weights = thermal_quadrature.triangle_quadrature_points_mm(
        nodes_mm,
        elements,
        rule="centroid",
    )
    three_points, three_weights = thermal_quadrature.triangle_quadrature_points_mm(
        nodes_mm,
        elements,
        rule="triangle_3point",
    )

    centroid_density = torch.full((centroid_points.shape[0],), 42.0, dtype=DTYPE)
    three_density = torch.full((three_points.shape[0],), 42.0, dtype=DTYPE)

    assert thermal_quadrature.weighted_mean_density(centroid_density, centroid_weights).item() == pytest.approx(42.0)
    assert thermal_quadrature.weighted_mean_density(three_density, three_weights).item() == pytest.approx(42.0)


def test_h1_trained_correction_diagnostics_are_finite_and_nonnegative():
    result = train_heat_only.run_h1_quadrature_diagnostic_case(
        quadrature_rule="centroid",
        regularization=None,
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    for key in (
        "max_abs_correction_K",
        "l2_correction_K",
        "max_abs_temperature_error_K",
        "l2_temperature_error_K",
        "max_abs_strong_residual_W_per_m3",
        "rms_strong_residual_W_per_m3",
        "max_temperature_gradient_K_per_m",
        "mean_temperature_gradient_K_per_m",
        "max_correction_gradient_K_per_m",
        "mean_correction_gradient_K_per_m",
        "normalized_residual_max",
        "normalized_residual_rms",
    ):
        assert math.isfinite(result[key]), key
        assert result[key] >= 0.0, key

    assert result["quadrature_rule"] == "centroid"
    assert result["primary_loss"] == train_heat_only.PRIMARY_LOSS
    assert result["used_strong_residual_as_primary_loss"] is False
    assert result["max_abs_correction_K"] > 0.0


def test_h1_triangle_3point_diagnostic_runs_and_reports_finite_normalized_residual():
    result = train_heat_only.run_h1_quadrature_diagnostic_case(
        quadrature_rule="triangle_3point",
        regularization=None,
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["quadrature_rule"] == "triangle_3point"
    assert math.isfinite(result["normalized_residual_max"])
    assert math.isfinite(result["normalized_residual_rms"])
    assert result["normalized_residual_scale_W_per_m3"] > 0.0


def test_h1_correction_l2_regularization_reduces_correction_without_residual_loss():
    unregularized = train_heat_only.run_h1_quadrature_diagnostic_case(
        quadrature_rule="triangle_3point",
        regularization=None,
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )
    regularized = train_heat_only.run_h1_quadrature_diagnostic_case(
        quadrature_rule="triangle_3point",
        regularization="correction_l2",
        regularization_weight=1.0e12,
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert regularized["regularization"] == "correction_l2"
    assert regularized["used_strong_residual_as_primary_loss"] is False
    assert regularized["max_abs_correction_K"] <= unregularized["max_abs_correction_K"]
    assert regularized["max_abs_temperature_error_K"] < 0.5


def test_strong_residual_is_not_primary_loss_and_no_forbidden_coupling_tokens():
    source = Path(train_heat_only.__file__).read_text(encoding="utf-8")
    result = train_heat_only.run_h1_quadrature_diagnostic_case(
        quadrature_rule="centroid",
        regularization=None,
        nx=2,
        ny=2,
        num_epochs=2,
        dtype=DTYPE,
    )

    assert result["primary_loss"] == "thermal_functional_area_weighted_mean"
    assert result["used_strong_residual_as_primary_loss"] is False
    forbidden_tokens = [
        "strong_residual_loss",
        "train_mixed_tm",
        "compute_energy_mixed_tm",
        "k(d)",
        "g(d)",
        "damage_dependent_conductivity",
    ]
    for token in forbidden_tokens:
        assert token not in source


def test_h1_review_public_api_has_no_damage_dependent_conductivity_inputs():
    public_functions = [
        thermal_quadrature.triangle_quadrature_points_mm,
        thermal_quadrature.weighted_mean_density,
        thermal_quadrature.integrate_density_with_weights,
        train_heat_only.run_h1_no_training_lift_baseline,
        train_heat_only.run_h1_quadrature_diagnostic_case,
    ]
    forbidden_parameter_names = {"alpha", "damage", "d", "g_d", "k_d"}

    for fn in public_functions:
        assert not (set(inspect.signature(fn).parameters) & forbidden_parameter_names)
