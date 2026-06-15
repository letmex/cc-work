from pathlib import Path
import inspect
import math
import sys

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heat_pde  # noqa: E402
import thermal_field  # noqa: E402
import thermal_quadrature  # noqa: E402
import train_heat_only  # noqa: E402


DTYPE = torch.float64


class ConstantNet(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, coords):
        return torch.full((coords.shape[0], 1), self.value, device=coords.device, dtype=coords.dtype)


def test_triangle_areas_convert_square_mm_to_square_meters():
    nodes_mm = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=DTYPE)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)

    area_m2 = thermal_quadrature.triangle_areas_m2(nodes_mm, elements)

    assert torch.allclose(area_m2, torch.tensor([3.0e-6], dtype=DTYPE))


def test_triangle_centroids_return_average_node_coordinates_in_mm():
    nodes_mm = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [3.0, 3.0],
        ],
        dtype=DTYPE,
    )
    elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

    centroids = thermal_quadrature.triangle_centroids_mm(nodes_mm, elements)

    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=DTYPE)
    assert torch.allclose(centroids, expected)


def test_area_weighted_mean_density_uses_triangle_area_weights():
    density = torch.tensor([10.0, 100.0], dtype=DTYPE)
    area_m2 = torch.tensor([1.0, 3.0], dtype=DTYPE)

    weighted = thermal_quadrature.area_weighted_mean_density(density, area_m2)
    unweighted = density.mean()

    assert torch.allclose(weighted, torch.tensor(77.5, dtype=DTYPE))
    assert not torch.allclose(weighted, unweighted)


def test_top_bottom_dirichlet_ansatz_hard_satisfies_both_boundaries():
    coords_mm = torch.tensor(
        [[0.0, 0.0], [0.005, 0.005], [0.01, 0.01]],
        dtype=DTYPE,
    )
    bounds_mm = torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE)
    net = ConstantNet(50.0)

    temperature = thermal_field.evaluate_top_bottom_dirichlet_temperature(
        coords_mm,
        net,
        bounds_mm,
        T_bottom_K=300.0,
        T_top_K=320.0,
        y_min_mm=0.0,
        y_max_mm=0.01,
    )

    assert torch.allclose(temperature[0], torch.tensor(300.0, dtype=DTYPE), atol=1.0e-12)
    assert torch.allclose(temperature[2], torch.tensor(320.0, dtype=DTYPE), atol=1.0e-12)
    assert torch.allclose(temperature[1], torch.tensor(322.5, dtype=DTYPE), atol=1.0e-12)


def test_bottom_only_dirichlet_ansatz_hard_satisfies_bottom_boundary():
    coords_mm = torch.tensor(
        [[0.0, 0.0], [0.005, 0.0], [0.01, 0.01]],
        dtype=DTYPE,
    )
    bounds_mm = torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE)
    net = ConstantNet(10.0)

    temperature = thermal_field.evaluate_bottom_dirichlet_temperature(
        coords_mm,
        net,
        bounds_mm,
        T_bottom_K=300.0,
    )

    assert torch.allclose(temperature[:2], torch.full((2,), 300.0, dtype=DTYPE), atol=1.0e-12)
    assert temperature[2].item() > 300.0


def test_h1_top_bottom_heat_only_patch_training_matches_linear_solution():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H1",
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["case_id"] == "H1"
    assert result["primary_loss"] == "thermal_functional_area_weighted_mean"
    assert result["used_strong_residual_as_primary_loss"] is False
    assert result["max_abs_temperature_error_K"] < 0.5
    assert result["l2_temperature_error_K"] < 0.2
    assert result["bottom_boundary_max_abs_error_K"] < 1.0e-10
    assert result["top_boundary_max_abs_error_K"] < 1.0e-10
    assert math.isfinite(result["max_abs_strong_residual_W_per_m3"])


def test_h2_bottom_only_heat_only_patch_training_matches_constant_solution():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H2",
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["case_id"] == "H2"
    assert result["primary_loss"] == "thermal_functional_area_weighted_mean"
    assert result["used_strong_residual_as_primary_loss"] is False
    assert result["max_abs_temperature_error_K"] < 0.2
    assert result["l2_temperature_error_K"] < 0.1
    assert result["bottom_boundary_max_abs_error_K"] < 1.0e-10
    assert math.isfinite(result["max_abs_strong_residual_W_per_m3"])
    assert result["max_abs_top_side_flux_W_per_m2"] < 1.0e-8


def test_heat_only_loss_uses_functional_density_not_strong_residual_as_primary_loss():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H2",
        nx=2,
        ny=2,
        num_epochs=2,
        dtype=DTYPE,
    )
    source = Path(train_heat_only.__file__).read_text(encoding="utf-8")

    assert result["primary_loss"] == "thermal_functional_area_weighted_mean"
    assert result["used_strong_residual_as_primary_loss"] is False
    assert "steady_thermal_energy_density_J_per_m3" in source
    assert "strong_residual_loss" not in source


def test_quadrature_sanity_case_integrates_constant_density_over_patch_area():
    result = train_heat_only.run_quadrature_sanity_case(
        x_min_mm=0.0,
        x_max_mm=0.02,
        y_min_mm=0.0,
        y_max_mm=0.01,
        constant_density=12.5,
        dtype=DTYPE,
    )

    assert result["area_weighted_mean_density"] == pytest.approx(12.5)
    assert result["integral"] == pytest.approx(12.5 * 2.0e-10)


def test_transient_storage_sanity_case_matches_incremental_density_formula():
    same = train_heat_only.run_transient_storage_sanity_case(delta_T_K=0.0, dt_s=2.0, dtype=DTYPE)
    changed = train_heat_only.run_transient_storage_sanity_case(delta_T_K=4.0, dt_s=2.0, dtype=DTYPE)

    expected = (
        heat_pde.DEFAULT_THERMAL_RHO_KG_PER_M3
        * heat_pde.DEFAULT_THERMAL_C_J_PER_KGK
        / (2.0 * 2.0)
        * 4.0**2
    )
    assert same["mean_incremental_density"] == pytest.approx(0.0)
    assert changed["mean_incremental_density"] == pytest.approx(expected)


def test_heat_only_public_api_has_no_damage_dependent_conductivity_inputs_or_tokens():
    public_functions = [
        thermal_field.normalize_coords_for_heat_net,
        thermal_field.evaluate_bottom_dirichlet_temperature,
        thermal_field.evaluate_top_bottom_dirichlet_temperature,
        thermal_quadrature.triangle_areas_m2,
        thermal_quadrature.area_weighted_mean_density,
        train_heat_only.run_steady_heat_patch_case,
        train_heat_only.run_quadrature_sanity_case,
        train_heat_only.run_transient_storage_sanity_case,
    ]
    forbidden_parameter_names = {"alpha", "damage", "d", "g_d", "k_d"}

    for fn in public_functions:
        parameter_names = set(inspect.signature(fn).parameters)
        assert not (parameter_names & forbidden_parameter_names)

    combined_source = "\n".join(
        [
            Path(thermal_field.__file__).read_text(encoding="utf-8"),
            Path(thermal_quadrature.__file__).read_text(encoding="utf-8"),
            Path(train_heat_only.__file__).read_text(encoding="utf-8"),
        ]
    )
    forbidden_active_tokens = [
        "k(d)",
        "g(d)",
        "k_d",
        "damage_dependent_conductivity",
        "alpha_conductivity",
        "train_mixed_tm",
        "compute_energy_mixed_tm",
    ]
    for token in forbidden_active_tokens:
        assert token not in combined_source
