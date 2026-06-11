from pathlib import Path
import inspect
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heat_pde  # noqa: E402


DTYPE = torch.float64


def _coords_mm():
    return torch.tensor(
        [
            [0.0, 0.0],
            [0.002, 0.0],
            [0.006, 0.004],
            [0.01, 0.01],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )


def test_constant_temperature_steady_residual_is_zero():
    coords_mm = _coords_mm()
    temperature_K = coords_mm[:, 0] * 0.0 + 300.0

    grad_T = heat_pde.temperature_gradient_m(temperature_K, coords_mm, coordinate_unit="mm")
    residual = heat_pde.steady_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        coordinate_unit="mm",
    )

    assert torch.allclose(grad_T, torch.zeros_like(grad_T), atol=1.0e-12)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-9)


def test_linear_1d_temperature_has_zero_steady_residual_and_expected_flux():
    coords_mm = _coords_mm()
    slope_K_per_m = 12.5
    x_m = heat_pde.coords_mm_to_m(coords_mm)[:, 0]
    temperature_K = 293.15 + slope_K_per_m * x_m

    residual = heat_pde.steady_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        coordinate_unit="mm",
    )
    flux = heat_pde.heat_flux_W_per_m2(temperature_K, coords_mm, coordinate_unit="mm")

    expected_flux_x = -heat_pde.DEFAULT_THERMAL_K0_W_PER_MK * slope_K_per_m
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-8)
    assert torch.allclose(flux[:, 0], torch.full_like(flux[:, 0], expected_flux_x), atol=1.0e-9)
    assert torch.allclose(flux[:, 1], torch.zeros_like(flux[:, 1]), atol=1.0e-12)


def test_quadratic_manufactured_source_cancels_steady_residual():
    coords_mm = _coords_mm()
    curvature_K_per_m2 = 7.0e5
    x_m = heat_pde.coords_mm_to_m(coords_mm)[:, 0]
    temperature_K = 293.15 + curvature_K_per_m2 * x_m**2
    matching_Q_W_per_m3 = -2.0 * heat_pde.DEFAULT_THERMAL_K0_W_PER_MK * curvature_K_per_m2

    no_source_residual = heat_pde.steady_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        coordinate_unit="mm",
    )
    matched_residual = heat_pde.steady_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        heat_source_Q_W_per_m3=matching_Q_W_per_m3,
        coordinate_unit="mm",
    )

    expected_no_source = torch.full_like(no_source_residual, matching_Q_W_per_m3)
    assert torch.allclose(no_source_residual, expected_no_source, rtol=1.0e-12, atol=1.0e-3)
    assert torch.allclose(matched_residual, torch.zeros_like(matched_residual), atol=1.0e-3)


def test_boundary_normal_heat_flux_uses_minus_k_grad_t_dot_n():
    coords_mm = _coords_mm()
    slope_K_per_m = 8.0
    x_m = heat_pde.coords_mm_to_m(coords_mm)[:, 0]
    linear_temperature = 300.0 + slope_K_per_m * x_m
    constant_temperature = coords_mm[:, 0] * 0.0 + 300.0
    normals = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=DTYPE,
    )

    constant_flux = heat_pde.normal_heat_flux_W_per_m2(
        constant_temperature,
        coords_mm,
        normals,
        coordinate_unit="mm",
    )
    linear_flux = heat_pde.normal_heat_flux_W_per_m2(
        linear_temperature,
        coords_mm,
        normals,
        coordinate_unit="mm",
    )

    expected = torch.tensor(
        [
            -heat_pde.DEFAULT_THERMAL_K0_W_PER_MK * slope_K_per_m,
            heat_pde.DEFAULT_THERMAL_K0_W_PER_MK * slope_K_per_m,
            0.0,
            0.0,
        ],
        dtype=DTYPE,
    )
    assert torch.allclose(constant_flux, torch.zeros_like(constant_flux), atol=1.0e-12)
    assert torch.allclose(linear_flux, expected, atol=1.0e-9)


def test_transient_uniform_no_source_residual_is_zero():
    coords_mm = _coords_mm()
    temperature_K = coords_mm[:, 0] * 0.0 + 300.0
    dTdt_K_per_s = torch.zeros(coords_mm.shape[0], dtype=DTYPE)

    residual = heat_pde.transient_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        dTdt_K_per_s,
        coordinate_unit="mm",
    )

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-9)


def test_transient_manufactured_source_cancels_residual():
    coords_mm = _coords_mm()
    curvature_K_per_m2 = 3.0e5
    dTdt_K_per_s = torch.full((coords_mm.shape[0],), 2.5, dtype=DTYPE)
    x_m = heat_pde.coords_mm_to_m(coords_mm)[:, 0]
    temperature_K = 300.0 + dTdt_K_per_s * 1.2 + curvature_K_per_m2 * x_m**2
    matching_Q_W_per_m3 = (
        heat_pde.DEFAULT_THERMAL_RHO_KG_PER_M3
        * heat_pde.DEFAULT_THERMAL_C_J_PER_KGK
        * dTdt_K_per_s
        - 2.0 * heat_pde.DEFAULT_THERMAL_K0_W_PER_MK * curvature_K_per_m2
    )

    residual = heat_pde.transient_heat_residual_W_per_m3(
        temperature_K,
        coords_mm,
        dTdt_K_per_s,
        heat_source_Q_W_per_m3=matching_Q_W_per_m3,
        coordinate_unit="mm",
    )

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-3)


def test_mm_to_m_chain_rule_matches_direct_meter_coordinates():
    coords_mm = _coords_mm()
    coords_m = heat_pde.coords_mm_to_m(coords_mm.detach()).requires_grad_(True)
    slope_K_per_m = 4.0
    curvature_K_per_m2 = 2.0e5

    x_mm_m = heat_pde.coords_mm_to_m(coords_mm)[:, 0]
    temperature_from_mm = 300.0 + slope_K_per_m * x_mm_m + curvature_K_per_m2 * x_mm_m**2

    x_m = coords_m[:, 0]
    temperature_from_m = 300.0 + slope_K_per_m * x_m + curvature_K_per_m2 * x_m**2

    grad_from_mm = heat_pde.temperature_gradient_m(temperature_from_mm, coords_mm, coordinate_unit="mm")
    grad_from_m = heat_pde.temperature_gradient_m(temperature_from_m, coords_m, coordinate_unit="m")
    residual_from_mm = heat_pde.steady_heat_residual_W_per_m3(
        temperature_from_mm,
        coords_mm,
        coordinate_unit="mm",
    )
    residual_from_m = heat_pde.steady_heat_residual_W_per_m3(
        temperature_from_m,
        coords_m,
        coordinate_unit="m",
    )

    native_grad_per_mm = torch.autograd.grad(
        temperature_from_mm,
        coords_mm,
        grad_outputs=torch.ones_like(temperature_from_mm),
        create_graph=True,
    )[0]

    assert torch.allclose(grad_from_mm, grad_from_m, rtol=1.0e-12, atol=1.0e-9)
    assert torch.allclose(residual_from_mm, residual_from_m, rtol=1.0e-12, atol=1.0e-3)
    assert torch.allclose(grad_from_mm[:, 0], native_grad_per_mm[:, 0] * 1.0e3)


def test_phase1_heat_pde_api_has_no_damage_dependent_conductivity_inputs():
    public_functions = [
        heat_pde.temperature_gradient_m,
        heat_pde.heat_flux_W_per_m2,
        heat_pde.steady_heat_residual_W_per_m3,
        heat_pde.transient_heat_residual_W_per_m3,
        heat_pde.normal_heat_flux_W_per_m2,
    ]
    forbidden_parameter_names = {"alpha", "damage", "d", "g_d", "k_d"}

    for fn in public_functions:
        parameter_names = set(inspect.signature(fn).parameters)
        assert not (parameter_names & forbidden_parameter_names)

    source = Path(heat_pde.__file__).read_text(encoding="utf-8")
    forbidden_active_tokens = [
        "k(d)",
        "g(d)",
        "k_d",
        "damage_dependent_conductivity",
        "alpha_conductivity",
    ]
    for token in forbidden_active_tokens:
        assert token not in source
