from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from mixed_mode_tm import mixed_mode_energy_split, tm_source_effective_stress_fields  # noqa: E402
from compute_energy_mixed_tm import compute_mixed_tm_fields  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from pff_model import PFFModel  # noqa: E402
from thermal_prescribed import (  # noqa: E402
    DEFAULT_ALPHA_T,
    DEFAULT_TREF_K,
    apply_thermal_strain,
    delta_T_from_temperature,
    thermal_strain_2d,
)


def _matprop(dtype=torch.float64):
    return MaterialProperties(
        mat_E=torch.tensor(81.5, dtype=dtype),
        mat_nu=torch.tensor(0.38, dtype=dtype),
        w1=torch.tensor(2.4e-6 / 1.5e-4, dtype=dtype),
        l0=torch.tensor(1.5e-4, dtype=dtype),
    )


def _pffmodel():
    return PFFModel(PFF_model="AT2", se_split="volumetric", tol_ir=torch.tensor(5.0e-3))


def test_zero_delta_t_is_no_thermal_equivalent_for_strain_and_split():
    exx = torch.tensor([1.0e-4, -2.0e-5], dtype=torch.float64)
    eyy = torch.tensor([3.0e-5, 4.0e-5], dtype=torch.float64)
    exy = torch.tensor([5.0e-5, -1.0e-5], dtype=torch.float64)

    elastic = apply_thermal_strain(exx, eyy, exy, delta_T=torch.zeros_like(exx))

    assert torch.equal(elastic["eps_xx"], exx)
    assert torch.equal(elastic["eps_yy"], eyy)
    assert torch.equal(elastic["eps_xy"], exy)

    matprop = _matprop()
    baseline = mixed_mode_energy_split(exx, eyy, exy, matprop, eps_r=1.0e-5)
    thermal = mixed_mode_energy_split(
        elastic["eps_xx"],
        elastic["eps_yy"],
        elastic["eps_xy"],
        matprop,
        eps_r=1.0e-5,
    )
    for key in ("psiI", "psiII", "psi_minus", "psi_total", "eps_zz"):
        assert torch.allclose(thermal[key], baseline[key], rtol=0.0, atol=0.0)


def test_temperature_uses_t_minus_tref_not_raw_temperature():
    Tref = torch.tensor(DEFAULT_TREF_K, dtype=torch.float64)
    temperature = torch.tensor([DEFAULT_TREF_K, DEFAULT_TREF_K + 20.0], dtype=torch.float64)

    delta_T = delta_T_from_temperature(temperature, Tref=Tref)
    eps_th = thermal_strain_2d(delta_T)

    assert torch.allclose(delta_T, torch.tensor([0.0, 20.0], dtype=torch.float64))
    assert torch.allclose(eps_th["eps_xx"], DEFAULT_ALPHA_T * delta_T)
    assert torch.allclose(eps_th["eps_yy"], DEFAULT_ALPHA_T * delta_T)
    assert torch.equal(eps_th["eps_xy"], torch.zeros_like(delta_T))


def test_free_uniform_thermal_expansion_has_near_zero_elastic_stress_and_energy():
    matprop = _matprop()
    alpha = torch.zeros(1, dtype=torch.float64)
    delta_T = torch.tensor([40.0], dtype=torch.float64)
    free_expansion = DEFAULT_ALPHA_T * delta_T

    elastic = apply_thermal_strain(
        free_expansion,
        free_expansion,
        torch.zeros_like(delta_T),
        delta_T=delta_T,
    )
    split = mixed_mode_energy_split(elastic["eps_xx"], elastic["eps_yy"], elastic["eps_xy"], matprop)
    stress = tm_source_effective_stress_fields(
        elastic["eps_xx"],
        elastic["eps_yy"],
        elastic["eps_xy"],
        alpha,
        matprop,
    )

    assert torch.allclose(elastic["eps_xx"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.allclose(elastic["eps_yy"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.equal(elastic["eps_xy"], torch.zeros_like(delta_T))
    assert torch.allclose(stress["sigma_xx_tm_total"], torch.zeros_like(delta_T), atol=1.0e-12)
    assert torch.allclose(stress["sigma_yy_tm_total"], torch.zeros_like(delta_T), atol=1.0e-12)
    assert torch.allclose(stress["sigma_xy_tm_total"], torch.zeros_like(delta_T), atol=1.0e-12)
    assert torch.allclose(split["psi_total"], torch.zeros_like(delta_T), atol=1.0e-18)


def test_constrained_uniform_heating_is_compressive_under_project_convention():
    matprop = _matprop()
    alpha = torch.zeros(1, dtype=torch.float64)
    delta_T = torch.tensor([50.0], dtype=torch.float64)
    zero = torch.zeros_like(delta_T)

    elastic = apply_thermal_strain(zero, zero, zero, delta_T=delta_T)
    stress = tm_source_effective_stress_fields(
        elastic["eps_xx"],
        elastic["eps_yy"],
        elastic["eps_xy"],
        alpha,
        matprop,
    )

    expected_elastic_normal = -DEFAULT_ALPHA_T * delta_T
    eps_zz = -matprop.mat_nu / (1.0 - matprop.mat_nu) * (2.0 * expected_elastic_normal)
    expected_sigma = matprop.mat_lmbda * (2.0 * expected_elastic_normal + eps_zz) + (
        2.0 * matprop.mat_mu * expected_elastic_normal
    )

    assert torch.allclose(elastic["eps_xx"], expected_elastic_normal)
    assert torch.allclose(elastic["eps_yy"], expected_elastic_normal)
    assert torch.equal(elastic["eps_xy"], zero)
    assert stress["sigma_xx_tm_total"].item() < 0.0
    assert stress["sigma_yy_tm_total"].item() < 0.0
    assert torch.allclose(stress["sigma_xx_tm_total"], expected_sigma)
    assert torch.allclose(stress["sigma_yy_tm_total"], expected_sigma)
    assert torch.equal(stress["sigma_xy_tm_total"], zero)


def test_thermal_strain_does_not_directly_modify_shear_component():
    exx = torch.tensor([1.0e-5], dtype=torch.float64)
    eyy = torch.tensor([-2.0e-5], dtype=torch.float64)
    exy = torch.tensor([3.0e-4], dtype=torch.float64)
    delta_T = torch.tensor([25.0], dtype=torch.float64)

    elastic = apply_thermal_strain(exx, eyy, exy, delta_T=delta_T)

    assert torch.allclose(elastic["eps_xx"], exx - DEFAULT_ALPHA_T * delta_T)
    assert torch.allclose(elastic["eps_yy"], eyy - DEFAULT_ALPHA_T * delta_T)
    assert torch.equal(elastic["eps_xy"], exy)


def test_compute_mixed_tm_fields_feeds_elastic_strain_into_split_route():
    dtype = torch.float64
    matprop = _matprop(dtype=dtype)
    pffmodel = _pffmodel()
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    area = torch.tensor([0.5], dtype=dtype)
    delta_T = torch.tensor([30.0], dtype=dtype)
    free_expansion = DEFAULT_ALPHA_T * delta_T
    u = free_expansion[0] * inp[:, 0]
    v = free_expansion[0] * inp[:, 1]
    alpha = torch.zeros(3, dtype=dtype)
    history = torch.zeros(1, dtype=dtype)

    fields = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        matprop,
        pffmodel,
        area,
        tri,
        thermal_delta_T=delta_T,
    )

    assert torch.allclose(fields["strain_11"], free_expansion)
    assert torch.allclose(fields["strain_22"], free_expansion)
    assert torch.allclose(fields["strain_12"], torch.zeros_like(delta_T))
    assert torch.allclose(fields["eps_xx_elastic"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.allclose(fields["eps_yy_elastic"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.allclose(fields["eps_xy_elastic"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.allclose(fields["psi_total"], torch.zeros_like(delta_T), atol=1.0e-18)
    assert torch.allclose(fields["sigma_xx_tm_total"], torch.zeros_like(delta_T), atol=1.0e-12)
    assert torch.allclose(fields["sigma_yy_tm_total"], torch.zeros_like(delta_T), atol=1.0e-12)


def test_compute_mixed_tm_fields_zero_delta_t_matches_default_route():
    dtype = torch.float64
    matprop = _matprop(dtype=dtype)
    pffmodel = _pffmodel()
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    area = torch.tensor([0.5], dtype=dtype)
    u = 1.0e-4 * inp[:, 0] + 2.0e-5 * inp[:, 1]
    v = -3.0e-5 * inp[:, 0] + 4.0e-5 * inp[:, 1]
    alpha = torch.zeros(3, dtype=dtype)
    history = torch.zeros(1, dtype=dtype)

    baseline = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        matprop,
        pffmodel,
        area,
        tri,
    )
    zero_thermal = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        matprop,
        pffmodel,
        area,
        tri,
        thermal_delta_T=torch.zeros(1, dtype=dtype),
    )

    for key in (
        "psiI",
        "psiII",
        "psi_minus",
        "psi_total",
        "He_current",
        "elastic_energy_density",
        "sigma_xx_tm_total",
        "sigma_yy_tm_total",
        "sigma_xy_tm_total",
    ):
        assert torch.equal(zero_thermal[key], baseline[key])


def test_no_heat_pde_or_damage_dependent_conductivity_was_added():
    scanned_files = [
        path
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in scanned_files)

    forbidden_active_tokens = [
        "heat_equation_residual",
        "thermal_pde_residual",
        "temperature_pde",
        "trainable_temperature",
        "k_d = g(d)*k0",
        "k_d=g(d)*k0",
        "damage_dependent_conductivity = True",
    ]
    for token in forbidden_active_tokens:
        assert token not in combined
