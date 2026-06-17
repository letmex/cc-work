from pathlib import Path
import inspect
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy_mixed_tm import compute_mixed_tm_fields  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from pff_model import PFFModel  # noqa: E402
from thermal_prescribed import DEFAULT_ALPHA_T, apply_thermal_strain, prescribed_delta_T  # noqa: E402
import thermal_solution_bridge  # noqa: E402


DTYPE = torch.float64


def _matprop(dtype=DTYPE):
    return MaterialProperties(
        mat_E=torch.tensor(81.5, dtype=dtype),
        mat_nu=torch.tensor(0.38, dtype=dtype),
        w1=torch.tensor(2.4e-6 / 1.5e-4, dtype=dtype),
        l0=torch.tensor(1.5e-4, dtype=dtype),
    )


def _pffmodel():
    return PFFModel(PFF_model="AT2", se_split="volumetric", tol_ir=torch.tensor(5.0e-3))


def _mechanics_patch_inputs(dtype=DTYPE):
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    area = torch.tensor([0.5], dtype=dtype)
    alpha = torch.zeros(3, dtype=dtype)
    history = torch.zeros(1, dtype=dtype)
    return inp, tri, area, alpha, history


def test_uniform_solved_temperature_bridge_matches_prescribed_delta_t_and_strain():
    coords_mm = torch.tensor([[0.0, 0.0], [0.004, 0.007], [0.01, 0.01]], dtype=DTYPE)

    result = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        coords_mm,
        source_mode="prescribed_uniform",
        uniform_temperature_K=320.0,
        T_ref_K=300.0,
    )
    prescribed = prescribed_delta_T(mode="uniform", y=coords_mm[:, 1], delta_T0=20.0)
    bridge_elastic = apply_thermal_strain(
        torch.zeros_like(prescribed),
        torch.zeros_like(prescribed),
        torch.zeros_like(prescribed),
        delta_T=result["thermal_delta_T"],
    )
    prescribed_elastic = apply_thermal_strain(
        torch.zeros_like(prescribed),
        torch.zeros_like(prescribed),
        torch.zeros_like(prescribed),
        delta_T=prescribed,
    )

    assert result["source_mode"] == "prescribed_uniform"
    assert torch.allclose(result["temperature_K"], torch.full((3,), 320.0, dtype=DTYPE))
    assert torch.allclose(result["thermal_delta_T"], prescribed)
    assert torch.allclose(bridge_elastic["eps_xx"], prescribed_elastic["eps_xx"])
    assert torch.allclose(bridge_elastic["eps_yy"], prescribed_elastic["eps_yy"])
    assert torch.equal(bridge_elastic["eps_xy"], prescribed_elastic["eps_xy"])


def test_zero_temperature_increment_bridge_matches_default_mechanics_route():
    inp, tri, area, alpha, history = _mechanics_patch_inputs()
    u = 1.0e-4 * inp[:, 0] + 2.0e-5 * inp[:, 1]
    v = -3.0e-5 * inp[:, 0] + 4.0e-5 * inp[:, 1]

    bridge = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        inp,
        source_mode="prescribed_uniform",
        uniform_temperature_K=300.0,
        T_ref_K=300.0,
    )
    baseline = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        tri,
    )
    bridged = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        tri,
        thermal_delta_T=bridge["thermal_delta_T"],
    )

    assert torch.allclose(bridge["thermal_delta_T"], torch.zeros_like(bridge["thermal_delta_T"]))
    for key in ("psi_total", "elastic_energy_density", "sigma_xx_tm_total", "sigma_yy_tm_total"):
        assert torch.equal(bridged[key], baseline[key])


def test_h1_solved_frozen_lift_bridge_returns_linear_delta_t_without_training():
    coords_mm = torch.tensor(
        [[0.0, 0.0], [0.0025, 0.0025], [0.005, 0.005], [0.01, 0.01]],
        dtype=DTYPE,
    )
    bounds_mm = torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE)

    result = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        coords_mm,
        source_mode="solved_frozen_lift",
        case_id="H1",
        bounds_mm=bounds_mm,
        T_bottom_K=300.0,
        T_top_K=320.0,
        T_ref_K=300.0,
    )
    eta = coords_mm[:, 1] / 0.01

    assert result["source_mode"] == "solved_frozen_lift"
    assert result["case_id"] == "H1"
    assert result["correction_policy"] == "frozen_lift"
    assert result["network_training_run"] is False
    assert torch.allclose(result["temperature_K"], 300.0 + 20.0 * eta)
    assert torch.allclose(result["thermal_delta_T"], 20.0 * eta)
    assert torch.allclose(result["correction_K"], torch.zeros_like(eta))


def test_h2_solved_frozen_lift_bridge_returns_uniform_bottom_temperature():
    coords_mm = torch.tensor([[0.0, 0.0], [0.006, 0.004], [0.01, 0.01]], dtype=DTYPE)
    bounds_mm = torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE)

    result = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        coords_mm,
        source_mode="solved_frozen_lift",
        case_id="H2",
        bounds_mm=bounds_mm,
        T_bottom_K=300.0,
        T_ref_K=300.0,
    )

    assert result["correction_policy"] == "frozen_lift"
    assert result["network_training_run"] is False
    assert torch.allclose(result["temperature_K"], torch.full((3,), 300.0, dtype=DTYPE))
    assert torch.allclose(result["thermal_delta_T"], torch.zeros(3, dtype=DTYPE))
    assert torch.allclose(result["correction_K"], torch.zeros(3, dtype=DTYPE))


def test_bridge_uniform_delta_t_matches_prescribed_mechanics_fields():
    inp, tri, area, alpha, history = _mechanics_patch_inputs()
    delta_T = torch.tensor([20.0], dtype=DTYPE)
    free_expansion = DEFAULT_ALPHA_T * delta_T
    u = free_expansion[0] * inp[:, 0]
    v = free_expansion[0] * inp[:, 1]

    bridge = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        inp,
        source_mode="prescribed_uniform",
        uniform_delta_T_K=20.0,
        T_ref_K=300.0,
    )
    prescribed = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        tri,
        thermal_delta_T=delta_T,
    )
    bridged = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        tri,
        thermal_delta_T=bridge["thermal_delta_T"],
    )

    for key in (
        "thermal_delta_T",
        "thermal_eps_xx",
        "thermal_eps_yy",
        "eps_xx_elastic",
        "eps_yy_elastic",
        "psi_total",
        "sigma_xx_tm_total",
        "sigma_yy_tm_total",
    ):
        assert torch.allclose(bridged[key], prescribed[key], atol=1.0e-18)


def test_bridge_defaults_do_not_switch_mechanics_to_solved_temperature():
    assert thermal_solution_bridge.DEFAULT_TEMPERATURE_SOURCE_MODE == "prescribed_uniform"

    signature = inspect.signature(compute_mixed_tm_fields)
    assert signature.parameters["thermal_mode"].default == "off"
    assert signature.parameters["thermal_temperature"].default is None
    assert signature.parameters["thermal_delta_T"].default is None


def test_bridge_source_has_no_forbidden_heat_fracture_or_damage_tokens():
    source = Path(thermal_solution_bridge.__file__).read_text(encoding="utf-8")
    forbidden_tokens = [
        "k(d)",
        "g(d)",
        "k_d",
        "damage_dependent_conductivity",
        "alpha_conductivity",
        "heat_fracture",
        "D0040",
        "seed study",
    ]
    for token in forbidden_tokens:
        assert token not in source
