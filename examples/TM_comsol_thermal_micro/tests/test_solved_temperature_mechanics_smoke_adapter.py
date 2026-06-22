from pathlib import Path
import ast
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
from thermal_mechanics_adapter import (  # noqa: E402
    build_element_thermal_delta_T_from_bridge,
    build_mechanics_thermal_kwargs_from_bridge,
    element_centroid_coords_mm,
)


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


def _two_element_patch_inputs(dtype=DTYPE):
    nodes = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]],
        dtype=dtype,
    )
    elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    area = torch.tensor([0.5e-4, 0.5e-4], dtype=dtype)
    alpha = torch.zeros(4, dtype=dtype)
    history = torch.zeros(2, dtype=dtype)
    u = 1.0e-4 * nodes[:, 0] + 2.0e-5 * nodes[:, 1]
    v = -3.0e-5 * nodes[:, 0] + 4.0e-5 * nodes[:, 1]
    return nodes, elements, area, alpha, history, u, v


def test_element_centroid_coords_mm_returns_one_point_per_triangle():
    nodes = torch.tensor(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 6.0], [3.0, 6.0]],
        dtype=DTYPE,
    )
    elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

    centroids = element_centroid_coords_mm(nodes, elements)

    expected = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=DTYPE)
    assert centroids.shape == (2, 2)
    assert torch.allclose(centroids, expected)


def test_uniform_bridge_adapter_returns_element_sized_delta_t():
    nodes, elements, *_ = _two_element_patch_inputs()

    result = build_element_thermal_delta_T_from_bridge(
        nodes,
        elements,
        source_mode="prescribed_uniform",
        uniform_delta_T_K=20.0,
        T_ref_K=300.0,
    )

    assert result["source_mode"] == "prescribed_uniform"
    assert result["evaluation_location"] == "element_centroid"
    assert result["element_centroid_coords_mm"].shape == (2, 2)
    assert result["thermal_delta_T"].shape == (2,)
    assert torch.allclose(result["thermal_delta_T"], torch.full((2,), 20.0, dtype=DTYPE))


def test_h1_frozen_lift_adapter_samples_delta_t_at_element_centroids():
    nodes, elements, *_ = _two_element_patch_inputs()
    bounds_mm = torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE)

    result = build_element_thermal_delta_T_from_bridge(
        nodes,
        elements,
        source_mode="solved_frozen_lift",
        case_id="H1",
        bounds_mm=bounds_mm,
        T_bottom_K=300.0,
        T_top_K=320.0,
        T_ref_K=300.0,
    )
    eta = result["element_centroid_coords_mm"][:, 1] / 0.01

    assert result["source_mode"] == "solved_frozen_lift"
    assert result["case_id"] == "H1"
    assert result["correction_policy"] == "frozen_lift"
    assert result["network_training_run"] is False
    assert torch.allclose(result["thermal_delta_T"], 20.0 * eta)
    assert torch.allclose(result["correction_K"], torch.zeros_like(eta))


def test_adapter_thermal_kwargs_match_direct_element_delta_t_mechanics_fields():
    nodes, elements, area, alpha, history, u, v = _two_element_patch_inputs()
    adapter = build_mechanics_thermal_kwargs_from_bridge(
        nodes,
        elements,
        source_mode="solved_frozen_lift",
        case_id="H1",
        bounds_mm=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE),
        T_bottom_K=300.0,
        T_top_K=320.0,
        T_ref_K=300.0,
    )

    direct = compute_mixed_tm_fields(
        nodes,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        elements,
        thermal_delta_T=adapter["diagnostics"]["thermal_delta_T"],
    )
    bridged = compute_mixed_tm_fields(
        nodes,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        elements,
        **adapter["thermal_kwargs"],
    )

    assert set(adapter["thermal_kwargs"]) == {"thermal_delta_T"}
    assert adapter["thermal_kwargs"]["thermal_delta_T"].shape == area.shape
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
        assert torch.allclose(bridged[key], direct[key], atol=1.0e-18)


def test_adapter_does_not_change_mechanics_defaults_or_training_entrypoint():
    signature = inspect.signature(compute_mixed_tm_fields)
    assert signature.parameters["thermal_mode"].default == "off"
    assert signature.parameters["thermal_temperature"].default is None
    assert signature.parameters["thermal_delta_T"].default is None

    train_source = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    tree = ast.parse(train_source)
    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(alias.name == "thermal_mechanics_adapter" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "thermal_mechanics_adapter"
            )
        )
    ]
    assert module_imports == []


def test_adapter_source_has_no_forbidden_heat_fracture_or_study_tokens():
    source = (ROOT / "thermal_mechanics_adapter.py").read_text(encoding="utf-8")
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
