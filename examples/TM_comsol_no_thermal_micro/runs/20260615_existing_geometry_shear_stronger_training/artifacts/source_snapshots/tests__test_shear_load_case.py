from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from field_computation import FieldComputation  # noqa: E402


class ConstantNet(nn.Module):
    def forward(self, inp):
        out = torch.zeros((inp.shape[0], 3), dtype=inp.dtype, device=inp.device)
        out[:, 0] = 0.25
        out[:, 1] = 0.40
        return out


def _shear_field():
    return FieldComputation(
        net=ConstantNet(),
        domain_extrema=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=torch.float32),
        lmbda=torch.tensor([2.0e-6], dtype=torch.float32),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32),
        alpha_constraint="nonsmooth",
        top_u_mode="free",
        coord_normalization="unit_box",
        load_case="shear",
    )


def test_shear_ansatz_imposes_horizontal_top_displacement_and_bottom_fixity():
    field = _shear_field()
    inp = torch.tensor(
        [
            [0.0, 0.0],
            [0.005, 0.005],
            [0.01, 0.01],
        ],
        dtype=torch.float32,
    )

    u, v, _alpha = field.fieldCalculation(inp)

    assert torch.isclose(u[0], torch.tensor(0.0), atol=1.0e-12)
    assert torch.isclose(v[0], torch.tensor(0.0), atol=1.0e-12)
    assert torch.isclose(u[-1], field.lmbda, atol=1.0e-12)
    assert torch.isclose(v[-1], 0.40 * field.lmbda, atol=1.0e-12)
    assert not torch.allclose(v[-1:], torch.zeros_like(v[-1:]))


def test_tension_ansatz_keeps_existing_top_vertical_displacement():
    field = FieldComputation(
        net=ConstantNet(),
        domain_extrema=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=torch.float32),
        lmbda=torch.tensor([2.0e-6], dtype=torch.float32),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32),
        alpha_constraint="nonsmooth",
        top_u_mode="free",
        coord_normalization="unit_box",
        load_case="tension",
    )
    inp = torch.tensor([[0.01, 0.01]], dtype=torch.float32)

    _u, v, _alpha = field.fieldCalculation(inp)

    assert torch.isclose(v[0], field.lmbda, atol=1.0e-12)


def test_config_exposes_load_case_without_old_route_flags():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")

    assert "--load-case" in config_text
    assert '"tension"' in config_text
    assert '"shear"' in config_text
    for removed in ["--mixed-mechanics-mode", "--solve-scheme", "legacy_top_sigma"]:
        assert removed not in config_text


def test_shear_stress_strain_curve_uses_shear_labels():
    from postprocess_results import build_stress_strain_curve

    curve = build_stress_strain_curve(
        pd.DataFrame(
            [
                {
                    "seed": 23,
                    "step": 0,
                    "load_case": "shear",
                    "Delta": 2.0e-6,
                    "Delta_s": 2.0e-6,
                    "reaction_N_energy": 0.05,
                }
            ]
        ),
        reference_length_mm=0.01,
        reference_area_mm2=0.01,
        load_case="shear",
    )

    row = curve.iloc[0]
    assert row["stress_strain_primary_metric"] == "nominal_shear_stress_energy_MPa"
    assert abs(row["engineering_shear_strain"] - 2.0e-4) <= 1.0e-12
    assert abs(row["nominal_shear_stress_energy_MPa"] - 5.0) <= 1.0e-12
    assert "nominal_stress_energy_MPa" not in curve.columns


def test_normal_shear_outputs_do_not_restore_old_top_sigma_columns():
    combined = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ["postprocess_results.py", "plot_results.py"]
    )
    for removed in ["legacy_top_sigma", "reaction_N_legacy_top_sigma", "legacy_curve_status"]:
        assert removed not in combined


def test_shear_plot_label_supports_fuller_schedule_codes():
    from plot_results import infer_short_run_label

    label = infer_short_run_label("outputs/results/seed23_S0030_shear")

    assert label == "seed23_shear"
