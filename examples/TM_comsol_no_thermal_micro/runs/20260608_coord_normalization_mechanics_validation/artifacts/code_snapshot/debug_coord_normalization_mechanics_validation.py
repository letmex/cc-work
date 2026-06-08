"""Coordinate-normalization mechanics-only validation.

This diagnostic promotes coordinate normalization as an NN-input ansatz option
and validates it against the accepted alpha=0 exact-FE target. It keeps the
physical mechanics model unchanged: alpha is fixed to zero, T3 gradients use
physical coordinates, and no local notch/lip loss or geometry-target guidance
is used.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_exact_fe_elastic_solve import (  # noqa: E402
    _assemble_stiffness,
    _classify,
    _load_mesh,
    _region_masks,
    _stats,
)
from debug_exact_fe_target_prefit import (  # noqa: E402
    _energy_loss,
    _field_from_state,
    _fit,
    _prefit_metric_row,
    _save_case_npz,
    _state_dict,
)
from debug_prefit_then_energy_mechanics import _evaluate, _load_target, _make_field  # noqa: E402
from validate_mechanics_target import exact_reference_from_npz, write_rows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Validate unit-box coordinate normalization on alpha=0 mechanics.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=ROOT / "geo_coarse_with_groups_mm.msh")
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--E", type=float, default=81.5)
    parser.add_argument("--nu", type=float, default=0.38)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--eta-residual", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--prefit-epochs", type=int, default=800)
    parser.add_argument("--random-energy-epochs", type=int, default=300)
    parser.add_argument("--continuation-epochs", default="10,30,100,300")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--strain-weight", type=float, default=1.0e-5)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _prepare_output_dirs(out_dir):
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _write_csv(path, rows):
    write_rows(path, rows)


def _parse_epochs(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _quick_trim(args):
    if not args.quick:
        return
    args.hidden_layers = min(args.hidden_layers, 2)
    args.neurons = min(args.neurons, 40)
    args.prefit_epochs = min(args.prefit_epochs, 20)
    args.random_energy_epochs = min(args.random_energy_epochs, 10)
    args.continuation_epochs = "2,5"


def _target_numpy_exact(target_path):
    with np.load(target_path) as data:
        return {
            "u": np.asarray(data["u"], dtype=np.float64),
            "v": np.asarray(data["v"], dtype=np.float64),
            "eps_xx": np.asarray(data["eps_xx"], dtype=np.float64),
            "eps_yy": np.asarray(data["eps_yy"], dtype=np.float64),
            "eps_xy": np.asarray(data["eps_xy"], dtype=np.float64),
            "He_current": np.asarray(data["He_current"], dtype=np.float64),
            "standard_internal_energy": float(data["standard_energy"]),
            "current_pinn_mechanics_energy": float(data["pinn_mechanics_energy"]),
            "reaction_top_v_N": float(data["reaction_top_v_N"]),
        }


def _case_field(args, coord_normalization, device):
    args.coord_normalization = coord_normalization
    return _make_field(args, "free", device)


def _fit_energy(field, target, args, device, epochs, energy_mode, trace_rows, case_id, energy_scale=None):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    for epoch in range(epochs):

        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target, args, device)
            loss = _energy_loss(state, energy_mode, energy_scale)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) in {10, 30, 100, 300}:
            with torch.no_grad():
                state = _evaluate(field, target, args, device)
            trace_rows.append(
                {
                    "case_id": case_id,
                    "epoch": epoch + 1,
                    "loss_kind": "energy",
                    "energy_mode": energy_mode,
                    "loss": float(loss.detach().cpu()),
                    "displacement_mse": float(state["disp_loss"].detach().cpu()),
                    "strain_mse": float(state["strain_loss"].detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )


def _metric(case_id, stage, coord_normalization, prefit_kind, state, target, exact, K, x, y, tri, area, args):
    row = _prefit_metric_row(case_id, stage, prefit_kind, state, target, exact, K, x, y, tri, area, args)
    row["coord_normalization"] = coord_normalization
    row["t3_gradients_use_physical_xy"] = True
    row["local_geometry_loss"] = "none"
    row["success_displacement_rel_rmse_lt_0p05"] = row["displacement_rel_rmse"] < 0.05
    row["success_strain_rel_rmse_lt_0p2"] = row["strain_rel_rmse"] < 0.2
    row["success_He_current_corr_gt_0p8"] = row["He_current_corr"] > 0.8
    row["success_energy_ratio_lt_1p5"] = row["pinn_mechanics_energy_ratio_vs_exact"] < 1.5
    row["success_reaction_ratio_0p8_1p2"] = (
        np.isfinite(row["reaction_ratio"]) and 0.8 <= row["reaction_ratio"] <= 1.2
    )
    return row


def _mapping_row(case_id, field, target):
    row = {"case_id": case_id}
    row.update(field.coord_mapping_diagnostics(target["inp"]))
    row["physical_x_min"] = float(torch.min(target["inp"][:, 0]).detach().cpu())
    row["physical_x_max"] = float(torch.max(target["inp"][:, 0]).detach().cpu())
    row["physical_y_min"] = float(torch.min(target["inp"][:, 1]).detach().cpu())
    row["physical_y_max"] = float(torch.max(target["inp"][:, 1]).detach().cpu())
    return row


def _boundary_row(row):
    return {
        "case_id": row["case_id"],
        "stage": row["stage"],
        "coord_normalization": row["coord_normalization"],
        "boundary_residual_abs_max": row["boundary_residual_abs_max"],
        "bottom_u_abs_max": row.get("bottom_u_abs_max", np.nan),
        "bottom_v_abs_max": row.get("bottom_v_abs_max", np.nan),
        "top_v_minus_delta_abs_max": row.get("top_v_minus_delta_abs_max", np.nan),
        "reaction_N": row["reaction_N"],
        "reaction_ratio": row["reaction_ratio"],
        "reaction_sign_match": row["reaction_sign_match"],
    }


def _drift_row(initial, final, energy_mode, continuation_epochs):
    return {
        "case_id": final["case_id"],
        "energy_mode": energy_mode,
        "continuation_epochs": continuation_epochs,
        "initial_strain_rel_rmse": initial["strain_rel_rmse"],
        "final_strain_rel_rmse": final["strain_rel_rmse"],
        "strain_rel_rmse_drift": final["strain_rel_rmse"] - initial["strain_rel_rmse"],
        "initial_He_current_corr": initial["He_current_corr"],
        "final_He_current_corr": final["He_current_corr"],
        "He_current_corr_drift": final["He_current_corr"] - initial["He_current_corr"],
        "initial_reaction_ratio": initial["reaction_ratio"],
        "final_reaction_ratio": final["reaction_ratio"],
        "reaction_ratio_drift": final["reaction_ratio"] - initial["reaction_ratio"],
        "initial_pinn_mechanics_energy_ratio_vs_exact": initial["pinn_mechanics_energy_ratio_vs_exact"],
        "final_pinn_mechanics_energy_ratio_vs_exact": final["pinn_mechanics_energy_ratio_vs_exact"],
        "energy_ratio_reduction": (
            initial["pinn_mechanics_energy_ratio_vs_exact"] - final["pinn_mechanics_energy_ratio_vs_exact"]
        ),
        "initial_max_He_current_x": initial["max_He_current_x"],
        "initial_max_He_current_y": initial["max_He_current_y"],
        "final_max_He_current_x": final["max_He_current_x"],
        "final_max_He_current_y": final["max_He_current_y"],
        "initial_classification": initial["classification"],
        "final_classification": final["classification"],
        "boundary_dominated_branch": final["classification"] == "boundary-dominated",
    }


def _initial_case(args, target, device, coord_normalization, epochs, trace_rows):
    field = _case_field(args, coord_normalization, device)
    _fit_energy(
        field,
        target,
        args,
        device,
        epochs,
        "log10",
        trace_rows,
        f"{coord_normalization}_random_init_energy_log10_e{epochs}",
    )
    return field, _evaluate(field, target, args, device)


def main():
    args = parse_args()
    _quick_trim(args)
    args.top_u_mode = "free"
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    commands = {
        "script": str(Path(__file__).resolve()),
        "device": device,
        "args": vars(args),
        "physics_model_changes": "none",
        "training_loss_geometry_guidance": "none",
        "alpha_mode": "zero_fixed",
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    target = _load_target(args.target, device)
    x, y, tri, area = _load_mesh(args.mesh)
    K = _assemble_stiffness(x, y, tri, area, args.E, args.nu)
    exact = _target_numpy_exact(args.target)
    exact_checked = exact_reference_from_npz(args.target, args, K=K, area=area)
    exact.update(
        {
            "eps_xx": exact_checked["eps_xx"],
            "eps_yy": exact_checked["eps_yy"],
            "eps_xy": exact_checked["eps_xy"],
            "He_current": exact_checked["He_current"],
            "standard_internal_energy": exact_checked["standard_internal_energy"],
            "current_pinn_mechanics_energy": exact_checked["current_pinn_mechanics_energy"],
            "reaction_top_v_N": exact_checked["reaction_top_v_N"],
        }
    )

    all_rows = []
    boundary_rows = []
    mapping_rows = []
    drift_rows = []
    trace_rows = []

    for coord_normalization in ("none", "unit_box"):
        field, state = _initial_case(
            args,
            target,
            device,
            coord_normalization,
            args.random_energy_epochs,
            trace_rows,
        )
        case_id = f"{coord_normalization}_random_init_energy_log10_e{args.random_energy_epochs}"
        row = _metric(case_id, "energy_end", coord_normalization, "none", state, target, exact, K, x, y, tri, area, args)
        all_rows.append(row)
        boundary_rows.append(_boundary_row(row))
        mapping_rows.append(_mapping_row(case_id, field, target))
        _save_case_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target, state)

    field = _case_field(args, "unit_box", device)
    _fit(
        field,
        target,
        args,
        device,
        args.prefit_epochs,
        "disp_strain_global",
        trace_rows,
        "unit_box_disp_strain_prefit",
    )
    prefit_state = _evaluate(field, target, args, device)
    prefit_row = _metric(
        "unit_box_disp_strain_prefit",
        "prefit_end",
        "unit_box",
        "disp_strain_global",
        prefit_state,
        target,
        exact,
        K,
        x,
        y,
        tri,
        area,
        args,
    )
    all_rows.append(prefit_row)
    boundary_rows.append(_boundary_row(prefit_row))
    mapping_rows.append(_mapping_row("unit_box_disp_strain_prefit", field, target))
    start_state = _state_dict(field)
    torch.save(start_state, args.out_dir / "artifacts" / "unit_box_disp_strain_prefit_state.pt")
    _save_case_npz(args.out_dir / "artifacts" / "unit_box_disp_strain_prefit_fields.npz", target, prefit_state)

    energy_scale = torch.clamp(
        prefit_state["E_el"] + prefit_state["E_d"],
        min=torch.finfo(prefit_state["E_el"].dtype).tiny,
    ).detach()
    for energy_mode in ("raw", "log10", "normalized"):
        for epochs in _parse_epochs(args.continuation_epochs):
            cont_field = _field_from_state(args, start_state, device)
            args.coord_normalization = "unit_box"
            case_id = f"unit_box_prefit_energy_{energy_mode}_e{epochs}"
            _fit_energy(
                cont_field,
                target,
                args,
                device,
                epochs,
                energy_mode,
                trace_rows,
                case_id,
                energy_scale=energy_scale,
            )
            state = _evaluate(cont_field, target, args, device)
            row = _metric(
                case_id,
                "energy_end",
                "unit_box",
                "disp_strain_global",
                state,
                target,
                exact,
                K,
                x,
                y,
                tri,
                area,
                args,
            )
            row["energy_mode"] = energy_mode
            row["continuation_epochs"] = epochs
            all_rows.append(row)
            boundary_rows.append(_boundary_row(row))
            drift_rows.append(_drift_row(prefit_row, row, energy_mode, epochs))
            _save_case_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target, state)

    _write_csv(args.out_dir / "tables" / "coord_normalization_case_comparison.csv", all_rows)
    _write_csv(args.out_dir / "tables" / "mechanics_validation_metrics.csv", all_rows)
    _write_csv(args.out_dir / "tables" / "energy_continuation_drift.csv", drift_rows)
    _write_csv(args.out_dir / "tables" / "boundary_residuals.csv", boundary_rows)
    _write_csv(args.out_dir / "tables" / "coord_mapping_diagnostics.csv", mapping_rows)
    _write_csv(args.out_dir / "logs" / "loss_trace.csv", trace_rows)
    (args.out_dir / "figures" / "figure_summary.md").write_text(
        "# Figure Summary\n\nNo PNG figures were generated. Diagnostic evidence is in CSV tables and compact NPZ artifacts.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
