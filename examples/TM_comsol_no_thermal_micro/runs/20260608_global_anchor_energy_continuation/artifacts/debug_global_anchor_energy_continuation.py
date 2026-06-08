"""Global-anchor mechanics-energy continuation diagnostics.

This script tests whether an FE-DOF-like alpha=0 mechanics branch can be
preserved during energy continuation using only global, non-geometry-specific
controls. No notch/lip masks, local displacement jumps, local weights, or
geometry labels are used in any training loss. Region metrics are imported from
the previous diagnostic only for postprocessing classification.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from debug_prefit_then_energy_mechanics import (
    _evaluate,
    _load_target,
    _make_field,
    _metrics,
    _plot_figures,
    _prepare_output_dirs,
    _save_case_npz,
    _write_rows,
)


SUCCESS_THRESHOLDS = {
    "He_current_corr_min": 0.8,
    "strain_corr_min": 0.9,
    "bulk_to_notch_He_current_max": 0.35,
    "bottom_to_notch_He_current_max": 0.1,
    "displacement_rel_rmse_max": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run global-anchor alpha-zero mechanics-energy continuation diagnostics."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-free", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--prefit-epochs", type=int, default=1000)
    parser.add_argument("--continuation-epochs", type=int, default=300)
    parser.add_argument("--trust-chunks", type=int, default=10)
    parser.add_argument("--trust-epochs-per-chunk", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--strain-weight", type=float, default=1.0e-5)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument(
        "--lambda-u",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
    )
    parser.add_argument(
        "--lambda-eps",
        type=float,
        nargs="+",
        default=[1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
    )
    parser.add_argument(
        "--combo-pairs",
        default="1e-4:1e-6,1e-3:1e-5,1e-2:1e-4,1e-1:1e-3",
        help="Comma-separated lambda_u:lambda_eps pairs for combined global anchors.",
    )
    parser.add_argument(
        "--trust-lambda-u",
        type=float,
        nargs="+",
        default=[1.0e-3, 1.0e-2, 1.0e-1],
    )
    parser.add_argument(
        "--trust-lambda-eps",
        type=float,
        default=0.0,
        help="Optional global strain-change penalty for trust-region continuation.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small subset for script smoke testing.",
    )
    return parser.parse_args()


def _parse_combo_pairs(text):
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        left, right = item.split(":")
        pairs.append((float(left), float(right)))
    return pairs


def _refs(target):
    disp_ref = torch.mean(target["u"] ** 2 + target["v"] ** 2)
    strain_ref = torch.mean(target["eps_xx"] ** 2 + target["eps_yy"] ** 2 + target["eps_xy"] ** 2)
    return {
        "disp_ref": torch.clamp(disp_ref, min=torch.finfo(disp_ref.dtype).tiny),
        "strain_ref": torch.clamp(strain_ref, min=torch.finfo(strain_ref.dtype).tiny),
    }


def _energy_objective(state, energy_mode, energy_scale):
    raw = state["E_el"] + state["E_d"]
    if energy_mode == "raw":
        return raw
    if energy_mode == "log10":
        return state["energy_loss"]
    if energy_mode == "normalized":
        return raw / energy_scale
    raise ValueError(f"unknown energy_mode {energy_mode!r}")


def _state_dict(field):
    return {key: value.detach().cpu().clone() for key, value in field.net.state_dict().items()}


def _field_from_state(args, state_dict, device):
    field = _make_field(args, "free", device)
    field.net.load_state_dict(state_dict)
    return field


def _save_checkpoint(path, field):
    torch.save(_state_dict(field), path)


def _top_boundary_residuals(target, state, args):
    tol = 1.0e-9
    y = target["y"]
    top = torch.abs(y - 0.01) <= tol
    bottom = torch.abs(y) <= tol
    u = state["u"]
    v = state["v"]
    result = {
        "bottom_u_abs_max": np.nan,
        "bottom_v_abs_max": np.nan,
        "top_v_minus_delta_abs_max": np.nan,
        "top_u_free_abs_mean": np.nan,
    }
    if torch.any(bottom):
        result["bottom_u_abs_max"] = float(torch.max(torch.abs(u[bottom])).detach().cpu())
        result["bottom_v_abs_max"] = float(torch.max(torch.abs(v[bottom])).detach().cpu())
    if torch.any(top):
        result["top_v_minus_delta_abs_max"] = float(torch.max(torch.abs(v[top] - args.delta)).detach().cpu())
        result["top_u_free_abs_mean"] = float(torch.mean(torch.abs(u[top])).detach().cpu())
    return result


def _threshold_status(row):
    checks = {
        "pass_He_current_corr": row["He_current_corr"] >= SUCCESS_THRESHOLDS["He_current_corr_min"],
        "pass_strain_corr": row["strain_corr"] >= SUCCESS_THRESHOLDS["strain_corr_min"],
        "pass_bulk_to_notch": row["bulk_to_notch_He_current"]
        <= SUCCESS_THRESHOLDS["bulk_to_notch_He_current_max"],
        "pass_bottom_to_notch": row["bottom_to_notch_He_current"]
        <= SUCCESS_THRESHOLDS["bottom_to_notch_He_current_max"],
        "pass_displacement_rel_rmse": row["displacement_rel_rmse"]
        < SUCCESS_THRESHOLDS["displacement_rel_rmse_max"],
    }
    checks["success_all_thresholds"] = all(bool(v) for v in checks.values())
    return checks


def _classify_with_thresholds(row):
    checks = _threshold_status(row)
    if checks["success_all_thresholds"]:
        return "target-like"
    max_x = row["max_He_current_x"]
    max_y = row["max_He_current_y"]
    boundary = max_x <= 5.0e-4 or max_x >= 9.5e-3 or max_y <= 5.0e-4 or max_y >= 9.5e-3
    if boundary:
        return "boundary-dominated"
    if (
        np.isfinite(row["bulk_to_notch_He_current"])
        and np.isfinite(row["bottom_to_notch_He_current"])
        and row["bulk_to_notch_He_current"] <= 0.35
        and row["bottom_to_notch_He_current"] <= 0.1
    ):
        return "notch-amplified"
    if row["displacement_rel_rmse"] > 0.9 and row["strain_rel_rmse"] > 0.9 and row["He_current_corr"] < 0.2:
        return "collapsed/non-target"
    if np.isfinite(row["bulk_to_notch_He_current"]) and 0.5 <= row["bulk_to_notch_He_current"] <= 2.0:
        return "broad/background"
    return "collapsed/non-target"


def _record_metrics(
    rows,
    sweep_rows,
    checkpoint_rows,
    success_rows,
    target,
    state,
    args,
    case_id,
    case_group,
    start_prefit,
    continuation_mode,
    energy_mode,
    energy_scale,
    lambda_u=0.0,
    lambda_eps=0.0,
    lambda_trust_u=0.0,
    lambda_trust_eps=0.0,
    checkpoint="final",
    save_npz=True,
):
    row, _energy_row, _strain_row = _metrics(
        case_id=case_id,
        top_u_mode="free",
        case_name=case_group,
        prefit_kind=start_prefit,
        stage=checkpoint,
        target=target,
        state=state,
        args=args,
    )
    raw_energy = float((state["E_el"] + state["E_d"]).detach().cpu())
    row.update(
        {
            "case_group": case_group,
            "start_prefit": start_prefit,
            "continuation_mode": continuation_mode,
            "energy_mode": energy_mode,
            "energy_scale": energy_scale,
            "normalized_mechanics_energy": raw_energy / energy_scale if energy_scale else np.nan,
            "lambda_u": lambda_u,
            "lambda_eps": lambda_eps,
            "lambda_trust_u": lambda_trust_u,
            "lambda_trust_eps": lambda_trust_eps,
            "checkpoint": checkpoint,
        }
    )
    row.update(_top_boundary_residuals(target, state, args))
    row.update(_threshold_status(row))
    row["classification"] = _classify_with_thresholds(row)
    rows.append(row)
    if checkpoint == "final":
        sweep_rows.append(row.copy())
    checkpoint_rows.append(row.copy())
    success_rows.append(
        {
            "case_id": case_id,
            "case_group": case_group,
            "start_prefit": start_prefit,
            "continuation_mode": continuation_mode,
            "energy_mode": energy_mode,
            "lambda_u": lambda_u,
            "lambda_eps": lambda_eps,
            "lambda_trust_u": lambda_trust_u,
            "lambda_trust_eps": lambda_trust_eps,
            "displacement_rel_rmse": row["displacement_rel_rmse"],
            "strain_corr": row["strain_corr"],
            "He_current_corr": row["He_current_corr"],
            "bulk_to_notch_He_current": row["bulk_to_notch_He_current"],
            "bottom_to_notch_He_current": row["bottom_to_notch_He_current"],
            "success_all_thresholds": row["success_all_thresholds"],
            "classification": row["classification"],
        }
    )
    if save_npz:
        _save_case_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target, state)


def _fit_prefit(field, target, args, device, kind, trace_rows):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    refs = _refs(target)
    for epoch in range(args.prefit_epochs):
        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target, args, device)
            disp_rel = state["disp_loss"] / refs["disp_ref"]
            strain_rel = state["strain_loss"] / refs["strain_ref"]
            if kind == "disp_global":
                loss = disp_rel
            elif kind == "disp_strain_global":
                loss = disp_rel + args.strain_weight * strain_rel
            else:
                raise ValueError(kind)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == args.prefit_epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target, args, device)
            trace_rows.append(
                {
                    "case_id": f"{kind}_prefit",
                    "loss_kind": kind,
                    "epoch": epoch + 1,
                    "checkpoint": "prefit",
                    "loss": float(loss.detach().cpu()),
                    "disp_rel": float((state["disp_loss"] / refs["disp_ref"]).detach().cpu()),
                    "strain_rel": float((state["strain_loss"] / refs["strain_ref"]).detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )


def _fit_continuation(
    field,
    target,
    args,
    device,
    epochs,
    trace_rows,
    case_id,
    energy_mode,
    energy_scale,
    lambda_u=0.0,
    lambda_eps=0.0,
    prev_anchor=None,
    lambda_trust_u=0.0,
    lambda_trust_eps=0.0,
):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    refs = _refs(target)
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target, args, device)
            disp_rel = state["disp_loss"] / refs["disp_ref"]
            strain_rel = state["strain_loss"] / refs["strain_ref"]
            loss = _energy_objective(state, energy_mode, energy_scale)
            loss = loss + lambda_u * disp_rel + lambda_eps * strain_rel
            if prev_anchor is not None and lambda_trust_u:
                du = torch.mean((state["u"] - prev_anchor["u"]) ** 2 + (state["v"] - prev_anchor["v"]) ** 2)
                loss = loss + lambda_trust_u * du / refs["disp_ref"]
            if prev_anchor is not None and lambda_trust_eps:
                eps_xx, eps_yy, eps_xy = state["strains"]
                p_xx, p_yy, p_xy = prev_anchor["strains"]
                deps = torch.mean((eps_xx - p_xx) ** 2 + (eps_yy - p_yy) ** 2 + (eps_xy - p_xy) ** 2)
                loss = loss + lambda_trust_eps * deps / refs["strain_ref"]
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 30 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target, args, device)
            trace_rows.append(
                {
                    "case_id": case_id,
                    "loss_kind": "continuation",
                    "epoch": epoch + 1,
                    "checkpoint": "continuation",
                    "loss": float(loss.detach().cpu()),
                    "disp_rel": float((state["disp_loss"] / refs["disp_ref"]).detach().cpu()),
                    "strain_rel": float((state["strain_loss"] / refs["strain_ref"]).detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                    "energy_mode": energy_mode,
                    "lambda_u": lambda_u,
                    "lambda_eps": lambda_eps,
                    "lambda_trust_u": lambda_trust_u,
                    "lambda_trust_eps": lambda_trust_eps,
                }
            )


def _detached_anchor(state):
    return {
        "u": state["u"].detach().clone(),
        "v": state["v"].detach().clone(),
        "strains": tuple(item.detach().clone() for item in state["strains"]),
    }


def _run_from_prefit(
    start_state_dict,
    target,
    args,
    device,
    case_id,
    case_group,
    start_prefit,
    continuation_mode,
    energy_mode,
    energy_scale,
    rows,
    sweep_rows,
    checkpoint_rows,
    success_rows,
    trace_rows,
    lambda_u=0.0,
    lambda_eps=0.0,
    lambda_trust_u=0.0,
    lambda_trust_eps=0.0,
):
    field = _field_from_state(args, start_state_dict, device)
    _fit_continuation(
        field,
        target,
        args,
        device,
        args.continuation_epochs,
        trace_rows,
        case_id,
        energy_mode,
        energy_scale,
        lambda_u=lambda_u,
        lambda_eps=lambda_eps,
    )
    state = _evaluate(field, target, args, device)
    _record_metrics(
        rows,
        sweep_rows,
        checkpoint_rows,
        success_rows,
        target,
        state,
        args,
        case_id,
        case_group,
        start_prefit,
        continuation_mode,
        energy_mode,
        energy_scale,
        lambda_u=lambda_u,
        lambda_eps=lambda_eps,
        lambda_trust_u=lambda_trust_u,
        lambda_trust_eps=lambda_trust_eps,
    )


def _run_trust_region(
    start_state_dict,
    target,
    args,
    device,
    case_id,
    start_prefit,
    energy_scale,
    rows,
    sweep_rows,
    checkpoint_rows,
    success_rows,
    trace_rows,
    lambda_trust_u,
    lambda_trust_eps,
):
    field = _field_from_state(args, start_state_dict, device)
    prev_anchor = _detached_anchor(_evaluate(field, target, args, device))
    for chunk in range(args.trust_chunks):
        _fit_continuation(
            field,
            target,
            args,
            device,
            args.trust_epochs_per_chunk,
            trace_rows,
            f"{case_id}_chunk{chunk + 1:02d}",
            "normalized",
            energy_scale,
            prev_anchor=prev_anchor,
            lambda_trust_u=lambda_trust_u,
            lambda_trust_eps=lambda_trust_eps,
        )
        state = _evaluate(field, target, args, device)
        checkpoint = f"chunk_{chunk + 1:02d}"
        _record_metrics(
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            target,
            state,
            args,
            f"{case_id}_{checkpoint}",
            "trust_region_to_previous_step",
            start_prefit,
            "trust_region_to_previous_step",
            "normalized",
            energy_scale,
            lambda_trust_u=lambda_trust_u,
            lambda_trust_eps=lambda_trust_eps,
            checkpoint=checkpoint if chunk + 1 < args.trust_chunks else "final",
            save_npz=chunk + 1 == args.trust_chunks,
        )
        prev_anchor = _detached_anchor(state)


def _quick_trim(args):
    if not args.quick:
        return
    args.prefit_epochs = min(args.prefit_epochs, 100)
    args.continuation_epochs = min(args.continuation_epochs, 50)
    args.trust_chunks = 2
    args.trust_epochs_per_chunk = min(args.trust_epochs_per_chunk, 15)
    args.lambda_u = [1.0e-4, 1.0e-2]
    args.lambda_eps = [1.0e-6, 1.0e-4]
    args.combo_pairs = "1e-3:1e-5"
    args.trust_lambda_u = [1.0e-2]


def _write_outputs(args, rows, sweep_rows, checkpoint_rows, success_rows, trace_rows):
    _write_rows(args.out_dir / "tables" / "anchor_case_comparison.csv", rows)
    _write_rows(args.out_dir / "tables" / "anchor_sweep_metrics.csv", sweep_rows)
    _write_rows(args.out_dir / "tables" / "continuation_checkpoint_metrics.csv", checkpoint_rows)
    _write_rows(args.out_dir / "tables" / "success_threshold_summary.csv", success_rows)
    _write_rows(args.out_dir / "logs" / "loss_trace.csv", trace_rows)


def main():
    args = parse_args()
    _quick_trim(args)
    args.energy_epochs = args.continuation_epochs
    args.curriculum_epochs = 0
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = _load_target(args.target_free, device)
    rows = []
    sweep_rows = []
    checkpoint_rows = []
    success_rows = []
    trace_rows = []

    commands = {
        "script": str(Path(__file__).resolve()),
        "device": device,
        "args": vars(args),
        "training_loss_geometry_guidance": "none; region masks are postprocessing only",
        "success_thresholds": SUCCESS_THRESHOLDS,
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    prefit_states = {}
    prefit_energy_scales = {}
    for kind in ("disp_global", "disp_strain_global"):
        field = _make_field(args, "free", device)
        _fit_prefit(field, target, args, device, kind, trace_rows)
        state = _evaluate(field, target, args, device)
        prefit_states[kind] = _state_dict(field)
        raw_energy = float((state["E_el"] + state["E_d"]).detach().cpu())
        prefit_energy_scales[kind] = max(raw_energy, np.finfo(float).tiny)
        _save_checkpoint(args.out_dir / "artifacts" / f"{kind}_prefit_state.pt", field)
        _record_metrics(
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            target,
            state,
            args,
            f"prefit_{kind}",
            "prefit",
            kind,
            "prefit_only",
            "none",
            prefit_energy_scales[kind],
            checkpoint="prefit_end",
        )

    start_kind = "disp_strain_global"
    start_state = prefit_states[start_kind]
    scale = prefit_energy_scales[start_kind]

    _run_from_prefit(
        start_state,
        target,
        args,
        device,
        "pure_energy_baseline_log10",
        "pure_energy_baseline",
        start_kind,
        "pure_energy_baseline",
        "log10",
        scale,
        rows,
        sweep_rows,
        checkpoint_rows,
        success_rows,
        trace_rows,
    )

    for lam in args.lambda_u:
        _run_from_prefit(
            start_state,
            target,
            args,
            device,
            f"global_displacement_anchor_lamU_{lam:.0e}",
            "global_displacement_anchor",
            start_kind,
            "global_displacement_anchor",
            "normalized",
            scale,
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            trace_rows,
            lambda_u=lam,
        )

    for lam in args.lambda_eps:
        _run_from_prefit(
            start_state,
            target,
            args,
            device,
            f"global_strain_anchor_lamEps_{lam:.0e}",
            "global_strain_anchor",
            start_kind,
            "global_strain_anchor",
            "normalized",
            scale,
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            trace_rows,
            lambda_eps=lam,
        )

    for lam_u, lam_eps in _parse_combo_pairs(args.combo_pairs):
        _run_from_prefit(
            start_state,
            target,
            args,
            device,
            f"global_disp_strain_anchor_lamU_{lam_u:.0e}_lamEps_{lam_eps:.0e}",
            "global_displacement_plus_strain_anchor",
            start_kind,
            "global_displacement_plus_strain_anchor",
            "normalized",
            scale,
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            trace_rows,
            lambda_u=lam_u,
            lambda_eps=lam_eps,
        )

    for lam in args.trust_lambda_u:
        _run_trust_region(
            start_state,
            target,
            args,
            device,
            f"trust_region_lamU_{lam:.0e}",
            start_kind,
            scale,
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            trace_rows,
            lambda_trust_u=lam,
            lambda_trust_eps=args.trust_lambda_eps,
        )

    for mode in ("raw", "normalized"):
        _run_from_prefit(
            start_state,
            target,
            args,
            device,
            f"energy_normalization_{mode}",
            "energy_normalization_variants",
            start_kind,
            "energy_normalization_variants",
            mode,
            scale,
            rows,
            sweep_rows,
            checkpoint_rows,
            success_rows,
            trace_rows,
        )

    # Include a small check from the displacement-only prefit branch without
    # expanding the full sweep. This uses global displacement anchor only.
    _run_from_prefit(
        prefit_states["disp_global"],
        target,
        args,
        device,
        "disp_global_start_displacement_anchor_lamU_1e-02",
        "global_displacement_anchor",
        "disp_global",
        "global_displacement_anchor",
        "normalized",
        prefit_energy_scales["disp_global"],
        rows,
        sweep_rows,
        checkpoint_rows,
        success_rows,
        trace_rows,
        lambda_u=1.0e-2,
    )

    _write_outputs(args, rows, sweep_rows, checkpoint_rows, success_rows, trace_rows)
    if not args.skip_figures:
        generated = _plot_figures(args.out_dir, sweep_rows)
        (args.out_dir / "figures" / "generated_figures.json").write_text(
            json.dumps(generated, indent=2, default=str), encoding="utf-8"
        )
    print(f"wrote {args.out_dir / 'tables' / 'anchor_case_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'anchor_sweep_metrics.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'continuation_checkpoint_metrics.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'success_threshold_summary.csv'}")


if __name__ == "__main__":
    main()
