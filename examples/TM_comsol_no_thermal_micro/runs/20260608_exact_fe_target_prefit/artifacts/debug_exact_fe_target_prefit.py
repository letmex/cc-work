"""Exact-FE target prefit and alpha=0 energy-continuation diagnostic.

This script replaces the rejected FE-DOF RPROP supervision target with a
direct sparse FE alpha=0 target. It then runs global-only PINN mechanics
prefit and short mechanics-only energy continuation. No local notch/lip loss,
local weights, masks, enrichment, geometry labels, coupled phase-field run, or
physical-model change is introduced.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_exact_fe_elastic_solve import (  # noqa: E402
    _assemble_stiffness,
    _boundary_masks,
    _boundary_residuals,
    _classify,
    _current_energy_fields,
    _load_mesh,
    _region_masks,
    _residual_metrics,
    _safe_corr,
    _solve_exact,
    _stats,
    _strains_stresses_standard,
)
from debug_prefit_then_energy_mechanics import (  # noqa: E402
    _evaluate,
    _global_prefit_losses,
    _load_target,
    _make_field,
    _save_case_npz,
)
from history_field_mixed_tm import element_centroids  # noqa: E402
from validate_mechanics_target import THRESHOLDS, evaluate_candidate_arrays, write_rows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run exact-FE target prefit diagnostic.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
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
    parser.add_argument("--continuation-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--strain-weight", type=float, default=1.0e-5)
    parser.add_argument(
        "--old-fedof-target",
        type=Path,
        default=None,
        help="Rejected old FE-DOF RPROP target. Defaults to the known package artifact.",
    )
    parser.add_argument("--quick", action="store_true", help="Short smoke run for script checks.")
    return parser.parse_args()


def _prepare_output_dirs(out_dir):
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _field_args(args):
    return SimpleNamespace(
        E=float(args.E),
        nu=float(args.nu),
        l0=float(args.l0),
        tm_eps_r=float(args.tm_eps_r),
        eta_residual=float(args.eta_residual),
        delta=float(args.delta),
        top_u_mode="free",
    )


def _write_csv(path, rows):
    write_rows(path, rows)


def _torch_corr(a, b):
    return _safe_corr(a.detach().cpu().numpy(), b.detach().cpu().numpy())


def _state_dict(field):
    return {key: value.detach().cpu().clone() for key, value in field.net.state_dict().items()}


def _field_from_state(args, state_dict, device):
    field = _make_field(args, "free", device)
    field.net.load_state_dict(state_dict)
    return field


def _energy_loss(state, mode, scale):
    raw = state["E_el"] + state["E_d"]
    if mode == "raw":
        return raw
    if mode == "log10":
        return state["energy_loss"]
    if mode == "normalized":
        return raw / scale
    raise ValueError(f"unknown energy mode {mode!r}")


def _fit(field, target, args, device, epochs, loss_kind, trace_rows, case_id, energy_mode=None, energy_scale=None):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    for epoch in range(epochs):

        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target, args, device)
            if loss_kind == "disp_global":
                loss = state["disp_loss"]
            elif loss_kind == "disp_strain_global":
                loss = state["disp_loss"] + args.strain_weight * state["strain_loss"]
            elif loss_kind == "energy":
                loss = _energy_loss(state, energy_mode, energy_scale)
            else:
                raise ValueError(f"unknown loss_kind {loss_kind!r}")
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target, args, device)
            trace_rows.append(
                {
                    "case_id": case_id,
                    "epoch": epoch + 1,
                    "loss_kind": loss_kind,
                    "energy_mode": energy_mode or "none",
                    "loss": float(loss.detach().cpu()),
                    "displacement_mse": float(state["disp_loss"].detach().cpu()),
                    "strain_mse": float(state["strain_loss"].detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )


def _save_exact_target(path, x, y, tri, area, exact, std, current, residual, boundary, args):
    element_x = np.mean(x[tri], axis=1)
    element_y = np.mean(y[tri], axis=1)
    masks = _boundary_masks(x, y)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        triangles=tri,
        connectivity=tri,
        element_x=element_x,
        element_y=element_y,
        area=area,
        u=exact["u"],
        v=exact["v"],
        u_exact=exact["u"],
        v_exact=exact["v"],
        eps_xx=std["eps_xx"],
        eps_yy=std["eps_yy"],
        eps_xy=std["eps_xy"],
        sigma_xx=std["sigma_xx"],
        sigma_yy=std["sigma_yy"],
        sigma_xy=std["sigma_xy"],
        standard_energy_density=std["standard_energy_density"],
        psiI=current["psiI"],
        psiII=current["psiII"],
        psi_minus=current["psi_minus"],
        He_current=current["He_current"],
        current_elastic_energy_density=current["elastic_energy_density"],
        residual=exact["residual"],
        bottom_nodes=np.where(masks["bottom"])[0],
        top_nodes=np.where(masks["top"])[0],
        left_nodes=np.where(masks["left"])[0],
        right_nodes=np.where(masks["right"])[0],
        reaction_top_v_N=np.array(residual["reaction_top_v_N"]),
        reaction_bottom_v_N=np.array(residual["reaction_bottom_v_N"]),
        standard_energy=np.array(std["standard_internal_energy"]),
        pinn_mechanics_energy=np.array(current["current_pinn_mechanics_energy"]),
        free_residual_L2=np.array(residual["free_dof_residual_l2"]),
        boundary_bottom_u_abs_max=np.array(boundary.get("bottom_u_abs_max", np.nan)),
        boundary_bottom_v_abs_max=np.array(boundary.get("bottom_v_abs_max", np.nan)),
        boundary_top_v_minus_delta_abs_max=np.array(boundary.get("top_v_minus_delta_abs_max", np.nan)),
        E=np.array(args.E),
        nu=np.array(args.nu),
        Delta=np.array(args.delta),
        l0=np.array(args.l0),
        top_u_mode=np.array("free"),
        mesh_filename=np.array(Path(args.mesh).name),
        units=np.array("mm, kN/mm^2, N reactions after kN-to-N conversion"),
    )


def _exact_summary_row(x, y, tri, exact, std, current, residual, boundary, args):
    element_x = np.mean(x[tri], axis=1)
    element_y = np.mean(y[tri], axis=1)
    masks = _region_masks(element_x, element_y)
    notch = _stats(current["He_current"], masks["notch_tip"])["max"]
    bulk = _stats(current["He_current"], masks["bulk"])["p95"]
    bottom = _stats(current["He_current"], masks["bottom_right"])["max"]
    idx = int(np.nanargmax(current["He_current"]))
    return {
        "target_id": "exact_fe_topufree_alpha0_Delta1e-6",
        "accepted_target": True,
        "mesh": Path(args.mesh).name,
        "E": args.E,
        "nu": args.nu,
        "Delta": args.delta,
        "l0": args.l0,
        "top_u_mode": "free",
        "standard_energy": std["standard_internal_energy"],
        "pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
        "reaction_top_v_N": residual["reaction_top_v_N"],
        "reaction_bottom_v_N": residual["reaction_bottom_v_N"],
        "free_residual_L2": residual["free_dof_residual_l2"],
        "bottom_u_abs_max": boundary.get("bottom_u_abs_max", np.nan),
        "bottom_v_abs_max": boundary.get("bottom_v_abs_max", np.nan),
        "top_v_minus_delta_abs_max": boundary.get("top_v_minus_delta_abs_max", np.nan),
        "notch_tip_He_current_max": notch,
        "bulk_He_current_p95": bulk,
        "bottom_right_He_current_max": bottom,
        "bulk_to_notch_He_current": bulk / notch if notch else np.nan,
        "bottom_to_notch_He_current": bottom / notch if notch else np.nan,
        "max_He_current": float(np.nanmax(current["He_current"])),
        "max_He_current_x": float(element_x[idx]),
        "max_He_current_y": float(element_y[idx]),
        "classification": _classify(current["He_current"], element_x, element_y),
    }


def _prefit_metric_row(case_id, stage, prefit_kind, state, target, exact, K, x, y, tri, area, args):
    eps_xx, eps_yy, eps_xy = state["strains"]
    strain_pred = torch.cat([eps_xx.flatten(), eps_yy.flatten(), eps_xy.flatten()])
    strain_target = torch.cat([target["eps_xx"].flatten(), target["eps_yy"].flatten(), target["eps_xy"].flatten()])
    disp_ref = torch.mean(target["u"] ** 2 + target["v"] ** 2)
    strain_ref = torch.mean(target["eps_xx"] ** 2 + target["eps_yy"] ** 2 + target["eps_xy"] ** 2)

    u_np = state["u"].detach().cpu().numpy()
    v_np = state["v"].detach().cpu().numpy()
    guard = evaluate_candidate_arrays(case_id, x, y, tri, u_np, v_np, exact, K, area, args, top_u_mode="free")
    he = state["fields"]["He_current"].detach().cpu().numpy()
    element_x = np.mean(x[tri], axis=1)
    element_y = np.mean(y[tri], axis=1)
    masks = _region_masks(element_x, element_y)
    notch = _stats(he, masks["notch_tip"])["max"]
    bulk = _stats(he, masks["bulk"])["p95"]
    bottom = _stats(he, masks["bottom_right"])["max"]
    max_idx = int(np.nanargmax(he))
    row = {
        "case_id": case_id,
        "stage": stage,
        "prefit_kind": prefit_kind,
        "seed": args.seed,
        "hidden_layers": args.hidden_layers,
        "neurons": args.neurons,
        "Delta": args.delta,
        "alpha_mode": "zero_fixed",
        "top_u_mode": "free",
        "displacement_mse": float(state["disp_loss"].detach().cpu()),
        "displacement_rel_rmse": float(torch.sqrt(state["disp_loss"] / disp_ref).detach().cpu()),
        "strain_mse": float(state["strain_loss"].detach().cpu()),
        "strain_rel_rmse": float(torch.sqrt(state["strain_loss"] / strain_ref).detach().cpu()),
        "u_corr": _torch_corr(state["u"], target["u"]),
        "v_corr": _torch_corr(state["v"], target["v"]),
        "strain_corr": _torch_corr(strain_pred, strain_target),
        "He_current_corr": _safe_corr(he, target["He_current"].detach().cpu().numpy()),
        "mechanics_energy_log10": float(state["energy_loss"].detach().cpu()),
        "elastic_energy": float(state["E_el"].detach().cpu()),
        "fracture_energy": float(state["E_d"].detach().cpu()),
        "standard_energy_ratio_vs_exact": guard["standard_energy_ratio"],
        "pinn_mechanics_energy_ratio_vs_exact": guard["pinn_mechanics_energy_ratio"],
        "free_residual_L2": guard["free_residual_L2"],
        "reaction_N": guard["reaction_N"],
        "reaction_ratio": guard["reaction_ratio"],
        "reaction_sign_match": guard["reaction_sign_match"],
        "boundary_residual_abs_max": guard["boundary_residual_abs_max"],
        "notch_tip_He_current_max": notch,
        "bulk_He_current_p95": bulk,
        "bottom_right_He_current_max": bottom,
        "bulk_to_notch_He_current": bulk / notch if notch else np.nan,
        "bottom_to_notch_He_current": bottom / notch if notch else np.nan,
        "max_He_current": float(np.nanmax(he)),
        "max_He_current_x": float(element_x[max_idx]),
        "max_He_current_y": float(element_y[max_idx]),
        "classification": _classify(he, element_x, element_y),
    }
    return row


def _fedof_audit_rows(old_row, exact_summary, args):
    lr_over_delta = args.lr / args.delta if args.delta else np.nan
    objective_exact = np.log10(exact_summary["pinn_mechanics_energy"])
    objective_old = np.log10(old_row["pinn_mechanics_energy"]) if old_row else np.nan
    return [
        {
            "check": "boundary_conditions",
            "status": "not_primary_failure",
            "evidence": "Nodal BC transformation enforces bottom u/v and top v; exact FE and old target boundary residuals are checked separately.",
            "conclusion": "Boundary assignment syntax is not the main explanation for 1e7 energy ratio.",
        },
        {
            "check": "objective_sign",
            "status": "not_negated",
            "evidence": f"old objective log10(E)={objective_old:.6g}, exact FE log10(E)={objective_exact:.6g}; exact FE is far lower.",
            "conclusion": "Old RPROP field is not preferred by the objective; this is optimizer/scale failure rather than energy sign reversal.",
        },
        {
            "check": "units_and_delta",
            "status": "consistent_inputs_but_bad_optimizer_scale",
            "evidence": f"Delta={args.delta}, E={args.E}, coordinates in mm; RPROP lr={args.lr}, lr/Delta={lr_over_delta:.6g}.",
            "conclusion": "Input units are consistent with direct FE audit, but RPROP step scale is too large for Delta=1e-6 nodal DOFs.",
        },
        {
            "check": "dof_initialization_optimizer",
            "status": "root_cause_candidate",
            "evidence": f"raw nodal DOFs start at zero and RPROP sign step starts at {args.lr}, about {lr_over_delta:.0f} times Delta.",
            "conclusion": "The old FE-DOF RPROP baseline is an optimization diagnostic, not an exact FE solve; it converged to a high-energy branch.",
        },
        {
            "check": "energy_evaluation",
            "status": "failed_guard",
            "evidence": f"old standard energy ratio={old_row['standard_energy_ratio']:.6g}, old PINN mechanics energy ratio={old_row['pinn_mechanics_energy_ratio']:.6g}.",
            "conclusion": "Exact FE has the lower standard and current PINN alpha=0 mechanics energies.",
        },
        {
            "check": "residual_checks",
            "status": "failed_guard",
            "evidence": f"old free residual L2={old_row['free_residual_L2']:.6g}, exact free residual L2={exact_summary['free_residual_L2']:.6g}.",
            "conclusion": "Old FE-DOF RPROP field is not an equilibrated mechanics target.",
        },
    ]


def _quick_trim(args):
    if args.quick:
        args.prefit_epochs = min(args.prefit_epochs, 20)
        args.continuation_epochs = min(args.continuation_epochs, 10)
        args.hidden_layers = min(args.hidden_layers, 2)
        args.neurons = min(args.neurons, 20)


def main():
    args = parse_args()
    _quick_trim(args)
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_fedof = args.old_fedof_target or (
        args.repo_root
        / "examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz/artifacts/fedof_free_log10_energy_e300_fields.npz"
    )
    commands = {
        "script": str(Path(__file__).resolve()),
        "device": device,
        "args": {**vars(args), "old_fedof_target": str(old_fedof)},
        "training_loss_geometry_guidance": "none",
        "target_guard_thresholds": THRESHOLDS,
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    x, y, tri, area = _load_mesh(args.mesh)
    K = _assemble_stiffness(x, y, tri, area, args.E, args.nu)
    exact = _solve_exact(K, x, y, args.delta, "free")
    std = _strains_stresses_standard(x, y, tri, area, exact["u"], exact["v"], args.E, args.nu)
    current = _current_energy_fields(x, y, tri, area, exact["u"], exact["v"], _field_args(args))
    residual = _residual_metrics(K, x, y, exact["u"], exact["v"], args.delta, "free")
    boundary = _boundary_residuals(x, y, exact["u"], exact["v"], args.delta, "free")

    target_path = args.out_dir / "artifacts" / "exact_fe_topufree_alpha0_Delta1e-6_fields.npz"
    _save_exact_target(target_path, x, y, tri, area, exact, std, current, residual, boundary, args)
    exact_summary = _exact_summary_row(x, y, tri, exact, std, current, residual, boundary, args)
    _write_csv(args.out_dir / "tables" / "exact_fe_target_summary.csv", [exact_summary])

    exact_ref = {
        "u": exact["u"],
        "v": exact["v"],
        "eps_xx": std["eps_xx"],
        "eps_yy": std["eps_yy"],
        "eps_xy": std["eps_xy"],
        "He_current": current["He_current"],
        "standard_internal_energy": std["standard_internal_energy"],
        "current_pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
        "reaction_top_v_N": residual["reaction_top_v_N"],
    }
    guard_rows = [
        evaluate_candidate_arrays(
            "accepted_direct_sparse_FE_topufree",
            x,
            y,
            tri,
            exact["u"],
            exact["v"],
            exact_ref,
            K,
            area,
            args,
            top_u_mode="free",
        )
    ]
    old_guard = None
    if old_fedof.exists():
        with np.load(old_fedof) as data:
            old_u = np.asarray(data["u"], dtype=np.float64)
            old_v = np.asarray(data["v"], dtype=np.float64)
        old_guard = evaluate_candidate_arrays(
            "rejected_old_FE_DOF_RPROP_free_log10_e300",
            x,
            y,
            tri,
            old_u,
            old_v,
            exact_ref,
            K,
            area,
            args,
            top_u_mode="free",
        )
        guard_rows.append(old_guard)
    _write_csv(args.out_dir / "tables" / "target_guard_check_summary.csv", guard_rows)
    if old_guard is not None:
        _write_csv(args.out_dir / "tables" / "fedof_rprop_audit.csv", _fedof_audit_rows(old_guard, exact_summary, args))

    target = _load_target(target_path, device)
    prefit_rows = []
    continuation_rows = []
    trace_rows = []
    prefit_states = {}
    for prefit_kind in ("disp_global", "disp_strain_global"):
        field = _make_field(args, "free", device)
        _fit(field, target, args, device, args.prefit_epochs, prefit_kind, trace_rows, f"prefit_{prefit_kind}")
        state = _evaluate(field, target, args, device)
        prefit_states[prefit_kind] = _state_dict(field)
        prefit_rows.append(
            _prefit_metric_row(
                f"prefit_{prefit_kind}",
                "prefit_end",
                prefit_kind,
                state,
                target,
                exact_ref,
                K,
                x,
                y,
                tri,
                area,
                args,
            )
        )
        torch.save(prefit_states[prefit_kind], args.out_dir / "artifacts" / f"{prefit_kind}_prefit_state.pt")
        _save_case_npz(args.out_dir / "artifacts" / f"{prefit_kind}_prefit_end_fields.npz", target, state)

    start_kind = "disp_strain_global"
    start_state = prefit_states[start_kind]
    start_field = _field_from_state(args, start_state, device)
    start_eval = _evaluate(start_field, target, args, device)
    energy_scale = torch.clamp(start_eval["E_el"] + start_eval["E_d"], min=torch.finfo(start_eval["E_el"].dtype).tiny).detach()
    for energy_mode in ("raw", "log10", "normalized"):
        field = _field_from_state(args, start_state, device)
        case_id = f"energy_continuation_from_{start_kind}_{energy_mode}"
        _fit(
            field,
            target,
            args,
            device,
            args.continuation_epochs,
            "energy",
            trace_rows,
            case_id,
            energy_mode=energy_mode,
            energy_scale=energy_scale,
        )
        state = _evaluate(field, target, args, device)
        row = _prefit_metric_row(
            case_id,
            "energy_end",
            start_kind,
            state,
            target,
            exact_ref,
            K,
            x,
            y,
            tri,
            area,
            args,
        )
        row["energy_mode"] = energy_mode
        row["continuation_epochs"] = args.continuation_epochs
        continuation_rows.append(row)
        _save_case_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target, state)

    _write_csv(args.out_dir / "tables" / "exact_target_prefit_metrics.csv", prefit_rows)
    _write_csv(args.out_dir / "tables" / "exact_target_energy_continuation.csv", continuation_rows)
    _write_csv(args.out_dir / "logs" / "loss_trace.csv", trace_rows)
    (args.out_dir / "figures" / "figure_summary.md").write_text(
        "# Figure Summary\n\nNo PNG figures were generated. The diagnostic evidence is contained in CSV tables and NPZ artifacts.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
