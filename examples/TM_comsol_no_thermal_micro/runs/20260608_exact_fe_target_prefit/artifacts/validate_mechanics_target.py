"""Validate alpha=0 mechanics target fields against a direct FE reference.

This diagnostic is intentionally postprocessing-only. It does not change the
physical model, training losses, history update logic, phase-field notch
handling, material parameters, or l0.
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_exact_fe_elastic_solve import (  # noqa: E402
    _assemble_stiffness,
    _boundary_residuals,
    _current_energy_fields,
    _load_mesh,
    _residual_metrics,
    _safe_corr,
    _strains_stresses_standard,
)


THRESHOLDS = {
    "energy_ratio_max": 10.0,
    "displacement_rel_rmse_max": 0.1,
    "strain_rel_rmse_max": 0.2,
    "free_residual_l2_max": 1.0e-6,
    "boundary_residual_abs_max": 1.0e-10,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a mechanics target against exact FE.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--top-u-mode", choices=["free", "fixed"], default="free")
    parser.add_argument("--mesh", type=Path, default=ROOT / "geo_coarse_with_groups_mm.msh")
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--E", type=float, default=81.5)
    parser.add_argument("--nu", type=float, default=0.38)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--eta-residual", type=float, default=1.0e-5)
    return parser.parse_args()


def _areas_from_triangles(x, y, tri):
    x0 = x[tri[:, 0]]
    y0 = y[tri[:, 0]]
    x1 = x[tri[:, 1]]
    y1 = y[tri[:, 1]]
    x2 = x[tri[:, 2]]
    y2 = y[tri[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _read_field(path, prefer_exact=False):
    with np.load(path) as data:
        keys = set(data.files)
        u_key = "u_exact" if prefer_exact and "u_exact" in keys else "u"
        v_key = "v_exact" if prefer_exact and "v_exact" in keys else "v"
        if u_key not in keys and "u_pred" in keys:
            u_key = "u_pred"
        if v_key not in keys and "v_pred" in keys:
            v_key = "v_pred"
        tri_key = "triangles" if "triangles" in keys else "connectivity"
        field = {
            "x": np.asarray(data["x"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
            "triangles": np.asarray(data[tri_key], dtype=np.int64),
            "u": np.asarray(data[u_key], dtype=np.float64),
            "v": np.asarray(data[v_key], dtype=np.float64),
        }
        for key in ("eps_xx", "eps_yy", "eps_xy", "He_current", "psiI", "psiII", "psi_minus"):
            if key in keys:
                field[key] = np.asarray(data[key], dtype=np.float64)
    return field


def _field_args(args):
    return SimpleNamespace(
        E=float(args.E),
        nu=float(args.nu),
        l0=float(args.l0),
        tm_eps_r=float(args.tm_eps_r),
        eta_residual=float(args.eta_residual),
    )


def _max_boundary_residual(boundary_row):
    values = [
        boundary_row.get("bottom_u_abs_max", np.nan),
        boundary_row.get("bottom_v_abs_max", np.nan),
        boundary_row.get("top_v_minus_delta_abs_max", np.nan),
    ]
    finite = [abs(float(v)) for v in values if np.isfinite(v)]
    return max(finite) if finite else np.nan


def _append_rejection(reasons, condition, message):
    if condition:
        reasons.append(message)


def evaluate_candidate_arrays(
    candidate_id,
    x,
    y,
    tri,
    u,
    v,
    exact,
    K,
    area,
    args,
    top_u_mode="free",
):
    std = _strains_stresses_standard(x, y, tri, area, u, v, args.E, args.nu)
    current = _current_energy_fields(x, y, tri, area, u, v, _field_args(args))
    residual = _residual_metrics(K, x, y, u, v, args.delta, top_u_mode)
    boundary = _boundary_residuals(x, y, u, v, args.delta, top_u_mode)

    disp_mse = float(np.mean((u - exact["u"]) ** 2 + (v - exact["v"]) ** 2))
    disp_ref = float(np.mean(exact["u"] ** 2 + exact["v"] ** 2))
    strain_mse = float(
        np.mean(
            (std["eps_xx"] - exact["eps_xx"]) ** 2
            + (std["eps_yy"] - exact["eps_yy"]) ** 2
            + (std["eps_xy"] - exact["eps_xy"]) ** 2
        )
    )
    strain_ref = float(np.mean(exact["eps_xx"] ** 2 + exact["eps_yy"] ** 2 + exact["eps_xy"] ** 2))

    reaction_exact = float(exact["reaction_top_v_N"])
    reaction = float(residual["reaction_top_v_N"])
    reaction_sign_match = bool(np.sign(reaction) == np.sign(reaction_exact)) if reaction != 0.0 else False
    reaction_ratio = reaction / reaction_exact if reaction_exact != 0.0 else np.nan
    boundary_abs_max = _max_boundary_residual(boundary)
    standard_energy_ratio = std["standard_internal_energy"] / exact["standard_internal_energy"]
    pinn_energy_ratio = current["current_pinn_mechanics_energy"] / exact["current_pinn_mechanics_energy"]

    reasons = []
    _append_rejection(
        reasons,
        standard_energy_ratio > THRESHOLDS["energy_ratio_max"],
        f"standard_energy_ratio>{THRESHOLDS['energy_ratio_max']}",
    )
    _append_rejection(
        reasons,
        pinn_energy_ratio > THRESHOLDS["energy_ratio_max"],
        f"pinn_mechanics_energy_ratio>{THRESHOLDS['energy_ratio_max']}",
    )
    displacement_rel_rmse = float(np.sqrt(disp_mse / disp_ref)) if disp_ref > 0.0 else np.nan
    strain_rel_rmse = float(np.sqrt(strain_mse / strain_ref)) if strain_ref > 0.0 else np.nan
    _append_rejection(
        reasons,
        displacement_rel_rmse > THRESHOLDS["displacement_rel_rmse_max"],
        f"displacement_rel_rmse>{THRESHOLDS['displacement_rel_rmse_max']}",
    )
    _append_rejection(
        reasons,
        strain_rel_rmse > THRESHOLDS["strain_rel_rmse_max"],
        f"strain_rel_rmse>{THRESHOLDS['strain_rel_rmse_max']}",
    )
    _append_rejection(reasons, not reaction_sign_match, "reaction_sign_mismatch")
    _append_rejection(
        reasons,
        residual["free_dof_residual_l2"] > THRESHOLDS["free_residual_l2_max"],
        f"free_residual_L2>{THRESHOLDS['free_residual_l2_max']}",
    )
    _append_rejection(
        reasons,
        np.isfinite(boundary_abs_max) and boundary_abs_max > THRESHOLDS["boundary_residual_abs_max"],
        f"boundary_residual>{THRESHOLDS['boundary_residual_abs_max']}",
    )

    return {
        "candidate_id": candidate_id,
        "top_u_mode": top_u_mode,
        "displacement_mse": disp_mse,
        "displacement_rel_rmse": displacement_rel_rmse,
        "strain_mse": strain_mse,
        "strain_rel_rmse": strain_rel_rmse,
        "u_corr": _safe_corr(u, exact["u"]),
        "v_corr": _safe_corr(v, exact["v"]),
        "strain_corr": _safe_corr(
            np.concatenate([std["eps_xx"], std["eps_yy"], std["eps_xy"]]),
            np.concatenate([exact["eps_xx"], exact["eps_yy"], exact["eps_xy"]]),
        ),
        "He_current_corr": _safe_corr(current["He_current"], exact["He_current"]),
        "standard_energy": std["standard_internal_energy"],
        "standard_energy_ratio": standard_energy_ratio,
        "pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
        "pinn_mechanics_energy_ratio": pinn_energy_ratio,
        "free_residual_L2": residual["free_dof_residual_l2"],
        "free_residual_max_abs": residual["free_dof_residual_max_abs"],
        "reaction_N": reaction,
        "reaction_exact_N": reaction_exact,
        "reaction_ratio": reaction_ratio,
        "reaction_sign_match": reaction_sign_match,
        "boundary_residual_abs_max": boundary_abs_max,
        "bottom_u_abs_max": boundary.get("bottom_u_abs_max", np.nan),
        "bottom_v_abs_max": boundary.get("bottom_v_abs_max", np.nan),
        "top_v_minus_delta_abs_max": boundary.get("top_v_minus_delta_abs_max", np.nan),
        "accepted_target": len(reasons) == 0,
        "rejection_reason": "accepted" if not reasons else "; ".join(reasons),
        "threshold_energy_ratio_max": THRESHOLDS["energy_ratio_max"],
        "threshold_displacement_rel_rmse_max": THRESHOLDS["displacement_rel_rmse_max"],
        "threshold_strain_rel_rmse_max": THRESHOLDS["strain_rel_rmse_max"],
        "threshold_free_residual_l2_max": THRESHOLDS["free_residual_l2_max"],
        "threshold_boundary_residual_abs_max": THRESHOLDS["boundary_residual_abs_max"],
    }


def exact_reference_from_npz(exact_path, args, K=None, area=None):
    exact_field = _read_field(exact_path, prefer_exact=True)
    if area is None:
        area = _areas_from_triangles(exact_field["x"], exact_field["y"], exact_field["triangles"])
    if K is None:
        K = _assemble_stiffness(exact_field["x"], exact_field["y"], exact_field["triangles"], area, args.E, args.nu)
    std = _strains_stresses_standard(
        exact_field["x"], exact_field["y"], exact_field["triangles"], area, exact_field["u"], exact_field["v"], args.E, args.nu
    )
    current = _current_energy_fields(
        exact_field["x"], exact_field["y"], exact_field["triangles"], area, exact_field["u"], exact_field["v"], _field_args(args)
    )
    residual = _residual_metrics(K, exact_field["x"], exact_field["y"], exact_field["u"], exact_field["v"], args.delta, args.top_u_mode)
    exact = {
        **exact_field,
        "eps_xx": std["eps_xx"],
        "eps_yy": std["eps_yy"],
        "eps_xy": std["eps_xy"],
        "He_current": current["He_current"],
        "standard_internal_energy": std["standard_internal_energy"],
        "current_pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
        "reaction_top_v_N": residual["reaction_top_v_N"],
    }
    return exact


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    x, y, tri, area = _load_mesh(args.mesh)
    K = _assemble_stiffness(x, y, tri, area, args.E, args.nu)
    exact = exact_reference_from_npz(args.exact, args, K=K, area=area)
    candidate = _read_field(args.candidate)
    candidate_id = args.candidate_id or args.candidate.stem
    row = evaluate_candidate_arrays(
        candidate_id,
        x,
        y,
        tri,
        candidate["u"],
        candidate["v"],
        exact,
        K,
        area,
        args,
        top_u_mode=args.top_u_mode,
    )
    write_rows(args.out, [row])
    print(f"wrote {args.out}")
    print(f"accepted_target={row['accepted_target']} reason={row['rejection_reason']}")


if __name__ == "__main__":
    main()
