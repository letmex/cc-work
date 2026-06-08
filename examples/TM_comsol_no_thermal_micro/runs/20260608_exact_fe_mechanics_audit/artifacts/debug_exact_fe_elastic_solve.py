"""Exact alpha=0 T3 FE mechanics audit for the COMSOL micro-notch case.

This diagnostic assembles a standard plane-stress P1/T3 linear elasticity
system on the same mesh, solves it directly, and compares that exact FE
solution with previous FE-DOF/PINN fields. Region metrics are postprocessing
only and are never used to train or guide any field.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy_mixed_tm import compute_mixed_tm_energy  # noqa: E402
from history_field_mixed_tm import element_centroids  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from mixed_mode_tm import mixed_mode_ratio  # noqa: E402
from pff_model import PFFModel  # noqa: E402
from utils import parse_mesh  # noqa: E402


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
BOUNDARY_WINDOW_MM = 5.0e-4
BOUNDARY_TOL_MM = 1.0e-9


def parse_args():
    parser = argparse.ArgumentParser(description="Audit exact alpha=0 FE mechanics solution.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=ROOT / "geo_coarse_with_groups_mm.msh")
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--E", type=float, default=81.5)
    parser.add_argument("--nu", type=float, default=0.38)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--eta-residual", type=float, default=1.0e-5)
    return parser.parse_args()


def _prepare_output_dirs(out_dir):
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _load_mesh(mesh):
    x, y, tri, area = parse_mesh(str(mesh), gradient_type="numerical")
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(tri, dtype=np.int64),
        np.asarray(area, dtype=np.float64),
    )


def _plane_stress_D(E, nu):
    factor = E / (1.0 - nu**2)
    return factor * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=np.float64,
    )


def _element_B(xe, ye, area):
    b = np.array([ye[1] - ye[2], ye[2] - ye[0], ye[0] - ye[1]], dtype=np.float64)
    c = np.array([xe[2] - xe[1], xe[0] - xe[2], xe[1] - xe[0]], dtype=np.float64)
    B = np.array(
        [
            [b[0], 0.0, b[1], 0.0, b[2], 0.0],
            [0.0, c[0], 0.0, c[1], 0.0, c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]],
        ],
        dtype=np.float64,
    )
    return B / (2.0 * area)


def _assemble_stiffness(x, y, tri, area, E, nu):
    D = _plane_stress_D(E, nu)
    rows = []
    cols = []
    data = []
    for elem, nodes in enumerate(tri):
        A = area[elem]
        xe = x[nodes]
        ye = y[nodes]
        B = _element_B(xe, ye, A)
        ke = A * (B.T @ D @ B)
        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        data.extend(ke.ravel())
    ndof = 2 * len(x)
    return coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()


def _boundary_masks(x, y):
    return {
        "bottom": np.abs(y) <= BOUNDARY_TOL_MM,
        "top": np.abs(y - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL_MM,
        "left": np.abs(x) <= BOUNDARY_TOL_MM,
        "right": np.abs(x - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL_MM,
    }


def _constraints(x, y, delta, top_u_mode):
    masks = _boundary_masks(x, y)
    constrained = {}
    for node in np.where(masks["bottom"])[0]:
        constrained[2 * int(node)] = 0.0
        constrained[2 * int(node) + 1] = 0.0
    for node in np.where(masks["top"])[0]:
        constrained[2 * int(node) + 1] = float(delta)
        if top_u_mode == "fixed":
            constrained[2 * int(node)] = 0.0
    constrained_dofs = np.array(sorted(constrained.keys()), dtype=np.int64)
    constrained_values = np.array([constrained[int(dof)] for dof in constrained_dofs], dtype=np.float64)
    all_dofs = np.arange(2 * len(x), dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=True)
    return free_dofs, constrained_dofs, constrained_values


def _solve_exact(K, x, y, delta, top_u_mode):
    free, cons, vals = _constraints(x, y, delta, top_u_mode)
    U = np.zeros(2 * len(x), dtype=np.float64)
    U[cons] = vals
    rhs = -K[free][:, cons] @ vals
    Kff = K[free][:, free]
    U[free] = spsolve(Kff, rhs)
    residual = K @ U
    return {
        "u": U[0::2].copy(),
        "v": U[1::2].copy(),
        "U": U,
        "free_dofs": free,
        "constrained_dofs": cons,
        "residual": residual,
        "solver_status": "spsolve_completed",
        "n_free_dofs": int(len(free)),
        "n_constrained_dofs": int(len(cons)),
    }


def _strains_stresses_standard(x, y, tri, area, u, v, E, nu):
    D = _plane_stress_D(E, nu)
    U = np.empty(2 * len(x), dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    eps_xx = np.empty(len(tri), dtype=np.float64)
    eps_yy = np.empty(len(tri), dtype=np.float64)
    eps_xy = np.empty(len(tri), dtype=np.float64)
    sigma_xx = np.empty(len(tri), dtype=np.float64)
    sigma_yy = np.empty(len(tri), dtype=np.float64)
    sigma_xy = np.empty(len(tri), dtype=np.float64)
    energy_density = np.empty(len(tri), dtype=np.float64)
    for elem, nodes in enumerate(tri):
        B = _element_B(x[nodes], y[nodes], area[elem])
        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        strain_eng = B @ U[dofs]
        stress = D @ strain_eng
        eps_xx[elem] = strain_eng[0]
        eps_yy[elem] = strain_eng[1]
        eps_xy[elem] = 0.5 * strain_eng[2]
        sigma_xx[elem] = stress[0]
        sigma_yy[elem] = stress[1]
        sigma_xy[elem] = stress[2]
        energy_density[elem] = 0.5 * float(strain_eng @ stress)
    return {
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "standard_energy_density": energy_density,
        "standard_internal_energy": float(np.sum(area * energy_density)),
    }


def _torch_material(device, E, nu, l0):
    matprop = MaterialProperties(
        torch.tensor(E, dtype=torch.float32, device=device),
        torch.tensor(nu, dtype=torch.float32, device=device),
        torch.tensor(2.4e-6 / l0, dtype=torch.float32, device=device),
        torch.tensor(l0, dtype=torch.float32, device=device),
    )
    pffmodel = PFFModel("AT2", "volumetric", torch.tensor(5.0e-3, dtype=torch.float32, device=device))
    gcII = 2.0 * (1.0 + nu) * (0.60**2) * 2.4e-6
    return matprop, pffmodel, gcII


def _current_energy_fields(x, y, tri, area, u, v, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.tensor(np.column_stack([x, y]), dtype=torch.float32, device=device)
    T_conn = torch.tensor(tri, dtype=torch.long, device=device)
    area_T = torch.tensor(area, dtype=torch.float32, device=device)
    uu = torch.tensor(u, dtype=torch.float32, device=device)
    vv = torch.tensor(v, dtype=torch.float32, device=device)
    alpha = torch.zeros(inp.shape[0], dtype=torch.float32, device=device)
    HI = torch.zeros_like(area_T)
    HII = torch.zeros_like(area_T)
    matprop, pffmodel, gcII = _torch_material(device, args.E, args.nu, args.l0)
    E_el, E_d, fields = compute_mixed_tm_energy(
        inp,
        uu,
        vv,
        alpha,
        HI,
        HII,
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=args.eta_residual,
        gcII=gcII,
        split_mode="tm_source",
        tm_eps_r=args.tm_eps_r,
        mechanics_mode="history",
    )
    keys = [
        "eps_xx",
        "eps_yy",
        "eps_xy",
        "eps_zz",
        "psiI",
        "psiII",
        "psi_minus",
        "psi_total",
        "He_current",
        "mechanics_current_energy_density",
        "history_elastic_energy_density",
        "elastic_energy_density",
        "sigma_yy_tm_eff",
    ]
    out = {key: fields[key].detach().cpu().numpy() for key in keys}
    out["current_pinn_mechanics_energy"] = float(E_el.detach().cpu())
    out["fracture_energy"] = float(E_d.detach().cpu())
    out["mixed_mode_ratio"] = mixed_mode_ratio(matprop, gcII=gcII)
    return out


def _safe_corr(a, b):
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 2:
        return np.nan
    aa = aa[mask] - aa[mask].mean()
    bb = bb[mask] - bb[mask].mean()
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(aa * bb) / denom)


def _region_masks(element_x, element_y):
    notch_tip = (
        (element_x >= NOTCH_TIP_X_MM - TIP_HALF_WINDOW_MM)
        & (element_x <= NOTCH_TIP_X_MM + TIP_HALF_WINDOW_MM)
        & (np.abs(element_y - NOTCH_CENTER_Y_MM) <= TIP_HALF_WINDOW_MM)
    )
    bottom_right = (
        (element_x >= SPECIMEN_SIZE_MM - BOTTOM_RIGHT_WINDOW_MM)
        & (element_x <= SPECIMEN_SIZE_MM)
        & (element_y >= 0.0)
        & (element_y <= BOTTOM_RIGHT_WINDOW_MM)
    )
    boundary = (
        (element_x <= BOUNDARY_WINDOW_MM)
        | (element_x >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM)
        | (element_y <= BOUNDARY_WINDOW_MM)
        | (element_y >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM)
    )
    bulk = (~notch_tip) & (~bottom_right) & (~boundary)
    return {"notch_tip": notch_tip, "bottom_right": bottom_right, "bulk": bulk, "boundary": boundary}


def _stats(values, mask):
    vals = np.asarray(values)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"max": np.nan, "p95": np.nan, "mean": np.nan}
    return {"max": float(np.max(vals)), "p95": float(np.quantile(vals, 0.95)), "mean": float(np.mean(vals))}


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return np.nan
    return float(num / den)


def _classify(he, element_x, element_y):
    masks = _region_masks(element_x, element_y)
    notch = _stats(he, masks["notch_tip"])["max"]
    bulk = _stats(he, masks["bulk"])["p95"]
    bottom = _stats(he, masks["bottom_right"])["max"]
    max_idx = int(np.nanargmax(he))
    max_x = float(element_x[max_idx])
    max_y = float(element_y[max_idx])
    boundary_max = (
        max_x <= BOUNDARY_WINDOW_MM
        or max_x >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
        or max_y <= BOUNDARY_WINDOW_MM
        or max_y >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
    )
    bulk_ratio = _ratio(bulk, notch)
    bottom_ratio = _ratio(bottom, notch)
    if boundary_max:
        return "boundary-dominated"
    if np.isfinite(bulk_ratio) and np.isfinite(bottom_ratio) and bulk_ratio <= 0.35 and bottom_ratio <= 0.1:
        return "notch-amplified"
    if np.isfinite(bulk_ratio) and 0.5 <= bulk_ratio <= 2.0:
        return "broad/background"
    return "other"


def _residual_metrics(K, x, y, u, v, delta, top_u_mode):
    free, cons, _vals = _constraints(x, y, delta, top_u_mode)
    U = np.empty(2 * len(x), dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    residual = K @ U
    top = np.where(_boundary_masks(x, y)["top"])[0]
    bottom = np.where(_boundary_masks(x, y)["bottom"])[0]
    top_v_dofs = 2 * top + 1
    bottom_v_dofs = 2 * bottom + 1
    return {
        "free_dof_residual_l2": float(np.linalg.norm(residual[free])),
        "free_dof_residual_max_abs": float(np.max(np.abs(residual[free]))) if len(free) else 0.0,
        "constrained_dof_residual_l2": float(np.linalg.norm(residual[cons])),
        "reaction_top_v_kN": float(np.sum(residual[top_v_dofs])),
        "reaction_top_v_N": float(1000.0 * np.sum(residual[top_v_dofs])),
        "reaction_bottom_v_kN": float(np.sum(residual[bottom_v_dofs])),
        "reaction_bottom_v_N": float(1000.0 * np.sum(residual[bottom_v_dofs])),
    }


def _boundary_residuals(x, y, u, v, delta, top_u_mode):
    masks = _boundary_masks(x, y)
    row = {"top_u_mode": top_u_mode}
    if np.any(masks["bottom"]):
        row["bottom_u_abs_max"] = float(np.max(np.abs(u[masks["bottom"]])))
        row["bottom_v_abs_max"] = float(np.max(np.abs(v[masks["bottom"]])))
    if np.any(masks["top"]):
        row["top_v_minus_delta_abs_max"] = float(np.max(np.abs(v[masks["top"]] - delta)))
        row["top_u_abs_max"] = float(np.max(np.abs(u[masks["top"]])))
        row["top_u_abs_mean"] = float(np.mean(np.abs(u[masks["top"]])))
        row["top_u_expected"] = 0.0 if top_u_mode == "fixed" else np.nan
    return row


def _load_candidate_fields(repo_root):
    specs = [
        (
            "fedof_rprop_free",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz/artifacts/fedof_free_log10_energy_e300_fields.npz",
            "u",
            "v",
        ),
        (
            "pinn_prefit_disp_global",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_prefit_energy_continuation/artifacts/free_disp_global_prefit_then_energy_prefit_end_fields.npz",
            "u_pred",
            "v_pred",
        ),
        (
            "pinn_prefit_disp_strain",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_prefit_energy_continuation/artifacts/free_disp_strain_global_prefit_then_energy_prefit_end_fields.npz",
            "u_pred",
            "v_pred",
        ),
        (
            "pinn_collapsed_pure_energy_log10",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation/artifacts/pure_energy_baseline_log10_fields.npz",
            "u_pred",
            "v_pred",
        ),
        (
            "pinn_collapsed_energy_raw",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation/artifacts/energy_normalization_raw_fields.npz",
            "u_pred",
            "v_pred",
        ),
        (
            "pinn_collapsed_energy_normalized",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation/artifacts/energy_normalization_normalized_fields.npz",
            "u_pred",
            "v_pred",
        ),
        (
            "pinn_strong_displacement_anchor",
            repo_root
            / "examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation/artifacts/global_displacement_anchor_lamU_1e-01_fields.npz",
            "u_pred",
            "v_pred",
        ),
    ]
    fields = []
    for name, path, u_key, v_key in specs:
        if not path.exists():
            continue
        with np.load(path) as data:
            fields.append(
                {
                    "field_id": name,
                    "source_path": str(path),
                    "u": np.asarray(data[u_key], dtype=np.float64),
                    "v": np.asarray(data[v_key], dtype=np.float64),
                }
            )
    return fields


def _save_npz(path, x, y, tri, element_x, element_y, u, v, std, current, extra=None):
    payload = {
        "x": x,
        "y": y,
        "triangles": tri,
        "element_x": element_x,
        "element_y": element_y,
        "u": u,
        "v": v,
        "eps_xx": std["eps_xx"],
        "eps_yy": std["eps_yy"],
        "eps_xy": std["eps_xy"],
        "standard_energy_density": std["standard_energy_density"],
        "He_current": current["He_current"],
        "psiI": current["psiI"],
        "psiII": current["psiII"],
        "psi_minus": current["psi_minus"],
        "psi_total": current["psi_total"],
        "current_elastic_energy_density": current["elastic_energy_density"],
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def _write_rows(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _field_rows(field_id, field_type, top_u_mode, source_path, x, y, tri, area, K, exact_ref, u, v, args):
    element_x = (x[tri[:, 0]] + x[tri[:, 1]] + x[tri[:, 2]]) / 3.0
    element_y = (y[tri[:, 0]] + y[tri[:, 1]] + y[tri[:, 2]]) / 3.0
    std = _strains_stresses_standard(x, y, tri, area, u, v, args.E, args.nu)
    current = _current_energy_fields(x, y, tri, area, u, v, args)
    residual = _residual_metrics(K, x, y, u, v, args.delta, top_u_mode)
    boundary = _boundary_residuals(x, y, u, v, args.delta, top_u_mode)
    masks = _region_masks(element_x, element_y)
    he = current["He_current"]
    max_idx = int(np.nanargmax(he))
    notch = _stats(he, masks["notch_tip"])["max"]
    bulk = _stats(he, masks["bulk"])["p95"]
    bottom = _stats(he, masks["bottom_right"])["max"]
    disp_mse = np.mean((u - exact_ref["u"]) ** 2 + (v - exact_ref["v"]) ** 2)
    disp_ref = np.mean(exact_ref["u"] ** 2 + exact_ref["v"] ** 2)
    strain_mse = np.mean(
        (std["eps_xx"] - exact_ref["std"]["eps_xx"]) ** 2
        + (std["eps_yy"] - exact_ref["std"]["eps_yy"]) ** 2
        + (std["eps_xy"] - exact_ref["std"]["eps_xy"]) ** 2
    )
    strain_ref = np.mean(
        exact_ref["std"]["eps_xx"] ** 2 + exact_ref["std"]["eps_yy"] ** 2 + exact_ref["std"]["eps_xy"] ** 2
    )
    strain_pred = np.concatenate([std["eps_xx"], std["eps_yy"], std["eps_xy"]])
    strain_exact = np.concatenate(
        [exact_ref["std"]["eps_xx"], exact_ref["std"]["eps_yy"], exact_ref["std"]["eps_xy"]]
    )
    comparison = {
        "field_id": field_id,
        "field_type": field_type,
        "source_path": source_path,
        "top_u_mode_for_residual": top_u_mode,
        "displacement_rmse_vs_exact": float(np.sqrt(disp_mse)),
        "displacement_rel_rmse_vs_exact": float(np.sqrt(disp_mse / disp_ref)),
        "strain_rmse_vs_exact": float(np.sqrt(strain_mse)),
        "strain_rel_rmse_vs_exact": float(np.sqrt(strain_mse / strain_ref)),
        "u_corr_vs_exact": _safe_corr(u, exact_ref["u"]),
        "v_corr_vs_exact": _safe_corr(v, exact_ref["v"]),
        "strain_corr_vs_exact": _safe_corr(strain_pred, strain_exact),
        "He_current_corr_vs_exact": _safe_corr(he, exact_ref["current"]["He_current"]),
        "notch_tip_He_current_max": notch,
        "bulk_He_current_p95": bulk,
        "bottom_right_He_current_max": bottom,
        "bulk_to_notch_He_current": _ratio(bulk, notch),
        "bottom_to_notch_He_current": _ratio(bottom, notch),
        "max_He_current": float(np.nanmax(he)),
        "max_He_current_x": float(element_x[max_idx]),
        "max_He_current_y": float(element_y[max_idx]),
        "classification": _classify(he, element_x, element_y),
        "reaction_top_v_N": residual["reaction_top_v_N"],
    }
    energy = {
        "field_id": field_id,
        "field_type": field_type,
        "standard_internal_energy": std["standard_internal_energy"],
        "standard_energy_normalized_vs_exact": _ratio(
            std["standard_internal_energy"], exact_ref["std"]["standard_internal_energy"]
        ),
        "current_pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
        "current_pinn_energy_normalized_vs_exact": _ratio(
            current["current_pinn_mechanics_energy"], exact_ref["current"]["current_pinn_mechanics_energy"]
        ),
        "current_pinn_log10_energy": float(np.log10(max(current["current_pinn_mechanics_energy"], np.finfo(float).tiny))),
        "sum_area_psiI": float(np.sum(area * current["psiI"])),
        "sum_area_psiII": float(np.sum(area * current["psiII"])),
        "sum_area_psi_minus": float(np.sum(area * current["psi_minus"])),
        "sum_area_psi_total": float(np.sum(area * current["psi_total"])),
        "sum_area_He_current": float(np.sum(area * current["He_current"])),
        "sum_area_mechanics_current_density": float(np.sum(area * current["mechanics_current_energy_density"])),
        "sum_area_history_elastic_density": float(np.sum(area * current["history_elastic_energy_density"])),
        "fracture_energy_alpha0": current["fracture_energy"],
        "mixed_mode_ratio": current["mixed_mode_ratio"],
    }
    residual_row = {"field_id": field_id, "field_type": field_type, **residual}
    boundary_row = {"field_id": field_id, "field_type": field_type, **boundary}
    return comparison, energy, residual_row, boundary_row, std, current


def main():
    args = parse_args()
    _prepare_output_dirs(args.out_dir)
    x, y, tri, area = _load_mesh(args.mesh)
    K = _assemble_stiffness(x, y, tri, area, args.E, args.nu)
    element_x = (x[tri[:, 0]] + x[tri[:, 1]] + x[tri[:, 2]]) / 3.0
    element_y = (y[tri[:, 0]] + y[tri[:, 1]] + y[tri[:, 2]]) / 3.0

    exact = {}
    exact_rows = []
    for top_u_mode in ("free", "fixed"):
        sol = _solve_exact(K, x, y, args.delta, top_u_mode)
        std = _strains_stresses_standard(x, y, tri, area, sol["u"], sol["v"], args.E, args.nu)
        current = _current_energy_fields(x, y, tri, area, sol["u"], sol["v"], args)
        residual = _residual_metrics(K, x, y, sol["u"], sol["v"], args.delta, top_u_mode)
        boundary = _boundary_residuals(x, y, sol["u"], sol["v"], args.delta, top_u_mode)
        he = current["He_current"]
        masks = _region_masks(element_x, element_y)
        max_idx = int(np.nanargmax(he))
        notch = _stats(he, masks["notch_tip"])["max"]
        bulk = _stats(he, masks["bulk"])["p95"]
        bottom = _stats(he, masks["bottom_right"])["max"]
        row = {
            "field_id": f"exact_fe_{top_u_mode}",
            "top_u_mode": top_u_mode,
            "solver_status": sol["solver_status"],
            "n_free_dofs": sol["n_free_dofs"],
            "n_constrained_dofs": sol["n_constrained_dofs"],
            "standard_internal_energy": std["standard_internal_energy"],
            "current_pinn_mechanics_energy": current["current_pinn_mechanics_energy"],
            "current_pinn_log10_energy": float(np.log10(max(current["current_pinn_mechanics_energy"], np.finfo(float).tiny))),
            "reaction_top_v_N": residual["reaction_top_v_N"],
            "free_dof_residual_l2": residual["free_dof_residual_l2"],
            "free_dof_residual_max_abs": residual["free_dof_residual_max_abs"],
            "notch_tip_He_current_max": notch,
            "bulk_He_current_p95": bulk,
            "bottom_right_He_current_max": bottom,
            "bulk_to_notch_He_current": _ratio(bulk, notch),
            "bottom_to_notch_He_current": _ratio(bottom, notch),
            "max_He_current": float(np.nanmax(he)),
            "max_He_current_x": float(element_x[max_idx]),
            "max_He_current_y": float(element_y[max_idx]),
            "classification": _classify(he, element_x, element_y),
            **boundary,
        }
        exact_rows.append(row)
        exact[top_u_mode] = {"u": sol["u"], "v": sol["v"], "std": std, "current": current, "residual": residual}
        _save_npz(
            args.out_dir / "artifacts" / f"exact_fe_{top_u_mode}_fields.npz",
            x,
            y,
            tri,
            element_x,
            element_y,
            sol["u"],
            sol["v"],
            std,
            current,
            extra={"residual": sol["residual"]},
        )

    comparison_rows = []
    energy_rows = []
    residual_rows = []
    boundary_rows = []
    exact_ref = exact["free"]
    for top_u_mode in ("free", "fixed"):
        comparison, energy, residual, boundary, _std, _current = _field_rows(
            f"exact_fe_{top_u_mode}",
            "exact_fe",
            top_u_mode,
            "direct_sparse_solve",
            x,
            y,
            tri,
            area,
            K,
            exact_ref,
            exact[top_u_mode]["u"],
            exact[top_u_mode]["v"],
            args,
        )
        comparison_rows.append(comparison)
        energy_rows.append(energy)
        residual_rows.append(residual)
        boundary_rows.append(boundary)

    for field in _load_candidate_fields(args.repo_root):
        comparison, energy, residual, boundary, std, current = _field_rows(
            field["field_id"],
            "candidate",
            "free",
            field["source_path"],
            x,
            y,
            tri,
            area,
            K,
            exact_ref,
            field["u"],
            field["v"],
            args,
        )
        comparison_rows.append(comparison)
        energy_rows.append(energy)
        residual_rows.append(residual)
        boundary_rows.append(boundary)
        _save_npz(
            args.out_dir / "artifacts" / f"{field['field_id']}_recomputed_fields.npz",
            x,
            y,
            tri,
            element_x,
            element_y,
            field["u"],
            field["v"],
            std,
            current,
        )

    _write_rows(args.out_dir / "tables" / "exact_fe_summary.csv", exact_rows)
    _write_rows(args.out_dir / "tables" / "mechanics_field_comparison.csv", comparison_rows)
    _write_rows(args.out_dir / "tables" / "energy_decomposition_comparison.csv", energy_rows)
    _write_rows(args.out_dir / "tables" / "residual_comparison.csv", residual_rows)
    _write_rows(args.out_dir / "tables" / "boundary_residuals.csv", boundary_rows)

    commands = {
        "script": str(Path(__file__).resolve()),
        "mesh": str(args.mesh),
        "delta": args.delta,
        "E": args.E,
        "nu": args.nu,
        "l0": args.l0,
        "tm_eps_r": args.tm_eps_r,
        "eta_residual": args.eta_residual,
        "repo_root": str(args.repo_root),
        "out_dir": str(args.out_dir),
        "notes": "Region metrics are postprocessing only; no training was run.",
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir / 'tables' / 'exact_fe_summary.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'mechanics_field_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'energy_decomposition_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'residual_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'boundary_residuals.csv'}")


if __name__ == "__main__":
    main()
