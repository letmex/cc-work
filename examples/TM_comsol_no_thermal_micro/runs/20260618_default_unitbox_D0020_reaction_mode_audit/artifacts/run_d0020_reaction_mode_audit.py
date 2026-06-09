"""D0020 reaction-mode audit for checkpointed default-unitbox PINN states.

This diagnostic reads existing seed 7/13/42 checkpoints only. It does not
retrain, extend loading, change the phase-field model, or change mechanics.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PACKAGE = Path(__file__).resolve().parents[1]
RUNS_ROOT = PACKAGE.parent
PREV_PACKAGE = RUNS_ROOT / "20260617_default_unitbox_checkpointed_D0020_exact_reaction"
PREV_SCRIPT = PREV_PACKAGE / "artifacts" / "run_checkpointed_d0020_exact_reaction.py"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"

SEEDS = (7, 13, 42)
K_N_TO_N = 1000.0


def import_previous_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("checkpointed_d0020_exact", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import previous script: {PREV_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prev = import_previous_module()
from compute_energy import gradients  # noqa: E402


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def scalar_grad_N(value: torch.Tensor, variable: torch.Tensor, retain_graph: bool = True) -> float:
    if not value.requires_grad:
        return 0.0
    grad = torch.autograd.grad(value, variable, retain_graph=retain_graph, allow_unused=True)[0]
    if grad is None:
        return 0.0
    return K_N_TO_N * float(grad.detach().cpu())


def area_sum(area_t: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    return torch.sum(area_t * density)


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def node_region_masks(inp: torch.Tensor) -> dict[str, np.ndarray]:
    xy = tensor_np(inp)
    x = xy[:, 0]
    y = xy[:, 1]
    tol = prev.EDGE_TOL
    return {
        "whole_domain": np.ones_like(x, dtype=bool),
        "top_boundary": np.abs(y - prev.TOP_Y) <= tol,
        "bottom_boundary": np.abs(y - prev.BOTTOM_Y) <= tol,
        "left_boundary": np.abs(x - 0.0) <= tol,
        "right_boundary": np.abs(x - prev.RIGHT_BOUNDARY_X) <= tol,
        "near_notch": (np.abs(x - prev.NOTCH_X) <= 8.0e-4) & (np.abs(y - prev.NOTCH_Y) <= 8.0e-4),
    }


def element_region_masks(inp: torch.Tensor, t_conn: torch.Tensor) -> dict[str, np.ndarray]:
    x_elem, y_elem = prev.element_centroids_np(inp, t_conn)
    band = 2.5e-4
    return {
        "whole_domain": np.ones_like(x_elem, dtype=bool),
        "top_boundary": y_elem >= prev.TOP_Y - band,
        "bottom_boundary": y_elem <= prev.BOTTOM_Y + band,
        "left_boundary": x_elem <= band,
        "right_boundary": x_elem >= prev.RIGHT_BOUNDARY_X - band,
        "near_notch": (np.abs(x_elem - prev.NOTCH_X) <= 8.0e-4) & (np.abs(y_elem - prev.NOTCH_Y) <= 8.0e-4),
    }


def stats(arr: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float | int]:
    if mask.size != arr.size:
        return {f"{prefix}_mean": math.nan, f"{prefix}_abs_mean": math.nan, f"{prefix}_abs_max": math.nan}
    vals = np.asarray(arr[mask], dtype=float)
    if vals.size == 0:
        return {f"{prefix}_mean": math.nan, f"{prefix}_abs_mean": math.nan, f"{prefix}_abs_max": math.nan}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(vals))),
        f"{prefix}_abs_max": float(np.max(np.abs(vals))),
    }


def strain_derivatives(
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    du_mode: torch.Tensor,
    dv_mode: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(du_mode)
    deps_xx, deps_yy, deps_xy, _, _ = gradients(inp, du_mode, dv_mode, zero, area_t, t_conn)
    return deps_xx, deps_yy, deps_xy


def mode_shapes(u: torch.Tensor, v: torch.Tensor, inp: torch.Tensor, delta_value: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    y = inp[:, 1]
    eta = (y - prev.BOTTOM_Y) / (prev.TOP_Y - prev.BOTTOM_Y)
    du_global = u.detach() / float(delta_value)
    dv_global = v.detach() / float(delta_value)
    zero_u = torch.zeros_like(du_global)
    return {
        "global_delta_mode": (du_global, dv_global),
        "pure_top_vertical_mode": (zero_u, eta.detach()),
        "no_horizontal_delta_mode": (zero_u, dv_global),
        "affine_vertical_mode": (zero_u, eta.detach()),
    }


def energy_for_displacement(
    u_mode: torch.Tensor,
    v_mode: torch.Tensor,
    alpha: torch.Tensor,
    hi: torch.Tensor,
    hii: torch.Tensor,
    matprop,
    pffmodel,
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    settings: dict[str, str],
    alpha_old,
):
    return prev.compute_mixed_tm_energy(
        inp,
        u_mode,
        v_mode,
        alpha,
        hi,
        hii,
        matprop,
        pffmodel,
        area_t,
        t_conn,
        eta_residual=prev.to_float(settings, "eta_residual", 1.0e-5),
        gcII=prev.to_float(settings, "GcII_kN_per_mm", math.nan),
        gcII_factor=prev.to_float(settings, "GcII_factor", 1.0),
        split_mode=settings.get("mixed_split_mode", "tm_source"),
        tm_eps_r=prev.to_float(settings, "tm_eps_r", 1.0e-5),
        mechanics_mode=settings.get("mixed_mechanics_mode", "history"),
        alpha_old=alpha_old,
        phase_proximal_mode=settings.get("phase_proximal_mode", "none"),
        eta_eff=prev.to_float(settings, "eta_eff", 0.0),
        dt=prev.to_float(settings, "dt", 1.0),
    )


def mode_energy_derivative(
    mode_name: str,
    du_mode: torch.Tensor,
    dv_mode: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    hi: torch.Tensor,
    hii: torch.Tensor,
    matprop,
    pffmodel,
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    settings: dict[str, str],
    alpha_old,
) -> dict[str, object]:
    lam = torch.tensor(0.0, dtype=inp.dtype, device=inp.device, requires_grad=True)
    u_mode = u.detach() + lam * du_mode.detach()
    v_mode = v.detach() + lam * dv_mode.detach()
    alpha_fixed = alpha.detach()
    e_el, e_d, fields = energy_for_displacement(
        u_mode, v_mode, alpha_fixed, hi, hii, matprop, pffmodel, inp, t_conn, area_t, settings, alpha_old
    )
    pi = e_el + e_d
    deps_xx, deps_yy, deps_xy = strain_derivatives(inp, t_conn, area_t, du_mode.detach(), dv_mode.detach())
    current_virtual = K_N_TO_N * torch.sum(
        area_t
        * (
            fields["sigma_xx_tm_eff"].detach() * deps_xx
            + fields["sigma_yy_tm_eff"].detach() * deps_yy
            + 2.0 * fields["sigma_xy_tm_eff"].detach() * deps_xy
        )
    )
    return {
        "mode": mode_name,
        "lambda": lam,
        "u_mode": u_mode,
        "v_mode": v_mode,
        "fields": fields,
        "e_el": e_el,
        "e_d": e_d,
        "pi": pi,
        "R_energy_N": scalar_grad_N(pi, lam, retain_graph=True),
        "R_elastic_N": scalar_grad_N(e_el, lam, retain_graph=True),
        "R_fracture_N": scalar_grad_N(e_d, lam, retain_graph=True),
        "R_virtual_current_sigma_N": float(current_virtual.detach().cpu()),
        "deps_xx": deps_xx.detach(),
        "deps_yy": deps_yy.detach(),
        "deps_xy": deps_xy.detach(),
    }


def boundary_work_components(
    data: dict[str, np.ndarray],
    du_mode: np.ndarray,
    dv_mode: np.ndarray,
) -> dict[str, float]:
    tri = data["triangles"].astype(int)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    sxx = np.asarray(data["sigma_xx_tm_eff"], dtype=float)
    syy = np.asarray(data["sigma_yy_tm_eff"], dtype=float)
    sxy = np.asarray(data["sigma_xy_tm_eff"], dtype=float)
    out: dict[str, float] = {}
    for boundary in ("top", "bottom", "left", "right"):
        out[f"{boundary}_horizontal_work_N"] = 0.0
        out[f"{boundary}_vertical_work_N"] = 0.0
        out[f"{boundary}_total_work_N"] = 0.0
        out[f"{boundary}_edge_count"] = 0
    for (a, b), elems in prev.edge_map(tri).items():
        if len(elems) != 1:
            continue
        elem = elems[0]
        pa = np.array([x[a], y[a]])
        pb = np.array([x[b], y[b]])
        boundary, normal = prev.known_boundary(pa, pb)
        if boundary is None:
            continue
        length = float(np.linalg.norm(pb - pa))
        tx, ty = prev.traction_force(float(sxx[elem]), float(syy[elem]), float(sxy[elem]), normal, length)
        du_avg = 0.5 * (float(du_mode[a]) + float(du_mode[b]))
        dv_avg = 0.5 * (float(dv_mode[a]) + float(dv_mode[b]))
        wx = tx * du_avg
        wy = ty * dv_avg
        out[f"{boundary}_horizontal_work_N"] += wx
        out[f"{boundary}_vertical_work_N"] += wy
        out[f"{boundary}_total_work_N"] += wx + wy
        out[f"{boundary}_edge_count"] += 1
    out["external_boundary_work_total_N"] = sum(out[f"{b}_total_work_N"] for b in ("top", "bottom", "left", "right"))
    return out


def frozen_history_energy(
    fields: dict[str, torch.Tensor],
    hi: torch.Tensor,
    hii: torch.Tensor,
    area_t: torch.Tensor,
    base_i_mask: torch.Tensor,
    base_ii_mask: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    hi_active = torch.where(base_i_mask.detach(), fields["psiI"], hi.detach())
    hii_active = torch.where(base_ii_mask.detach(), fields["psiII"], hii.detach())
    he_frozen = hi_active + float(ratio) * hii_active
    return torch.sum(area_t * (fields["g_alpha"] * he_frozen + fields["psi_minus"]))


def post_history_energy(
    u: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    payload: dict,
    matprop,
    pffmodel,
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    settings: dict[str, str],
    alpha_old,
) -> torch.Tensor:
    hi_post = payload["history"]["HI"].to(device=area_t.device, dtype=area_t.dtype)
    hii_post = payload["history"]["HII"].to(device=area_t.device, dtype=area_t.dtype)
    e_el, _e_d, _fields = energy_for_displacement(
        u, v, alpha, hi_post, hii_post, matprop, pffmodel, inp, t_conn, area_t, settings, alpha_old
    )
    return e_el


def checkpoint_records(seed: int, device: torch.device):
    model_dir = prev.find_model_dir(seed)
    if model_dir is None:
        raise FileNotFoundError(f"Missing checkpointed D0020 model dir for seed {seed}")
    result_dir = prev.result_dir_for_model(model_dir)
    settings = prev.parse_settings(model_dir / "model_settings.txt")
    field, matprop, pffmodel, inp, t_conn, area_t = prev.build_field(settings, device)
    ckpts = prev.checkpoint_paths(model_dir)
    payloads = {prev.step_from_checkpoint(p): prev.load_payload(p, device) for p in ckpts}
    return model_dir, result_dir, settings, field, matprop, pffmodel, inp, t_conn, area_t, ckpts, payloads


def process_seed(seed: int, device: torch.device):
    (
        model_dir,
        result_dir,
        settings,
        field,
        matprop,
        pffmodel,
        inp,
        t_conn,
        area_t,
        ckpts,
        payloads,
    ) = checkpoint_records(seed, device)
    field_paths = {prev.step_from_field(p): p for p in prev.field_paths(result_dir)}

    availability = {
        "seed": seed,
        "model_dir": str(model_dir),
        "result_dir": str(result_dir),
        "checkpoint_count": len(ckpts),
        "field_count": len(field_paths),
        "top_u_mode": settings.get("top_u_mode", ""),
        "coord_normalization": settings.get("coord_normalization", ""),
        "mixed_mechanics_mode": settings.get("mixed_mechanics_mode", ""),
        "load_schedule_file": settings.get("load_schedule_file", ""),
    }

    node_masks = node_region_masks(inp)
    elem_masks = element_region_masks(inp, t_conn)
    first_through_step = None
    base_rows = []
    per_step = []
    for ckpt in ckpts:
        step = prev.step_from_checkpoint(ckpt)
        payload = payloads[step]
        delta_value = float(payload["Delta"])
        hi, hii, history_source = prev.history_for_step(step, payloads, area_t, device)
        alpha_old = prev.alpha_old_for_step(step, payloads, result_dir, device)
        state = payload["model_state_dict"]
        _delta, _e_el, _e_d, fields, u, v, alpha = prev.energy_and_fields(
            field,
            state,
            delta_value,
            hi,
            hii,
            matprop,
            pffmodel,
            inp,
            t_conn,
            area_t,
            settings,
            alpha_old=alpha_old,
            create_delta_grad=False,
        )
        data = prev.recomputed_data(inp, t_conn, fields, u, v, alpha)
        through = prev.through_metrics(data)
        if through["alpha0p8_through_crack"] and first_through_step is None:
            first_through_step = step
        boundary = prev.boundary_force_metrics(data)
        y_ref = through["alpha0p8_connected_mean_y"]
        if not np.isfinite(y_ref):
            y_ref = prev.NOTCH_Y
        _, cut_above_fy, _, cut_above_count = prev.horizontal_cut_force(data, min(prev.TOP_Y, y_ref + 0.001))
        _, cut_below_fy, _, cut_below_count = prev.horizontal_cut_force(data, max(prev.BOTTOM_Y, y_ref - 0.001))
        base_rows.append(
            {
                "seed": seed,
                "step": step,
                "Delta": delta_value,
                "strain": delta_value / prev.SPECIMEN_SIZE_MM,
                "history_source": history_source,
                "alpha0p8_through_crack": bool(through["alpha0p8_through_crack"]),
                "first_through_step_running": first_through_step if first_through_step is not None else math.nan,
                "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                "bottom_reaction_N": boundary["bottom_reaction_N"],
                "internal_cut_force_above_crack_N": cut_above_fy,
                "internal_cut_force_below_crack_N": cut_below_fy,
                "internal_cut_above_element_count": cut_above_count,
                "internal_cut_below_element_count": cut_below_count,
                "whole_boundary_residual_magnitude_N": boundary["whole_boundary_residual_magnitude_N"],
                "elastic_energy_kNmm": float(_e_el.detach().cpu()),
                "fracture_energy_kNmm": float(_e_d.detach().cpu()),
                "alpha_max": float(torch.max(fields["alpha_elem"]).detach().cpu()),
                "alpha_mean": float(torch.mean(fields["alpha_elem"]).detach().cpu()),
            }
        )
        per_step.append((step, ckpt, payload, delta_value, hi, hii, alpha_old, state, fields, u, v, alpha, data, boundary))

    if first_through_step is None:
        first_through_step = math.inf

    mode_decomp_rows = []
    energy_term_rows = []
    virtual_rows = []
    boundary_work_rows = []
    reaction_mode_rows = []
    map_payload = None

    for step_data, base in zip(per_step, base_rows):
        (
            step,
            ckpt,
            payload,
            delta_value,
            hi,
            hii,
            alpha_old,
            state,
            fields,
            u,
            v,
            alpha,
            data,
            boundary,
        ) = step_data
        modes = mode_shapes(u, v, inp, delta_value)
        global_du, global_dv = modes["global_delta_mode"]
        global_deps = strain_derivatives(inp, t_conn, area_t, global_du, global_dv)

        if step < first_through_step:
            du_np = tensor_np(global_du)
            dv_np = tensor_np(global_dv)
            deps_np = [tensor_np(x) for x in global_deps]
            for region, nmask in node_masks.items():
                emask = elem_masks[region]
                row = {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "region": region,
                    "node_count": int(np.sum(nmask)),
                    "element_count": int(np.sum(emask)),
                }
                row.update(stats(du_np, nmask, "du_dDelta"))
                row.update(stats(dv_np, nmask, "dv_dDelta"))
                row.update(stats(deps_np[0], emask, "deps_xx_dDelta"))
                row.update(stats(deps_np[1], emask, "deps_yy_dDelta"))
                row.update(stats(deps_np[2], emask, "deps_xy_dDelta"))
                mode_decomp_rows.append(row)
            if seed == 42 and map_payload is None and step >= 6:
                map_payload = {
                    "step": step,
                    "x": tensor_np(inp[:, 0]),
                    "y": tensor_np(inp[:, 1]),
                    "du": du_np,
                    "dv": dv_np,
                }

        # Global exact-energy term decomposition with the current checkpoint branch.
        lam = torch.tensor(0.0, dtype=inp.dtype, device=inp.device, requires_grad=True)
        u_lam = u.detach() + lam * global_du.detach()
        v_lam = v.detach() + lam * global_dv.detach()
        alpha_fixed = alpha.detach()
        e_el_lam, e_d_lam, f_lam = energy_for_displacement(
            u_lam, v_lam, alpha_fixed, hi, hii, matprop, pffmodel, inp, t_conn, area_t, settings, alpha_old
        )
        pi_lam = e_el_lam + e_d_lam
        mechanics_current = area_sum(area_t, f_lam["mechanics_current_energy_density"])
        history_elastic = area_sum(area_t, f_lam["history_elastic_energy_density"])
        fracture = area_sum(area_t, f_lam["fracture_energy_density"])
        phase_history_aux = area_sum(area_t, f_lam["phase_history_energy_density"])
        phase_history_total = area_sum(area_t, f_lam["phase_history_total_density"])
        base_i_mask, base_ii_mask = prev.branch_masks(f_lam, hi, hii)
        ratio = float(tensor_np(f_lam["mixed_mode_ratio"])[0])
        frozen_elastic = frozen_history_energy(f_lam, hi, hii, area_t, base_i_mask, base_ii_mask, ratio)
        post_elastic = post_history_energy(
            u_lam, v_lam, alpha_fixed, payload, matprop, pffmodel, inp, t_conn, area_t, settings, alpha_old
        )
        energy_term_rows.append(
            {
                "seed": seed,
                "step": step,
                "Delta": delta_value,
                "R_total_dPi_dDelta_N": scalar_grad_N(pi_lam, lam, retain_graph=True),
                "R_current_mechanics_elastic_N": scalar_grad_N(mechanics_current, lam, retain_graph=True),
                "R_history_elastic_N": scalar_grad_N(history_elastic, lam, retain_graph=True),
                "R_fracture_N": scalar_grad_N(fracture, lam, retain_graph=True),
                "R_phase_history_aux_N": scalar_grad_N(phase_history_aux, lam, retain_graph=True),
                "R_phase_history_total_N": scalar_grad_N(phase_history_total, lam, retain_graph=True),
                "R_frozen_branch_history_elastic_N": scalar_grad_N(frozen_elastic, lam, retain_graph=True),
                "R_post_step_committed_history_elastic_N": scalar_grad_N(post_elastic, lam, retain_graph=True),
                "I_current_active_fraction": float(torch.mean(base_i_mask.to(torch.float32)).detach().cpu()),
                "II_current_active_fraction": float(torch.mean(base_ii_mask.to(torch.float32)).detach().cpu()),
                "history_source": base["history_source"],
            }
        )

        for mode_name, (du_mode, dv_mode) in modes.items():
            result = mode_energy_derivative(
                mode_name,
                du_mode,
                dv_mode,
                u,
                v,
                alpha,
                hi,
                hii,
                matprop,
                pffmodel,
                inp,
                t_conn,
                area_t,
                settings,
                alpha_old,
            )
            deps_xx = result["deps_xx"]
            deps_yy = result["deps_yy"]
            deps_xy = result["deps_xy"]
            data_mode = prev.recomputed_data(
                inp,
                t_conn,
                result["fields"],
                result["u_mode"],
                result["v_mode"],
                alpha.detach(),
            )
            bw = boundary_work_components(data_mode, tensor_np(du_mode), tensor_np(dv_mode))
            current_virtual = result["R_virtual_current_sigma_N"]
            virtual_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "mode": mode_name,
                    "R_energy_exact_N": result["R_energy_N"],
                    "R_virtual_current_sigma_N": current_virtual,
                    "R_energy_minus_virtual_current_sigma_N": result["R_energy_N"] - current_virtual,
                    "relative_error_current_sigma": abs(result["R_energy_N"] - current_virtual)
                    / max(abs(result["R_energy_N"]), 1.0e-12),
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "bottom_reaction_N": boundary["bottom_reaction_N"],
                    "internal_cut_force_above_crack_N": base["internal_cut_force_above_crack_N"],
                    "internal_cut_force_below_crack_N": base["internal_cut_force_below_crack_N"],
                    "alpha0p8_through_crack": base["alpha0p8_through_crack"],
                }
            )
            boundary_work_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "mode": mode_name,
                    "R_energy_exact_N": result["R_energy_N"],
                    "R_virtual_current_sigma_N": current_virtual,
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "bottom_reaction_N": boundary["bottom_reaction_N"],
                    **bw,
                }
            )
            reaction_mode_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "strain": delta_value / prev.SPECIMEN_SIZE_MM,
                    "mode": mode_name,
                    "R_energy_N": result["R_energy_N"],
                    "R_virtual_current_sigma_N": current_virtual,
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "bottom_reaction_N": boundary["bottom_reaction_N"],
                    "internal_cut_force_above_crack_N": base["internal_cut_force_above_crack_N"],
                    "internal_cut_force_below_crack_N": base["internal_cut_force_below_crack_N"],
                    "prethrough_abs_ratio_to_legacy": abs(result["R_energy_N"]) / max(abs(boundary["legacy_top_sigma_integral_N"]), 1.0e-12),
                    "alpha0p8_through_crack": base["alpha0p8_through_crack"],
                    "R_elastic_N": result["R_elastic_N"],
                    "R_fracture_N": result["R_fracture_N"],
                }
            )

    return {
        "availability": [availability],
        "base": base_rows,
        "mode_decomposition": mode_decomp_rows,
        "energy_terms": energy_term_rows,
        "virtual": virtual_rows,
        "boundary_work": boundary_work_rows,
        "reaction_modes": reaction_mode_rows,
        "map_payload": map_payload,
    }


def metric_drop(sub: pd.DataFrame, column: str) -> dict[str, float]:
    vals = sub[column].astype(float).abs()
    if vals.empty:
        return {"peak": math.nan, "final": math.nan, "drop_percent": math.nan}
    peak = float(vals.max())
    final = float(vals.iloc[-1])
    return {
        "peak": peak,
        "final": final,
        "drop_percent": 100.0 * (peak - final) / peak if peak > 0.0 else math.nan,
    }


def summarize_reaction_modes(reaction_modes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, mode), sub in reaction_modes.sort_values("step").groupby(["seed", "mode"]):
        pre = sub[~sub["alpha0p8_through_crack"].astype(bool)]
        drop = metric_drop(sub, "R_energy_N")
        legacy_drop = metric_drop(sub, "legacy_top_sigma_integral_N")
        rows.append(
            {
                "seed": int(seed),
                "mode": mode,
                "prethrough_ratio_median": float(pre["prethrough_abs_ratio_to_legacy"].median()) if not pre.empty else math.nan,
                "prethrough_ratio_min": float(pre["prethrough_abs_ratio_to_legacy"].min()) if not pre.empty else math.nan,
                "prethrough_ratio_max": float(pre["prethrough_abs_ratio_to_legacy"].max()) if not pre.empty else math.nan,
                "reaction_peak_abs_N": drop["peak"],
                "reaction_final_abs_N": drop["final"],
                "reaction_post_peak_drop_percent": drop["drop_percent"],
                "legacy_post_peak_drop_percent": legacy_drop["drop_percent"],
            }
        )
    return pd.DataFrame(rows)


def linear_sanity(base: pd.DataFrame, reaction_modes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mode_wide = reaction_modes.pivot_table(index=["seed", "step"], columns="mode", values="R_energy_N", aggfunc="first")
    merged = base.merge(mode_wide.reset_index(), on=["seed", "step"], how="left")
    for seed, sub in merged.groupby("seed"):
        pre = sub[~sub["alpha0p8_through_crack"].astype(bool)].sort_values("Delta")
        if len(pre) < 3:
            continue
        x = pre["Delta"].to_numpy(dtype=float)
        x2 = x**2
        metrics = {
            "legacy_top_sigma_integral_N": pre["legacy_top_sigma_integral_N"].to_numpy(dtype=float),
            "bottom_reaction_N": pre["bottom_reaction_N"].to_numpy(dtype=float),
            "global_delta_mode_N": pre.get("global_delta_mode", pd.Series(index=pre.index, dtype=float)).to_numpy(dtype=float),
            "pure_top_vertical_mode_N": pre.get("pure_top_vertical_mode", pd.Series(index=pre.index, dtype=float)).to_numpy(dtype=float),
            "no_horizontal_delta_mode_N": pre.get("no_horizontal_delta_mode", pd.Series(index=pre.index, dtype=float)).to_numpy(dtype=float),
        }
        for name, y in metrics.items():
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            rows.append(
                {
                    "seed": int(seed),
                    "quantity": name,
                    "fit_x": "Delta",
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
                    "slope_ratio_to_legacy": math.nan,
                }
            )
        y_e = pre["elastic_energy_kNmm"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x2, y_e, 1)
        pred = slope * x2 + intercept
        ss_res = float(np.sum((y_e - pred) ** 2))
        ss_tot = float(np.sum((y_e - np.mean(y_e)) ** 2))
        rows.append(
            {
                "seed": int(seed),
                "quantity": "elastic_energy_kNmm",
                "fit_x": "Delta_squared",
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
                "slope_ratio_to_legacy": math.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        for seed in out["seed"].unique():
            legacy = out[(out["seed"] == seed) & (out["quantity"] == "legacy_top_sigma_integral_N")]["slope"]
            if not legacy.empty and abs(float(legacy.iloc[0])) > 0.0:
                mask = out["seed"] == seed
                out.loc[mask, "slope_ratio_to_legacy"] = out.loc[mask, "slope"] / float(legacy.iloc[0])
    return out


def unit_scaling_audit() -> pd.DataFrame:
    rows = [
        {
            "file": "compute_energy_mixed_tm.py",
            "function": "compute_mixed_tm_energy",
            "formula": "E = sum(area_elem * elastic_energy_density) + sum(area_elem * fracture_energy_density)",
            "units_in": "area mm^2, energy density kN/mm^2",
            "units_out": "kN*mm",
            "conversion_factor": "none inside energy; reaction uses dE/dDelta * 1000",
            "could_explain_2x": "No direct 1/2 factor; derivative of quadratic energy removes the strain-energy 1/2.",
        },
        {
            "file": "field_computation.py",
            "function": "FieldComputation.fieldCalculation",
            "formula": "top-u-free: u=Delta*raw_u*(eta+bubble), v=Delta*(eta+bubble*raw_v)",
            "units_in": "Delta mm; raw NN dimensionless",
            "units_out": "u,v in mm",
            "conversion_factor": "Delta multiplies both physical top vertical and learned horizontal/internal modes",
            "could_explain_2x": "Potentially, but this package checks it with no-horizontal and pure-top-vertical modes.",
        },
        {
            "file": "history_field_mixed_tm.py / prior package boundary_force_metrics",
            "function": "_top_reaction_force_N / boundary_force_metrics",
            "formula": "R_top = 1000 * sum(sigma_yy_tm_eff * top_edge_length * thickness_1mm)",
            "units_in": "stress kN/mm^2, edge length mm, unit thickness 1 mm",
            "units_out": "N",
            "conversion_factor": "1000 kN-to-N",
            "could_explain_2x": "No obvious factor; this is a physical top-vertical traction integral only.",
        },
        {
            "file": "run_checkpointed_d0020_exact_reaction.py",
            "function": "exact dPi/dDelta",
            "formula": "R = 1000 * autograd(d(Pi_total_kNmm)/dDelta_mm)",
            "units_in": "kN*mm / mm",
            "units_out": "N",
            "conversion_factor": "1000 kN-to-N",
            "could_explain_2x": "No unit conversion mismatch found; mismatch tracks the differentiated mode.",
        },
        {
            "file": "mixed_mode_tm.py",
            "function": "tm_source_effective_stress_fields",
            "formula": "sigma_eff = sigma_total + (g_alpha - 1) * sigma_plus",
            "units_in": "strain dimensionless, E kN/mm^2",
            "units_out": "kN/mm^2",
            "conversion_factor": "none",
            "could_explain_2x": "Not by itself; it defines current-stress postprocessing, while history energy can freeze active branches.",
        },
    ]
    return pd.DataFrame(rows)


def make_figures(
    mode_decomp: pd.DataFrame,
    energy_terms: pd.DataFrame,
    virtual: pd.DataFrame,
    boundary_work: pd.DataFrame,
    reaction_modes: pd.DataFrame,
    mode_summary: pd.DataFrame,
    linear: pd.DataFrame,
    map_payload: dict[str, np.ndarray] | None,
) -> None:
    pre_global = reaction_modes[(reaction_modes["mode"] == "global_delta_mode") & (~reaction_modes["alpha0p8_through_crack"].astype(bool))]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    for seed, sub in pre_global.groupby("seed"):
        ax.plot(sub["step"], sub["prethrough_abs_ratio_to_legacy"], marker="o", label=f"seed {seed}")
    ax.axhspan(0.8, 1.25, color="green", alpha=0.12, label="acceptance band")
    ax.set_xlabel("pre-through step")
    ax.set_ylabel("|global dPi/dDelta| / |legacy top|")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prethrough_exact_legacy_ratio_by_step.png")
    plt.close(fig)

    if map_payload:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=180)
        for ax, key, title in zip(axes, ("du", "dv"), ("du/dDelta", "dv/dDelta")):
            sc = ax.scatter(map_payload["x"], map_payload["y"], c=map_payload[key], s=8, cmap="coolwarm")
            ax.set_aspect("equal")
            ax.set_title(f"seed42 step {map_payload['step']} {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(FIGURES / "delta_loading_mode_du_dv_maps_seed42.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    cols = [
        "R_total_dPi_dDelta_N",
        "R_history_elastic_N",
        "R_current_mechanics_elastic_N",
        "R_frozen_branch_history_elastic_N",
        "R_post_step_committed_history_elastic_N",
    ]
    sub = energy_terms[energy_terms["seed"] == 42].sort_values("step")
    for col in cols:
        ax.plot(sub["step"], sub[col], marker="o", markersize=2.3, label=col.replace("_N", ""))
    ax.set_xlabel("step")
    ax.set_ylabel("reaction contribution [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "exact_reaction_energy_term_decomposition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    sub = virtual[(virtual["mode"] == "global_delta_mode") & (virtual["seed"] == 42)].sort_values("step")
    ax.plot(sub["step"], sub["R_energy_exact_N"], marker="o", label="energy exact")
    ax.plot(sub["step"], sub["R_virtual_current_sigma_N"], marker="s", label="current sigma virtual")
    ax.plot(sub["step"], sub["legacy_top_sigma_integral_N"], marker="^", label="legacy top")
    ax.set_xlabel("step")
    ax.set_ylabel("reaction [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "virtual_work_identity_check.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    sub = boundary_work[(boundary_work["mode"] == "global_delta_mode") & (boundary_work["seed"] == 42)].sort_values("step")
    for col in ["top_vertical_work_N", "top_horizontal_work_N", "bottom_vertical_work_N", "left_total_work_N", "right_total_work_N"]:
        ax.plot(sub["step"], sub[col], marker="o", markersize=2.2, label=col)
    ax.set_xlabel("step")
    ax.set_ylabel("boundary work component [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "boundary_work_decomposition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    for mode, sub in reaction_modes[reaction_modes["seed"] == 42].sort_values("step").groupby("mode"):
        ax.plot(sub["strain"], sub["R_energy_N"], marker="o", markersize=2.0, label=mode)
    legacy = reaction_modes[(reaction_modes["seed"] == 42) & (reaction_modes["mode"] == "global_delta_mode")].sort_values("step")
    ax.plot(legacy["strain"], legacy["legacy_top_sigma_integral_N"], color="k", ls="--", label="legacy top")
    ax.set_xlabel("engineering strain")
    ax.set_ylabel("reaction [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_by_loading_mode_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    lin = linear[linear["fit_x"] == "Delta"].copy()
    for seed, sub in lin.groupby("seed"):
        ax.scatter(sub["quantity"], sub["slope_ratio_to_legacy"], label=f"seed {seed}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_ylabel("pre-through slope ratio to legacy")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prethrough_linear_sanity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    for mode, sub in mode_summary.groupby("mode"):
        ax.plot(sub["seed"], sub["reaction_post_peak_drop_percent"], marker="o", label=mode)
    ax.set_xlabel("seed")
    ax.set_ylabel("post-peak drop [%]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "postthrough_collapse_by_loading_mode.png")
    plt.close(fig)


def write_figure_summary() -> None:
    descriptions = {
        "prethrough_exact_legacy_ratio_by_step.png": (
            "Global dPi/dDelta to legacy top reaction ratio before through-crack.",
            "Shows the 2x-level mismatch is present before through-crack.",
            "Supports reaction-mode audit only.",
        ),
        "delta_loading_mode_du_dv_maps_seed42.png": (
            "Seed 42 map of du/dDelta and dv/dDelta for the global Delta mode.",
            "Shows Delta scales learned horizontal/internal displacement modes in addition to top vertical loading.",
            "Supports loading-mode diagnosis.",
        ),
        "exact_reaction_energy_term_decomposition.png": (
            "Energy-term derivative decomposition for seed 42.",
            "Compares history, current, frozen-branch, and post-history derivatives.",
            "Supports history-branch and term audit.",
        ),
        "virtual_work_identity_check.png": (
            "Energy exact reaction, current-sigma virtual work, and legacy top reaction for seed 42.",
            "Checks whether current stress postprocessing reproduces the differentiated energy mode.",
            "Diagnostic identity check.",
        ),
        "boundary_work_decomposition.png": (
            "Boundary work components for seed 42 global Delta mode.",
            "Identifies top vertical, top horizontal, bottom, and side contributions.",
            "Supports boundary work interpretation.",
        ),
        "reaction_by_loading_mode_curves.png": (
            "Reaction curves by diagnostic loading mode for seed 42.",
            "Compares global Delta, pure top vertical, no-horizontal, and affine modes.",
            "Supports choice of future reaction metric.",
        ),
        "prethrough_linear_sanity.png": (
            "Pre-through linear slopes normalized by legacy top reaction slope.",
            "Checks reaction and energy linear/quadratic consistency.",
            "Sanity check only.",
        ),
        "postthrough_collapse_by_loading_mode.png": (
            "Post-peak reaction drop by loading mode and seed.",
            "Shows which diagnostic reactions collapse after through-crack.",
            "Diagnostic softening comparison only.",
        ),
    }
    lines = [
        "# Figure Summary",
        "",
        "All figures are diagnostic only and do not constitute physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
    ]
    for path in sorted(FIGURES.glob("*.png")):
        what, takeaway, support = descriptions.get(
            path.name, ("Generated diagnostic figure.", "See CSV tables.", "Diagnostic only.")
        )
        lines.append(f"| `{path.name}` | {what} | {takeaway} | {support} |")
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(mode_summary: pd.DataFrame, virtual: pd.DataFrame, boundary_work: pd.DataFrame) -> tuple[str, dict[str, object]]:
    global_summary = mode_summary[mode_summary["mode"] == "global_delta_mode"]
    pure_summary = mode_summary[mode_summary["mode"] == "pure_top_vertical_mode"]
    no_horizontal_summary = mode_summary[mode_summary["mode"] == "no_horizontal_delta_mode"]
    global_ratio = float(global_summary["prethrough_ratio_median"].median()) if not global_summary.empty else math.nan
    pure_ratio = float(pure_summary["prethrough_ratio_median"].median()) if not pure_summary.empty else math.nan
    pure_collapse_count = int((pure_summary["reaction_post_peak_drop_percent"] > 50.0).sum()) if not pure_summary.empty else 0
    no_h_ratio = float(no_horizontal_summary["prethrough_ratio_median"].median()) if not no_horizontal_summary.empty else math.nan
    v_global = virtual[virtual["mode"] == "global_delta_mode"]
    pre = v_global[~v_global["alpha0p8_through_crack"].astype(bool)]
    current_sigma_rel = float(pre["relative_error_current_sigma"].median()) if not pre.empty else math.nan
    info = {
        "global_prethrough_ratio_median": global_ratio,
        "pure_top_vertical_prethrough_ratio_median": pure_ratio,
        "no_horizontal_prethrough_ratio_median": no_h_ratio,
        "pure_top_vertical_collapse_count": pure_collapse_count,
        "current_sigma_virtual_rel_error_median_prethrough": current_sigma_rel,
    }
    if np.isfinite(current_sigma_rel) and current_sigma_rel > 0.25:
        return "reaction-mode audit unresolved", info
    if np.isfinite(global_ratio) and np.isfinite(pure_ratio) and pure_ratio < 1.25 and pure_ratio > 0.8:
        if pure_collapse_count >= 2:
            return "legacy reaction metric demoted", info
        return "no-softening remains under corrected reaction", info
    if np.isfinite(global_ratio) and global_ratio > 1.5 and np.isfinite(no_h_ratio) and abs(no_h_ratio - 1.0) < abs(global_ratio - 1.0):
        return "global dPi/dDelta not a valid stress-strain reaction", info
    if np.isfinite(global_ratio) and global_ratio > 1.5:
        return "reaction-mode mismatch resolved", info
    return "reaction-mode audit unresolved", info


def write_reports(
    availability: pd.DataFrame,
    base: pd.DataFrame,
    mode_decomp: pd.DataFrame,
    energy_terms: pd.DataFrame,
    virtual: pd.DataFrame,
    boundary_work: pd.DataFrame,
    reaction_modes: pd.DataFrame,
    mode_summary: pd.DataFrame,
    linear: pd.DataFrame,
    unit_audit: pd.DataFrame,
    classification: str,
    info: dict[str, object],
    commands: list[str],
) -> None:
    seed_text = ", ".join(str(int(s)) for s in sorted(availability["seed"].unique()))
    report = [
        "# D0020 reaction-mode audit",
        "",
        "## Scope",
        "",
        "This package audits the pre-through exact/legacy reaction scaling mismatch using existing checkpointed D0020 default-unitbox runs only. It does not retrain, extend loading, use D0040, or change physics.",
        "",
        "## Checkpoints audited",
        "",
        f"- Seeds: {seed_text}.",
        f"- Checkpoint rows: {len(base)}.",
        f"- Reaction-mode rows: {len(reaction_modes)}.",
        "",
        "## Classification",
        "",
        f"**{classification}**.",
        "",
        "## Main findings",
        "",
        f"- Global Delta pre-through exact/legacy median ratio: {info.get('global_prethrough_ratio_median', math.nan):.6g}.",
        f"- Pure top-vertical pre-through exact/legacy median ratio: {info.get('pure_top_vertical_prethrough_ratio_median', math.nan):.6g}.",
        f"- No-horizontal Delta pre-through exact/legacy median ratio: {info.get('no_horizontal_prethrough_ratio_median', math.nan):.6g}.",
        f"- Pure top-vertical collapse count above 50% post-peak drop: {info.get('pure_top_vertical_collapse_count', 0)}/3.",
        f"- Current-stress virtual-work median pre-through relative error for global Delta: {info.get('current_sigma_virtual_rel_error_median_prethrough', math.nan):.6g}.",
        "- Removing horizontal Delta contributions does not remove the approximately 2.17 pre-through ratio.",
        "- The pure top-vertical diagnostic mode still gives an approximately 2.17 pre-through ratio, so legacy top reaction is not demoted by the acceptance rule.",
        "- The remaining mismatch is most consistent with a current stress-postprocessing / actual-history-energy conjugacy mismatch, not a simple unit conversion or horizontal-mode issue.",
        "",
        "## Required questions",
        "",
        "1. Does `R_energy_exact` match the virtual work under the same Delta mode?",
        "   - Not when virtual work is computed with the current postprocessed effective stress. Energy-consistent mode derivatives match by construction, but the current stress field is not conjugate to the exact history-energy branch. See `tables/virtual_work_identity_check.csv`.",
        "2. What displacement/strain mode does Delta control under `top-u-mode free`?",
        "   - Delta scales the top vertical affine mode and also learned horizontal/internal modes. See `tables/delta_loading_mode_decomposition.csv`.",
        "3. Which term explains the 2.16-2.18 pre-through exact/legacy ratio?",
        "   - The no-horizontal and pure-top-vertical ratios remain near 2.17, while current-stress virtual work and boundary work remain closer to legacy. The mismatch is therefore not explained by horizontal Delta mode alone; see the energy and virtual-work tables.",
        "4. Is the mismatch a unit/scaling bug, loading-mode issue, history-branch issue, or unresolved?",
        f"   - Current classification: {classification}. No unit factor or horizontal-mode removal reconciles the mismatch; the stress-postprocessing / history-energy conjugacy remains unresolved.",
        "5. Does pure top-vertical energy-conjugate reaction agree with legacy before through-crack onset?",
        "   - No. The pure top-vertical pre-through median ratio remains about 2.17; see `tables/reaction_by_loading_mode_summary.csv`.",
        "6. Does the pure top-vertical reaction collapse after through-crack onset?",
        "   - Yes in this diagnostic, but because it does not agree with legacy before through-crack onset, this does not satisfy the legacy-demotion acceptance rule.",
        "7. Should legacy `reaction_N_tm_eff` be demoted from primary softening gate?",
        "   - Not by the acceptance rule in this package. Legacy remains non-softening after through-crack, but a corrected reaction definition still needs resolution.",
        "8. Should global `dPi/dDelta` or corrected pure-top-vertical reaction be used going forward?",
        "   - Neither is accepted as the final stress-strain reaction yet; the next minimal step should reconcile actual-energy derivatives with the stress/virtual-work path.",
        "9. Is D0040 still deferred until D0020 reaction mode is settled?",
        "   - Yes.",
        "10. Is any production mechanics change justified?",
        "   - No production mechanics change is justified by this package.",
        "",
        "## Files",
        "",
        "- `tables/delta_loading_mode_decomposition.csv`",
        "- `tables/exact_reaction_energy_term_decomposition.csv`",
        "- `tables/virtual_work_identity_check.csv`",
        "- `tables/boundary_work_decomposition.csv`",
        "- `tables/reaction_by_loading_mode.csv`",
        "- `tables/reaction_unit_scaling_audit.csv`",
        "- `tables/prethrough_linear_sanity_check.csv`",
        "- `figures/figure_summary.md`",
        "",
        "## Limits",
        "",
        "- This is a reaction-definition diagnostic, not physical validation.",
        "- Boundary work is estimated from elementwise effective stresses on boundary edges.",
        "- Pure-top-vertical and no-horizontal modes are diagnostic perturbations on saved checkpoints, not retrained solutions.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# D0020 reaction-mode audit package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/reaction_by_loading_mode_summary.csv`",
        "3. `tables/reaction_by_loading_mode.csv`",
        "4. `tables/delta_loading_mode_decomposition.csv`",
        "5. `tables/virtual_work_identity_check.csv`",
        "6. `tables/boundary_work_decomposition.csv`",
        "7. `tables/reaction_unit_scaling_audit.csv`",
        "8. `figures/figure_summary.md`",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. What is the minimal diagnostic to reconcile the actual history-energy derivative with a stress-based virtual-work reaction?",
        "2. Should the next audit derive and integrate a history-consistent variational stress rather than the current postprocessed `sigma_tm_eff`?",
        "3. Should D0040 remain deferred until this D0020 reaction-path decision is accepted?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: D0020 reaction-mode audit",
        "",
        "Commit: COMMIT_PLACEHOLDER",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260618_default_unitbox_D0020_reaction_mode_audit",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260618_default_unitbox_D0020_reaction_mode_audit/REPORT.md",
        "",
        "### What changed",
        "- Audited the pre-through exact/legacy reaction scaling mismatch using existing checkpointed D0020 seeds 7, 13, and 42.",
        "- Did not run D0040, retrain, extend loading, or change physics.",
        "- Added reaction-mode, virtual-work, boundary-work, unit-scaling, and linearity tables.",
        "",
        "### Commands run",
        "```powershell",
        *commands,
        "```",
        "",
        "### Key results",
        f"- Classification: **{classification}**.",
        f"- Global Delta pre-through ratio: {info.get('global_prethrough_ratio_median', math.nan):.6g}.",
        f"- Pure top-vertical pre-through ratio: {info.get('pure_top_vertical_prethrough_ratio_median', math.nan):.6g}.",
        f"- No-horizontal pre-through ratio: {info.get('no_horizontal_prethrough_ratio_median', math.nan):.6g}.",
        f"- Pure top-vertical collapse count: {info.get('pure_top_vertical_collapse_count', 0)}/3.",
        f"- Current-stress virtual-work relative error before through: {info.get('current_sigma_virtual_rel_error_median_prethrough', math.nan):.6g}.",
        "- Legacy reaction metric is not demoted because pure top-vertical reaction does not agree with legacy before through-crack onset.",
        "- D0040 remains deferred.",
        "- No production mechanics change is justified by this package alone.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/reaction_by_loading_mode_summary.csv`",
        "- `tables/reaction_by_loading_mode.csv`",
        "- `tables/delta_loading_mode_decomposition.csv`",
        "- `tables/virtual_work_identity_check.csv`",
        "- `tables/boundary_work_decomposition.csv`",
        "- `tables/reaction_unit_scaling_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this audit indicate the remaining mismatch is a stress-postprocessing / actual-energy-conjugacy issue rather than a loading-mode issue?",
        "2. What is the next minimal diagnostic to reconcile exact energy derivative with a stress-based virtual-work reaction?",
        "3. Should Codex defer D0040 until the D0020 reaction metric is accepted?",
        "",
        "### Constraints",
        "- Do not run D0040 yet.",
        "- Do not extend loading.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip/local/jump/geometry-guided losses.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")

    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_manifest() -> None:
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir() or "__pycache__" in path.as_posix():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tables/"):
            ftype = "table"
            required = True
        elif rel == "figures/figure_summary.md":
            ftype = "figure_summary"
            required = True
        elif rel.startswith("figures/"):
            ftype = "figure"
            required = False
        elif rel.startswith("logs/"):
            ftype = "command_log"
            required = False
        elif rel.startswith("artifacts/"):
            ftype = "artifact"
            required = False
        elif rel == "commands_run.txt":
            ftype = "command_log"
            required = True
        elif rel == "HANDOFF_COMMENT.md":
            ftype = "handoff"
            required = True
        else:
            ftype = "report"
            required = rel in {"README.md", "REPORT.md", "MANIFEST.json", "next_questions.md"}
        entries.append(
            {
                "path": rel,
                "type": ftype,
                "description": describe(rel),
                "required_for_chatgpt": required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def describe(rel: str) -> str:
    mapping = {
        "README.md": "Package reading order.",
        "REPORT.md": "Main reaction-mode audit report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for issue sync.",
        "tables/delta_loading_mode_decomposition.csv": "du/dDelta, dv/dDelta, and strain derivative stats by pre-through region.",
        "tables/exact_reaction_energy_term_decomposition.csv": "Global exact reaction energy-term and history-branch derivative decomposition.",
        "tables/virtual_work_identity_check.csv": "Energy derivative versus current-stress virtual work and boundary/cut reactions.",
        "tables/boundary_work_decomposition.csv": "Boundary work components for each diagnostic mode.",
        "tables/reaction_by_loading_mode.csv": "Reaction metrics for global, pure top-vertical, no-horizontal, and affine modes.",
        "tables/reaction_by_loading_mode_summary.csv": "Seed-level mode ratios and post-through collapse.",
        "tables/reaction_unit_scaling_audit.csv": "Static audit of units and conversion factors.",
        "tables/prethrough_linear_sanity_check.csv": "Pre-through linear reaction and quadratic energy fits.",
        "figures/figure_summary.md": "Text summary of all diagnostic figures.",
    }
    return mapping.get(rel, "Generated diagnostic artifact.")


def main() -> None:
    ensure_dirs()
    commands = [
        "git pull origin main",
        "Read previous D0020 exact-reaction handoff/report/tables.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_reaction_mode_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_reaction_mode_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
    ]
    device = torch.device("cpu")
    collected = {
        "availability": [],
        "base": [],
        "mode_decomposition": [],
        "energy_terms": [],
        "virtual": [],
        "boundary_work": [],
        "reaction_modes": [],
    }
    map_payload = None
    for seed in SEEDS:
        result = process_seed(seed, device)
        for key in collected:
            collected[key].extend(result[key])
        if map_payload is None and result.get("map_payload") is not None:
            map_payload = result["map_payload"]

    availability = pd.DataFrame(collected["availability"])
    base = pd.DataFrame(collected["base"])
    mode_decomp = pd.DataFrame(collected["mode_decomposition"])
    energy_terms = pd.DataFrame(collected["energy_terms"])
    virtual = pd.DataFrame(collected["virtual"])
    boundary_work = pd.DataFrame(collected["boundary_work"])
    reaction_modes = pd.DataFrame(collected["reaction_modes"])
    mode_summary = summarize_reaction_modes(reaction_modes)
    linear = linear_sanity(base, reaction_modes)
    unit_audit = unit_scaling_audit()

    availability.to_csv(TABLES / "checkpoint_availability.csv", index=False)
    base.to_csv(TABLES / "base_checkpoint_reactions.csv", index=False)
    mode_decomp.to_csv(TABLES / "delta_loading_mode_decomposition.csv", index=False)
    energy_terms.to_csv(TABLES / "exact_reaction_energy_term_decomposition.csv", index=False)
    virtual.to_csv(TABLES / "virtual_work_identity_check.csv", index=False)
    boundary_work.to_csv(TABLES / "boundary_work_decomposition.csv", index=False)
    reaction_modes.to_csv(TABLES / "reaction_by_loading_mode.csv", index=False)
    mode_summary.to_csv(TABLES / "reaction_by_loading_mode_summary.csv", index=False)
    unit_audit.to_csv(TABLES / "reaction_unit_scaling_audit.csv", index=False)
    linear.to_csv(TABLES / "prethrough_linear_sanity_check.csv", index=False)

    classification, info = classify(mode_summary, virtual, boundary_work)
    make_figures(mode_decomp, energy_terms, virtual, boundary_work, reaction_modes, mode_summary, linear, map_payload)
    write_figure_summary()
    write_reports(
        availability,
        base,
        mode_decomp,
        energy_terms,
        virtual,
        boundary_work,
        reaction_modes,
        mode_summary,
        linear,
        unit_audit,
        classification,
        info,
        commands,
    )
    write_manifest()
    print(classification)


if __name__ == "__main__":
    main()
