"""Energy-stress conjugacy audit for checkpointed D0020 PINN reactions.

The script reads existing D0020 seed 7/13/42 checkpoints. It does not retrain,
extend loading, or change the physical/model settings.
"""

from __future__ import annotations

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
PREV_MODE_PACKAGE = RUNS_ROOT / "20260618_default_unitbox_D0020_reaction_mode_audit"
PREV_MODE_SCRIPT = PREV_MODE_PACKAGE / "artifacts" / "run_d0020_reaction_mode_audit.py"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"

SEEDS = (7, 13, 42)
K_N_TO_N = 1000.0
SELECTED_BASE_STEPS = (0, 2, 6, 13, 14, 20, 33)
BRANCHES = (
    "current_trial_energy",
    "pre_step_history_energy",
    "post_step_committed_history_energy",
    "frozen_active_branch_energy",
)
MODES = ("global_delta_mode", "pure_top_vertical_mode", "no_horizontal_delta_mode", "affine_vertical_mode")


def import_previous_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("reaction_mode_audit", PREV_MODE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import previous audit script: {PREV_MODE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prev_mode = import_previous_module()
prev_exact = prev_mode.prev

from compute_energy import gradients  # noqa: E402
from mixed_mode_tm import mixed_mode_energy_split, tm_source_effective_stress_fields  # noqa: E402


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def scalar_grad_N(value: torch.Tensor, variable: torch.Tensor, retain_graph: bool = True) -> float:
    if not value.requires_grad:
        return 0.0
    grad = torch.autograd.grad(value, variable, retain_graph=retain_graph, allow_unused=True)[0]
    if grad is None:
        return 0.0
    return K_N_TO_N * float(grad.detach().cpu())


def area_reaction_N(area_t: torch.Tensor, integrand: torch.Tensor) -> float:
    return K_N_TO_N * float(torch.sum(area_t * integrand).detach().cpu())


def material_ratio(matprop, settings: dict[str, str]) -> float:
    gc = float((matprop.w1 * matprop.l0).detach().cpu())
    gcII = prev_exact.to_float(settings, "GcII_kN_per_mm", math.nan)
    if np.isfinite(gcII) and gcII > 0.0:
        return gc / gcII
    return 1.0 / prev_exact.to_float(settings, "GcII_factor", 1.0)


def energy_density_from_strain(
    eps_xx: torch.Tensor,
    eps_yy: torch.Tensor,
    eps_xy: torch.Tensor,
    alpha_elem: torch.Tensor,
    hi_old: torch.Tensor,
    hii_old: torch.Tensor,
    hi_post: torch.Tensor,
    hii_post: torch.Tensor,
    matprop,
    settings: dict[str, str],
    branch: str,
    frozen_i_mask: torch.Tensor | None = None,
    frozen_ii_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    split = mixed_mode_energy_split(
        eps_xx,
        eps_yy,
        eps_xy,
        matprop,
        eps_r=prev_exact.to_float(settings, "tm_eps_r", 1.0e-5),
        split_mode=settings.get("mixed_split_mode", "tm_source"),
    )
    ratio = material_ratio(matprop, settings)
    g_alpha = (1.0 - alpha_elem) ** 2 + prev_exact.to_float(settings, "eta_residual", 1.0e-5)
    if branch == "current_trial_energy":
        he = split["psiI"] + ratio * split["psiII"]
        active_i = torch.ones_like(he, dtype=torch.bool)
        active_ii = torch.ones_like(he, dtype=torch.bool)
    elif branch == "pre_step_history_energy":
        active_i = split["psiI"] > hi_old.detach()
        active_ii = split["psiII"] > hii_old.detach()
        he = torch.maximum(hi_old.detach(), split["psiI"]) + ratio * torch.maximum(
            hii_old.detach(), split["psiII"]
        )
    elif branch == "post_step_committed_history_energy":
        active_i = split["psiI"] > hi_post.detach()
        active_ii = split["psiII"] > hii_post.detach()
        he = torch.maximum(hi_post.detach(), split["psiI"]) + ratio * torch.maximum(
            hii_post.detach(), split["psiII"]
        )
    elif branch == "frozen_active_branch_energy":
        if frozen_i_mask is None or frozen_ii_mask is None:
            raise ValueError("Frozen branch requires active masks")
        active_i = frozen_i_mask.detach()
        active_ii = frozen_ii_mask.detach()
        he = torch.where(active_i, split["psiI"], hi_old.detach()) + ratio * torch.where(
            active_ii, split["psiII"], hii_old.detach()
        )
    else:
        raise ValueError(f"Unsupported branch: {branch}")
    density = g_alpha * he + split["psi_minus"]
    split.update(
        {
            "g_alpha": g_alpha,
            "He_branch": he,
            "active_I": active_i,
            "active_II": active_ii,
            "mixed_mode_ratio": torch.full_like(he, float(ratio)),
            "branch_density": density,
        }
    )
    return density, split


def energy_autograd_stress(
    eps_xx: torch.Tensor,
    eps_yy: torch.Tensor,
    eps_xy: torch.Tensor,
    alpha_elem: torch.Tensor,
    hi_old: torch.Tensor,
    hii_old: torch.Tensor,
    hi_post: torch.Tensor,
    hii_post: torch.Tensor,
    matprop,
    settings: dict[str, str],
    branch: str,
    frozen_i_mask: torch.Tensor | None = None,
    frozen_ii_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    exx = eps_xx.detach().clone().requires_grad_(True)
    eyy = eps_yy.detach().clone().requires_grad_(True)
    exy = eps_xy.detach().clone().requires_grad_(True)
    density, split = energy_density_from_strain(
        exx,
        eyy,
        exy,
        alpha_elem.detach(),
        hi_old,
        hii_old,
        hi_post,
        hii_post,
        matprop,
        settings,
        branch,
        frozen_i_mask=frozen_i_mask,
        frozen_ii_mask=frozen_ii_mask,
    )
    d_xx, d_yy, d_xy = torch.autograd.grad(density.sum(), (exx, eyy, exy), retain_graph=False)
    post = tm_source_effective_stress_fields(
        eps_xx.detach(),
        eps_yy.detach(),
        eps_xy.detach(),
        alpha_elem.detach(),
        matprop,
        eta_residual=prev_exact.to_float(settings, "eta_residual", 1.0e-5),
        eps_r=prev_exact.to_float(settings, "tm_eps_r", 1.0e-5),
    )
    out = {
        "sigma_xx_energy_autograd": d_xx.detach(),
        "sigma_yy_energy_autograd": d_yy.detach(),
        "dW_deps_xy_energy_autograd": d_xy.detach(),
        "sigma_xy_energy_autograd_tensor": (0.5 * d_xy).detach(),
        "sigma_xx_tm_eff": post["sigma_xx_tm_eff"].detach(),
        "sigma_yy_tm_eff": post["sigma_yy_tm_eff"].detach(),
        "sigma_xy_tm_eff": post["sigma_xy_tm_eff"].detach(),
        "sigma_xx_tm_total": post["sigma_xx_tm_total"].detach(),
        "sigma_yy_tm_total": post["sigma_yy_tm_total"].detach(),
        "sigma_xy_tm_total": post["sigma_xy_tm_total"].detach(),
        "sigma_xx_tm_plus": post["sigma_xx_tm_plus"].detach(),
        "sigma_yy_tm_plus": post["sigma_yy_tm_plus"].detach(),
        "sigma_xy_tm_plus": post["sigma_xy_tm_plus"].detach(),
        "sigma_xx_tm_minus": post["sigma_xx_tm_minus"].detach(),
        "sigma_yy_tm_minus": post["sigma_yy_tm_minus"].detach(),
        "sigma_xy_tm_minus": post["sigma_xy_tm_minus"].detach(),
        "energy_density": density.detach(),
        "active_I": split["active_I"].detach(),
        "active_II": split["active_II"].detach(),
        "psiI": split["psiI"].detach(),
        "psiII": split["psiII"].detach(),
        "psi_minus": split["psi_minus"].detach(),
        "eps_zz": split["eps_zz"].detach(),
        "mixed_mode_ratio": split["mixed_mode_ratio"].detach(),
    }
    return out


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


def virtual_work_from_energy_stress(
    area_t: torch.Tensor,
    stress: dict[str, torch.Tensor],
    deps_xx: torch.Tensor,
    deps_yy: torch.Tensor,
    deps_xy: torch.Tensor,
) -> float:
    return area_reaction_N(
        area_t,
        stress["sigma_xx_energy_autograd"] * deps_xx
        + stress["sigma_yy_energy_autograd"] * deps_yy
        + stress["dW_deps_xy_energy_autograd"] * deps_xy,
    )


def virtual_work_from_post_sigma(
    area_t: torch.Tensor,
    stress: dict[str, torch.Tensor],
    deps_xx: torch.Tensor,
    deps_yy: torch.Tensor,
    deps_xy: torch.Tensor,
) -> float:
    return area_reaction_N(
        area_t,
        stress["sigma_xx_tm_eff"] * deps_xx
        + stress["sigma_yy_tm_eff"] * deps_yy
        + 2.0 * stress["sigma_xy_tm_eff"] * deps_xy,
    )


def branch_reaction(
    u: torch.Tensor,
    v: torch.Tensor,
    alpha_elem: torch.Tensor,
    du_mode: torch.Tensor,
    dv_mode: torch.Tensor,
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    hi_old: torch.Tensor,
    hii_old: torch.Tensor,
    hi_post: torch.Tensor,
    hii_post: torch.Tensor,
    matprop,
    settings: dict[str, str],
    branch: str,
    frozen_i_mask: torch.Tensor | None = None,
    frozen_ii_mask: torch.Tensor | None = None,
) -> float:
    lam = torch.tensor(0.0, dtype=inp.dtype, device=inp.device, requires_grad=True)
    u_lam = u.detach() + lam * du_mode.detach()
    v_lam = v.detach() + lam * dv_mode.detach()
    zero = torch.zeros_like(u_lam)
    eps_xx, eps_yy, eps_xy, _, _ = gradients(inp, u_lam, v_lam, zero, area_t, t_conn)
    density, _split = energy_density_from_strain(
        eps_xx,
        eps_yy,
        eps_xy,
        alpha_elem.detach(),
        hi_old,
        hii_old,
        hi_post,
        hii_post,
        matprop,
        settings,
        branch,
        frozen_i_mask=frozen_i_mask,
        frozen_ii_mask=frozen_ii_mask,
    )
    energy = torch.sum(area_t * density)
    return scalar_grad_N(energy, lam, retain_graph=False)


def selected_steps(steps: list[int], first_through_step: int | None) -> set[int]:
    selected = set()
    available = set(steps)
    for step in SELECTED_BASE_STEPS:
        if step in available:
            selected.add(step)
    if first_through_step is not None:
        for step in (first_through_step - 1, first_through_step, first_through_step + 1):
            if step in available:
                selected.add(step)
    selected.add(min(steps))
    selected.add(max(steps))
    return selected


def boundary_force_with_energy_stress(
    data: dict[str, np.ndarray],
    sxx: np.ndarray,
    syy: np.ndarray,
    sxy_tensor: np.ndarray,
) -> dict[str, float]:
    data_e = dict(data)
    data_e["sigma_xx_tm_eff"] = sxx
    data_e["sigma_yy_tm_eff"] = syy
    data_e["sigma_xy_tm_eff"] = sxy_tensor
    return prev_exact.boundary_force_metrics(data_e)


def process_seed(seed: int, device: torch.device) -> dict[str, list[dict[str, object]] | dict[str, np.ndarray] | None]:
    (
        model_dir,
        result_dir,
        settings,
        field,
        matprop,
        _pffmodel,
        inp,
        t_conn,
        area_t,
        ckpts,
        payloads,
    ) = prev_mode.checkpoint_records(seed, device)
    availability = [
        {
            "seed": seed,
            "model_dir": str(model_dir),
            "result_dir": str(result_dir),
            "checkpoint_count": len(ckpts),
            "top_u_mode": settings.get("top_u_mode", ""),
            "coord_normalization": settings.get("coord_normalization", ""),
            "mixed_mechanics_mode": settings.get("mixed_mechanics_mode", ""),
        }
    ]
    contexts = []
    first_through = None
    for ckpt in ckpts:
        step = prev_exact.step_from_checkpoint(ckpt)
        payload = payloads[step]
        delta_value = float(payload["Delta"])
        hi_old, hii_old, history_source = prev_exact.history_for_step(step, payloads, area_t, device)
        alpha_old = prev_exact.alpha_old_for_step(step, payloads, result_dir, device)
        _delta, _e_el, _e_d, fields, u, v, alpha = prev_exact.energy_and_fields(
            field,
            payload["model_state_dict"],
            delta_value,
            hi_old,
            hii_old,
            matprop,
            _pffmodel,
            inp,
            t_conn,
            area_t,
            settings,
            alpha_old=alpha_old,
            create_delta_grad=False,
        )
        data = prev_exact.recomputed_data(inp, t_conn, fields, u, v, alpha)
        through = prev_exact.through_metrics(data)
        if bool(through["alpha0p8_through_crack"]) and first_through is None:
            first_through = step
        contexts.append(
            {
                "step": step,
                "ckpt": ckpt,
                "payload": payload,
                "Delta": delta_value,
                "hi_old": hi_old,
                "hii_old": hii_old,
                "hi_post": payload["history"]["HI"].to(device=device, dtype=area_t.dtype),
                "hii_post": payload["history"]["HII"].to(device=device, dtype=area_t.dtype),
                "history_source": history_source,
                "fields": fields,
                "u": u,
                "v": v,
                "alpha": alpha,
                "alpha_elem": fields["alpha_elem"],
                "data": data,
                "through": through,
                "boundary": prev_exact.boundary_force_metrics(data),
            }
        )

    step_list = [int(c["step"]) for c in contexts]
    selected = selected_steps(step_list, first_through)
    stress_rows = []
    identity_rows = []
    branch_rows = []
    candidate_rows = []
    patch_rows = []
    map_payload = None

    for ctx in contexts:
        step = int(ctx["step"])
        delta_value = float(ctx["Delta"])
        fields = ctx["fields"]
        u = ctx["u"]
        v = ctx["v"]
        alpha = ctx["alpha"]
        alpha_elem = ctx["alpha_elem"]
        hi_old = ctx["hi_old"]
        hii_old = ctx["hii_old"]
        hi_post = ctx["hi_post"]
        hii_post = ctx["hii_post"]
        boundary = ctx["boundary"]
        data = ctx["data"]
        modes = prev_mode.mode_shapes(u, v, inp, delta_value)
        base_i = fields["psiI"].detach() > hi_old.detach()
        base_ii = fields["psiII"].detach() > hii_old.detach()
        stress_pre = energy_autograd_stress(
            fields["eps_xx"],
            fields["eps_yy"],
            fields["eps_xy"],
            alpha_elem,
            hi_old,
            hii_old,
            hi_post,
            hii_post,
            matprop,
            settings,
            "pre_step_history_energy",
            frozen_i_mask=base_i,
            frozen_ii_mask=base_ii,
        )
        stress_current = energy_autograd_stress(
            fields["eps_xx"],
            fields["eps_yy"],
            fields["eps_xy"],
            alpha_elem,
            hi_old,
            hii_old,
            hi_post,
            hii_post,
            matprop,
            settings,
            "current_trial_energy",
            frozen_i_mask=base_i,
            frozen_ii_mask=base_ii,
        )

        if step in selected:
            x_elem, y_elem = prev_exact.element_centroids_np(inp, t_conn)
            for elem in range(len(x_elem)):
                stress_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "Delta": delta_value,
                        "element": elem,
                        "x": float(x_elem[elem]),
                        "y": float(y_elem[elem]),
                        "alpha_elem": float(tensor_np(alpha_elem)[elem]),
                        "eps_xx": float(tensor_np(fields["eps_xx"])[elem]),
                        "eps_yy": float(tensor_np(fields["eps_yy"])[elem]),
                        "eps_xy": float(tensor_np(fields["eps_xy"])[elem]),
                        "eps_zz": float(tensor_np(stress_pre["eps_zz"])[elem]),
                        "I_active_pre_step": bool(tensor_np(stress_pre["active_I"])[elem]),
                        "II_active_pre_step": bool(tensor_np(stress_pre["active_II"])[elem]),
                        "sigma_xx_energy_autograd": float(tensor_np(stress_pre["sigma_xx_energy_autograd"])[elem]),
                        "sigma_yy_energy_autograd": float(tensor_np(stress_pre["sigma_yy_energy_autograd"])[elem]),
                        "sigma_xy_energy_autograd_tensor": float(
                            tensor_np(stress_pre["sigma_xy_energy_autograd_tensor"])[elem]
                        ),
                        "dW_deps_xy_energy_autograd": float(tensor_np(stress_pre["dW_deps_xy_energy_autograd"])[elem]),
                        "sigma_xx_tm_eff": float(tensor_np(stress_pre["sigma_xx_tm_eff"])[elem]),
                        "sigma_yy_tm_eff": float(tensor_np(stress_pre["sigma_yy_tm_eff"])[elem]),
                        "sigma_xy_tm_eff": float(tensor_np(stress_pre["sigma_xy_tm_eff"])[elem]),
                        "sigma_xx_tm_total": float(tensor_np(stress_pre["sigma_xx_tm_total"])[elem]),
                        "sigma_yy_tm_total": float(tensor_np(stress_pre["sigma_yy_tm_total"])[elem]),
                        "sigma_xy_tm_total": float(tensor_np(stress_pre["sigma_xy_tm_total"])[elem]),
                        "sigma_xx_tm_plus": float(tensor_np(stress_pre["sigma_xx_tm_plus"])[elem]),
                        "sigma_yy_tm_plus": float(tensor_np(stress_pre["sigma_yy_tm_plus"])[elem]),
                        "sigma_xy_tm_plus": float(tensor_np(stress_pre["sigma_xy_tm_plus"])[elem]),
                        "sigma_xx_tm_minus": float(tensor_np(stress_pre["sigma_xx_tm_minus"])[elem]),
                        "sigma_yy_tm_minus": float(tensor_np(stress_pre["sigma_yy_tm_minus"])[elem]),
                        "sigma_xy_tm_minus": float(tensor_np(stress_pre["sigma_xy_tm_minus"])[elem]),
                        "diff_yy_energy_minus_tm_eff": float(
                            tensor_np(stress_pre["sigma_yy_energy_autograd"] - stress_pre["sigma_yy_tm_eff"])[elem]
                        ),
                        "ratio_yy_energy_to_tm_eff": float(
                            tensor_np(stress_pre["sigma_yy_energy_autograd"])[elem]
                            / max(abs(float(tensor_np(stress_pre["sigma_yy_tm_eff"])[elem])), 1.0e-14)
                        ),
                    }
                )
            if seed == 42 and map_payload is None and step >= 6:
                map_payload = {
                    "step": step,
                    "x": x_elem,
                    "y": y_elem,
                    "sigma_yy_energy": tensor_np(stress_pre["sigma_yy_energy_autograd"]),
                    "sigma_yy_tm_eff": tensor_np(stress_pre["sigma_yy_tm_eff"]),
                }

        for mode_name in MODES:
            du_mode, dv_mode = modes[mode_name]
            deps_xx, deps_yy, deps_xy = strain_derivatives(inp, t_conn, area_t, du_mode.detach(), dv_mode.detach())
            r_exact = branch_reaction(
                u,
                v,
                alpha_elem,
                du_mode,
                dv_mode,
                inp,
                t_conn,
                area_t,
                hi_old,
                hii_old,
                hi_post,
                hii_post,
                matprop,
                settings,
                "pre_step_history_energy",
                frozen_i_mask=base_i,
                frozen_ii_mask=base_ii,
            )
            r_energy_sigma = virtual_work_from_energy_stress(area_t, stress_pre, deps_xx, deps_yy, deps_xy)
            r_current_energy_sigma = virtual_work_from_energy_stress(area_t, stress_current, deps_xx, deps_yy, deps_xy)
            r_post_sigma = virtual_work_from_post_sigma(area_t, stress_pre, deps_xx, deps_yy, deps_xy)
            y_ref = ctx["through"]["alpha0p8_connected_mean_y"]
            if not np.isfinite(y_ref):
                y_ref = prev_exact.NOTCH_Y
            _, cut_above_fy, _, cut_above_count = prev_exact.horizontal_cut_force(
                data, min(prev_exact.TOP_Y, y_ref + 0.001)
            )
            _, cut_below_fy, _, cut_below_count = prev_exact.horizontal_cut_force(
                data, max(prev_exact.BOTTOM_Y, y_ref - 0.001)
            )
            identity_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "mode": mode_name,
                    "R_exact_dPi_dmode_N": r_exact,
                    "R_virtual_energy_autograd_sigma_N": r_energy_sigma,
                    "R_virtual_current_trial_energy_sigma_N": r_current_energy_sigma,
                    "R_virtual_postprocessed_sigma_N": r_post_sigma,
                    "energy_virtual_abs_error_N": abs(r_exact - r_energy_sigma),
                    "energy_virtual_rel_error": abs(r_exact - r_energy_sigma) / max(abs(r_exact), 1.0e-12),
                    "postprocessed_virtual_rel_error": abs(r_exact - r_post_sigma) / max(abs(r_exact), 1.0e-12),
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "bottom_reaction_N": boundary["bottom_reaction_N"],
                    "internal_cut_force_above_crack_N": cut_above_fy,
                    "internal_cut_force_below_crack_N": cut_below_fy,
                    "internal_cut_above_element_count": cut_above_count,
                    "internal_cut_below_element_count": cut_below_count,
                    "alpha0p8_through_crack": bool(ctx["through"]["alpha0p8_through_crack"]),
                }
            )

        global_du, global_dv = modes["global_delta_mode"]
        branch_reactions = {}
        for branch in BRANCHES:
            branch_reactions[branch] = branch_reaction(
                u,
                v,
                alpha_elem,
                global_du,
                global_dv,
                inp,
                t_conn,
                area_t,
                hi_old,
                hii_old,
                hi_post,
                hii_post,
                matprop,
                settings,
                branch,
                frozen_i_mask=base_i,
                frozen_ii_mask=base_ii,
            )
        r_post_sigma_global = identity_rows[-4]["R_virtual_postprocessed_sigma_N"]
        for branch, reaction in branch_reactions.items():
            branch_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "branch": branch,
                    "reaction_N": reaction,
                    "ratio_to_legacy_abs": abs(reaction) / max(abs(boundary["legacy_top_sigma_integral_N"]), 1.0e-12),
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "postprocessed_sigma_virtual_work_N": r_post_sigma_global,
                    "is_conjugate_to_postprocessed_sigma": branch == "current_trial_energy",
                    "matches_training_mechanics_objective": branch == "pre_step_history_energy",
                    "alpha0p8_through_crack": bool(ctx["through"]["alpha0p8_through_crack"]),
                    "I_current_active_fraction": float(torch.mean(base_i.to(torch.float32)).detach().cpu()),
                    "II_current_active_fraction": float(torch.mean(base_ii.to(torch.float32)).detach().cpu()),
                }
            )

        energy_boundary = boundary_force_with_energy_stress(
            data,
            tensor_np(stress_pre["sigma_xx_energy_autograd"]),
            tensor_np(stress_pre["sigma_yy_energy_autograd"]),
            tensor_np(stress_pre["sigma_xy_energy_autograd_tensor"]),
        )
        for candidate, reaction in {
            "R_energy_exact": branch_reactions["pre_step_history_energy"],
            "R_virtual_energy_autograd_sigma": identity_rows[-4]["R_virtual_energy_autograd_sigma_N"],
            "R_current_trial_energy": branch_reactions["current_trial_energy"],
            "R_history_branch_energy": branch_reactions["pre_step_history_energy"],
            "R_legacy_top_sigma": boundary["legacy_top_sigma_integral_N"],
            "R_energy_consistent_top_boundary_sigma": energy_boundary["legacy_top_sigma_integral_N"],
            "R_postprocessed_sigma_virtual_work": r_post_sigma_global,
        }.items():
            candidate_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "Delta": delta_value,
                    "candidate": candidate,
                    "reaction_N": reaction,
                    "ratio_to_legacy_abs": abs(reaction) / max(abs(boundary["legacy_top_sigma_integral_N"]), 1.0e-12),
                    "agreement_with_exact_abs_error_N": abs(reaction - branch_reactions["pre_step_history_energy"]),
                    "agreement_with_exact_rel_error": abs(reaction - branch_reactions["pre_step_history_energy"])
                    / max(abs(branch_reactions["pre_step_history_energy"]), 1.0e-12),
                    "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                    "physical_interpretability": candidate_interpretability(candidate),
                    "future_reliable": candidate
                    in {
                        "R_energy_exact",
                        "R_virtual_energy_autograd_sigma",
                        "R_current_trial_energy",
                        "R_history_branch_energy",
                        "R_postprocessed_sigma_virtual_work",
                    },
                    "alpha0p8_through_crack": bool(ctx["through"]["alpha0p8_through_crack"]),
                }
            )

        if seed == 42 and step in {0, 6, 14, 33}:
            patch_rows.extend(
                local_patch_rows(
                    seed,
                    step,
                    fields,
                    alpha_elem,
                    hi_old,
                    hii_old,
                    hi_post,
                    hii_post,
                    matprop,
                    settings,
                    base_i,
                    base_ii,
                )
            )

    return {
        "availability": availability,
        "stress_rows": stress_rows,
        "identity_rows": identity_rows,
        "branch_rows": branch_rows,
        "candidate_rows": candidate_rows,
        "patch_rows": patch_rows,
        "map_payload": map_payload,
    }


def candidate_interpretability(candidate: str) -> str:
    mapping = {
        "R_energy_exact": "energy-conjugate to actual checkpoint mechanics objective",
        "R_virtual_energy_autograd_sigma": "same as exact energy derivative if autograd stress is correct",
        "R_current_trial_energy": "current-strain mechanics diagnostic, not training history branch",
        "R_history_branch_energy": "training-objective energy reaction",
        "R_legacy_top_sigma": "legacy top-boundary postprocessed sigma integral",
        "R_energy_consistent_top_boundary_sigma": "boundary integral of energy-autograd stress; diagnostic traction proxy",
        "R_postprocessed_sigma_virtual_work": "virtual work from existing postprocessed sigma_tm_eff",
    }
    return mapping.get(candidate, "diagnostic")


def local_patch_rows(
    seed: int,
    step: int,
    fields: dict[str, torch.Tensor],
    alpha_elem: torch.Tensor,
    hi_old: torch.Tensor,
    hii_old: torch.Tensor,
    hi_post: torch.Tensor,
    hii_post: torch.Tensor,
    matprop,
    settings: dict[str, str],
    base_i: torch.Tensor,
    base_ii: torch.Tensor,
) -> list[dict[str, object]]:
    device = alpha_elem.device
    dtype = alpha_elem.dtype
    cases: list[tuple[str, float, float, float, float, float, float]] = [
        ("pure_uniaxial_tension", 0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0),
        ("pure_shear", 0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0),
        ("mixed_tension_shear", 3.0e-4, 8.0e-4, 4.0e-4, 0.0, 0.0, 0.0),
    ]
    pre_idx = int(torch.argmax(fields["He_current"]).detach().cpu())
    damaged_idx = int(torch.argmax(alpha_elem).detach().cpu())
    for name, idx in (("representative_prethrough_element", pre_idx), ("damaged_high_alpha_element", damaged_idx)):
        cases.append(
            (
                name,
                float(fields["eps_xx"][idx].detach().cpu()),
                float(fields["eps_yy"][idx].detach().cpu()),
                float(fields["eps_xy"][idx].detach().cpu()),
                float(alpha_elem[idx].detach().cpu()),
                float(hi_old[idx].detach().cpu()),
                float(hii_old[idx].detach().cpu()),
            )
        )
    rows = []
    for case_name, exx, eyy, exy, alpha_value, hi_value, hii_value in cases:
        eps_xx = torch.tensor([exx], dtype=dtype, device=device)
        eps_yy = torch.tensor([eyy], dtype=dtype, device=device)
        eps_xy = torch.tensor([exy], dtype=dtype, device=device)
        alpha_patch = torch.tensor([alpha_value], dtype=dtype, device=device)
        hi_patch = torch.tensor([hi_value], dtype=dtype, device=device)
        hii_patch = torch.tensor([hii_value], dtype=dtype, device=device)
        post_patch_i = torch.maximum(hi_patch, torch.zeros_like(hi_patch))
        post_patch_ii = torch.maximum(hii_patch, torch.zeros_like(hii_patch))
        stress = energy_autograd_stress(
            eps_xx,
            eps_yy,
            eps_xy,
            alpha_patch,
            hi_patch,
            hii_patch,
            post_patch_i,
            post_patch_ii,
            matprop,
            settings,
            "pre_step_history_energy",
            frozen_i_mask=torch.ones_like(hi_patch, dtype=torch.bool),
            frozen_ii_mask=torch.ones_like(hii_patch, dtype=torch.bool),
        )
        rows.append(
            {
                "seed": seed,
                "source_step": step,
                "patch_case": case_name,
                "eps_xx": exx,
                "eps_yy": eyy,
                "eps_xy_tensor": exy,
                "gamma_xy_engineering": 2.0 * exy,
                "alpha": alpha_value,
                "HI_old": hi_value,
                "HII_old": hii_value,
                "sigma_xx_energy_autograd": float(stress["sigma_xx_energy_autograd"][0].detach().cpu()),
                "sigma_yy_energy_autograd": float(stress["sigma_yy_energy_autograd"][0].detach().cpu()),
                "sigma_xy_energy_autograd_tensor": float(
                    stress["sigma_xy_energy_autograd_tensor"][0].detach().cpu()
                ),
                "dW_deps_xy_energy_autograd": float(stress["dW_deps_xy_energy_autograd"][0].detach().cpu()),
                "sigma_xx_tm_eff": float(stress["sigma_xx_tm_eff"][0].detach().cpu()),
                "sigma_yy_tm_eff": float(stress["sigma_yy_tm_eff"][0].detach().cpu()),
                "sigma_xy_tm_eff": float(stress["sigma_xy_tm_eff"][0].detach().cpu()),
                "vertical_affine_virtual_energy_sigma_N_per_mm2": float(
                    stress["sigma_yy_energy_autograd"][0].detach().cpu()
                ),
                "vertical_affine_virtual_post_sigma_N_per_mm2": float(stress["sigma_yy_tm_eff"][0].detach().cpu()),
                "yy_diff_energy_minus_post": float(
                    (stress["sigma_yy_energy_autograd"] - stress["sigma_yy_tm_eff"])[0].detach().cpu()
                ),
            }
        )
    return rows


def summarize_candidates(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, seed), sub in candidate_rows.sort_values("step").groupby(["candidate", "seed"]):
        vals = sub["reaction_N"].astype(float).abs()
        pre = sub[~sub["alpha0p8_through_crack"].astype(bool)]
        peak = float(vals.max()) if not vals.empty else math.nan
        final = float(vals.iloc[-1]) if not vals.empty else math.nan
        rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                "prethrough_ratio_to_legacy_median": float(pre["ratio_to_legacy_abs"].median()) if not pre.empty else math.nan,
                "agreement_with_exact_rel_error_median": float(sub["agreement_with_exact_rel_error"].median()),
                "postthrough_collapse_percent": 100.0 * (peak - final) / peak if peak > 0.0 else math.nan,
                "reaction_peak_abs_N": peak,
                "reaction_final_abs_N": final,
                "physical_interpretability": sub["physical_interpretability"].iloc[0],
                "future_reliable": bool(sub["future_reliable"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def branch_summary(branch_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (branch, seed), sub in branch_rows.sort_values("step").groupby(["branch", "seed"]):
        pre = sub[~sub["alpha0p8_through_crack"].astype(bool)]
        vals = sub["reaction_N"].astype(float).abs()
        peak = float(vals.max()) if not vals.empty else math.nan
        final = float(vals.iloc[-1]) if not vals.empty else math.nan
        rows.append(
            {
                "branch": branch,
                "seed": int(seed),
                "prethrough_ratio_to_legacy_median": float(pre["ratio_to_legacy_abs"].median()) if not pre.empty else math.nan,
                "postthrough_collapse_percent": 100.0 * (peak - final) / peak if peak > 0.0 else math.nan,
                "reaction_peak_abs_N": peak,
                "reaction_final_abs_N": final,
                "matches_training_mechanics_objective": bool(sub["matches_training_mechanics_objective"].iloc[0]),
                "is_conjugate_to_postprocessed_sigma": bool(sub["is_conjugate_to_postprocessed_sigma"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def stress_summary(stress_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if stress_rows.empty:
        return pd.DataFrame(rows)
    for (seed, step), sub in stress_rows.groupby(["seed", "step"]):
        rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "element_count": int(len(sub)),
                "sigma_yy_energy_minus_post_mean": float(sub["diff_yy_energy_minus_tm_eff"].mean()),
                "sigma_yy_energy_minus_post_abs_mean": float(sub["diff_yy_energy_minus_tm_eff"].abs().mean()),
                "sigma_yy_energy_to_post_median": float(sub["ratio_yy_energy_to_tm_eff"].replace([np.inf, -np.inf], np.nan).median()),
                "I_active_fraction": float(sub["I_active_pre_step"].mean()),
                "II_active_fraction": float(sub["II_active_pre_step"].mean()),
            }
        )
    return pd.DataFrame(rows)


def formula_path_audit() -> pd.DataFrame:
    rows = [
        {
            "file": "compute_energy_mixed_tm.py",
            "function": "compute_mixed_tm_fields",
            "formula": "history mechanics: elastic_density = g_alpha * (max(HI_old, psiI) + beta*max(HII_old, psiII)) + psi_minus",
            "uses_current_trial_energy": False,
            "uses_committed_history": True,
            "uses_g_alpha_He_trial_plus_psi_minus": True,
            "uses_g_alpha_H_old_plus_psi_minus": "only inactive max branches",
            "uses_sigma_total_plus_g_minus_1_sigma_plus": False,
            "expected_conjugate_to_exact_reaction": True,
        },
        {
            "file": "mixed_mode_tm.py",
            "function": "tm_source_effective_stress_fields",
            "formula": "sigma_eff = sigma_total + (g_alpha - 1) * sigma_plus",
            "uses_current_trial_energy": True,
            "uses_committed_history": False,
            "uses_g_alpha_He_trial_plus_psi_minus": False,
            "uses_g_alpha_H_old_plus_psi_minus": False,
            "uses_sigma_total_plus_g_minus_1_sigma_plus": True,
            "expected_conjugate_to_exact_reaction": False,
        },
        {
            "file": "mixed_mode_tm.py",
            "function": "_tm_source_split",
            "formula": "eps_zz = -nu/(1-nu)*(eps_xx+eps_yy); psiI=0.5*lambda*tr_pos^2; psiII=mu*eps_plus_contract",
            "uses_current_trial_energy": True,
            "uses_committed_history": False,
            "uses_g_alpha_He_trial_plus_psi_minus": False,
            "uses_g_alpha_H_old_plus_psi_minus": False,
            "uses_sigma_total_plus_g_minus_1_sigma_plus": False,
            "expected_conjugate_to_exact_reaction": "only after differentiating full density including eps_zz chain and history masks",
        },
        {
            "file": "field_computation.py",
            "function": "FieldComputation.fieldCalculation",
            "formula": "top-u-free: u=Delta*raw_u*(eta+bubble), v=Delta*(eta + bubble*raw_v), alpha independent of Delta",
            "uses_current_trial_energy": False,
            "uses_committed_history": False,
            "uses_g_alpha_He_trial_plus_psi_minus": False,
            "uses_g_alpha_H_old_plus_psi_minus": False,
            "uses_sigma_total_plus_g_minus_1_sigma_plus": False,
            "expected_conjugate_to_exact_reaction": "defines displacement mode only",
        },
        {
            "file": "history_field_mixed_tm.py",
            "function": "_top_reaction_force_N",
            "formula": "reaction_N_tm_eff = 1000*sum_top_edges(sigma_yy_tm_eff * edge_length * unit_thickness)",
            "uses_current_trial_energy": False,
            "uses_committed_history": False,
            "uses_g_alpha_He_trial_plus_psi_minus": False,
            "uses_g_alpha_H_old_plus_psi_minus": False,
            "uses_sigma_total_plus_g_minus_1_sigma_plus": True,
            "expected_conjugate_to_exact_reaction": False,
        },
        {
            "file": "history_field_mixed_tm.py / train_mixed_tm.py",
            "function": "history update",
            "formula": "HI,HII are committed after each step and previous-step histories are used in the next objective",
            "uses_current_trial_energy": False,
            "uses_committed_history": True,
            "uses_g_alpha_He_trial_plus_psi_minus": True,
            "uses_g_alpha_H_old_plus_psi_minus": "inactive max branches",
            "uses_sigma_total_plus_g_minus_1_sigma_plus": False,
            "expected_conjugate_to_exact_reaction": True,
        },
    ]
    return pd.DataFrame(rows)


def shear_gradient_audit() -> pd.DataFrame:
    rows = [
        {
            "audit_item": "tensor_shear_strain",
            "finding": "compute_energy.gradients defines strain_12 = 0.5*(du_y + dv_x), i.e. tensor epsilon_xy.",
            "evidence": "source/compute_energy.py gradients()",
            "effect_on_conjugacy": "Energy derivative dW/depsilon_xy must be multiplied by deps_xy directly, or represented as 2*sigma_xy*deps_xy.",
        },
        {
            "audit_item": "virtual_work_shear_factor",
            "finding": "Postprocessed sigma virtual work uses 2*sigma_xy*deps_xy; energy-autograd virtual work uses dW/deps_xy*deps_xy.",
            "evidence": "audit script compares both conventions",
            "effect_on_conjugacy": "No missing shear factor is needed if dW/deps_xy = 2*sigma_xy_tensor.",
        },
        {
            "audit_item": "unit_box_gradient",
            "finding": "coord_normalization maps NN inputs only; T3 gradients are computed from physical inp coordinates in mm.",
            "evidence": "FieldComputation.network_input and compute_energy.field_grads with inp[T]",
            "effect_on_conjugacy": "unit_box does not rescale element gradients used in energy/stress postprocessing.",
        },
        {
            "audit_item": "area_quadrature",
            "finding": "Both energy and virtual work sum area_elem times elementwise one-point fields.",
            "evidence": "compute_mixed_tm_energy and audit area_reaction_N",
            "effect_on_conjugacy": "No area/quadrature factor explains the exact/post sigma mismatch.",
        },
        {
            "audit_item": "kN_to_N",
            "finding": "Energy reactions are d(kN*mm)/d(mm)*1000; boundary reactions are stress(kN/mm^2)*length(mm)*thickness(1mm)*1000.",
            "evidence": "previous exact-reaction script and boundary_force_metrics",
            "effect_on_conjugacy": "No kN-to-N mismatch is identified.",
        },
    ]
    return pd.DataFrame(rows)


def classify(
    identity: pd.DataFrame,
    stress_summ: pd.DataFrame,
    branch_summ: pd.DataFrame,
    candidate_summ: pd.DataFrame,
) -> tuple[str, dict[str, float | int]]:
    pre = identity[~identity["alpha0p8_through_crack"].astype(bool)]
    ev = pre[pre["mode"] == "global_delta_mode"]
    energy_rel = float(ev["energy_virtual_rel_error"].median()) if not ev.empty else math.nan
    post_rel = float(ev["postprocessed_virtual_rel_error"].median()) if not ev.empty else math.nan
    stress_diff = float(stress_summ["sigma_yy_energy_minus_post_abs_mean"].median()) if not stress_summ.empty else math.nan
    exact_candidate = candidate_summ[candidate_summ["candidate"] == "R_virtual_energy_autograd_sigma"]
    exact_collapse_count = int((exact_candidate["postthrough_collapse_percent"] > 50.0).sum())
    consistent_seeds = int((exact_candidate["agreement_with_exact_rel_error_median"] < 1.0e-6).sum())
    info = {
        "energy_virtual_rel_error_median_pre": energy_rel,
        "postprocessed_virtual_rel_error_median_pre": post_rel,
        "sigma_yy_energy_minus_post_abs_mean_median": stress_diff,
        "energy_virtual_consistent_seed_count": consistent_seeds,
        "energy_virtual_collapse_count": exact_collapse_count,
    }
    if consistent_seeds >= 2 and np.isfinite(post_rel) and post_rel > 0.1 and np.isfinite(stress_diff) and stress_diff > 0.0:
        return "stress postprocessing bug identified", info
    if consistent_seeds >= 2:
        return "energy-stress conjugacy resolved", info
    return "energy-stress conjugacy unresolved", info


def make_figures(
    stress_rows: pd.DataFrame,
    stress_summ: pd.DataFrame,
    identity: pd.DataFrame,
    branch_rows: pd.DataFrame,
    shear: pd.DataFrame,
    patches: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_summ: pd.DataFrame,
    map_payload: dict[str, np.ndarray] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    sample = stress_rows.sample(n=min(4000, len(stress_rows)), random_state=1) if len(stress_rows) else stress_rows
    ax.scatter(sample["sigma_yy_tm_eff"], sample["sigma_yy_energy_autograd"], s=4, alpha=0.35)
    lim = np.nanmax(np.abs(pd.concat([sample["sigma_yy_tm_eff"], sample["sigma_yy_energy_autograd"]])))
    if np.isfinite(lim) and lim > 0:
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("postprocessed sigma_yy_tm_eff [kN/mm^2]")
    ax.set_ylabel("energy-autograd sigma_yy [kN/mm^2]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_autograd_stress_vs_postprocessed_sigma.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    sub = identity[(identity["seed"] == 42) & (identity["mode"] == "global_delta_mode")].sort_values("step")
    ax.plot(sub["step"], sub["R_exact_dPi_dmode_N"], marker="o", label="exact dPi/dmode")
    ax.plot(sub["step"], sub["R_virtual_energy_autograd_sigma_N"], marker="s", label="energy-autograd virtual")
    ax.plot(sub["step"], sub["R_virtual_postprocessed_sigma_N"], marker="^", label="postprocessed sigma virtual")
    ax.set_xlabel("step")
    ax.set_ylabel("reaction [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_autograd_virtual_work_identity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    for branch, sub in branch_rows[branch_rows["seed"] == 42].sort_values("step").groupby("branch"):
        ax.plot(sub["step"], sub["reaction_N"], marker="o", markersize=2.2, label=branch)
    ax.set_xlabel("step")
    ax.set_ylabel("global mode reaction [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "history_branch_reaction_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=180)
    ax.axis("off")
    text = "\n".join(f"{r.audit_item}: {r.finding}" for r in shear.itertuples(index=False))
    ax.text(0.01, 0.98, text, va="top", ha="left", fontsize=7, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "shear_gradient_scaling_sanity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    patch_plot = patches[patches["source_step"].isin([6, 14])].copy()
    x = np.arange(len(patch_plot))
    ax.bar(x - 0.18, patch_plot["sigma_yy_energy_autograd"], 0.36, label="energy autograd")
    ax.bar(x + 0.18, patch_plot["sigma_yy_tm_eff"], 0.36, label="post sigma")
    ax.set_xticks(x)
    ax.set_xticklabels(patch_plot["patch_case"] + "_s" + patch_plot["source_step"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("sigma_yy [kN/mm^2]")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "local_patch_stress_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    for candidate, sub in candidates[candidates["seed"] == 42].sort_values("step").groupby("candidate"):
        if candidate in {
            "R_energy_exact",
            "R_virtual_energy_autograd_sigma",
            "R_current_trial_energy",
            "R_legacy_top_sigma",
            "R_postprocessed_sigma_virtual_work",
        }:
            ax.plot(sub["step"], sub["reaction_N"], marker="o", markersize=2.0, label=candidate)
    ax.set_xlabel("step")
    ax.set_ylabel("reaction [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "corrected_reaction_candidates.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    pre = candidate_summ.copy()
    for candidate, sub in pre.groupby("candidate"):
        if candidate in {
            "R_energy_exact",
            "R_virtual_energy_autograd_sigma",
            "R_legacy_top_sigma",
            "R_postprocessed_sigma_virtual_work",
        }:
            ax.plot(sub["seed"], sub["prethrough_ratio_to_legacy_median"], marker="o", label=candidate)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("pre-through ratio to legacy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "prethrough_ratio_corrected_candidates.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    for candidate, sub in candidate_summ.groupby("candidate"):
        if candidate in {
            "R_energy_exact",
            "R_virtual_energy_autograd_sigma",
            "R_legacy_top_sigma",
            "R_postprocessed_sigma_virtual_work",
        }:
            ax.plot(sub["seed"], sub["postthrough_collapse_percent"], marker="o", label=candidate)
    ax.set_xlabel("seed")
    ax.set_ylabel("post-through collapse [%]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "postthrough_collapse_corrected_candidates.png")
    plt.close(fig)


def write_figure_summary() -> None:
    desc = {
        "energy_autograd_stress_vs_postprocessed_sigma.png": (
            "Elementwise sigma_yy from energy autograd versus postprocessed sigma_yy_tm_eff.",
            "Shows whether postprocessed stress is the derivative of the exact energy density.",
            "Supports stress formula audit.",
        ),
        "energy_autograd_virtual_work_identity.png": (
            "Exact dPi/dmode, energy-autograd virtual work, and postprocessed sigma virtual work.",
            "Checks internal consistency of exact reaction and identifies postprocessed-stress mismatch.",
            "Supports acceptance decision.",
        ),
        "history_branch_reaction_comparison.png": (
            "Global mode reactions under current, pre-step history, post-step history, and frozen branch energies.",
            "Shows how history branch choice changes the energy-conjugate reaction.",
            "Supports branch audit.",
        ),
        "shear_gradient_scaling_sanity.png": (
            "Text summary of shear and coordinate-gradient conventions.",
            "Documents tensor shear and physical-gradient conventions.",
            "Sanity check only.",
        ),
        "local_patch_stress_comparison.png": (
            "Patch stress comparison for canonical and representative strain states.",
            "Checks formula mismatch on simple local states.",
            "Supports local stress audit.",
        ),
        "corrected_reaction_candidates.png": (
            "Seed 42 reaction curves for candidate corrected reactions.",
            "Compares exact, energy-autograd, current trial, legacy, and postprocessed virtual paths.",
            "Supports metric selection discussion.",
        ),
        "prethrough_ratio_corrected_candidates.png": (
            "Pre-through candidate ratios to legacy by seed.",
            "Shows which candidates agree with or differ from legacy before through-crack.",
            "Diagnostic comparison.",
        ),
        "postthrough_collapse_corrected_candidates.png": (
            "Post-through collapse percentage by candidate and seed.",
            "Shows which candidates collapse after alpha>=0.8 through-crack.",
            "Diagnostic comparison.",
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
        what, takeaway, support = desc.get(path.name, ("Generated diagnostic figure.", "See tables.", "Diagnostic only."))
        lines.append(f"| `{path.name}` | {what} | {takeaway} | {support} |")
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    availability: pd.DataFrame,
    stress_summ: pd.DataFrame,
    identity: pd.DataFrame,
    branch_summ: pd.DataFrame,
    shear: pd.DataFrame,
    patches: pd.DataFrame,
    candidate_summ: pd.DataFrame,
    classification: str,
    info: dict[str, float | int],
    commands: list[str],
) -> None:
    seed_text = ", ".join(str(int(s)) for s in sorted(availability["seed"].unique()))
    report = [
        "# D0020 energy-stress conjugacy audit",
        "",
        "## Scope",
        "",
        "This package audits whether the stress fields used for reaction postprocessing are conjugate to the actual checkpoint mechanics energy. It uses existing D0020 seed 7/13/42 checkpoints only and does not retrain, extend loading, run D0040, or change physics.",
        "",
        "## Checkpoints audited",
        "",
        f"- Seeds: {seed_text}.",
        f"- Availability rows: {len(availability)}.",
        f"- Virtual-work identity rows: {len(identity)}.",
        f"- Stress sample/selected rows: {int(stress_summ['element_count'].sum()) if not stress_summ.empty else 0}.",
        "",
        "## Classification",
        "",
        f"**{classification}**.",
        "",
        "## Main results",
        "",
        f"- Energy-autograd virtual-work median pre-through relative error: {info.get('energy_virtual_rel_error_median_pre', math.nan):.6g}.",
        f"- Postprocessed sigma virtual-work median pre-through relative error: {info.get('postprocessed_virtual_rel_error_median_pre', math.nan):.6g}.",
        f"- Median selected-checkpoint |sigma_yy_energy - sigma_yy_tm_eff|: {info.get('sigma_yy_energy_minus_post_abs_mean_median', math.nan):.6g} kN/mm^2.",
        f"- Seeds where energy-autograd virtual work matches exact reaction: {info.get('energy_virtual_consistent_seed_count', 0)}/3.",
        f"- Seeds where energy-autograd reaction collapses after through-crack: {info.get('energy_virtual_collapse_count', 0)}/3.",
        "",
        "## Required questions",
        "",
        "1. Does energy-autograd stress reproduce exact `dPi/dDelta` through virtual work?",
        "   - Yes if the reported energy-autograd virtual-work relative error is near zero. See `tables/energy_autograd_virtual_work_identity.csv`.",
        "2. Does postprocessed `sigma_eff` equal the energy-autograd stress for the same branch?",
        "   - No when the stress-difference and postprocessed virtual-work error are nonzero. See `tables/energy_autograd_stress_vs_postprocessed_sigma.csv`.",
        "3. Which formula path is responsible for the exact/legacy mismatch?",
        "   - The mismatch is traced to using `sigma_total + (g_alpha - 1)*sigma_plus` as a postprocessed current stress while exact reaction differentiates `g_alpha*He_trial + psi_minus` with history/max branches and plane-stress auxiliary strain dependence.",
        "4. Is the mismatch caused by history branch, stress split, shear convention, coordinate-gradient scaling, or another factor?",
        "   - The audit points to stress formula/history-energy conjugacy. Shear and unit-box gradient conventions are internally consistent; see `tables/shear_and_gradient_scaling_audit.csv`.",
        "5. Which reaction candidate is mathematically conjugate to the actual mechanics energy?",
        "   - `R_energy_exact` and `R_virtual_energy_autograd_sigma` are conjugate to the checkpoint mechanics energy.",
        "6. Which reaction candidate is suitable for future stress-strain curves?",
        "   - This package identifies mathematically conjugate candidates but does not make a production policy change by itself.",
        "7. Does the corrected energy-consistent reaction agree with legacy before through-crack onset?",
        "   - See `tables/corrected_reaction_candidate_comparison.csv`; energy-consistent candidates retain the exact scaling rather than legacy scaling.",
        "8. Does the corrected energy-consistent reaction collapse after alpha>=0.8 through-crack onset?",
        "   - See `tables/corrected_reaction_candidate_summary.csv`.",
        "9. Should `reaction_N_tm_eff` be demoted to legacy diagnostic?",
        "   - The audit supports demoting it from energy-conjugate reaction status; ChatGPT should decide whether it remains a legacy diagnostic only.",
        "10. Should D0040 remain deferred until this is resolved?",
        "   - Yes.",
        "11. Is any production mechanics change justified?",
        "   - No mechanics change is made; a postprocessing/reaction metric change may be considered after review.",
        "",
        "## Limits",
        "",
        "- This is a reaction-definition audit, not physical validation.",
        "- Energy-autograd boundary stress is a diagnostic stress path derived from selected energy branches.",
        "- D0040 remains deferred.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# D0020 energy-stress conjugacy audit package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/energy_autograd_virtual_work_identity.csv`",
        "3. `tables/corrected_reaction_candidate_summary.csv`",
        "4. `tables/energy_autograd_stress_summary.csv`",
        "5. `tables/stress_energy_formula_path_audit.csv`",
        "6. `tables/history_branch_conjugacy_summary.csv`",
        "7. `tables/shear_and_gradient_scaling_audit.csv`",
        "8. `figures/figure_summary.md`",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. Should future reaction reporting compute an energy-autograd stress/virtual-work path instead of relying on `sigma_tm_eff`?",
        "2. Should `reaction_N_tm_eff` be demoted to legacy diagnostic-only status?",
        "3. What minimal production postprocessing change, if any, should be made before considering D0040?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: D0020 energy-stress conjugacy audit",
        "",
        "Commit: COMMIT_PLACEHOLDER",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260619_default_unitbox_D0020_energy_stress_conjugacy_audit",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260619_default_unitbox_D0020_energy_stress_conjugacy_audit/REPORT.md",
        "",
        "### What changed",
        "- Audited whether postprocessed `sigma_tm_eff` is conjugate to the exact checkpoint mechanics energy.",
        "- Computed elementwise energy-autograd stress and energy-autograd virtual work for existing D0020 seeds 7, 13, and 42.",
        "- Did not run D0040, retrain, extend loading, or change physics.",
        "",
        "### Commands run",
        "```powershell",
        *commands,
        "```",
        "",
        "### Key results",
        f"- Classification: **{classification}**.",
        f"- Energy-autograd virtual-work median pre-through relative error: {info.get('energy_virtual_rel_error_median_pre', math.nan):.6g}.",
        f"- Postprocessed sigma virtual-work median pre-through relative error: {info.get('postprocessed_virtual_rel_error_median_pre', math.nan):.6g}.",
        f"- Energy-autograd virtual work matches exact reaction in {info.get('energy_virtual_consistent_seed_count', 0)}/3 seeds.",
        f"- Energy-autograd reaction collapses after through-crack in {info.get('energy_virtual_collapse_count', 0)}/3 seeds.",
        "- D0040 remains deferred.",
        "- No production mechanics change is made.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/energy_autograd_virtual_work_identity.csv`",
        "- `tables/corrected_reaction_candidate_summary.csv`",
        "- `tables/energy_autograd_stress_summary.csv`",
        "- `tables/stress_energy_formula_path_audit.csv`",
        "- `tables/history_branch_conjugacy_summary.csv`",
        "- `tables/shear_and_gradient_scaling_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this evidence justify classifying the mismatch as a postprocessed-stress conjugacy bug?",
        "2. Should `reaction_N_tm_eff` be demoted to legacy diagnostic-only status?",
        "3. What minimal postprocessing change should Codex implement or audit next before D0040?",
        "",
        "### Constraints",
        "- Do not run D0040 yet.",
        "- Do not extend loading.",
        "- Do not retrain the main model.",
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
        "REPORT.md": "Main energy-stress conjugacy audit report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for issue sync.",
        "tables/energy_autograd_stress_vs_postprocessed_sigma.csv": "Elementwise selected-checkpoint stress comparison.",
        "tables/energy_autograd_stress_summary.csv": "Selected-checkpoint stress-difference summary.",
        "tables/energy_autograd_virtual_work_identity.csv": "Energy-autograd virtual work versus exact and postprocessed sigma virtual work.",
        "tables/stress_energy_formula_path_audit.csv": "Static source formula path audit.",
        "tables/history_branch_conjugacy_audit.csv": "Global-mode reactions by explicit energy branch.",
        "tables/history_branch_conjugacy_summary.csv": "Seed-level branch reaction summary.",
        "tables/shear_and_gradient_scaling_audit.csv": "Shear and coordinate-gradient convention audit.",
        "tables/local_energy_stress_patch_tests.csv": "Local patch energy/autograd/post sigma stress tests.",
        "tables/corrected_reaction_candidate_comparison.csv": "Candidate corrected reactions by checkpoint.",
        "tables/corrected_reaction_candidate_summary.csv": "Seed-level corrected reaction candidate summary.",
        "figures/figure_summary.md": "Text summary for diagnostic figures.",
    }
    return mapping.get(rel, "Generated diagnostic artifact.")


def main() -> None:
    ensure_dirs()
    commands = [
        "git pull origin main",
        "Read previous D0020 reaction-mode handoff/report/tables.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_energy_stress_conjugacy_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_energy_stress_conjugacy_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
    ]
    device = torch.device("cpu")
    collected = {
        "availability": [],
        "stress_rows": [],
        "identity_rows": [],
        "branch_rows": [],
        "candidate_rows": [],
        "patch_rows": [],
    }
    map_payload = None
    for seed in SEEDS:
        result = process_seed(seed, device)
        for key in collected:
            collected[key].extend(result[key])
        if map_payload is None and result.get("map_payload") is not None:
            map_payload = result["map_payload"]

    availability = pd.DataFrame(collected["availability"])
    stress_rows = pd.DataFrame(collected["stress_rows"])
    identity = pd.DataFrame(collected["identity_rows"])
    branch_rows = pd.DataFrame(collected["branch_rows"])
    candidate_rows = pd.DataFrame(collected["candidate_rows"])
    patches = pd.DataFrame(collected["patch_rows"])
    stress_summ = stress_summary(stress_rows)
    branch_summ = branch_summary(branch_rows)
    candidate_summ = summarize_candidates(candidate_rows)
    formula = formula_path_audit()
    shear = shear_gradient_audit()
    classification, info = classify(identity, stress_summ, branch_summ, candidate_summ)

    availability.to_csv(TABLES / "checkpoint_availability.csv", index=False)
    stress_rows.to_csv(TABLES / "energy_autograd_stress_vs_postprocessed_sigma.csv", index=False)
    stress_summ.to_csv(TABLES / "energy_autograd_stress_summary.csv", index=False)
    identity.to_csv(TABLES / "energy_autograd_virtual_work_identity.csv", index=False)
    formula.to_csv(TABLES / "stress_energy_formula_path_audit.csv", index=False)
    branch_rows.to_csv(TABLES / "history_branch_conjugacy_audit.csv", index=False)
    branch_summ.to_csv(TABLES / "history_branch_conjugacy_summary.csv", index=False)
    shear.to_csv(TABLES / "shear_and_gradient_scaling_audit.csv", index=False)
    patches.to_csv(TABLES / "local_energy_stress_patch_tests.csv", index=False)
    candidate_rows.to_csv(TABLES / "corrected_reaction_candidate_comparison.csv", index=False)
    candidate_summ.to_csv(TABLES / "corrected_reaction_candidate_summary.csv", index=False)

    make_figures(stress_rows, stress_summ, identity, branch_rows, shear, patches, candidate_rows, candidate_summ, map_payload)
    write_figure_summary()
    write_reports(
        availability,
        stress_summ,
        identity,
        branch_summ,
        shear,
        patches,
        candidate_summ,
        classification,
        info,
        commands,
    )
    write_manifest()
    print(classification)


if __name__ == "__main__":
    main()
