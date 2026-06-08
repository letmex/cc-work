import csv

import numpy as np
import torch

from compute_energy_mixed_tm import TM_STRESS_FIELD_KEYS, compute_mixed_tm_fields


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
NOTCH_ZONE_HALF_HEIGHT_MM = 3.0e-4
CORNER_WINDOW_MM = 5.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
TOP_Y_MM = SPECIMEN_SIZE_MM
BOUNDARY_TOL_MM = 1.0e-9


LOCAL_DIAGNOSTIC_FIELDS = (
    ("eps_xx", "eps_xx"),
    ("eps_yy", "eps_yy"),
    ("eps_xy", "eps_xy"),
    ("psiI", "psiI"),
    ("psiII", "psiII"),
    ("psi_minus", "psi_minus"),
    ("He_current", "He_current"),
    ("He_history", "He_history"),
    ("mechanics_drive", "mechanics_drive"),
    ("phase_history_drive", "phase_history_drive"),
    ("HI", "HI"),
    ("HII", "HII"),
    ("alpha", "alpha_elem"),
    ("grad_alpha_norm", "grad_alpha_norm"),
    ("g_alpha", "g_alpha"),
    ("mechanics_current_energy_density", "mechanics_current_energy_density"),
    ("phase_history_energy_density", "phase_history_energy_density"),
    ("phase_history_total_density", "phase_history_total_density"),
    ("elastic_energy_density", "elastic_energy_density"),
    ("fracture_energy_density", "fracture_energy_density"),
    ("phase_proximal_energy_density", "phase_proximal_energy_density"),
    ("alpha_step_change", "alpha_step_change"),
)


def element_centroids(inp, T_conn):
    if T_conn is None:
        return inp[:, 0], inp[:, 1]
    x = (inp[T_conn[:, 0], 0] + inp[T_conn[:, 1], 0] + inp[T_conn[:, 2], 0]) / 3.0
    y = (inp[T_conn[:, 0], 1] + inp[T_conn[:, 1], 1] + inp[T_conn[:, 2], 1]) / 3.0
    return x, y


def element_areas(inp, T_conn):
    if T_conn is None:
        return None
    x1 = inp[T_conn[:, 0], 0]
    y1 = inp[T_conn[:, 0], 1]
    x2 = inp[T_conn[:, 1], 0]
    y2 = inp[T_conn[:, 1], 1]
    x3 = inp[T_conn[:, 2], 0]
    y3 = inp[T_conn[:, 2], 1]
    return 0.5 * torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _to_numpy(value):
    if value is None:
        return np.array([])
    return value.detach().cpu().numpy()


def _zone_masks(x_elem, y_elem):
    notch_tip_region = (
        (x_elem >= NOTCH_TIP_X_MM - TIP_HALF_WINDOW_MM)
        & (x_elem <= NOTCH_TIP_X_MM + TIP_HALF_WINDOW_MM)
        & (torch.abs(y_elem - NOTCH_CENTER_Y_MM) <= TIP_HALF_WINDOW_MM)
    )
    tip_box = notch_tip_region
    notch_zone = (
        (x_elem >= 0.0)
        & (x_elem <= NOTCH_TIP_X_MM + TIP_HALF_WINDOW_MM)
        & (torch.abs(y_elem - NOTCH_CENTER_Y_MM) <= NOTCH_ZONE_HALF_HEIGHT_MM)
    )
    bottom_right_region = (
        (x_elem >= SPECIMEN_SIZE_MM - BOTTOM_RIGHT_WINDOW_MM)
        & (x_elem <= SPECIMEN_SIZE_MM)
        & (y_elem >= 0.0)
        & (y_elem <= BOTTOM_RIGHT_WINDOW_MM)
    )
    corner_zone = (
        ((x_elem <= CORNER_WINDOW_MM) | (x_elem >= SPECIMEN_SIZE_MM - CORNER_WINDOW_MM))
        & ((y_elem <= CORNER_WINDOW_MM) | (y_elem >= SPECIMEN_SIZE_MM - CORNER_WINDOW_MM))
    )
    bulk_zone = (
        (x_elem >= NOTCH_TIP_X_MM + 1.0e-3)
        & (y_elem >= 0.0035)
        & (y_elem <= 0.0065)
        & (~corner_zone)
    )
    return {
        "notch_tip_region": notch_tip_region,
        "tip_box": tip_box,
        "notch_zone": notch_zone,
        "bottom_right_region": bottom_right_region,
        "corner_zone": corner_zone,
        "bulk_zone": bulk_zone,
    }


def _field_stats(values, mask):
    if not torch.any(mask):
        return np.nan, np.nan, None
    subset = values[mask]
    max_local = torch.argmax(subset)
    global_idx = torch.nonzero(mask, as_tuple=False).flatten()[max_local]
    return (
        float(torch.mean(subset).detach().cpu()),
        float(torch.max(subset).detach().cpu()),
        int(global_idx.detach().cpu()),
    )


def _coord_for_index(x_elem, y_elem, idx):
    if idx is None:
        return np.nan, np.nan
    return (
        float(x_elem[idx].detach().cpu()),
        float(y_elem[idx].detach().cpu()),
    )


def _add_max_location(row, prefix, values, x_elem, y_elem):
    idx, max_value = _global_max(values)
    max_x, max_y = _coord_for_index(x_elem, y_elem, idx)
    row[f"max_{prefix}"] = max_value
    row[f"max_{prefix}_x"] = max_x
    row[f"max_{prefix}_y"] = max_y


def _add_local_region_stats(row, prefix, mask, fields):
    for output_name, field_key in LOCAL_DIAGNOSTIC_FIELDS:
        if field_key not in fields:
            row[f"{prefix}_{output_name}_mean"] = np.nan
            row[f"{prefix}_{output_name}_max"] = np.nan
            continue
        mean_value, max_value, _idx = _field_stats(fields[field_key], mask)
        row[f"{prefix}_{output_name}_mean"] = mean_value
        row[f"{prefix}_{output_name}_max"] = max_value


def _energy_sum(inp, T_conn, density):
    if density is None or T_conn is None:
        return np.nan
    areas = element_areas(inp, T_conn)
    if areas is None or areas.numel() == 0:
        return np.nan
    if density.numel() != areas.numel():
        return np.nan
    return float(torch.sum(areas * density).detach().cpu())


def initialize_mixed_history_fields(area_T):
    zeros = torch.zeros_like(area_T).detach()
    return {
        "HI": zeros.clone(),
        "HII": zeros.clone(),
        "He": zeros.clone(),
    }


def commit_mixed_history_from_fields(HI_old, HII_old, psiI, psiII, ratio=1.0):
    HI = torch.maximum(HI_old.detach(), psiI.detach()).detach()
    HII = torch.maximum(HII_old.detach(), psiII.detach()).detach()
    He = (HI + float(ratio) * HII).detach()
    return {"HI": HI, "HII": HII, "He": He}


def commit_mixed_tm_history_from_model(
    field_comp,
    inp,
    history_old,
    matprop,
    pffmodel,
    area_T,
    T_conn,
    eta_residual=1.0e-8,
    gcII=None,
    gcII_factor=1.0,
    split_mode="current",
    tm_eps_r=0.0,
    mechanics_mode="history",
    phase_proximal_mode="none",
    eta_eff=0.0,
    dt=1.0,
):
    if T_conn is None:
        inp_eval = inp.detach().clone().requires_grad_(True)
        u, v, alpha = field_comp.fieldCalculation(inp_eval)
    else:
        inp_eval = inp
        with torch.no_grad():
            u, v, alpha = field_comp.fieldCalculation(inp_eval)

    fields = compute_mixed_tm_fields(
        inp_eval,
        u,
        v,
        alpha,
        history_old["HI"],
        history_old["HII"],
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=eta_residual,
        gcII=gcII,
        gcII_factor=gcII_factor,
        split_mode=split_mode,
        tm_eps_r=tm_eps_r,
        mechanics_mode=mechanics_mode,
        alpha_old=history_old.get("alpha_old"),
        phase_proximal_mode=phase_proximal_mode,
        eta_eff=eta_eff,
        dt=dt,
    )
    ratio = float(fields["mixed_mode_ratio"][0].detach().cpu()) if fields["mixed_mode_ratio"].numel() else 1.0
    history_new = commit_mixed_history_from_fields(
        history_old["HI"], history_old["HII"], fields["psiI"], fields["psiII"], ratio=ratio
    )
    fields["HI"] = history_new["HI"]
    fields["HII"] = history_new["HII"]
    fields["He"] = history_new["He"]
    fields["He_history"] = history_new["He"]
    return history_new, alpha.detach(), u.detach(), v.detach(), fields


def _global_max(values):
    idx = int(torch.argmax(values).detach().cpu())
    return idx, float(torch.max(values).detach().cpu())


def _field_or(fields, key, fallback_key):
    return fields[key] if key in fields else fields[fallback_key]


def _boundary_displacement_stats(inp, u, v, displacement, top_y=TOP_Y_MM, bottom_y=0.0, tol=BOUNDARY_TOL_MM):
    if u is None or v is None:
        return {}
    top_mask = torch.abs(inp[:, 1] - top_y) <= tol
    bottom_mask = torch.abs(inp[:, 1] - bottom_y) <= tol

    def _masked_stats(values, mask, prefix):
        if not torch.any(mask):
            return {
                f"{prefix}_min": np.nan,
                f"{prefix}_max": np.nan,
                f"{prefix}_mean": np.nan,
                f"{prefix}_abs_max": np.nan,
            }
        subset = values[mask]
        return {
            f"{prefix}_min": float(torch.min(subset).detach().cpu()),
            f"{prefix}_max": float(torch.max(subset).detach().cpu()),
            f"{prefix}_mean": float(torch.mean(subset).detach().cpu()),
            f"{prefix}_abs_max": float(torch.max(torch.abs(subset)).detach().cpu()),
        }

    stats = {}
    stats.update(_masked_stats(u, top_mask, "top_u"))
    stats.update(_masked_stats(v, top_mask, "top_v"))
    stats.update(_masked_stats(u, bottom_mask, "bottom_u"))
    stats.update(_masked_stats(v, bottom_mask, "bottom_v"))
    target_top_v = torch.as_tensor(float(displacement), dtype=v.dtype, device=v.device)
    if torch.any(top_mask):
        stats["top_v_error_max"] = float(torch.max(torch.abs(v[top_mask] - target_top_v)).detach().cpu())
    else:
        stats["top_v_error_max"] = np.nan
    stats["bottom_u_abs_max"] = stats.get("bottom_u_abs_max", np.nan)
    stats["bottom_v_abs_max"] = stats.get("bottom_v_abs_max", np.nan)
    stats["top_u_abs_max"] = stats.get("top_u_abs_max", np.nan)
    return stats


def _top_reaction_force_N(inp, T_conn, sigma_yy, top_y=TOP_Y_MM, tol=BOUNDARY_TOL_MM):
    if T_conn is None:
        return np.nan
    reaction_kN = torch.zeros((), device=sigma_yy.device, dtype=sigma_yy.dtype)
    pairs = ((0, 1), (1, 2), (2, 0))
    for elem_id in range(T_conn.shape[0]):
        nodes = T_conn[elem_id]
        for a_local, b_local in pairs:
            a = nodes[a_local]
            b = nodes[b_local]
            ya = inp[a, 1]
            yb = inp[b, 1]
            if torch.abs(ya - top_y) <= tol and torch.abs(yb - top_y) <= tol:
                edge_length = torch.linalg.norm(inp[a] - inp[b])
                reaction_kN = reaction_kN + sigma_yy[elem_id] * edge_length
    return float((1000.0 * reaction_kN).detach().cpu())


def append_mixed_tm_summary(
    summary_path,
    step,
    displacement,
    inp,
    T_conn,
    fields,
    reaction_N=np.nan,
    u=None,
    v=None,
    top_u_mode="fixed",
    coord_mapping=None,
):
    x_elem, y_elem = element_centroids(inp, T_conn)
    masks = _zone_masks(x_elem, y_elem)
    he_current = _field_or(fields, "He_current", "He")
    he_history = _field_or(fields, "He_history", "He")
    mechanics_drive = _field_or(fields, "mechanics_drive", "He")
    alpha_elem = fields["alpha_elem"]
    alpha_idx, alpha_max = _global_max(fields["alpha_elem"])
    he_idx, he_max = _global_max(fields["He"])
    max_alpha_x, max_alpha_y = _coord_for_index(x_elem, y_elem, alpha_idx)
    max_He_x, max_He_y = _coord_for_index(x_elem, y_elem, he_idx)
    elastic_energy = _energy_sum(inp, T_conn, fields.get("elastic_energy_density"))
    fracture_energy = _energy_sum(inp, T_conn, fields.get("fracture_energy_density"))
    mechanics_current_energy = _energy_sum(inp, T_conn, fields.get("mechanics_current_energy_density"))
    phase_history_elastic_energy = _energy_sum(inp, T_conn, fields.get("phase_history_energy_density"))
    phase_history_energy = _energy_sum(inp, T_conn, fields.get("phase_history_total_density"))
    phase_proximal_energy = _energy_sum(inp, T_conn, fields.get("phase_proximal_energy_density"))
    loss_total = (
        elastic_energy + fracture_energy
        if np.isfinite(elastic_energy) and np.isfinite(fracture_energy)
        else np.nan
    )
    loss_log10 = np.log10(loss_total) if np.isfinite(loss_total) and loss_total > 0.0 else np.nan
    if not np.isfinite(reaction_N):
        reaction_N = _top_reaction_force_N(inp, T_conn, fields["sigma_yy"])
    reaction_N_tm_eff = np.nan
    if "sigma_yy_tm_eff" in fields:
        reaction_N_tm_eff = _top_reaction_force_N(inp, T_conn, fields["sigma_yy_tm_eff"])

    row = {
        "step": step,
        "Delta": float(displacement),
        "top_u_mode": top_u_mode,
        "coord_normalization": (coord_mapping or {}).get("coord_normalization", "none"),
        "x_hat_min": (coord_mapping or {}).get("x_hat_min", np.nan),
        "x_hat_max": (coord_mapping or {}).get("x_hat_max", np.nan),
        "y_hat_min": (coord_mapping or {}).get("y_hat_min", np.nan),
        "y_hat_max": (coord_mapping or {}).get("y_hat_max", np.nan),
        "t3_gradients_use_physical_xy": (coord_mapping or {}).get("t3_gradients_use_physical_xy", True),
        "loss_total": loss_total,
        "loss_log10": loss_log10,
        "elastic_energy": elastic_energy,
        "fracture_energy": fracture_energy,
        "mechanics_current_energy": mechanics_current_energy,
        "phase_history_elastic_energy": phase_history_elastic_energy,
        "phase_history_energy": phase_history_energy,
        "phase_proximal_energy": phase_proximal_energy,
        "alpha_step_change_mean": float(fields["alpha_step_change_mean"].detach().cpu()),
        "alpha_step_change_max": float(fields["alpha_step_change_max"].detach().cpu()),
        "alpha_min": float(torch.min(alpha_elem).detach().cpu()),
        "alpha_max": alpha_max,
        "alpha_mean": float(torch.mean(alpha_elem).detach().cpu()),
        "n_alpha_lt_0": int(torch.sum(alpha_elem < 0.0).detach().cpu()),
        "n_alpha_gt_1": int(torch.sum(alpha_elem > 1.0).detach().cpu()),
        "HI_max": float(torch.max(fields["HI"]).detach().cpu()),
        "HI_mean": float(torch.mean(fields["HI"]).detach().cpu()),
        "HII_max": float(torch.max(fields["HII"]).detach().cpu()),
        "HII_mean": float(torch.mean(fields["HII"]).detach().cpu()),
        "He_max": he_max,
        "He_mean": float(torch.mean(fields["He"]).detach().cpu()),
        "He_current_max": float(torch.max(he_current).detach().cpu()),
        "He_current_mean": float(torch.mean(he_current).detach().cpu()),
        "He_history_max": float(torch.max(he_history).detach().cpu()),
        "He_history_mean": float(torch.mean(he_history).detach().cpu()),
        "mechanics_drive_max": float(torch.max(mechanics_drive).detach().cpu()),
        "mechanics_drive_mean": float(torch.mean(mechanics_drive).detach().cpu()),
        "psiI_max": float(torch.max(fields["psiI"]).detach().cpu()),
        "psiII_max": float(torch.max(fields["psiII"]).detach().cpu()),
        "psi_minus_max": float(torch.max(fields["psi_minus"]).detach().cpu()),
        "psi_residual_raw_min": float(torch.min(fields["psi_residual_raw"]).detach().cpu()),
        "psi_residual_raw_negative_count": int(torch.sum(fields["psi_residual_raw"] < 0.0).detach().cpu()),
        "max_alpha_x": max_alpha_x,
        "max_alpha_y": max_alpha_y,
        "max_He_x": max_He_x,
        "max_He_y": max_He_y,
        "reaction_N": float(reaction_N),
        "reaction_N_tm_eff": float(reaction_N_tm_eff),
    }
    row.update(_boundary_displacement_stats(inp, u, v, displacement))
    _add_max_location(row, "He_current", he_current, x_elem, y_elem)
    _add_max_location(row, "He_history", he_history, x_elem, y_elem)
    _add_max_location(row, "mechanics_drive", mechanics_drive, x_elem, y_elem)
    for zone_name in ["tip_box", "bulk_zone", "corner_zone"]:
        mask = masks[zone_name]
        _alpha_mean, zone_alpha_max, _ = _field_stats(fields["alpha_elem"], mask)
        _he_mean, zone_he_max, _ = _field_stats(fields["He"], mask)
        prefix = "bulk" if zone_name == "bulk_zone" else ("corner" if zone_name == "corner_zone" else "tip_box")
        row[f"{prefix}_alpha_max"] = zone_alpha_max
        row[f"{prefix}_He_max"] = zone_he_max
    _add_local_region_stats(row, "notch_tip", masks["notch_tip_region"], fields)
    _add_local_region_stats(row, "bottom_right", masks["bottom_right_region"], fields)

    columns = [
        "step",
        "Delta",
        "top_u_mode",
        "coord_normalization",
        "x_hat_min",
        "x_hat_max",
        "y_hat_min",
        "y_hat_max",
        "t3_gradients_use_physical_xy",
        "top_u_min",
        "top_u_max",
        "top_u_mean",
        "top_u_abs_max",
        "top_v_min",
        "top_v_max",
        "top_v_mean",
        "top_v_abs_max",
        "top_v_error_max",
        "bottom_u_min",
        "bottom_u_max",
        "bottom_u_mean",
        "bottom_u_abs_max",
        "bottom_v_min",
        "bottom_v_max",
        "bottom_v_mean",
        "bottom_v_abs_max",
        "loss_total",
        "loss_log10",
        "elastic_energy",
        "fracture_energy",
        "mechanics_current_energy",
        "phase_history_elastic_energy",
        "phase_history_energy",
        "phase_proximal_energy",
        "alpha_step_change_mean",
        "alpha_step_change_max",
        "alpha_min",
        "alpha_max",
        "alpha_mean",
        "n_alpha_lt_0",
        "n_alpha_gt_1",
        "HI_max",
        "HI_mean",
        "HII_max",
        "HII_mean",
        "He_max",
        "He_mean",
        "He_current_max",
        "He_current_mean",
        "He_history_max",
        "He_history_mean",
        "mechanics_drive_max",
        "mechanics_drive_mean",
        "psiI_max",
        "psiII_max",
        "psi_minus_max",
        "psi_residual_raw_min",
        "psi_residual_raw_negative_count",
        "max_alpha_x",
        "max_alpha_y",
        "max_He_x",
        "max_He_y",
        "max_He_current",
        "max_He_current_x",
        "max_He_current_y",
        "max_He_history",
        "max_He_history_x",
        "max_He_history_y",
        "max_mechanics_drive",
        "max_mechanics_drive_x",
        "max_mechanics_drive_y",
        "tip_box_alpha_max",
        "tip_box_He_max",
        "bulk_alpha_max",
        "bulk_He_max",
        "corner_alpha_max",
        "corner_He_max",
        "reaction_N",
        "reaction_N_tm_eff",
    ]
    for prefix in ("notch_tip", "bottom_right"):
        for output_name, _field_key in LOCAL_DIAGNOSTIC_FIELDS:
            columns.append(f"{prefix}_{output_name}_mean")
            columns.append(f"{prefix}_{output_name}_max")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_mixed_tm_step_fields(results_path, step, displacement, inp, T_conn, alpha, u, v, fields):
    results_path.mkdir(parents=True, exist_ok=True)
    x_elem, y_elem = element_centroids(inp, T_conn)
    triangles = T_conn if T_conn is not None else None
    optional_fields = {
        key: _to_numpy(fields[key])
        for key in (
            "phase_history_drive",
            "mechanics_current_energy_density",
            "history_elastic_energy_density",
            "phase_history_energy_density",
            "phase_history_total_density",
            "split_gradient_elastic_energy_density",
        )
        if key in fields
    }
    np.savez_compressed(
        results_path / f"fields_mixed_tm_step_{step:04d}.npz",
        x=_to_numpy(inp[:, 0]),
        y=_to_numpy(inp[:, 1]),
        triangles=_to_numpy(triangles).astype(int) if triangles is not None else np.array([]),
        element_x=_to_numpy(x_elem),
        element_y=_to_numpy(y_elem),
        displacement_mm=float(displacement),
        alpha=_to_numpy(alpha),
        alpha_elem=_to_numpy(fields["alpha_elem"]),
        u=_to_numpy(u),
        v=_to_numpy(v),
        eps_xx=_to_numpy(fields["eps_xx"]),
        eps_yy=_to_numpy(fields["eps_yy"]),
        eps_xy=_to_numpy(fields["eps_xy"]),
        eps_zz=_to_numpy(fields["eps_zz"]),
        strain_11=_to_numpy(fields["strain_11"]),
        strain_22=_to_numpy(fields["strain_22"]),
        strain_12=_to_numpy(fields["strain_12"]),
        psiI=_to_numpy(fields["psiI"]),
        psiII=_to_numpy(fields["psiII"]),
        psi_minus=_to_numpy(fields["psi_minus"]),
        psi_residual_raw=_to_numpy(fields["psi_residual_raw"]),
        tm_eps_r=_to_numpy(fields["tm_eps_r"]),
        HI=_to_numpy(fields["HI"]),
        HII=_to_numpy(fields["HII"]),
        He=_to_numpy(fields["He"]),
        He_current=_to_numpy(fields["He_current"]),
        He_history=_to_numpy(fields["He_history"]),
        mechanics_drive=_to_numpy(fields["mechanics_drive"]),
        g_alpha=_to_numpy(fields["g_alpha"]),
        grad_alpha_norm=_to_numpy(fields["grad_alpha_norm"]),
        sigma_yy=_to_numpy(fields["sigma_yy"]),
        **{key: _to_numpy(fields[key]) for key in TM_STRESS_FIELD_KEYS},
        max_principal_strain=_to_numpy(fields["max_principal_strain"]),
        elastic_energy_density=_to_numpy(fields["elastic_energy_density"]),
        fracture_energy_density=_to_numpy(fields["fracture_energy_density"]),
        **optional_fields,
    )
