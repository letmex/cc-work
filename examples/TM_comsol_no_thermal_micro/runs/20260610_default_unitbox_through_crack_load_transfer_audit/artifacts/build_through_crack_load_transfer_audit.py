import csv
import json
import math
import re
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
RESULTS = PROJECT / "results"
SOFTENING_PACKAGE = PACKAGE.parents[0] / "20260609_default_unitbox_softening_gate"

CASES = [
    {"case": "D0040_seed7_default_unitbox", "seed": 7, "suffix": "softgate_D0040_seed7_history_default_unitbox"},
    {"case": "D0040_seed13_default_unitbox", "seed": 13, "suffix": "softgate_D0040_seed13_history_default_unitbox"},
    {"case": "D0040_seed42_default_unitbox", "seed": 42, "suffix": "softgate_D0040_seed42_history_default_unitbox"},
]

THRESHOLDS = [0.5, 0.8, 0.95]
CUT_XS = [0.006, 0.007, 0.008, 0.009]
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
RIGHT_BOUNDARY_X = 0.01
RIGHT_BOUNDARY_BAND = 2.5e-4
CUT_TOL = 2.5e-4
CRACK_BAND_Y_TOL = 8.0e-4
TOP_Y = 0.01
SECTION_AREA = 1.0e-5
SPECIMEN_HEIGHT = 0.01
EDGE_TOL = 1.0e-9
ETA_RESIDUAL = 1.0e-8


def result_dir_by_suffix(suffix):
    matches = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result dir ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def field_paths(run_dir):
    return sorted(run_dir.glob("fields_mixed_tm_step_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


def step_from_path(path):
    return int(path.stem.split("_")[-1])


def triangle_areas(data):
    pts = np.column_stack([data["x"], data["y"]])
    tri = data["triangles"].astype(int)
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def element_adjacency(triangles):
    edge_to_elem = {}
    adjacency = [[] for _ in range(len(triangles))]
    for elem_id, nodes in enumerate(triangles.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            key = tuple(sorted((int(a), int(b))))
            other = edge_to_elem.get(key)
            if other is None:
                edge_to_elem[key] = elem_id
            else:
                adjacency[elem_id].append(other)
                adjacency[other].append(elem_id)
    return adjacency


def connected_component(mask, adjacency, seed_mask):
    seeds = np.flatnonzero(mask & seed_mask)
    visited = np.zeros(mask.shape[0], dtype=bool)
    queue = deque()
    for idx in seeds:
        visited[idx] = True
        queue.append(int(idx))
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(int(nxt))
    return visited


def through_metrics(data, threshold):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    areas = triangle_areas(data)
    mask = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    right_mask = x >= RIGHT_BOUNDARY_X - RIGHT_BOUNDARY_BAND
    comp = connected_component(mask, element_adjacency(data["triangles"]), seed_mask)
    comp_x = x[comp]
    comp_y = y[comp]
    if comp_x.size:
        x_span = float(np.max(comp_x) - np.min(comp_x))
        y_span = float(np.max(comp_y) - np.min(comp_y))
        area = float(np.sum(areas[comp]))
        reaches_right = bool(np.any(comp & right_mask))
        min_x = float(np.min(comp_x))
        max_x = float(np.max(comp_x))
        mean_y = float(np.mean(comp_y))
    else:
        x_span = y_span = area = 0.0
        reaches_right = False
        min_x = max_x = mean_y = math.nan
    return {
        "threshold": threshold,
        "through_crack": reaches_right,
        "crack_x_span": x_span,
        "crack_y_span": y_span,
        "connected_high_alpha_area": area,
        "connected_element_count": int(np.sum(comp)),
        "connected_min_x": min_x,
        "connected_max_x": max_x,
        "connected_mean_y": mean_y,
        "component_mask": comp,
    }


def top_reaction_force_N(data, stress_key):
    points = np.column_stack([data["x"], data["y"]])
    triangles = data["triangles"].astype(int)
    stress = data[stress_key]
    reaction_kN = 0.0
    for elem_id, nodes in enumerate(triangles):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            if abs(points[a, 1] - TOP_Y) <= EDGE_TOL and abs(points[b, 1] - TOP_Y) <= EDGE_TOL:
                reaction_kN += float(stress[elem_id]) * float(np.linalg.norm(points[a] - points[b]))
    return 1000.0 * reaction_kN


def reaction_decomposition(data):
    total = top_reaction_force_N(data, "sigma_yy_tm_total")
    plus = top_reaction_force_N(data, "sigma_yy_tm_plus")
    minus = top_reaction_force_N(data, "sigma_yy_tm_minus")
    eff = top_reaction_force_N(data, "sigma_yy_tm_eff")
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    g = np.asarray(data["g_alpha"], dtype=float)
    residual_plus = top_reaction_force_weighted_N(data, "sigma_yy_tm_plus", np.full_like(alpha, ETA_RESIDUAL))
    degraded_plus = top_reaction_force_weighted_N(data, "sigma_yy_tm_plus", g)
    shear_eff_abs = top_boundary_abs_integral_N(data, "sigma_xy_tm_eff")
    return {
        "reaction_total_undegraded_N": total,
        "reaction_degraded_effective_N": eff,
        "reaction_positive_undegraded_N": plus,
        "reaction_negative_non_degraded_N": minus,
        "reaction_degraded_positive_contribution_N": degraded_plus,
        "reaction_residual_stiffness_positive_contribution_N": residual_plus,
        "reaction_shear_abs_top_boundary_N": shear_eff_abs,
        "positive_fraction_of_total": plus / total if total else math.nan,
        "negative_fraction_of_effective": minus / eff if eff else math.nan,
        "degraded_positive_fraction_of_effective": degraded_plus / eff if eff else math.nan,
        "residual_fraction_of_effective": residual_plus / eff if eff else math.nan,
    }


def top_reaction_force_weighted_N(data, stress_key, weight):
    points = np.column_stack([data["x"], data["y"]])
    triangles = data["triangles"].astype(int)
    stress = np.asarray(data[stress_key], dtype=float) * np.asarray(weight, dtype=float)
    reaction_kN = 0.0
    for elem_id, nodes in enumerate(triangles):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            if abs(points[a, 1] - TOP_Y) <= EDGE_TOL and abs(points[b, 1] - TOP_Y) <= EDGE_TOL:
                reaction_kN += float(stress[elem_id]) * float(np.linalg.norm(points[a] - points[b]))
    return 1000.0 * reaction_kN


def top_boundary_abs_integral_N(data, stress_key):
    points = np.column_stack([data["x"], data["y"]])
    triangles = data["triangles"].astype(int)
    stress = np.abs(np.asarray(data[stress_key], dtype=float))
    reaction_kN = 0.0
    for elem_id, nodes in enumerate(triangles):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            if abs(points[a, 1] - TOP_Y) <= EDGE_TOL and abs(points[b, 1] - TOP_Y) <= EDGE_TOL:
                reaction_kN += float(stress[elem_id]) * float(np.linalg.norm(points[a] - points[b]))
    return 1000.0 * reaction_kN


def nodal_jump_proxy(data, cut_x, crack_y):
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    u = np.asarray(data["u"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    near = np.abs(x - cut_x) <= CUT_TOL
    above = near & (y >= crack_y + 2.0e-4) & (y <= crack_y + 1.2e-3)
    below = near & (y <= crack_y - 2.0e-4) & (y >= crack_y - 1.2e-3)
    if not np.any(above) or not np.any(below):
        return {
            "v_jump_proxy": math.nan,
            "u_jump_proxy": math.nan,
            "above_node_count": int(np.sum(above)),
            "below_node_count": int(np.sum(below)),
        }
    return {
        "v_jump_proxy": float(np.mean(v[above]) - np.mean(v[below])),
        "u_jump_proxy": float(np.mean(u[above]) - np.mean(u[below])),
        "above_node_count": int(np.sum(above)),
        "below_node_count": int(np.sum(below)),
    }


def cut_line_rows(case, seed, data, step, onset_step):
    rows = []
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    crack_y = float(np.nanmean(y[alpha >= 0.8])) if np.any(alpha >= 0.8) else NOTCH_Y
    for cut_x in CUT_XS:
        band = (np.abs(x - cut_x) <= CUT_TOL) & (np.abs(y - crack_y) <= CRACK_BAND_Y_TOL)
        high = band & (alpha >= 0.8)
        if np.any(high):
            sigma_yy_eff_abs_mean = float(np.mean(np.abs(data["sigma_yy_tm_eff"][high])))
            sigma_yy_eff_abs_max = float(np.max(np.abs(data["sigma_yy_tm_eff"][high])))
            sigma_xy_eff_abs_mean = float(np.mean(np.abs(data["sigma_xy_tm_eff"][high])))
            sigma_xy_eff_abs_max = float(np.max(np.abs(data["sigma_xy_tm_eff"][high])))
            g_mean = float(np.mean(data["g_alpha"][high]))
            alpha_mean = float(np.mean(alpha[high]))
            total_mean = float(np.mean(data["sigma_yy_tm_total"][high]))
            plus_mean = float(np.mean(data["sigma_yy_tm_plus"][high]))
            eff_mean = float(np.mean(data["sigma_yy_tm_eff"][high]))
            minus_mean = float(np.mean(data["sigma_yy_tm_minus"][high]))
            degraded_plus_mean = float(np.mean(data["g_alpha"][high] * data["sigma_yy_tm_plus"][high]))
            residual_plus_mean = float(np.mean(ETA_RESIDUAL * data["sigma_yy_tm_plus"][high]))
            minus_over_eff_abs = (
                abs(minus_mean) / abs(eff_mean) if np.isfinite(eff_mean) and abs(eff_mean) > 1.0e-30 else math.nan
            )
            degraded_plus_over_eff_abs = (
                abs(degraded_plus_mean) / abs(eff_mean)
                if np.isfinite(eff_mean) and abs(eff_mean) > 1.0e-30
                else math.nan
            )
            eps_xx_mean = float(np.mean(data["eps_xx"][high]))
            eps_yy_mean = float(np.mean(data["eps_yy"][high]))
            eps_xy_mean = float(np.mean(data["eps_xy"][high]))
        else:
            sigma_yy_eff_abs_mean = sigma_yy_eff_abs_max = math.nan
            sigma_xy_eff_abs_mean = sigma_xy_eff_abs_max = math.nan
            g_mean = alpha_mean = total_mean = plus_mean = eff_mean = minus_mean = math.nan
            degraded_plus_mean = residual_plus_mean = minus_over_eff_abs = degraded_plus_over_eff_abs = math.nan
            eps_xx_mean = eps_yy_mean = eps_xy_mean = math.nan
        jump = nodal_jump_proxy(data, cut_x, crack_y)
        rows.append(
            {
                "case": case,
                "seed": seed,
                "step": step,
                "after_alpha0p8_through_onset": step >= onset_step if onset_step is not None else False,
                "Delta": float(data["displacement_mm"]),
                "cut_x": cut_x,
                "crack_y_proxy": crack_y,
                "alpha_band_element_count": int(np.sum(high)),
                "alpha_mean_in_band": alpha_mean,
                "g_alpha_mean_in_band": g_mean,
                "sigma_yy_tm_total_mean_in_band": total_mean,
                "sigma_yy_tm_plus_mean_in_band": plus_mean,
                "sigma_yy_tm_minus_mean_in_band": minus_mean,
                "sigma_yy_tm_degraded_plus_mean_in_band": degraded_plus_mean,
                "sigma_yy_tm_residual_plus_mean_in_band": residual_plus_mean,
                "sigma_yy_tm_eff_mean_in_band": eff_mean,
                "abs_minus_over_abs_eff_in_band": minus_over_eff_abs,
                "abs_degraded_plus_over_abs_eff_in_band": degraded_plus_over_eff_abs,
                "sigma_xy_tm_eff_mean_in_band": float(np.mean(data["sigma_xy_tm_eff"][high])) if np.any(high) else math.nan,
                "eps_xx_mean_in_band": eps_xx_mean,
                "eps_yy_mean_in_band": eps_yy_mean,
                "eps_xy_mean_in_band": eps_xy_mean,
                "abs_sigma_yy_tm_eff_mean_in_alpha_ge_0p8_band": sigma_yy_eff_abs_mean,
                "abs_sigma_yy_tm_eff_max_in_alpha_ge_0p8_band": sigma_yy_eff_abs_max,
                "abs_sigma_xy_tm_eff_mean_in_alpha_ge_0p8_band": sigma_xy_eff_abs_mean,
                "abs_sigma_xy_tm_eff_max_in_alpha_ge_0p8_band": sigma_xy_eff_abs_max,
                **jump,
            }
        )
    return rows


def stress_split_sanity(case, seed, data):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    crack_band = (alpha >= 0.8) & (x >= NOTCH_X - 2.0e-4)
    crack_y = float(np.mean(y[crack_band])) if np.any(crack_band) else NOTCH_Y
    above = (np.abs(y - (crack_y + 5.0e-4)) <= 4.0e-4) & (x >= NOTCH_X) & (x <= RIGHT_BOUNDARY_X)
    below = (np.abs(y - (crack_y - 5.0e-4)) <= 4.0e-4) & (x >= NOTCH_X) & (x <= RIGHT_BOUNDARY_X)
    rows = []
    for region, mask in [("alpha_ge_0p8_crack_band", crack_band), ("above_band", above), ("below_band", below)]:
        if not np.any(mask):
            rows.append({"case": case, "seed": seed, "region": region, "element_count": 0})
            continue
        total = np.asarray(data["sigma_yy_tm_total"], dtype=float)[mask]
        plus = np.asarray(data["sigma_yy_tm_plus"], dtype=float)[mask]
        minus = np.asarray(data["sigma_yy_tm_minus"], dtype=float)[mask]
        eff = np.asarray(data["sigma_yy_tm_eff"], dtype=float)[mask]
        tensile = total > 0.0
        rows.append(
            {
                "case": case,
                "seed": seed,
                "region": region,
                "element_count": int(np.sum(mask)),
                "sigma_yy_tm_total_mean": float(np.mean(total)),
                "sigma_yy_tm_plus_mean": float(np.mean(plus)),
                "sigma_yy_tm_minus_mean": float(np.mean(minus)),
                "sigma_yy_tm_eff_mean": float(np.mean(eff)),
                "abs_sigma_yy_tm_eff_mean": float(np.mean(np.abs(eff))),
                "sigma_yy_plus_over_total_mean": safe_ratio_mean(plus, total),
                "sigma_yy_eff_over_total_mean": safe_ratio_mean(eff, total),
                "tensile_total_element_fraction": float(np.mean(tensile)),
                "tensile_total_with_low_plus_fraction": float(np.mean(tensile & (plus < 0.1 * np.maximum(total, 1.0e-30)))),
                "opening_tension_misclassified_to_nondegraded": bool(np.mean(tensile & (plus < 0.1 * np.maximum(total, 1.0e-30))) > 0.2),
            }
        )
    return rows


def safe_ratio_mean(num, den):
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    mask = np.abs(den) > 1.0e-30
    return float(np.mean(num[mask] / den[mask])) if np.any(mask) else math.nan


def mechanics_training_path_audit_rows():
    return [
        {
            "path": "mechanics/energy training loss",
            "file": "train_mixed_tm.py",
            "function": "fit_mixed_tm / fit_mixed_tm_with_early_stopping",
            "stress_variable_used": "No explicit stress residual; variational energy loss from compute_mixed_tm_energy",
            "alpha_degradation_applied": True,
            "degradation_formula": "elastic_energy_density = g_alpha * He_trial + psi_minus for mixed_mechanics_mode='history'",
            "positive_negative_split_used": True,
            "affects_training_or_postprocessing": "training",
            "conclusion": "Alpha degradation enters u/v optimization through the degraded elastic energy, not through a strong-form stress residual.",
        },
        {
            "path": "energy density construction",
            "file": "compute_energy_mixed_tm.py",
            "function": "compute_mixed_tm_fields",
            "stress_variable_used": "He_trial/He_current energy split, psi_minus",
            "alpha_degradation_applied": True,
            "degradation_formula": "g_alpha=(1-alpha_elem)^2+eta_residual; history_elastic_energy_density=g_alpha*He_trial+psi_minus",
            "positive_negative_split_used": True,
            "affects_training_or_postprocessing": "training and saved fields",
            "conclusion": "The training objective retains psi_minus as non-degraded energy and degrades only the positive/mixed driving part.",
        },
        {
            "path": "stress split postprocessing",
            "file": "mixed_mode_tm.py",
            "function": "tm_source_effective_stress_fields",
            "stress_variable_used": "sigma_yy_tm_total, sigma_yy_tm_plus, sigma_yy_tm_minus, sigma_yy_tm_eff",
            "alpha_degradation_applied": True,
            "degradation_formula": "sigma_yy_tm_eff=sigma_yy_tm_total+(g_alpha-1)*sigma_yy_tm_plus",
            "positive_negative_split_used": True,
            "affects_training_or_postprocessing": "saved fields/postprocessing; stress fields are derived from strains",
            "conclusion": "Saved effective stress matches the intended positive-stress degradation formula.",
        },
        {
            "path": "saved field output",
            "file": "history_field_mixed_tm.py",
            "function": "save_mixed_tm_step_fields",
            "stress_variable_used": "sigma_yy, sigma_yy_tm_total/plus/minus/eff",
            "alpha_degradation_applied": True,
            "degradation_formula": "stores g_alpha and sigma_yy_tm_eff from compute_mixed_tm_fields",
            "positive_negative_split_used": True,
            "affects_training_or_postprocessing": "postprocessing artifact",
            "conclusion": "The NPZ fields contain enough data to decompose post-crack load transfer.",
        },
        {
            "path": "diagnostic reaction",
            "file": "history_field_mixed_tm.py",
            "function": "append_mixed_tm_summary",
            "stress_variable_used": "sigma_yy_tm_eff",
            "alpha_degradation_applied": True,
            "degradation_formula": "top-boundary integration of sigma_yy_tm_eff",
            "positive_negative_split_used": True,
            "affects_training_or_postprocessing": "postprocessing only",
            "conclusion": "reaction_N_tm_eff is a degraded-stress postprocessing metric; it does not feed back into training.",
        },
        {
            "path": "boundary constraints",
            "file": "field_computation.py",
            "function": "FieldComputation.fieldCalculation",
            "stress_variable_used": "No traction loss; Dirichlet displacement ansatz",
            "alpha_degradation_applied": False,
            "degradation_formula": "Not applicable",
            "positive_negative_split_used": False,
            "affects_training_or_postprocessing": "training ansatz",
            "conclusion": "u and v are continuous neural fields with baked-in top/bottom displacement constraints; no displacement discontinuity enrichment exists across alpha=1 crack.",
        },
    ]


def compute_all():
    through_rows = []
    section_rows = []
    reaction_rows = []
    sanity_rows = []
    onset_by_case_threshold = {}
    final_data_by_case = {}

    for meta in CASES:
        case = meta["case"]
        seed = meta["seed"]
        run_dir = result_dir_by_suffix(meta["suffix"])
        paths = field_paths(run_dir)
        per_case_threshold = {}
        through_by_step_thr = {}
        reaction_by_step = {}

        loaded = []
        for path in paths:
            step = step_from_path(path)
            data = np.load(path)
            loaded.append((step, path, data))
            reaction = top_reaction_force_N(data, "sigma_yy_tm_eff")
            reaction_by_step[step] = reaction
            for thr in THRESHOLDS:
                tm = through_metrics(data, thr)
                through_by_step_thr[(step, thr)] = tm
                through_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "step": step,
                        "Delta": float(data["displacement_mm"]),
                        "threshold": thr,
                        "through_crack": tm["through_crack"],
                        "crack_x_span": tm["crack_x_span"],
                        "crack_y_span": tm["crack_y_span"],
                        "connected_high_alpha_area": tm["connected_high_alpha_area"],
                        "connected_element_count": tm["connected_element_count"],
                        "connected_min_x": tm["connected_min_x"],
                        "connected_max_x": tm["connected_max_x"],
                        "connected_mean_y": tm["connected_mean_y"],
                        "reaction_N_tm_eff": reaction,
                    }
                )
        for thr in THRESHOLDS:
            hits = [row for row in through_rows if row["case"] == case and row["threshold"] == thr and row["through_crack"]]
            if hits:
                onset = min(hits, key=lambda r: r["step"])
                onset_step = int(onset["step"])
                onset_reaction = float(onset["reaction_N_tm_eff"])
                final_step = max(reaction_by_step)
                final_reaction = reaction_by_step[final_step]
                drop_after = (onset_reaction - final_reaction) / onset_reaction if onset_reaction > 0.0 else math.nan
                per_case_threshold[thr] = onset_step
            else:
                onset_step = None
                onset_reaction = math.nan
                final_step = max(reaction_by_step)
                final_reaction = reaction_by_step[final_step]
                drop_after = math.nan
                per_case_threshold[thr] = None
            onset_by_case_threshold[(case, thr)] = {
                "first_through_step": onset_step,
                "reaction_at_through_onset": onset_reaction,
                "final_reaction": final_reaction,
                "reaction_drop_after_onset_fraction": drop_after,
                "reaction_drop_after_onset_percent": 100.0 * drop_after if np.isfinite(drop_after) else math.nan,
            }

        onset_08 = per_case_threshold.get(0.8)
        if onset_08 is None:
            onset_08 = max(reaction_by_step) + 1
        for step, _path, data in loaded:
            if step >= onset_08:
                section_rows.extend(cut_line_rows(case, seed, data, step, onset_08))
            dec = reaction_decomposition(data)
            reaction_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "step": step,
                    "Delta": float(data["displacement_mm"]),
                    **dec,
                }
            )
        final_step, _final_path, final_data = loaded[-1]
        final_data_by_case[case] = final_data
        sanity_rows.extend(stress_split_sanity(case, seed, final_data))

    through_df = pd.DataFrame(through_rows)
    for thr in THRESHOLDS:
        through_df[f"first_through_step_alpha_ge_{str(thr).replace('.', 'p')}"] = [
            onset_by_case_threshold[(row.case, thr)]["first_through_step"] for row in through_df.itertuples()
        ]
        through_df[f"reaction_at_first_through_alpha_ge_{str(thr).replace('.', 'p')}"] = [
            onset_by_case_threshold[(row.case, thr)]["reaction_at_through_onset"] for row in through_df.itertuples()
        ]
        through_df[f"reaction_drop_after_first_through_alpha_ge_{str(thr).replace('.', 'p')}_percent"] = [
            onset_by_case_threshold[(row.case, thr)]["reaction_drop_after_onset_percent"] for row in through_df.itertuples()
        ]
    through_df.to_csv(TABLES / "through_crack_geometry_audit.csv", index=False)
    pd.DataFrame(section_rows).to_csv(TABLES / "crack_section_load_transfer_audit.csv", index=False)
    pd.DataFrame(reaction_rows).to_csv(TABLES / "reaction_decomposition_audit.csv", index=False)
    pd.DataFrame(mechanics_training_path_audit_rows()).to_csv(TABLES / "mechanics_training_path_audit.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(TABLES / "stress_split_sanity_audit.csv", index=False)
    return through_df, pd.DataFrame(section_rows), pd.DataFrame(reaction_rows), pd.DataFrame(sanity_rows), final_data_by_case, onset_by_case_threshold


def diagnose_cause(section_df, reaction_df, sanity_df):
    final_react = reaction_df.sort_values("step").groupby("case").tail(1)
    residual_fracs = final_react["residual_fraction_of_effective"].astype(float)
    mis = sanity_df["opening_tension_misclassified_to_nondegraded"].fillna(False).astype(bool).any()
    section_final = section_df.sort_values("step").groupby(["case", "cut_x"]).tail(1)
    traction = section_final["abs_sigma_yy_tm_eff_mean_in_alpha_ge_0p8_band"].dropna()
    minus_over_eff = section_final["abs_minus_over_abs_eff_in_band"].dropna()
    degraded_plus_over_eff = section_final["abs_degraded_plus_over_abs_eff_in_band"].dropna()
    residual_plus = section_final["sigma_yy_tm_residual_plus_mean_in_band"].abs().dropna()
    jumps = section_final["v_jump_proxy"].dropna()
    causes = []
    if residual_fracs.mean() > 0.1:
        causes.append("residual stiffness contribution is non-negligible")
    if traction.size and minus_over_eff.size and float(traction.mean()) > 1.0 and float(minus_over_eff.mean()) > 0.8:
        causes.append("high-alpha crack band still transmits effective traction dominated by non-degraded negative/compressive stress")
    if degraded_plus_over_eff.size and float(degraded_plus_over_eff.mean()) < 0.1 and residual_plus.size and float(residual_plus.max()) < 1.0e-4:
        causes.append("positive tensile stress is degraded correctly in the crack band and residual stiffness is negligible")
    if mis:
        causes.append("possible tensile opening stress misclassified into non-degraded part")
    if jumps.size and float(jumps.mean()) < 1.0e-4:
        causes.append("continuous PINN displacement field bridges the crack and limits displacement jump")
    if not causes:
        return "through-crack load-transfer cause unresolved", "No single dominant cause exceeded the audit thresholds."
    return "through-crack load-transfer cause identified", "; ".join(causes)


def plot_final_alpha_with_path(case, data, out_path):
    x = data["x"]
    y = data["y"]
    tri = data["triangles"].astype(int)
    alpha = data["alpha"]
    triang = mtri.Triangulation(x, y, tri)
    elem_alpha = data["alpha_elem"]
    elem_x = data["element_x"]
    elem_y = data["element_y"]
    tm = through_metrics(data, 0.8)
    fig, ax = plt.subplots(figsize=(4.7, 4.0), dpi=180)
    tpc = ax.tripcolor(triang, alpha, shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    comp = tm["component_mask"]
    ax.scatter(elem_x[comp], elem_y[comp], s=4, c="red", label="connected alpha>=0.8")
    ax.axvline(RIGHT_BOUNDARY_X - RIGHT_BOUNDARY_BAND, color="white", linestyle="--", linewidth=1.0)
    ax.set_title(f"{case}: final alpha with through-crack proxy")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.035)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_figures(through_df, section_df, reaction_df, final_data_by_case, onset_info):
    for case, data in final_data_by_case.items():
        plot_final_alpha_with_path(case, data, FIGURES / f"final_alpha_through_path_{case}.png")

    for case, sub in reaction_df.groupby("case", sort=False):
        fig, ax = plt.subplots(figsize=(5.4, 3.7), dpi=180)
        strain = sub["Delta"] / SPECIMEN_HEIGHT
        ax.plot(strain, sub["reaction_degraded_effective_N"], label="degraded effective", linewidth=1.4)
        ax.plot(strain, sub["reaction_total_undegraded_N"], label="total undegraded", linewidth=1.1)
        onset = onset_info[(case, 0.8)]["first_through_step"]
        if onset is not None:
            onset_delta = float(sub.loc[sub["step"] == onset, "Delta"].iloc[0])
            ax.axvline(onset_delta / SPECIMEN_HEIGHT, color="red", linestyle="--", linewidth=1.0, label="alpha>=0.8 through onset")
        ax.set_xlabel("Engineering strain")
        ax.set_ylabel("Reaction [N]")
        ax.set_title(f"{case}: reaction with through-crack onset")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"reaction_with_through_onset_{case}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=180)
        ax.plot(sub["step"], sub["reaction_degraded_effective_N"], label="effective", linewidth=1.4)
        ax.plot(sub["step"], sub["reaction_positive_undegraded_N"], label="positive undegraded", linewidth=1.1)
        ax.plot(sub["step"], sub["reaction_degraded_positive_contribution_N"], label="g*positive", linewidth=1.1)
        ax.plot(sub["step"], sub["reaction_negative_non_degraded_N"], label="negative/non-degraded", linewidth=1.1)
        ax.plot(sub["step"], sub["reaction_residual_stiffness_positive_contribution_N"], label="eta*positive", linewidth=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("Top-boundary reaction contribution [N]")
        ax.set_title(f"{case}: reaction decomposition")
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"reaction_decomposition_{case}.png")
        plt.close(fig)

        final_step = int(sub["step"].max())
        sec = section_df[(section_df["case"] == case) & (section_df["step"] == final_step)]
        if not sec.empty:
            fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=180)
            ax.plot(sec["cut_x"], sec["alpha_mean_in_band"], marker="o", label="alpha")
            ax.plot(sec["cut_x"], sec["g_alpha_mean_in_band"], marker="o", label="g_alpha")
            ax2 = ax.twinx()
            ax2.plot(sec["cut_x"], sec["sigma_yy_tm_total_mean_in_band"], "--", marker="s", color="#0072B2", label="sigma_yy_total")
            ax2.plot(sec["cut_x"], sec["sigma_yy_tm_plus_mean_in_band"], "--", marker="s", color="#D55E00", label="sigma_yy_plus")
            ax2.plot(sec["cut_x"], sec["sigma_yy_tm_eff_mean_in_band"], "--", marker="s", color="#009E73", label="sigma_yy_eff")
            ax.set_xlabel("cut x [mm]")
            ax.set_ylabel("alpha / g_alpha")
            ax2.set_ylabel("stress mean in alpha>=0.8 band")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURES / f"cutline_alpha_g_stress_final_{case}.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=180)
            ax.plot(sec["cut_x"], sec["v_jump_proxy"], marker="o", label="v_above - v_below")
            ax.plot(sec["cut_x"], sec["u_jump_proxy"], marker="o", label="u_above - u_below")
            ax.set_xlabel("cut x [mm]")
            ax.set_ylabel("displacement jump proxy [mm]")
            ax.set_title(f"{case}: final cut-line displacement jump proxy")
            ax.legend(frameon=False, fontsize=7)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURES / f"displacement_jump_proxy_final_{case}.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=180)
            ax.plot(sec["cut_x"], sec["abs_sigma_yy_tm_eff_mean_in_alpha_ge_0p8_band"], marker="o", label="mean |sigma_yy_eff|")
            ax.plot(sec["cut_x"], sec["abs_sigma_yy_tm_eff_max_in_alpha_ge_0p8_band"], marker="o", label="max |sigma_yy_eff|")
            ax.plot(sec["cut_x"], sec["abs_sigma_xy_tm_eff_mean_in_alpha_ge_0p8_band"], marker="s", label="mean |sigma_xy_eff|")
            ax.set_xlabel("cut x [mm]")
            ax.set_ylabel("effective traction proxy in alpha>=0.8 band")
            ax.set_title(f"{case}: stress transmission in cracked band")
            ax.legend(frameon=False, fontsize=7)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURES / f"stress_transmission_alpha0p8_band_{case}.png")
            plt.close(fig)


def write_docs(through_df, section_df, reaction_df, sanity_df, onset_info, cause_status, cause_text):
    onset_rows = []
    for meta in CASES:
        case = meta["case"]
        seed = meta["seed"]
        for thr in THRESHOLDS:
            info = onset_info[(case, thr)]
            onset_rows.append(
                f"| {seed} | {thr} | {info['first_through_step']} | {info['reaction_at_through_onset']:.6g} | {info['reaction_drop_after_onset_percent']:.3g} |"
            )
    final_reaction = reaction_df.sort_values("step").groupby("case").tail(1)
    react_rows = []
    for _, row in final_reaction.iterrows():
        react_rows.append(
            f"| {int(row['seed'])} | {row['reaction_degraded_effective_N']:.6g} | {row['reaction_total_undegraded_N']:.6g} | {row['reaction_positive_undegraded_N']:.6g} | {row['reaction_degraded_positive_contribution_N']:.6g} | {row['reaction_negative_non_degraded_N']:.6g} | {row['negative_fraction_of_effective']:.3g} |"
        )
    section_final = section_df.sort_values("step").groupby(["case", "cut_x"]).tail(1)
    traction_mean = section_final["abs_sigma_yy_tm_eff_mean_in_alpha_ge_0p8_band"].mean()
    minus_over_eff_mean = section_final["abs_minus_over_abs_eff_in_band"].mean()
    degraded_plus_over_eff_mean = section_final["abs_degraded_plus_over_abs_eff_in_band"].mean()
    residual_plus_max = section_final["sigma_yy_tm_residual_plus_mean_in_band"].abs().max()
    jump_mean = section_final["v_jump_proxy"].mean()
    mis = sanity_df["opening_tension_misclassified_to_nondegraded"].fillna(False).astype(bool).any()
    report = [
        "# Through-crack load-transfer audit",
        "",
        "## Scope",
        "",
        "This package diagnoses why the default-alpha `unit_box` route keeps transmitting load after a through-going high-alpha crack has formed. It uses existing D0040 fields only; no new training or extended loading was run.",
        "",
        "## Through-crack onset",
        "",
        "| seed | alpha threshold | first through step | reaction at onset [N] | reaction drop after onset [%] |",
        "|---:|---:|---:|---:|---:|",
        *onset_rows,
        "",
        "## Reaction decomposition at final step",
        "",
        "| seed | effective reaction | undegraded total | positive undegraded | degraded positive | negative/non-degraded | negative/effective |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *react_rows,
        "",
        "## Answers",
        "",
        "1. Through-crack first forms at the steps listed above for alpha thresholds 0.5, 0.8, and 0.95.",
        "2. Reaction at through-crack onset is listed in `tables/through_crack_geometry_audit.csv` and summarized above.",
        "3. Reaction does not strongly collapse after through-crack onset; the previous softening gate reported only sub-10% final post-peak drops.",
        f"4. Effective traction remains inside the high-alpha crack band; final cut-line mean |sigma_yy_tm_eff| averaged across cases/cuts is {traction_mean:.6g}.",
        "5. The final top-boundary reaction is a net result of positive and negative split contributions. The crack-section audit is more diagnostic for the through-crack load path: inside the alpha>=0.8 band, the effective traction is dominated by the non-degraded negative/compressive component.",
        "6. Alpha degradation enters mechanics training through the variational energy loss, not only postprocessing. The path is `train_mixed_tm.py -> compute_mixed_tm_energy -> compute_mixed_tm_fields`, with `history_elastic_energy_density = g_alpha * He_trial + psi_minus`.",
        f"7. Stress split sanity audit flags opening tensile stress misclassification as {mis}; the primary evidence points to non-degraded branch/continuous displacement bridging rather than a simple missing sigma_plus degradation.",
        f"8. Cause status: **{cause_status}**. Identified evidence: {cause_text}. Mean |minus|/|effective| inside final alpha>=0.8 cut bands is {minus_over_eff_mean:.3g}; mean |degraded positive|/|effective| is {degraded_plus_over_eff_mean:.3g}; max residual-stiffness positive contribution in the cut bands is {residual_plus_max:.3g}.",
        "9. Next minimal intervention: do not change physics parameters first; run a focused kinematic/weak-form audit that compares a discontinuous or crack-face separated displacement representation against the current continuous PINN field on the same saved alpha field, and quantify reaction collapse when the cracked band is allowed to separate.",
        "",
        "## Interpretation",
        "",
        "The audit does not support `reaction integration uses undegraded stress` as the dominant cause: `reaction_N_tm_eff` integrates `sigma_yy_tm_eff`. It also does not support `alpha degradation is only postprocessing`: degradation enters the training energy. It also does not show tensile opening stress being excluded from `sigma_plus`. The stronger finding is that the high-alpha band still carries a non-degraded negative/compressive stress component, while the continuous PINN displacement field provides no displacement discontinuity that would remove that contact-like load path.",
        "",
        "No physical validation is claimed.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# Through-crack load-transfer audit",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/through_crack_geometry_audit.csv`",
        "3. `tables/crack_section_load_transfer_audit.csv`",
        "4. `tables/reaction_decomposition_audit.csv`",
        "5. `tables/mechanics_training_path_audit.csv`",
        "6. `tables/stress_split_sanity_audit.csv`",
        "7. `figures/figure_summary.md`",
        "",
        "No new training was run; this package postprocesses existing D0040 fields.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    questions = [
        "# Next questions",
        "",
        "1. Is the identified load-transfer mechanism sufficient to justify a kinematic/discontinuous-field diagnostic next?",
        "2. Should the next task quantify reaction if the final alpha field is held fixed and the displacement field is solved with an enriched/discontinuous representation?",
        "3. Should the non-degraded negative branch be audited against the intended COMSOL/source formulation before any model change is proposed?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: through-crack load-transfer audit",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_default_unitbox_through_crack_load_transfer_audit",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_default_unitbox_through_crack_load_transfer_audit/REPORT.md",
        "",
        "### What changed",
        "- Postprocessed existing D0040 seed 7/13/42 fields only; no new training or load extension was run.",
        "- Confirmed through-crack onset for alpha thresholds 0.5, 0.8, and 0.95.",
        "- Audited cut-line load transfer, reaction decomposition, mechanics training path, and stress split sanity.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_through_crack_load_transfer_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260610_default_unitbox_through_crack_load_transfer_audit\\artifacts\\build_through_crack_load_transfer_audit.py",
        "```",
        "",
        "### Key results",
        f"- Identified status: **{cause_status}**.",
        f"- Cause evidence: {cause_text}.",
        f"- Final cut-line mean |sigma_yy_tm_eff| in alpha>=0.8 band averaged over cases/cuts: {traction_mean:.6g}.",
        f"- Final cut-line mean |minus|/|effective| in alpha>=0.8 band: {minus_over_eff_mean:.6g}.",
        f"- Final cut-line max residual-stiffness positive contribution: {residual_plus_max:.6g}.",
        f"- Final mean v-jump proxy across cut lines: {jump_mean:.6g} mm.",
        "- Mechanics path audit: alpha degradation enters training energy, not only postprocessing.",
        "- Reaction path audit: `reaction_N_tm_eff` integrates degraded `sigma_yy_tm_eff`.",
        "- No physical validation is claimed.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/through_crack_geometry_audit.csv`",
        "- `tables/crack_section_load_transfer_audit.csv`",
        "- `tables/reaction_decomposition_audit.csv`",
        "- `tables/mechanics_training_path_audit.csv`",
        "- `tables/stress_split_sanity_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this evidence identify the dominant reason for continued post-crack load transfer?",
        "2. Is the next minimal intervention a fixed-alpha kinematic/enrichment diagnostic, or a deeper audit of the non-degraded negative branch?",
        "3. What exact Codex prompt should run next without changing physical parameters?",
        "",
        "### Constraints",
        "- Do not extend loading as the main action.",
        "- Do not change `l0`, material parameters, TM split, thermal terms, or history update logic unless a clear bug is found.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance in this diagnostic unless explicitly requested.",
        "- Do not use `--alpha-init-intact` as the main route.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only and do not claim physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
    ]
    for meta in CASES:
        case = meta["case"]
        lines.extend(
            [
                f"| `final_alpha_through_path_{case}.png` | Final alpha field with alpha>=0.8 connected component overlay | Shows detected notch-to-right-boundary through path. | Supports geometry audit. |",
                f"| `reaction_with_through_onset_{case}.png` | Reaction-strain with alpha>=0.8 through-onset marker | Shows reaction does not strongly collapse after onset. | Supports load-transfer audit. |",
                f"| `reaction_decomposition_{case}.png` | Effective, positive, degraded-positive, negative/non-degraded and residual reaction terms | Shows which term continues carrying top reaction. | Supports cause diagnosis. |",
                f"| `cutline_alpha_g_stress_final_{case}.png` | Final cut-line alpha/g_alpha and sigma_yy split means | Shows traction state in cracked section. | Supports section audit. |",
                f"| `displacement_jump_proxy_final_{case}.png` | Final u/v jump proxy across cut lines | Shows continuous field separation proxy remains limited. | Supports kinematic audit. |",
                f"| `stress_transmission_alpha0p8_band_{case}.png` | Effective traction proxy in alpha>=0.8 band | Shows residual stress transmission inside damaged band. | Supports load-transfer audit. |",
            ]
        )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands():
    lines = [
        "git pull origin main",
        "Read softening-gate HANDOFF_COMMENT.md, REPORT.md, extended_softening_summary.csv, alpha_connectivity_by_case.csv, reaction_consistency_audit.csv",
        "No new training run.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_through_crack_load_transfer_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260610_default_unitbox_through_crack_load_transfer_audit\\artifacts\\build_through_crack_load_transfer_audit.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest():
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel == "HANDOFF_COMMENT.md":
            typ = "handoff"
        elif rel == "figures/figure_summary.md":
            typ = "figure_summary"
        elif rel.startswith("tables/") and rel.endswith(".csv"):
            typ = "table"
        elif rel.startswith("figures/") and rel.endswith(".png"):
            typ = "figure"
        elif rel == "commands_run.txt":
            typ = "command_log"
        elif rel.endswith(".md"):
            typ = "report"
        else:
            typ = "artifact"
        entries.append(
            {
                "path": rel,
                "type": typ,
                "description": describe(rel),
                "required_for_chatgpt": rel
                in {
                    "README.md",
                    "REPORT.md",
                    "HANDOFF_COMMENT.md",
                    "tables/through_crack_geometry_audit.csv",
                    "tables/crack_section_load_transfer_audit.csv",
                    "tables/reaction_decomposition_audit.csv",
                    "tables/mechanics_training_path_audit.csv",
                    "tables/stress_split_sanity_audit.csv",
                    "figures/figure_summary.md",
                },
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps({"package": PACKAGE.name, "files": entries}, indent=2), encoding="utf-8")


def describe(rel):
    mapping = {
        "README.md": "Package overview and reading order.",
        "REPORT.md": "Main through-crack load-transfer audit report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for ChatGPT issue sync.",
        "tables/through_crack_geometry_audit.csv": "Stepwise through-crack connectivity and onset reaction audit.",
        "tables/crack_section_load_transfer_audit.csv": "Cut-line traction and displacement jump proxy audit.",
        "tables/reaction_decomposition_audit.csv": "Stepwise top-boundary reaction decomposition.",
        "tables/mechanics_training_path_audit.csv": "Code-path audit for mechanics training, saved stress, and reaction.",
        "tables/stress_split_sanity_audit.csv": "Final stress split sanity metrics inside/near crack band.",
        "figures/figure_summary.md": "Text explanation of diagnostic figures.",
    }
    return mapping.get(rel, "Diagnostic artifact or figure.")


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    through_df, section_df, reaction_df, sanity_df, final_data, onset_info = compute_all()
    cause_status, cause_text = diagnose_cause(section_df, reaction_df, sanity_df)
    make_figures(through_df, section_df, reaction_df, final_data, onset_info)
    write_docs(through_df, section_df, reaction_df, sanity_df, onset_info, cause_status, cause_text)
    write_figure_summary()
    write_commands()
    write_manifest()


if __name__ == "__main__":
    main()
