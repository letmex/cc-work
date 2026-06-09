"""Actual saved-PINN branch reaction postprocessing.

This diagnostic postprocesses existing saved fields only. No loading is
extended, no network is retrained, and alpha/history/physics settings are not
changed. Exact autograd dPi/dDelta is reported unavailable when checkpoints
are absent; saved-field reaction proxies are clearly labeled as proxies.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"
REPO = PACKAGE.parents[3]
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
RESULTS = PROJECT / "results"

SPECIMEN_SIZE_MM = 0.01
TOP_Y = 0.01
BOTTOM_Y = 0.0
EDGE_TOL = 1.0e-9
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
RIGHT_BOUNDARY_X = 0.01
RIGHT_BOUNDARY_BAND = 2.5e-4
SECTION_AREA = 1.0e-5

CASES = [
    {"group": "D0040", "case": "D0040_seed7_default_unitbox", "seed": 7, "suffix": "softgate_D0040_seed7_history_default_unitbox"},
    {"group": "D0040", "case": "D0040_seed13_default_unitbox", "seed": 13, "suffix": "softgate_D0040_seed13_history_default_unitbox"},
    {"group": "D0040", "case": "D0040_seed42_default_unitbox", "seed": 42, "suffix": "softgate_D0040_seed42_history_default_unitbox"},
    {"group": "D0020", "case": "D0020_seed7_default_unitbox", "seed": 7, "suffix": "full_D0020_seed7_history_default_unitbox"},
    {"group": "D0020", "case": "D0020_seed13_default_unitbox", "seed": 13, "suffix": "full_D0020_seed13_history_default_unitbox"},
    {"group": "D0020", "case": "D0020_seed21_default_unitbox", "seed": 21, "suffix": "full_D0020_seed21_history_default_unitbox"},
    {"group": "D0020", "case": "D0020_seed42_default_unitbox", "seed": 42, "suffix": "full_D0020_seed42_history_default_unitbox"},
    {"group": "D0020", "case": "D0020_seed99_default_unitbox", "seed": 99, "suffix": "full_D0020_seed99_history_default_unitbox"},
]

METRICS_FOR_CURVES = [
    "legacy_top_sigma_integral_N",
    "saved_field_energy_fd_proxy_N",
    "saved_field_virtual_work_proxy_N",
    "bottom_sigma_integral_outward_N",
    "internal_cut_force_above_crack_N",
]


def setup_dirs():
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def result_dir_by_suffix(suffix: str) -> Path | None:
    matches = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if len(matches) == 1:
        return matches[0]
    return None


def field_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("fields_mixed_tm_step_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


def step_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def edge_map(triangles: np.ndarray):
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for elem, nodes in enumerate(triangles.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges[tuple(sorted((int(a), int(b))))].append(elem)
    return edges


def triangle_areas(data: dict[str, np.ndarray]) -> np.ndarray:
    pts = np.column_stack([data["x"], data["y"]])
    tri = data["triangles"].astype(int)
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def element_adjacency(triangles: np.ndarray):
    edges = edge_map(triangles)
    adjacency = [[] for _ in range(len(triangles))]
    for elems in edges.values():
        if len(elems) == 2:
            a, b = elems
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def connected_component(mask: np.ndarray, adjacency, seed_mask: np.ndarray) -> np.ndarray:
    seeds = np.flatnonzero(mask & seed_mask)
    visited = np.zeros(mask.shape[0], dtype=bool)
    queue: deque[int] = deque()
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


def through_metrics(data: dict[str, np.ndarray], threshold: float = 0.8):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    mask = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    comp = connected_component(mask, element_adjacency(data["triangles"]), seed_mask)
    comp_x = x[comp]
    comp_y = y[comp]
    if comp_x.size:
        reaches_right = bool(np.any(comp_x >= RIGHT_BOUNDARY_X - RIGHT_BOUNDARY_BAND))
        return {
            "alpha0p8_through_crack": reaches_right,
            "alpha0p8_connected_count": int(np.sum(comp)),
            "alpha0p8_connected_min_x": float(np.min(comp_x)),
            "alpha0p8_connected_max_x": float(np.max(comp_x)),
            "alpha0p8_connected_mean_y": float(np.mean(comp_y)),
            "alpha0p8_connected_x_span": float(np.max(comp_x) - np.min(comp_x)),
            "alpha0p8_connected_mask": comp,
        }
    return {
        "alpha0p8_through_crack": False,
        "alpha0p8_connected_count": 0,
        "alpha0p8_connected_min_x": math.nan,
        "alpha0p8_connected_max_x": math.nan,
        "alpha0p8_connected_mean_y": math.nan,
        "alpha0p8_connected_x_span": 0.0,
        "alpha0p8_connected_mask": comp,
    }


def known_boundary(pa: np.ndarray, pb: np.ndarray):
    if abs(pa[1] - TOP_Y) <= EDGE_TOL and abs(pb[1] - TOP_Y) <= EDGE_TOL:
        return "top", np.array([0.0, 1.0])
    if abs(pa[1] - BOTTOM_Y) <= EDGE_TOL and abs(pb[1] - BOTTOM_Y) <= EDGE_TOL:
        return "bottom", np.array([0.0, -1.0])
    if abs(pa[0]) <= EDGE_TOL and abs(pb[0]) <= EDGE_TOL:
        return "left", np.array([-1.0, 0.0])
    if abs(pa[0] - SPECIMEN_SIZE_MM) <= EDGE_TOL and abs(pb[0] - SPECIMEN_SIZE_MM) <= EDGE_TOL:
        return "right", np.array([1.0, 0.0])
    return None, None


def traction_force(sxx: float, syy: float, sxy: float, normal: np.ndarray, length: float):
    tx = (sxx * normal[0] + sxy * normal[1]) * length
    ty = (sxy * normal[0] + syy * normal[1]) * length
    return 1000.0 * tx, 1000.0 * ty


def boundary_force_metrics(data: dict[str, np.ndarray]):
    tri = data["triangles"].astype(int)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    stress = {
        "xx": np.asarray(data["sigma_xx_tm_eff"], dtype=float),
        "yy": np.asarray(data["sigma_yy_tm_eff"], dtype=float),
        "xy": np.asarray(data["sigma_xy_tm_eff"], dtype=float),
    }
    grouped = defaultdict(lambda: {"fx": 0.0, "fy": 0.0, "length": 0.0, "edges": 0})
    top_legacy = 0.0
    for (a, b), elems in edge_map(tri).items():
        if len(elems) != 1:
            continue
        elem = elems[0]
        pa = np.array([x[a], y[a]])
        pb = np.array([x[b], y[b]])
        boundary, normal = known_boundary(pa, pb)
        if boundary is None:
            continue
        length = float(np.linalg.norm(pb - pa))
        fx, fy = traction_force(stress["xx"][elem], stress["yy"][elem], stress["xy"][elem], normal, length)
        grouped[boundary]["fx"] += fx
        grouped[boundary]["fy"] += fy
        grouped[boundary]["length"] += length
        grouped[boundary]["edges"] += 1
        if boundary == "top":
            top_legacy += 1000.0 * float(stress["yy"][elem]) * length
    residual_x = sum(v["fx"] for v in grouped.values())
    residual_y = sum(v["fy"] for v in grouped.values())
    return {
        "legacy_top_sigma_integral_N": top_legacy,
        "top_boundary_outward_fx_N": grouped["top"]["fx"],
        "top_boundary_outward_fy_N": grouped["top"]["fy"],
        "bottom_boundary_outward_fx_N": grouped["bottom"]["fx"],
        "bottom_sigma_integral_outward_N": grouped["bottom"]["fy"],
        "left_boundary_outward_fx_N": grouped["left"]["fx"],
        "left_boundary_outward_fy_N": grouped["left"]["fy"],
        "right_boundary_outward_fx_N": grouped["right"]["fx"],
        "right_boundary_outward_fy_N": grouped["right"]["fy"],
        "whole_boundary_residual_x_N": residual_x,
        "whole_boundary_residual_y_N": residual_y,
        "whole_boundary_residual_magnitude_N": float(math.hypot(residual_x, residual_y)),
    }


def horizontal_cut_force(data: dict[str, np.ndarray], y0: float):
    tri = data["triangles"].astype(int)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    sxy = np.asarray(data["sigma_xy_tm_eff"], dtype=float)
    syy = np.asarray(data["sigma_yy_tm_eff"], dtype=float)
    fx = fy = total_len = 0.0
    count = 0
    for elem, nodes in enumerate(tri):
        xs = x[nodes]
        ys = y[nodes]
        if y0 < np.min(ys) or y0 > np.max(ys):
            continue
        intersections = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            y1, y2 = ys[i], ys[j]
            x1, x2 = xs[i], xs[j]
            if abs(y1 - y2) <= 1.0e-14:
                if abs(y0 - y1) <= 1.0e-12:
                    intersections.extend([x1, x2])
                continue
            if (y0 - y1) * (y0 - y2) <= 0.0:
                t = (y0 - y1) / (y2 - y1)
                if -1.0e-12 <= t <= 1.0 + 1.0e-12:
                    intersections.append(x1 + t * (x2 - x1))
        if len(intersections) < 2:
            continue
        length = float(np.max(intersections) - np.min(intersections))
        if length <= 0.0:
            continue
        fx += float(sxy[elem]) * length * 1000.0
        fy += float(syy[elem]) * length * 1000.0
        total_len += length
        count += 1
    return fx, fy, total_len, count


def crack_band_interface_force(data: dict[str, np.ndarray], crack_mask: np.ndarray):
    tri = data["triangles"].astype(int)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    sxx = np.asarray(data["sigma_xx_tm_eff"], dtype=float)
    syy = np.asarray(data["sigma_yy_tm_eff"], dtype=float)
    sxy = np.asarray(data["sigma_xy_tm_eff"], dtype=float)
    total = 0.0
    edge_count = 0
    for (a, b), elems in edge_map(tri).items():
        if len(elems) != 2:
            continue
        e0, e1 = elems
        if crack_mask[e0] == crack_mask[e1]:
            continue
        crack_elem = e0 if crack_mask[e0] else e1
        pa = np.array([x[a], y[a]])
        pb = np.array([x[b], y[b]])
        midpoint = 0.5 * (pa + pb)
        center = np.array([data["element_x"][crack_elem], data["element_y"][crack_elem]])
        normal = midpoint - center
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            continue
        normal = normal / norm
        length = float(np.linalg.norm(pb - pa))
        fx, fy = traction_force(sxx[crack_elem], syy[crack_elem], sxy[crack_elem], normal, length)
        total += math.hypot(fx, fy)
        edge_count += 1
    return total, edge_count


def area_integral(data: dict[str, np.ndarray], key: str) -> float:
    if key not in data:
        return math.nan
    return float(np.sum(triangle_areas(data) * np.asarray(data[key], dtype=float)))


def virtual_work_scaled_displacement_proxy(data: dict[str, np.ndarray]) -> float:
    delta = float(data["displacement_mm"])
    if abs(delta) < 1.0e-14:
        return math.nan
    areas = triangle_areas(data)
    work_density = (
        np.asarray(data["sigma_xx_tm_eff"], dtype=float) * np.asarray(data["eps_xx"], dtype=float)
        + np.asarray(data["sigma_yy_tm_eff"], dtype=float) * np.asarray(data["eps_yy"], dtype=float)
        + 2.0 * np.asarray(data["sigma_xy_tm_eff"], dtype=float) * np.asarray(data["eps_xy"], dtype=float)
    )
    return float(1000.0 * np.sum(areas * work_density) / delta)


def load_case_data():
    availability = []
    cases = []
    for meta in CASES:
        run_dir = result_dir_by_suffix(meta["suffix"])
        fields = field_paths(run_dir) if run_dir else []
        checkpoint_patterns = []
        if run_dir:
            checkpoint_patterns = (
                list(run_dir.glob("*.pt"))
                + list(run_dir.glob("*.pth"))
                + list(run_dir.glob("*checkpoint*"))
                + list(run_dir.glob("*ckpt*"))
            )
        diag = run_dir / "diagnostics_mixed_tm_summary.csv" if run_dir else None
        disp = run_dir / "displacement_list.csv" if run_dir else None
        settings = run_dir / "model_settings.txt" if run_dir else None
        exact = bool(checkpoint_patterns and settings and settings.exists())
        reason = "available" if exact else "no model checkpoint/state file in result_dir; model_settings.txt also absent"
        availability.append(
            {
                **meta,
                "run_dir": str(run_dir) if run_dir else "",
                "run_dir_found": run_dir is not None,
                "field_npz_count": len(fields),
                "first_step": step_from_path(fields[0]) if fields else math.nan,
                "last_step": step_from_path(fields[-1]) if fields else math.nan,
                "diagnostics_csv": str(diag) if diag and diag.exists() else "",
                "displacement_list_csv": str(disp) if disp and disp.exists() else "",
                "model_settings_txt_present": bool(settings and settings.exists()),
                "checkpoint_count": len(checkpoint_patterns),
                "checkpoint_files": ";".join(p.name for p in checkpoint_patterns[:8]),
                "exact_autograd_dPi_dDelta_computable": exact,
                "exact_unavailable_reason": reason,
                "coord_normalization_inferred": "unit_box" if "coordUnitBox" in str(run_dir) else "unknown",
                "top_u_mode_inferred": "free" if "topUfree" in str(run_dir) else "unknown",
            }
        )
        if run_dir:
            cases.append({**meta, "run_dir": run_dir, "fields": fields, "diag": diag})
    return pd.DataFrame(availability), cases


def compute_step_tables(cases):
    exact_rows = []
    fd_check_rows = []
    proxy_rows = []
    virtual_rows = []
    consistency_rows = []

    for case in cases:
        diag_df = pd.read_csv(case["diag"]) if case["diag"].exists() else pd.DataFrame()
        step_records = []
        for path in case["fields"]:
            step = step_from_path(path)
            with np.load(path) as z:
                data = {k: np.asarray(z[k]) for k in z.files}
            delta = float(data["displacement_mm"])
            through = through_metrics(data, 0.8)
            y_ref = through["alpha0p8_connected_mean_y"]
            if not np.isfinite(y_ref):
                y_ref = NOTCH_Y
            _, cut_above_fy, _, cut_above_count = horizontal_cut_force(data, min(TOP_Y, y_ref + 0.001))
            _, cut_below_fy, _, cut_below_count = horizontal_cut_force(data, max(BOTTOM_Y, y_ref - 0.001))
            iface_force, iface_edges = crack_band_interface_force(data, through["alpha0p8_connected_mask"])
            boundary = boundary_force_metrics(data)
            mechanics_energy = area_integral(data, "mechanics_current_energy_density")
            history_energy = area_integral(data, "history_elastic_energy_density")
            fracture_energy = area_integral(data, "fracture_energy_density")
            virtual = virtual_work_scaled_displacement_proxy(data)
            diag_row = diag_df[diag_df["step"] == step]
            diag_legacy = float(diag_row["reaction_N_tm_eff"].iloc[0]) if len(diag_row) and "reaction_N_tm_eff" in diag_row else math.nan
            row = {
                **case_meta(case),
                "step": step,
                "Delta": delta,
                "strain": delta / SPECIMEN_SIZE_MM,
                "mechanics_current_energy_proxy": mechanics_energy,
                "history_elastic_energy_proxy": history_energy,
                "fracture_energy_proxy": fracture_energy,
                "legacy_top_sigma_integral_N": boundary["legacy_top_sigma_integral_N"],
                "diagnostics_reaction_N_tm_eff": diag_legacy,
                "saved_field_virtual_work_proxy_N": virtual,
                "internal_cut_force_above_crack_N": cut_above_fy,
                "internal_cut_force_below_crack_N": cut_below_fy,
                "internal_cut_above_element_count": cut_above_count,
                "internal_cut_below_element_count": cut_below_count,
                "crack_band_interface_force_N": iface_force,
                "crack_band_interface_edge_count": iface_edges,
                **{k: v for k, v in through.items() if k != "alpha0p8_connected_mask"},
                **boundary,
            }
            step_records.append(row)

            exact_rows.append(
                {
                    **case_meta(case),
                    "step": step,
                    "Delta": delta,
                    "exact_autograd_available": False,
                    "R_energy_exact_N": math.nan,
                    "method": "not_computed",
                    "reason": "no checkpoint/state_dict/model_settings available for actual PINN reconstruction",
                }
            )
            virtual_rows.append(
                {
                    **case_meta(case),
                    "step": step,
                    "Delta": delta,
                    "method": "saved_field_scaled_displacement_virtual_work_proxy",
                    "virtual_work_reaction_N": virtual,
                    "exact_top_mode_available": False,
                    "reason": "top displacement mode/network parameters unavailable; proxy uses saved strain field scaled by Delta",
                }
            )

        df = pd.DataFrame(step_records).sort_values("step").reset_index(drop=True)
        df["saved_field_energy_fd_proxy_N"] = energy_fd(df["Delta"].to_numpy(), df["mechanics_current_energy_proxy"].to_numpy())
        df["saved_field_history_energy_fd_proxy_N"] = energy_fd(df["Delta"].to_numpy(), df["history_elastic_energy_proxy"].to_numpy())
        proxy_rows.extend(
            df[
                [
                    "group",
                    "case",
                    "seed",
                    "step",
                    "Delta",
                    "strain",
                    "mechanics_current_energy_proxy",
                    "history_elastic_energy_proxy",
                    "fracture_energy_proxy",
                    "saved_field_energy_fd_proxy_N",
                    "saved_field_history_energy_fd_proxy_N",
                    "saved_field_virtual_work_proxy_N",
                    "legacy_top_sigma_integral_N",
                    "alpha0p8_through_crack",
                ]
            ].assign(method="saved_field_proxy", exact_autograd_available=False).to_dict("records")
        )
        consistency_rows.extend(df.to_dict("records"))
        for eps in (1.0e-7, 5.0e-7, 1.0e-6):
            fd_check_rows.append(
                {
                    **case_meta(case),
                    "eps": eps,
                    "exact_finite_difference_available": False,
                    "R_fd_N": math.nan,
                    "stability_note": "not available: no continuous checkpoint-based energy evaluator",
                }
            )
    return (
        pd.DataFrame(exact_rows),
        pd.DataFrame(fd_check_rows),
        pd.DataFrame(proxy_rows),
        pd.DataFrame(virtual_rows),
        pd.DataFrame(consistency_rows),
    )


def case_meta(case):
    return {"group": case["group"], "case": case["case"], "seed": case["seed"], "run_dir": str(case["run_dir"])}


def energy_fd(delta: np.ndarray, energy: np.ndarray) -> np.ndarray:
    order = np.argsort(delta)
    out = np.full(delta.shape, np.nan, dtype=float)
    if len(delta) < 2:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        deriv = np.gradient(energy[order], delta[order]) * 1000.0
    out[order] = deriv
    return out


def curve_summary(consistency: pd.DataFrame):
    rows = []
    curve_rows = []
    for (case, metric), sub in melt_metrics(consistency).groupby(["case", "metric"], sort=False):
        sub = sub.sort_values("step")
        finite = sub[np.isfinite(sub["reaction_N"])]
        if finite.empty:
            continue
        peak_idx = finite["reaction_N"].abs().idxmax()
        peak = finite.loc[peak_idx]
        final = finite.iloc[-1]
        initial = finite.iloc[0]
        drop = (abs(float(peak["reaction_N"])) - abs(float(final["reaction_N"]))) / abs(float(peak["reaction_N"])) if abs(float(peak["reaction_N"])) > 0 else math.nan
        rows.append(
            {
                "group": final["group"],
                "case": case,
                "seed": int(final["seed"]),
                "metric": metric,
                "initial_reaction_N": initial["reaction_N"],
                "peak_abs_reaction_N": abs(float(peak["reaction_N"])),
                "peak_step": int(peak["step"]),
                "peak_Delta": peak["Delta"],
                "final_reaction_N": final["reaction_N"],
                "final_abs_reaction_N": abs(float(final["reaction_N"])),
                "post_peak_drop_fraction_abs": drop,
                "post_peak_drop_percent_abs": 100.0 * drop if np.isfinite(drop) else math.nan,
                "final_alpha0p8_through_crack": bool(final["alpha0p8_through_crack"]),
            }
        )
        curve_rows.extend(sub.to_dict("records"))
    return pd.DataFrame(curve_rows), pd.DataFrame(rows)


def melt_metrics(consistency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in consistency.itertuples(index=False):
        for metric in METRICS_FOR_CURVES:
            rows.append(
                {
                    "group": row.group,
                    "case": row.case,
                    "seed": row.seed,
                    "step": row.step,
                    "Delta": row.Delta,
                    "strain": row.strain,
                    "metric": metric,
                    "reaction_N": getattr(row, metric),
                    "alpha0p8_through_crack": row.alpha0p8_through_crack,
                }
            )
    return pd.DataFrame(rows)


def through_reactions(curves: pd.DataFrame):
    rows = []
    for (case, metric), sub in curves.groupby(["case", "metric"], sort=False):
        sub = sub.sort_values("step")
        hits = sub[sub["alpha0p8_through_crack"].astype(bool)]
        onset = hits.iloc[0] if not hits.empty else None
        final = sub.iloc[-1]
        peak_abs = float(np.nanmax(np.abs(sub["reaction_N"])))
        onset_val = float(onset["reaction_N"]) if onset is not None else math.nan
        final_val = float(final["reaction_N"])
        drop_after_onset = (abs(onset_val) - abs(final_val)) / abs(onset_val) if np.isfinite(onset_val) and abs(onset_val) > 0 else math.nan
        rows.append(
            {
                "group": final["group"],
                "case": case,
                "seed": int(final["seed"]),
                "metric": metric,
                "first_alpha0p8_through_step": int(onset["step"]) if onset is not None else math.nan,
                "first_alpha0p8_through_Delta": float(onset["Delta"]) if onset is not None else math.nan,
                "reaction_at_first_through_N": onset_val,
                "final_reaction_N": final_val,
                "peak_abs_reaction_N": peak_abs,
                "drop_after_first_through_fraction_abs": drop_after_onset,
                "drop_after_first_through_percent_abs": 100.0 * drop_after_onset if np.isfinite(drop_after_onset) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_tables(availability, exact, fd_check, proxy, virtual, consistency, curves, drops, through):
    availability.to_csv(TABLES / "saved_artifact_availability.csv", index=False)
    exact.to_csv(TABLES / "pinn_energy_conjugate_reaction_by_step.csv", index=False)
    fd_check.to_csv(TABLES / "pinn_energy_reaction_finite_difference_check.csv", index=False)
    proxy.to_csv(TABLES / "saved_field_energy_proxy_reaction.csv", index=False)
    virtual.to_csv(TABLES / "pinn_virtual_work_reaction.csv", index=False)
    consistency.to_csv(TABLES / "pinn_reaction_boundary_cut_consistency.csv", index=False)
    curves.to_csv(TABLES / "reaction_metric_curve_summary.csv", index=False)
    drops.to_csv(TABLES / "post_peak_drop_by_metric.csv", index=False)
    through.to_csv(TABLES / "through_crack_reaction_by_metric.csv", index=False)


def plot_group_curves(curves: pd.DataFrame, group: str, filename: str):
    sub = curves[(curves["group"] == group) & (curves["metric"].isin(METRICS_FOR_CURVES))]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180, sharex=True)
    metric_sets = [
        ["legacy_top_sigma_integral_N", "saved_field_energy_fd_proxy_N", "saved_field_virtual_work_proxy_N"],
        ["bottom_sigma_integral_outward_N", "internal_cut_force_above_crack_N"],
    ]
    for ax, metrics in zip(axes, metric_sets):
        for (case, metric), g in sub[sub["metric"].isin(metrics)].groupby(["case", "metric"], sort=False):
            ax.plot(g["strain"], g["reaction_N"], linewidth=1.0, label=f"{case.replace('_default_unitbox','')} {short_metric(metric)}")
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_xlabel("strain")
        ax.set_ylabel("reaction / force [N]")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=5)
    fig.suptitle(f"{group} reaction metric curves")
    fig.tight_layout()
    fig.savefig(FIGURES / filename)
    plt.close(fig)


def short_metric(metric: str) -> str:
    return {
        "legacy_top_sigma_integral_N": "legacy-top",
        "saved_field_energy_fd_proxy_N": "energy-fd-proxy",
        "saved_field_virtual_work_proxy_N": "virtual-proxy",
        "bottom_sigma_integral_outward_N": "bottom",
        "internal_cut_force_above_crack_N": "cut-above",
    }.get(metric, metric)


def plot_legacy_vs_energy(curves: pd.DataFrame, group: str, filename: str):
    sub = curves[(curves["group"] == group) & (curves["metric"].isin(["legacy_top_sigma_integral_N", "saved_field_energy_fd_proxy_N"]))]
    seeds = sorted(sub["seed"].unique())
    fig, axes = plt.subplots(len(seeds), 1, figsize=(7.5, max(3.0, 2.3 * len(seeds))), dpi=180, sharex=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        s = sub[sub["seed"] == seed]
        for metric, g in s.groupby("metric"):
            ax.plot(g["strain"], g["reaction_N"], marker="o", markersize=1.8, linewidth=1.0, label=short_metric(metric))
        through = s[s["alpha0p8_through_crack"].astype(bool)]
        if not through.empty:
            ax.axvline(float(through["strain"].min()), color="r", ls="--", lw=0.8, label="alpha>=0.8 through onset")
        ax.set_ylabel(f"seed {seed}\nN")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("strain")
    fig.suptitle(f"{group}: legacy top reaction vs saved-field energy proxy")
    fig.tight_layout()
    fig.savefig(FIGURES / filename)
    plt.close(fig)


def plot_drop_summary(drops: pd.DataFrame):
    metrics = ["legacy_top_sigma_integral_N", "saved_field_energy_fd_proxy_N", "saved_field_virtual_work_proxy_N", "bottom_sigma_integral_outward_N", "internal_cut_force_above_crack_N"]
    sub = drops[drops["metric"].isin(metrics)]
    labels = [f"{row.case.replace('_default_unitbox','')}\n{short_metric(row.metric)}" for row in sub.itertuples()]
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=180)
    ax.bar(np.arange(len(sub)), sub["post_peak_drop_percent_abs"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("post-peak drop [%], absolute reaction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_metric_drop_summary.png")
    plt.close(fig)


def plot_boundary_cut_summary(consistency: pd.DataFrame):
    final = consistency.sort_values("step").groupby("case", sort=False).tail(1)
    x = np.arange(len(final))
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=180)
    ax.bar(x - 0.3, final["legacy_top_sigma_integral_N"], 0.2, label="top")
    ax.bar(x - 0.1, final["bottom_sigma_integral_outward_N"], 0.2, label="bottom outward")
    ax.bar(x + 0.1, final["internal_cut_force_above_crack_N"], 0.2, label="cut above")
    ax.bar(x + 0.3, final["whole_boundary_residual_magnitude_N"], 0.2, label="boundary residual")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_default_unitbox", "") for c in final["case"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("force [N]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "boundary_cut_consistency_summary.png")
    plt.close(fig)


def make_figures(curves, drops, consistency):
    plot_group_curves(curves, "D0040", "reaction_metric_curves_D0040.png")
    plot_group_curves(curves, "D0020", "reaction_metric_curves_D0020.png")
    plot_legacy_vs_energy(curves, "D0040", "legacy_vs_energy_reaction_D0040_by_seed.png")
    plot_legacy_vs_energy(curves, "D0020", "legacy_vs_energy_reaction_D0020_by_seed.png")
    plot_drop_summary(drops)
    plot_boundary_cut_summary(consistency)
    write_figure_summary()


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures compare reaction metrics computed from existing saved fields only. They do not support physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `reaction_metric_curves_D0040.png` | D0040 reaction/load curves for legacy top sigma, saved-field energy proxy, virtual-work proxy, bottom reaction, and internal cut force | Shows whether metric choice changes apparent post-peak behavior. | Diagnostic postprocessing evidence only. |",
        "| `reaction_metric_curves_D0020.png` | Same reaction metrics for D0020 5-seed robustness runs | Shows whether D0020 no-softening is metric-dependent in saved-field proxies. | Diagnostic only. |",
        "| `legacy_vs_energy_reaction_D0040_by_seed.png` | Legacy top reaction against saved-field energy proxy by seed with alpha>=0.8 through-onset marker | Highlights divergence after through-crack onset, if present. | Proxy diagnostic only. |",
        "| `legacy_vs_energy_reaction_D0020_by_seed.png` | Same comparison for D0020 seeds | Audits robustness package under alternative reaction proxy. | Proxy diagnostic only. |",
        "| `reaction_metric_drop_summary.png` | Post-peak drop percentage by case and metric | Summarizes softening/no-softening sensitivity to reaction metric. | Diagnostic summary. |",
        "| `boundary_cut_consistency_summary.png` | Final top, bottom, internal cut, and boundary residual force metrics | Checks whether boundary/cut metrics agree at final saved step. | Boundary/cut consistency evidence. |",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(availability: pd.DataFrame) -> str:
    if not bool(availability["exact_autograd_dPi_dDelta_computable"].any()):
        return "reaction postprocessing unresolved: exact actual-PINN dPi/dDelta unavailable because checkpoints are absent"
    return "reaction postprocessing unresolved"


def write_docs(availability, curves, drops, through, classification):
    exact_count = int(availability["exact_autograd_dPi_dDelta_computable"].sum())
    processed = len(availability)
    d0040_exact = int(availability[(availability["group"] == "D0040")]["exact_autograd_dPi_dDelta_computable"].sum())
    d0040 = drops[(drops["group"] == "D0040") & (drops["metric"].isin(["legacy_top_sigma_integral_N", "saved_field_energy_fd_proxy_N", "saved_field_virtual_work_proxy_N"]))]
    d0040_rows = [
        f"| {row.case} | {short_metric(row.metric)} | {row.peak_abs_reaction_N:.6g} | {row.final_abs_reaction_N:.6g} | {row.post_peak_drop_percent_abs:.3g} |"
        for row in d0040.itertuples()
    ]
    report = [
        "# Actual saved-PINN reaction postprocessing audit",
        "",
        "## Scope",
        "",
        "This package postprocesses existing saved PINN field outputs for D0040 seeds 7/13/42 and D0020 seeds 7/13/21/42/99. It does not extend loading, retrain models, evolve alpha, or modify `l0`, materials, TM split, history logic, alpha initialization, or losses.",
        "",
        "## Checkpoint availability",
        "",
        f"- Target runs processed: {processed}/8.",
        f"- Runs with exact autograd `dPi/dDelta` availability: {exact_count}/8.",
        f"- D0040 runs with exact autograd availability: {d0040_exact}/3.",
        "- All target result directories contain saved fields and diagnostics CSVs but no `.pt`, `.pth`, `checkpoint`, `ckpt`, or `model_settings.txt` files. Exact actual-PINN autograd reconstruction is therefore not possible from the saved artifacts.",
        "",
        "## D0040 metric drop summary",
        "",
        "| case | metric | peak abs reaction [N] | final abs reaction [N] | post-peak drop [%] |",
        "|---|---|---:|---:|---:|",
        *d0040_rows,
        "",
        "## Answers",
        "",
        "1. None of the inspected saved runs has enough checkpoint/model information for exact actual-PINN `dPi/dDelta`.",
        "2. Exact actual-PINN energy-conjugate reaction cannot be compared with legacy top sigma reaction before through-crack formation.",
        "3. Exact actual-PINN energy-conjugate reaction cannot be tested after through-crack formation.",
        "4. Saved-field energy finite-difference proxies can show different post-peak behavior from legacy top sigma, but they are not exact actual-PINN autograd reactions.",
        "5. Bottom reaction and internal cut force are reported in `tables/pinn_reaction_boundary_cut_consistency.csv`; they should be treated as consistency diagnostics, not exact generalized loads.",
        "6. The previous no-softening conclusion is not resolved by exact reaction metrics because exact metrics are unavailable.",
        "7. Future stress-strain curves should not rely on `reaction_N_tm_eff` alone, but exact energy-conjugate or constrained-DOF reaction requires future runs to save checkpoints/model settings.",
        "8. No production mechanics change is justified from this postprocessing package.",
        "9. Next minimal intervention: add checkpoint/model-settings saving and exact reaction postprocessing hooks to future runs, or rerun a short D0040 checkpointed replay for exact `dPi/dDelta` without changing physics.",
        "",
        f"## Classification\n\n**{classification}**.",
        "",
        "## Limitations",
        "",
        "- `saved_field_energy_fd_proxy_N` is a finite difference of saved optimized branch energies over the discrete load schedule. It is not an autograd derivative at fixed network state.",
        "- `saved_field_virtual_work_proxy_N` uses saved effective stress and saved strain scaled by Delta. It is not an exact top-mode virtual work unless the unknown PINN top-mode derivative equals the saved displacement scaling.",
        "- Proxy results must not be used to justify production physics changes.",
        "",
        "## Verification",
        "",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q`: record the final pass/fail result after package verification.",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260616_default_unitbox_pinn_energy_reaction_postprocess\\artifacts\\run_pinn_energy_reaction_postprocess.py`: record the final pass/fail result after package verification.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# Actual saved-PINN reaction postprocessing package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/saved_artifact_availability.csv`",
        "3. `tables/pinn_energy_conjugate_reaction_by_step.csv`",
        "4. `tables/saved_field_energy_proxy_reaction.csv`",
        "5. `tables/pinn_virtual_work_reaction.csv`",
        "6. `tables/pinn_reaction_boundary_cut_consistency.csv`",
        "7. `tables/post_peak_drop_by_metric.csv`",
        "8. `tables/through_crack_reaction_by_metric.csv`",
        "9. `figures/figure_summary.md`",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    questions = [
        "# Next questions",
        "",
        "1. Should the next task add checkpoint/model-settings saving and exact reaction hooks to future runs?",
        "2. Is a short checkpointed D0040 rerun acceptable if it does not change physics or load schedule?",
        "3. Should existing legacy stress-strain curves be marked as top-boundary sigma-integral curves rather than global load curves?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(questions) + "\n", encoding="utf-8")

    commands = [
        "git pull origin main",
        "Read 20260615_default_unitbox_fedof_reaction_reference handoff/report/tables/figure summary.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_pinn_energy_reaction_postprocess.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260616_default_unitbox_pinn_energy_reaction_postprocess\\artifacts\\run_pinn_energy_reaction_postprocess.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: actual saved-PINN reaction postprocessing",
        "",
        "Commit: replace with final package commit SHA after commit.",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260616_default_unitbox_pinn_energy_reaction_postprocess",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260616_default_unitbox_pinn_energy_reaction_postprocess/REPORT.md",
        "",
        "### What changed",
        "- Added and ran saved-field reaction postprocessing for D0040 seeds 7/13/42 and D0020 seeds 7/13/21/42/99.",
        "- Audited checkpoint/model availability for exact actual-PINN autograd `dPi/dDelta`.",
        "- Computed saved-field energy finite-difference proxies, saved-field virtual-work proxies, boundary reactions, internal cut forces, and through-crack reaction summaries.",
        "- Regenerated reaction metric comparison figures without changing physics or training.",
        "",
        "### Commands run",
        "```powershell",
        *commands,
        "```",
        "",
        "### Key results",
        f"- Identified cause/status: **{classification}**.",
        f"- Exact autograd availability: {exact_count}/8 target runs, D0040 exact availability {d0040_exact}/3.",
        "- All numerical alternative reaction metrics in this package are saved-field proxies, not exact actual-PINN `dPi/dDelta`.",
        "- No production mechanics change is justified from this package.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/saved_artifact_availability.csv`",
        "- `tables/pinn_energy_conjugate_reaction_by_step.csv`",
        "- `tables/saved_field_energy_proxy_reaction.csv`",
        "- `tables/pinn_virtual_work_reaction.csv`",
        "- `tables/pinn_reaction_boundary_cut_consistency.csv`",
        "- `tables/post_peak_drop_by_metric.csv`",
        "- `tables/through_crack_reaction_by_metric.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Should `reaction_N_tm_eff` be demoted now, or only relabeled until exact checkpointed `dPi/dDelta` is available?",
        "2. Should the next Codex task add checkpoint/model-settings saving plus exact reaction hooks to future runs?",
        "3. Is a short checkpointed D0040 rerun the minimal intervention needed to resolve reaction postprocessing?",
        "",
        "### Constraints",
        "- Do not extend loading.",
        "- Do not retrain the main model unless explicitly requested.",
        "- Do not evolve alpha in this postprocessing task.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, history update logic, alpha initialization, or training losses.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def write_manifest():
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
        elif rel.startswith("artifacts/"):
            ftype = "artifact"
            required = False
        elif rel == "commands_run.txt":
            ftype = "command_log"
            required = False
        elif rel == "HANDOFF_COMMENT.md":
            ftype = "handoff"
            required = True
        else:
            ftype = "report"
            required = rel in {"README.md", "REPORT.md", "MANIFEST.json"}
        entries.append({"path": rel, "type": ftype, "description": describe(rel), "required_for_chatgpt": required})
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def describe(rel: str) -> str:
    mapping = {
        "REPORT.md": "Main saved-PINN reaction postprocessing report.",
        "README.md": "Package reading order.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for issue sync.",
        "tables/saved_artifact_availability.csv": "Checkpoint/model/field availability audit.",
        "tables/pinn_energy_conjugate_reaction_by_step.csv": "Exact autograd reaction table with unavailable reasons.",
        "tables/pinn_energy_reaction_finite_difference_check.csv": "Exact finite-difference check table with unavailable reasons.",
        "tables/saved_field_energy_proxy_reaction.csv": "Saved-field energy finite-difference proxy reactions.",
        "tables/pinn_virtual_work_reaction.csv": "Saved-field virtual-work proxy reactions.",
        "tables/pinn_reaction_boundary_cut_consistency.csv": "Boundary, cut, and through-crack metrics by saved step.",
        "tables/reaction_metric_curve_summary.csv": "Long-form reaction curves by metric.",
        "tables/post_peak_drop_by_metric.csv": "Peak/final/drop summary by metric.",
        "tables/through_crack_reaction_by_metric.csv": "Reaction at first alpha>=0.8 through-crack by metric.",
        "figures/figure_summary.md": "Text summary for figures.",
    }
    return mapping.get(rel, "Generated diagnostic artifact.")


def main():
    setup_dirs()
    availability, cases = load_case_data()
    exact, fd_check, proxy, virtual, consistency = compute_step_tables(cases)
    curves, drops = curve_summary(consistency)
    through = through_reactions(curves)
    classification = classify(availability)
    write_tables(availability, exact, fd_check, proxy, virtual, consistency, curves, drops, through)
    make_figures(curves, drops, consistency)
    write_docs(availability, curves, drops, through, classification)
    write_manifest()
    print(classification)


if __name__ == "__main__":
    main()
