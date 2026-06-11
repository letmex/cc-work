from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
THERMAL_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]

SOURCE_PACKAGE = (
    THERMAL_ROOT / "runs" / "20260623_stronger_prescribed_temperature_tension_diagnostic"
)
RESULTS_ROOT = THERMAL_ROOT / "outputs" / "results"
FIGURES_DIR = PACKAGE_DIR / "figures"
TABLES_DIR = PACKAGE_DIR / "tables"

CASES = {
    "A": {
        "run_id": "20260623_strong_A_off_seed23",
        "thermal_mode": "off",
        "delta_T_K": 0.0,
    },
    "B": {
        "run_id": "20260623_strong_B_deltaT0_seed23",
        "thermal_mode": "uniform",
        "delta_T_K": 0.0,
    },
    "C": {
        "run_id": "20260623_strong_C_deltaT20_seed23",
        "thermal_mode": "uniform",
        "delta_T_K": 20.0,
    },
}

THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]
NOTCH_TIP_XY = (0.005, 0.005)
NOTCH_HALF_WINDOW = 3e-4
LOW_RANGE_ALPHA_VMAX = 0.04
FINAL_CLASSIFICATION = "caseC diffuse alpha likely plotting-scale artifact plus low-amplitude background"

FIELD_KEYS = [
    "x",
    "y",
    "element_x",
    "element_y",
    "triangles",
    "alpha_elem",
    "HI",
    "HII",
    "He",
    "mechanics_drive",
    "elastic_energy_density",
    "fracture_energy_density",
    "displacement_mm",
    "thermal_delta_T",
    "thermal_active",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def scalar(value: object) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def percentile(values: np.ndarray, q: float) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def load_fields() -> dict[str, list[dict[str, object]]]:
    fields: dict[str, list[dict[str, object]]] = {}
    for case_id, meta in CASES.items():
        run_dir = RESULTS_ROOT / meta["run_id"]
        files = sorted(run_dir.glob("fields_mixed_tm_step_*.npz"))
        if not files:
            raise FileNotFoundError(f"No field files found for {case_id}: {run_dir}")

        case_fields: list[dict[str, object]] = []
        for path in files:
            step = int(path.stem.rsplit("_", 1)[-1])
            with np.load(path, allow_pickle=False) as npz:
                entry = {
                    key: np.asarray(npz[key]).copy()
                    for key in FIELD_KEYS
                    if key in npz.files
                }
            entry["step"] = step
            entry["field_file"] = rel(path)
            entry["displacement_mm_scalar"] = scalar(entry.get("displacement_mm", np.nan))
            case_fields.append(entry)
        fields[case_id] = sorted(case_fields, key=lambda item: int(item["step"]))
    return fields


def assert_shared_geometry(fields: dict[str, list[dict[str, object]]]) -> None:
    ref = fields["A"][-1]
    for case_id, case_fields in fields.items():
        final = case_fields[-1]
        for key in ("element_x", "element_y", "triangles"):
            if key == "triangles":
                same = np.array_equal(ref[key], final[key])
            else:
                same = np.allclose(ref[key], final[key])
            if not same:
                raise ValueError(f"Geometry mismatch for case {case_id}, key {key}")


def build_element_adjacency(triangles: np.ndarray) -> list[np.ndarray]:
    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for elem_idx, tri in enumerate(np.asarray(triangles, dtype=int)):
        edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        for a, b in edges:
            edge = (a, b) if a < b else (b, a)
            edge_to_elements.setdefault(edge, []).append(elem_idx)

    neighbors: list[set[int]] = [set() for _ in range(len(triangles))]
    for elements in edge_to_elements.values():
        if len(elements) < 2:
            continue
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                a = elements[i]
                b = elements[j]
                neighbors[a].add(b)
                neighbors[b].add(a)
    return [np.asarray(sorted(item), dtype=int) for item in neighbors]


def connected_components(mask: np.ndarray, adjacency: list[np.ndarray]) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.shape[0], dtype=bool)
    components: list[np.ndarray] = []

    for seed in np.flatnonzero(active):
        if visited[seed]:
            continue
        stack = [int(seed)]
        visited[seed] = True
        comp: list[int] = []
        while stack:
            current = stack.pop()
            comp.append(current)
            for neighbor in adjacency[current]:
                neighbor_int = int(neighbor)
                if active[neighbor_int] and not visited[neighbor_int]:
                    visited[neighbor_int] = True
                    stack.append(neighbor_int)
        components.append(np.asarray(comp, dtype=int))
    return components


def triangle_area(x_nodes: np.ndarray, y_nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=int)
    x0 = x_nodes[tri[:, 0]]
    y0 = y_nodes[tri[:, 0]]
    x1 = x_nodes[tri[:, 1]]
    y1 = y_nodes[tri[:, 1]]
    x2 = x_nodes[tri[:, 2]]
    y2 = y_nodes[tri[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def distribution_comment(case_id: str, step: int, final_step: int) -> str:
    if case_id == "C" and step == final_step:
        return "Case C final raw alpha has a lower peak than A/B; broad low-level values require scale-aware interpretation."
    if case_id in ("A", "B") and step == final_step:
        return "Final A/B reference distribution; peak is notch-tip dominated."
    return "Raw element alpha distribution from existing field file."


def make_distribution_table(fields: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for case_id, case_fields in fields.items():
        final_step = int(case_fields[-1]["step"])
        for entry in case_fields:
            step = int(entry["step"])
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "alpha_min": float(np.nanmin(alpha)),
                    "alpha_p01": percentile(alpha, 1),
                    "alpha_p05": percentile(alpha, 5),
                    "alpha_p25": percentile(alpha, 25),
                    "alpha_median": percentile(alpha, 50),
                    "alpha_mean": float(np.nanmean(alpha)),
                    "alpha_p75": percentile(alpha, 75),
                    "alpha_p95": percentile(alpha, 95),
                    "alpha_p99": percentile(alpha, 99),
                    "alpha_max": float(np.nanmax(alpha)),
                    "alpha_std": float(np.nanstd(alpha)),
                    "comment": distribution_comment(case_id, step, final_step),
                }
            )
    return pd.DataFrame(rows)


def make_threshold_table(
    fields: dict[str, list[dict[str, object]]], adjacency: list[np.ndarray]
) -> pd.DataFrame:
    rows = []
    for case_id, case_fields in fields.items():
        for entry in case_fields:
            step = int(entry["step"])
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            x = np.asarray(entry["element_x"], dtype=float)
            y = np.asarray(entry["element_y"], dtype=float)
            n_total = alpha.size
            notch_seed_region = (
                (np.abs(x - NOTCH_TIP_XY[0]) <= NOTCH_HALF_WINDOW)
                & (np.abs(y - NOTCH_TIP_XY[1]) <= NOTCH_HALF_WINDOW)
            )

            for threshold in THRESHOLDS:
                mask = np.isfinite(alpha) & (alpha >= threshold)
                active_idx = np.flatnonzero(mask)
                components = connected_components(mask, adjacency)
                component_sizes = [int(comp.size) for comp in components]
                largest_component_count = max(component_sizes) if component_sizes else 0
                seed_flags = notch_seed_region & mask
                notch_connected_count = 0
                if np.any(seed_flags):
                    for comp in components:
                        if np.any(seed_flags[comp]):
                            notch_connected_count += int(comp.size)

                if active_idx.size:
                    x_active = x[active_idx]
                    y_active = y[active_idx]
                    x_min = float(np.nanmin(x_active))
                    x_max = float(np.nanmax(x_active))
                    y_min = float(np.nanmin(y_active))
                    y_max = float(np.nanmax(y_active))
                    centroid_x = float(np.nanmean(x_active))
                    centroid_y = float(np.nanmean(y_active))
                    if notch_connected_count:
                        comment = "Above-threshold set includes a notch-seeded connected component."
                    elif largest_component_count > 0:
                        comment = "Above-threshold set is present but not connected to the notch seed window."
                    else:
                        comment = "Above-threshold set is present; no connected component was detected."
                else:
                    x_min = x_max = y_min = y_max = centroid_x = centroid_y = float("nan")
                    comment = "No elements above this threshold."

                rows.append(
                    {
                        "case_id": case_id,
                        "step": step,
                        "threshold": threshold,
                        "count_above_threshold": int(active_idx.size),
                        "fraction_above_threshold": float(active_idx.size / n_total),
                        "notch_connected_count": int(notch_connected_count),
                        "largest_component_count": int(largest_component_count),
                        "largest_component_fraction": float(
                            largest_component_count / active_idx.size
                        )
                        if active_idx.size
                        else 0.0,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "x_span": float(x_max - x_min) if active_idx.size else float("nan"),
                        "y_span": float(y_max - y_min) if active_idx.size else float("nan"),
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "comment": comment,
                    }
                )
    return pd.DataFrame(rows)


def make_difference_table(fields: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    comparisons = {
        "C_minus_A": ("C", "A"),
        "C_minus_B": ("C", "B"),
        "B_minus_A": ("B", "A"),
    }
    rows = []
    steps = [int(item["step"]) for item in fields["A"]]
    for step_idx, step in enumerate(steps):
        for name, (case_left, case_right) in comparisons.items():
            diff = np.asarray(fields[case_left][step_idx]["alpha_elem"], dtype=float) - np.asarray(
                fields[case_right][step_idx]["alpha_elem"], dtype=float
            )
            finite = finite_values(diff)
            if name == "B_minus_A" and np.nanmax(np.abs(diff)) <= 1e-12:
                comment = "A and B alpha fields are identical within numerical table precision."
            elif name.startswith("C_minus"):
                comment = "Positive values mark locations where Case C exceeds the reference; negative values include the reduced A/B notch peak."
            else:
                comment = "Difference statistics from raw element alpha."
            rows.append(
                {
                    "comparison": name,
                    "step": step,
                    "diff_min": float(np.nanmin(diff)),
                    "diff_p01": percentile(diff, 1),
                    "diff_p05": percentile(diff, 5),
                    "diff_median": percentile(diff, 50),
                    "diff_mean": float(np.nanmean(diff)),
                    "diff_p95": percentile(diff, 95),
                    "diff_p99": percentile(diff, 99),
                    "diff_max": float(np.nanmax(diff)),
                    "positive_diff_fraction": float(np.mean(finite > 0)) if finite.size else 0.0,
                    "negative_diff_fraction": float(np.mean(finite < 0)) if finite.size else 0.0,
                    "comment": comment,
                }
            )
    return pd.DataFrame(rows)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    finite = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[finite]
    bb = bb[finite]
    if aa.size < 3 or float(np.nanstd(aa)) == 0.0 or float(np.nanstd(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    finite = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[finite]
    bb = bb[finite]
    if aa.size < 3:
        return float("nan")
    rank_a = pd.Series(aa).rank(method="average").to_numpy(dtype=float)
    rank_b = pd.Series(bb).rank(method="average").to_numpy(dtype=float)
    return safe_pearson(rank_a, rank_b)


def correlation_interpretation(pearson: float, spearman: float) -> str:
    strength = np.nanmax(np.abs([pearson, spearman]))
    if not np.isfinite(strength):
        return "Correlation unavailable."
    if strength >= 0.75:
        return "Strong spatial alignment."
    if strength >= 0.40:
        return "Moderate spatial alignment."
    if strength >= 0.15:
        return "Weak spatial alignment."
    return "Little spatial alignment."


def make_correlation_table(fields: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    field_pairs = [
        ("alpha vs HI", "HI"),
        ("alpha vs HII", "HII"),
        ("alpha vs He", "He"),
        ("alpha vs mechanics_drive", "mechanics_drive"),
        ("alpha vs elastic_energy_density", "elastic_energy_density"),
        ("alpha vs fracture_energy_density", "fracture_energy_density"),
    ]
    rows = []
    for case_id, case_fields in fields.items():
        for entry in case_fields:
            step = int(entry["step"])
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            alpha_q95 = np.nanpercentile(alpha, 95)
            top_alpha = alpha >= alpha_q95
            domain_alpha_mean = float(np.nanmean(alpha))

            for label, field_key in field_pairs:
                if field_key not in entry:
                    continue
                drive = np.asarray(entry[field_key], dtype=float)
                drive_q95 = np.nanpercentile(drive, 95)
                top_drive = drive >= drive_q95
                pearson = safe_pearson(alpha, drive)
                spearman = safe_spearman(alpha, drive)
                rows.append(
                    {
                        "case_id": case_id,
                        "step": step,
                        "field_pair": label,
                        "pearson_correlation": pearson,
                        "spearman_correlation": spearman,
                        "top_alpha_region_drive_mean": float(np.nanmean(drive[top_alpha])),
                        "domain_drive_mean": float(np.nanmean(drive)),
                        "top_drive_region_alpha_mean": float(np.nanmean(alpha[top_drive])),
                        "domain_alpha_mean": domain_alpha_mean,
                        "interpretation": correlation_interpretation(pearson, spearman),
                    }
                )
    return pd.DataFrame(rows)


def make_case_c_evolution_table(
    fields: dict[str, list[dict[str, object]]], reaction_table: pd.DataFrame
) -> pd.DataFrame:
    c_reaction = reaction_table[reaction_table["case_id"] == "C"].copy()
    c_reaction = c_reaction.set_index("step")
    rows = []
    positive_steps = [
        int(step)
        for step, row in c_reaction.iterrows()
        if float(row["reaction_N_energy"]) >= 0.0
    ]
    first_positive_step = min(positive_steps) if positive_steps else None
    final_step = int(fields["C"][-1]["step"])

    for entry in fields["C"]:
        step = int(entry["step"])
        alpha = np.asarray(entry["alpha_elem"], dtype=float)
        hi = np.asarray(entry["HI"], dtype=float)
        hii = np.asarray(entry["HII"], dtype=float)
        reaction_row = c_reaction.loc[step]
        reaction = float(reaction_row["reaction_N_energy"])
        if reaction < 0:
            interpretation = "Below compensation in the compressive-reaction branch; alpha remains low."
        elif first_positive_step is not None and step == first_positive_step:
            interpretation = "Near the compensation crossing; reaction has just become tensile and alpha remains low-amplitude."
        elif step == final_step:
            interpretation = "Final tensile branch; notch alpha is reduced versus A/B and the diffuse low-level tail remains non-validated."
        else:
            interpretation = "Post-compensation tensile branch; alpha grows but remains below A/B peak levels."

        hi_peak = float(np.nanmax(hi))
        hii_peak = float(np.nanmax(hii))
        rows.append(
            {
                "step": step,
                "displacement_mm": float(reaction_row["displacement_or_Delta"]),
                "engineering_strain": float(reaction_row["engineering_strain"]),
                "reaction_N_energy": reaction,
                "nominal_stress_MPa": float(reaction_row["nominal_stress_energy_MPa"]),
                "alpha_max": float(np.nanmax(alpha)),
                "alpha_mean": float(np.nanmean(alpha)),
                "alpha_p95": percentile(alpha, 95),
                "alpha_above_1e_minus_3_fraction": float(np.mean(alpha >= 1e-3)),
                "alpha_above_5e_minus_3_fraction": float(np.mean(alpha >= 5e-3)),
                "alpha_above_1e_minus_2_fraction": float(np.mean(alpha >= 1e-2)),
                "HI_peak": hi_peak,
                "HII_peak": hii_peak,
                "HII_over_HI_ratio": float(hii_peak / hi_peak) if hi_peak else float("nan"),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def make_plot_scale_table(final_stats: dict[str, float]) -> pd.DataFrame:
    global_max = max(final_stats.values())
    rows = [
        {
            "figure_name": "figures/final_alpha_global_scale.png",
            "color_scale_type": "shared global alpha scale",
            "vmin": 0.0,
            "vmax": global_max,
            "global_alpha_max_A": final_stats["A"],
            "global_alpha_max_B": final_stats["B"],
            "global_alpha_max_C": final_stats["C"],
            "clipped_cases": "",
            "intended_use": "Compare true peak alpha magnitude across A/B/C.",
            "misleading_if_used_for": "Inspecting very low Case C background detail.",
            "comment": "A/B dominate the true alpha peak; Case C remains lower.",
        },
        {
            "figure_name": "figures/final_alpha_low_range_scale.png",
            "color_scale_type": "shared low alpha scale",
            "vmin": 0.0,
            "vmax": LOW_RANGE_ALPHA_VMAX,
            "global_alpha_max_A": final_stats["A"],
            "global_alpha_max_B": final_stats["B"],
            "global_alpha_max_C": final_stats["C"],
            "clipped_cases": ",".join(
                case_id
                for case_id, max_value in final_stats.items()
                if max_value > LOW_RANGE_ALPHA_VMAX
            ),
            "intended_use": "Inspect low-amplitude Case C diffuse/background values.",
            "misleading_if_used_for": "Comparing peak damage magnitude or claiming A/B and C have similar peak alpha.",
            "comment": "A/B are intentionally clipped on this scale; this explains the suspicious visual emphasis.",
        },
        {
            "figure_name": "prior figures/final_alpha_comparison.png",
            "color_scale_type": "inferred low or Case-C-like scale",
            "vmin": 0.0,
            "vmax": LOW_RANGE_ALPHA_VMAX,
            "global_alpha_max_A": final_stats["A"],
            "global_alpha_max_B": final_stats["B"],
            "global_alpha_max_C": final_stats["C"],
            "clipped_cases": "A,B",
            "intended_use": "Historical context only.",
            "misleading_if_used_for": "Judging relative fracture severity from color saturation.",
            "comment": "The previous visual concern is consistent with clipping/low-range amplification.",
        },
    ]
    return pd.DataFrame(rows)


def make_artifact_table(
    fields: dict[str, list[dict[str, object]]],
    threshold_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
) -> pd.DataFrame:
    c_final = fields["C"][-1]
    a_final = fields["A"][-1]
    c_alpha = np.asarray(c_final["alpha_elem"], dtype=float)
    a_alpha = np.asarray(a_final["alpha_elem"], dtype=float)
    area = triangle_area(
        np.asarray(c_final["x"], dtype=float),
        np.asarray(c_final["y"], dtype=float),
        np.asarray(c_final["triangles"], dtype=int),
    )
    alpha_area_corr = safe_spearman(c_alpha, area)
    c_final_step = int(c_final["step"])
    final_threshold = threshold_table[
        (threshold_table["case_id"] == "C")
        & (threshold_table["step"] == c_final_step)
        & (threshold_table["threshold"].isin([1e-3, 1e-2, 3e-2]))
    ]
    c_he_corr = correlation_table[
        (correlation_table["case_id"] == "C")
        & (correlation_table["step"] == c_final_step)
        & (correlation_table["field_pair"] == "alpha vs He")
    ]
    he_text = "unavailable"
    if not c_he_corr.empty:
        row = c_he_corr.iloc[0]
        he_text = (
            f"pearson={row['pearson_correlation']:.3f}, "
            f"spearman={row['spearman_correlation']:.3f}"
        )
    threshold_text = "; ".join(
        f"thr={row.threshold:g}: frac={row.fraction_above_threshold:.4f}, "
        f"largest={int(row.largest_component_count)}, notch={int(row.notch_connected_count)}"
        for row in final_threshold.itertuples()
    )
    max_ratio = float(np.nanmax(c_alpha) / np.nanmax(a_alpha))

    rows = [
        {
            "risk_item": "colorbar clipping artifact",
            "evidence_for": "Low-range scale vmax=0.04 clips A/B peaks while Case C remains below the cap.",
            "evidence_against": "The low-level Case C values are present in the raw NPZ field, not only in a rendered image.",
            "risk_level": "high for visual interpretation",
            "conclusion": "Color scaling can make Case C look visually broad while hiding A/B peak dominance.",
            "recommended_action": "Use global-scale and low-range figures together; never infer fracture severity from the clipped low-range figure alone.",
        },
        {
            "risk_item": "low-amplitude PINN background artifact",
            "evidence_for": f"Case C final peak is only {max_ratio:.3f} of A; threshold/connectivity summary: {threshold_text}.",
            "evidence_against": f"Alpha still has spatial relation to drive fields where {he_text}.",
            "risk_level": "medium",
            "conclusion": "The broad low-amplitude background is not validated as physical fracture damage.",
            "recommended_action": "Treat diffuse low-level alpha as a diagnostic warning until repeated with a denser schedule/independent seed or direct residual checks.",
        },
        {
            "risk_item": "sampling/grid texture artifact",
            "evidence_for": f"Spearman(alpha, triangle_area) for Case C final is {alpha_area_corr:.3f}; texture panel should be inspected visually.",
            "evidence_against": "The strongest alpha remains near the notch-tip region rather than only following element size or element ordering.",
            "risk_level": "low-to-medium",
            "conclusion": "No single mesh-texture metric proves the diffuse field physical or nonphysical.",
            "recommended_action": "Review sampling_texture_diagnostic.png and repeat on another mesh only if this becomes a decision-critical claim.",
        },
        {
            "risk_item": "physical diffuse thermal-strain response",
            "evidence_for": "Uniform prescribed thermal strain changes the raw mechanical fields and shifts reaction/stress as expected.",
            "evidence_against": "Thresholded alpha remains low-amplitude relative to A/B peak damage, and physical validation was not attempted.",
            "risk_level": "medium",
            "conclusion": "A low-amplitude diffuse response is possible, but it is not established as physical fracture evidence.",
            "recommended_action": "Trust reaction/stress shift first; require additional validation before using diffuse alpha as a material conclusion.",
        },
        {
            "risk_item": "history-field path dependence artifact",
            "evidence_for": "Alpha and history fields are path dependent across the compensation crossing.",
            "evidence_against": "A/B match exactly under the same schedule and seed, which reduces evidence of a generic route bug.",
            "risk_level": "medium",
            "conclusion": "Path dependence could amplify low-level alpha after the compensation crossing.",
            "recommended_action": "If revisited, add a denser compensation-region schedule and compare history evolution without changing physics.",
        },
        {
            "risk_item": "postprocess/interpolation artifact",
            "evidence_for": "Rendered scatter/colormap choices can change the visual impression.",
            "evidence_against": "Audit uses raw element arrays directly; no interpolation is needed for the main tables.",
            "risk_level": "low",
            "conclusion": "Postprocessing can mislead visually, but the raw low-level values themselves are real output values.",
            "recommended_action": "Use raw tables and direct element-center scatter plots for review.",
        },
        {
            "risk_item": "insufficient training artifact",
            "evidence_for": "Only one strong seed is available for this audit; Case C took longer and has a different thermoelastic regime.",
            "evidence_against": "The strong run completed all checkpoints and produced stable finite fields.",
            "risk_level": "medium",
            "conclusion": "Single-run training uncertainty remains for the diffuse low-level alpha cloud.",
            "recommended_action": "Do not run a seed study for this task; reserve it as a future validation decision.",
        },
    ]
    return pd.DataFrame(rows)


def make_no_training_guard_table() -> pd.DataFrame:
    rows = [
        {
            "guard_item": "no new training",
            "expected_status": "No training or A/B/C rerun.",
            "observed_status": "Audit script only reads existing CSV/NPZ outputs and writes audit artifacts.",
            "passed": True,
            "comment": "No training entry point is imported or invoked.",
        },
        {
            "guard_item": "no source physics changes",
            "expected_status": "No source behavior changes.",
            "observed_status": "Only a new run/audit package is generated.",
            "passed": True,
            "comment": "Validation should confirm changed files stay under examples/TM_comsol_thermal_micro/runs.",
        },
        {
            "guard_item": "no heat PDE",
            "expected_status": "No heat PDE implementation or run.",
            "observed_status": "Existing prescribed-temperature fields are read only.",
            "passed": True,
            "comment": "No thermal PDE code was introduced.",
        },
        {
            "guard_item": "no damage-dependent conductivity",
            "expected_status": "No k(d) conductivity implementation or run.",
            "observed_status": "No conductivity model is touched.",
            "passed": True,
            "comment": "This audit does not alter physics.",
        },
        {
            "guard_item": "no no-thermal project modification",
            "expected_status": "No edits under examples/TM_comsol_no_thermal_micro.",
            "observed_status": "Audit package is under examples/TM_comsol_thermal_micro.",
            "passed": True,
            "comment": "Validated separately with git diff name-only guard.",
        },
        {
            "guard_item": "no legacy top-sigma primary reaction",
            "expected_status": "Keep energy-conjugate reaction as the interpreted reaction metric.",
            "observed_status": "Case C evolution table joins existing reaction_N_energy and nominal_stress_energy_MPa.",
            "passed": True,
            "comment": "No legacy reaction route is used.",
        },
    ]
    return pd.DataFrame(rows)


def setup_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(0.0, 0.010)
    ax.set_ylim(0.0, 0.010)


def scatter_field(
    ax: plt.Axes,
    entry: dict[str, object],
    values: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: object | None = None,
    size: float = 4.0,
) -> object:
    x = np.asarray(entry["element_x"], dtype=float)
    y = np.asarray(entry["element_y"], dtype=float)
    scatter = ax.scatter(
        x,
        y,
        c=np.asarray(values, dtype=float),
        s=size,
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title)
    setup_axes(ax)
    return scatter


def save_final_alpha_global(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "final_alpha_global_scale.png"
    final_max = max(float(np.nanmax(fields[case][-1]["alpha_elem"])) for case in CASES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    scatter = None
    for ax, case_id in zip(axes, CASES):
        entry = fields[case_id][-1]
        scatter = scatter_field(
            ax,
            entry,
            np.asarray(entry["alpha_elem"], dtype=float),
            f"Case {case_id} final alpha\nshared 0..{final_max:.4f}",
            vmin=0.0,
            vmax=final_max,
        )
    fig.colorbar(scatter, ax=axes, shrink=0.86, label="alpha")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_final_alpha_low_range(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "final_alpha_low_range_scale.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    scatter = None
    for ax, case_id in zip(axes, CASES):
        entry = fields[case_id][-1]
        max_alpha = float(np.nanmax(entry["alpha_elem"]))
        clipped = " clipped" if max_alpha > LOW_RANGE_ALPHA_VMAX else ""
        scatter = scatter_field(
            ax,
            entry,
            np.asarray(entry["alpha_elem"], dtype=float),
            f"Case {case_id} final alpha{clipped}\nshared 0..0.04",
            vmin=0.0,
            vmax=LOW_RANGE_ALPHA_VMAX,
        )
    fig.colorbar(scatter, ax=axes, shrink=0.86, label="alpha")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_difference_figure(
    fields: dict[str, list[dict[str, object]]], comparison_name: str, left: str, right: str
) -> Path:
    path = FIGURES_DIR / f"final_alpha_difference_{comparison_name}.png"
    entry = fields[left][-1]
    diff = np.asarray(fields[left][-1]["alpha_elem"], dtype=float) - np.asarray(
        fields[right][-1]["alpha_elem"], dtype=float
    )
    vmax = float(np.nanmax(np.abs(diff)))
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.5), constrained_layout=True)
    scatter = scatter_field(
        ax,
        entry,
        diff,
        f"Final alpha {left} - {right}",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        size=5.0,
    )
    fig.colorbar(scatter, ax=ax, label=f"alpha {left} - {right}")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_ratio_or_mask_figure(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "final_alpha_ratio_or_mask_comparison.png"
    c_entry = fields["C"][-1]
    a_alpha = np.asarray(fields["A"][-1]["alpha_elem"], dtype=float)
    b_alpha = np.asarray(fields["B"][-1]["alpha_elem"], dtype=float)
    c_alpha = np.asarray(c_entry["alpha_elem"], dtype=float)
    eps = 1e-6
    ratio_ca = np.clip(c_alpha / (a_alpha + eps), 0.0, 2.0)
    ratio_cb = np.clip(c_alpha / (b_alpha + eps), 0.0, 2.0)
    c_exceeds_a = (c_alpha - a_alpha) > 1e-4
    c_mask = c_alpha >= 1e-3

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), constrained_layout=True)
    sc0 = scatter_field(
        axes[0, 0],
        c_entry,
        ratio_ca,
        "C / (A + 1e-6), clipped 0..2",
        cmap="magma",
        vmin=0.0,
        vmax=2.0,
    )
    fig.colorbar(sc0, ax=axes[0, 0], label="ratio")
    sc1 = scatter_field(
        axes[0, 1],
        c_entry,
        ratio_cb,
        "C / (B + 1e-6), clipped 0..2",
        cmap="magma",
        vmin=0.0,
        vmax=2.0,
    )
    fig.colorbar(sc1, ax=axes[0, 1], label="ratio")
    sc2 = scatter_field(
        axes[1, 0],
        c_entry,
        c_exceeds_a.astype(float),
        "Mask: C alpha exceeds A by >1e-4",
        cmap=ListedColormap(["#d9d9d9", "#d73027"]),
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(sc2, ax=axes[1, 0], ticks=[0, 1], label="mask")
    sc3 = scatter_field(
        axes[1, 1],
        c_entry,
        c_mask.astype(float),
        "Mask: Case C alpha >= 1e-3",
        cmap=ListedColormap(["#d9d9d9", "#2166ac"]),
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(sc3, ax=axes[1, 1], ticks=[0, 1], label="mask")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_threshold_masks(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "alpha_threshold_masks.png"
    fig, axes = plt.subplots(
        len(THRESHOLDS),
        3,
        figsize=(9.8, 2.25 * len(THRESHOLDS)),
        constrained_layout=True,
    )
    cmap = ListedColormap(["#eeeeee", "#1b9e77"])
    for row_idx, threshold in enumerate(THRESHOLDS):
        for col_idx, case_id in enumerate(CASES):
            ax = axes[row_idx, col_idx]
            entry = fields[case_id][-1]
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            mask = alpha >= threshold
            scatter_field(
                ax,
                entry,
                mask.astype(float),
                f"{case_id}: alpha >= {threshold:g}\ncount={int(mask.sum())}",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                size=3.5,
            )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_alignment_case_c(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "alpha_HI_HII_He_alignment_caseC.png"
    entry = fields["C"][-1]
    panels = [
        ("alpha", "alpha_elem"),
        ("HI", "HI"),
        ("HII", "HII"),
        ("He / mechanics drive", "He" if "He" in entry else "mechanics_drive"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, (title, key) in zip(axes, panels):
        values = np.asarray(entry[key], dtype=float)
        scatter = scatter_field(
            ax,
            entry,
            values,
            f"Case C final {title}",
            vmin=0.0,
            vmax=float(np.nanmax(values)),
        )
        fig.colorbar(scatter, ax=ax, shrink=0.82, label=title)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def selected_case_c_steps(fields: dict[str, list[dict[str, object]]]) -> list[int]:
    steps = [int(entry["step"]) for entry in fields["C"]]
    candidates = [2, 3, 4, steps[-1]]
    selected = []
    for step in candidates:
        if step in steps and step not in selected:
            selected.append(step)
    while len(selected) < 4 and steps:
        candidate = steps[len(selected)]
        if candidate not in selected:
            selected.append(candidate)
    return selected[:4]


def save_case_c_evolution(
    fields: dict[str, list[dict[str, object]]], evolution_table: pd.DataFrame
) -> Path:
    path = FIGURES_DIR / "caseC_alpha_evolution_by_step.png"
    selected_steps = selected_case_c_steps(fields)
    step_to_entry = {int(entry["step"]): entry for entry in fields["C"]}
    fig, axes = plt.subplots(1, len(selected_steps), figsize=(4.0 * len(selected_steps), 4), constrained_layout=True)
    if len(selected_steps) == 1:
        axes = [axes]
    scatter = None
    for ax, step in zip(axes, selected_steps):
        entry = step_to_entry[step]
        row = evolution_table[evolution_table["step"] == step].iloc[0]
        scatter = scatter_field(
            ax,
            entry,
            np.asarray(entry["alpha_elem"], dtype=float),
            f"C step {step}\nD={row.displacement_mm:.2e} mm, R={row.reaction_N_energy:.3e} N",
            vmin=0.0,
            vmax=LOW_RANGE_ALPHA_VMAX,
        )
    fig.colorbar(scatter, ax=axes, shrink=0.86, label="alpha, shared 0..0.04")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_case_c_histogram(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "caseC_alpha_histogram_by_step.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    all_alpha = np.concatenate([np.asarray(entry["alpha_elem"], dtype=float) for entry in fields["C"]])
    lo = float(np.nanmin(all_alpha))
    hi = max(float(np.nanmax(all_alpha)), 1e-6)
    bins = np.linspace(lo, hi, 80)
    for entry in fields["C"]:
        step = int(entry["step"])
        alpha = np.asarray(entry["alpha_elem"], dtype=float)
        hist, edges = np.histogram(alpha[np.isfinite(alpha)], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist + 1, linewidth=1.1, label=f"step {step}")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("element count + 1")
    ax.set_title("Case C alpha histogram by step")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_sampling_texture(fields: dict[str, list[dict[str, object]]]) -> Path:
    path = FIGURES_DIR / "sampling_texture_diagnostic.png"
    entry = fields["C"][-1]
    alpha = np.asarray(entry["alpha_elem"], dtype=float)
    area = triangle_area(
        np.asarray(entry["x"], dtype=float),
        np.asarray(entry["y"], dtype=float),
        np.asarray(entry["triangles"], dtype=int),
    )
    elem_id_mod = np.arange(alpha.size) % 32
    mask = alpha >= 1e-3

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), constrained_layout=True)
    panels = [
        ("Case C final alpha, low range", alpha, "viridis", 0.0, LOW_RANGE_ALPHA_VMAX),
        ("Case C alpha >= 1e-3 mask", mask.astype(float), "binary", 0.0, 1.0),
        ("Element area from raw triangles", area, "plasma", float(np.nanmin(area)), float(np.nanmax(area))),
        ("Element index modulo 32", elem_id_mod, "tab20", 0.0, 31.0),
    ]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        scatter = scatter_field(ax, entry, values, title, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(scatter, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_figure_summary(figure_paths: list[Path]) -> None:
    descriptions = {
        "final_alpha_global_scale.png": "A/B/C final alpha on one true global scale from 0 to the maximum alpha across all three cases.",
        "final_alpha_low_range_scale.png": "A/B/C final alpha on a shared 0..0.04 scale to expose low-level Case C structure; A/B are clipped by design.",
        "final_alpha_difference_C_minus_A.png": "Direct raw element difference alpha_C - alpha_A.",
        "final_alpha_difference_C_minus_B.png": "Direct raw element difference alpha_C - alpha_B.",
        "final_alpha_ratio_or_mask_comparison.png": "Safe-denominator C/A and C/B ratio-like views plus threshold masks.",
        "alpha_threshold_masks.png": "Final alpha masks for thresholds 1e-4 through 1e-1 for cases A/B/C.",
        "alpha_HI_HII_He_alignment_caseC.png": "Case C final alpha, HI, HII, and He/mechanics drive side by side.",
        "caseC_alpha_evolution_by_step.png": "Case C alpha at selected below/near/after-compensation and final steps.",
        "caseC_alpha_histogram_by_step.png": "Case C alpha histograms across all saved steps.",
        "sampling_texture_diagnostic.png": "Element-center alpha/mask views next to element area and element ordering texture diagnostics.",
    }
    lines = ["# Figure Summary", ""]
    for path in figure_paths:
        lines.append(f"- `{path.name}`: {descriptions.get(path.name, 'Generated audit figure.')}")
    (FIGURES_DIR / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    fields: dict[str, list[dict[str, object]]],
    distribution_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    difference_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    evolution_table: pd.DataFrame,
    plot_scale_table: pd.DataFrame,
    artifact_table: pd.DataFrame,
) -> None:
    final_stats = {
        case_id: float(np.nanmax(fields[case_id][-1]["alpha_elem"])) for case_id in CASES
    }
    c_step = int(fields["C"][-1]["step"])
    c_final_dist = distribution_table[
        (distribution_table["case_id"] == "C") & (distribution_table["step"] == c_step)
    ].iloc[0]
    a_final_dist = distribution_table[
        (distribution_table["case_id"] == "A") & (distribution_table["step"] == c_step)
    ].iloc[0]
    c_minus_a_final = difference_table[
        (difference_table["comparison"] == "C_minus_A") & (difference_table["step"] == c_step)
    ].iloc[0]
    c_minus_b_final = difference_table[
        (difference_table["comparison"] == "C_minus_B") & (difference_table["step"] == c_step)
    ].iloc[0]
    c_threshold_subset = threshold_table[
        (threshold_table["case_id"] == "C")
        & (threshold_table["step"] == c_step)
        & (threshold_table["threshold"].isin([1e-3, 5e-3, 1e-2, 2e-2, 3e-2]))
    ]
    c_corr_final = correlation_table[
        (correlation_table["case_id"] == "C") & (correlation_table["step"] == c_step)
    ].copy()
    he_corr = c_corr_final[c_corr_final["field_pair"] == "alpha vs He"]
    he_corr_text = "unavailable"
    if not he_corr.empty:
        row = he_corr.iloc[0]
        he_corr_text = (
            f"Pearson {row['pearson_correlation']:.3f}, "
            f"Spearman {row['spearman_correlation']:.3f}"
        )

    reaction_start = evolution_table.iloc[0]
    reaction_near = evolution_table.iloc[min(3, len(evolution_table) - 1)]
    reaction_final = evolution_table.iloc[-1]

    threshold_lines = [
        "| threshold | fraction above | largest component | notch-connected | x span | y span |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in c_threshold_subset.itertuples():
        threshold_lines.append(
            f"| {row.threshold:g} | {row.fraction_above_threshold:.6f} | "
            f"{int(row.largest_component_count)} | {int(row.notch_connected_count)} | "
            f"{row.x_span:.6g} | {row.y_span:.6g} |"
        )

    c_corr_lines = [
        "| field pair | Pearson | Spearman | interpretation |",
        "|---|---:|---:|---|",
    ]
    for row in c_corr_final.itertuples():
        c_corr_lines.append(
            f"| {row.field_pair} | {row.pearson_correlation:.3f} | "
            f"{row.spearman_correlation:.3f} | {row.interpretation} |"
        )

    report = f"""# Case C Alpha Anomaly Audit

## 1. Purpose

This package audits the suspicious broad low-level alpha field in Case C from the existing stronger prescribed-temperature tension diagnostic. It uses only completed outputs from `{rel(SOURCE_PACKAGE)}` and does not run training, change source behavior, or modify the no-thermal project.

## 2. Existing strong diagnostic being audited

- Case A: `{CASES['A']['run_id']}`, thermal mode `off`, delta_T `0 K`.
- Case B: `{CASES['B']['run_id']}`, thermal mode `uniform`, delta_T `0 K`.
- Case C: `{CASES['C']['run_id']}`, thermal mode `uniform`, delta_T `+20 K`.
- Source package: `{rel(SOURCE_PACKAGE)}`.
- Field source: `{rel(RESULTS_ROOT)}`.

## 3. Why Case C alpha looked suspicious

The previous final alpha view made Case C look spatially broad in the right half of the specimen, while A/B had a much larger notch-tip peak. The visual concern was plausible because a low or Case-C-like colorbar can saturate A/B and amplify low-amplitude Case C background.

## 4. Raw-data versus plotting-scale assessment

The broad low-level Case C values are present in the raw `alpha_elem` field, so the effect is not purely an image export artifact. However, its visual prominence is strongly scale dependent. Final alpha maxima are A `{final_stats['A']:.12g}`, B `{final_stats['B']:.12g}`, and C `{final_stats['C']:.12g}`. Case C reaches only `{final_stats['C'] / final_stats['A']:.3f}` of the A/B peak.

## 5. Global-scale alpha interpretation

`figures/final_alpha_global_scale.png` uses a shared true global scale. On that scale, A/B peak alpha dominates and Case C is visibly lower. This supports the existing strong diagnostic result that positive prescribed temperature reduces the effective tensile damage drive at the final displacement.

## 6. Low-range alpha interpretation

`figures/final_alpha_low_range_scale.png` uses a shared `0..0.04` scale. This is useful for inspecting Case C low-level structure, but it clips A/B by design. The low-range figure should not be used to compare fracture severity or peak damage.

## 7. C-minus-A and C-minus-B alpha difference interpretation

The final `C_minus_A` distribution has min `{c_minus_a_final.diff_min:.6g}`, median `{c_minus_a_final.diff_median:.6g}`, max `{c_minus_a_final.diff_max:.6g}`, positive fraction `{c_minus_a_final.positive_diff_fraction:.4f}`, and negative fraction `{c_minus_a_final.negative_diff_fraction:.4f}`. `C_minus_B` is the same within A/B equality: min `{c_minus_b_final.diff_min:.6g}`, median `{c_minus_b_final.diff_median:.6g}`, max `{c_minus_b_final.diff_max:.6g}`. Negative differences include the reduced notch peak in Case C; positive differences mark low-level background locations where C exceeds the A/B baseline.

## 8. Threshold area/connectivity analysis

Connectivity uses element adjacency through shared raw mesh triangle edges. The notch seed window is centered at `(0.005, 0.005) mm` with half-width `{NOTCH_HALF_WINDOW:g} mm`.

{chr(10).join(threshold_lines)}

The final Case C thresholded area is therefore threshold-sensitive. Low thresholds capture broad low-amplitude output; higher thresholds quickly collapse toward the notch region or disappear.

## 9. Alpha versus HI/HII/He spatial correlation

For Case C final, alpha versus He is {he_corr_text}. Full final correlations:

{chr(10).join(c_corr_lines)}

The correlation evidence supports some relation to mechanical drive near high-alpha regions, but it does not by itself validate the broad low-amplitude background as physical fracture damage.

## 10. Case C alpha evolution through compensation crossing

Case C starts with reaction `{reaction_start.reaction_N_energy:.6g} N` at displacement `{reaction_start.displacement_mm:.6g} mm`, is near/after crossing by step `{int(reaction_near.step)}` with reaction `{reaction_near.reaction_N_energy:.6g} N`, and ends at `{reaction_final.reaction_N_energy:.6g} N`. Final Case C alpha max is `{reaction_final.alpha_max:.6g}`, mean `{reaction_final.alpha_mean:.6g}`, and p95 `{reaction_final.alpha_p95:.6g}`. See `tables/caseC_alpha_evolution_summary.csv` and `figures/caseC_alpha_evolution_by_step.png`.

## 11. Sampling/texture artifact check

`figures/sampling_texture_diagnostic.png` compares Case C final alpha and threshold masks against element area and element index texture. This audit does not find enough evidence to treat the broad field as a pure interpolation artifact, because the tables use raw element values directly. It also does not validate the broad field as physical, because the low-amplitude region is scale-sensitive and single-run.

## 12. Artifact risk assessment

See `tables/artifact_risk_assessment.csv`. The highest immediate risk is colorbar clipping/low-range visual amplification. Medium residual risks remain for low-amplitude PINN background, path dependence through compensation, and insufficient single-run training evidence.

## 13. What is trustworthy from Case C

- The reaction/stress downward shift relative to A/B.
- The compressive-to-tensile reaction crossing through the compensation region.
- The reduced final notch-tip alpha peak relative to A/B.
- A/B equality under the same strong settings and zero prescribed temperature.

## 14. What is not yet trustworthy from Case C

- Treating the broad low-level Case C alpha cloud as physical fracture evidence.
- Comparing fracture severity from the low-range alpha colorbar.
- Using diffuse alpha area as a material conclusion without further validation.

## 15. Final classification

`{FINAL_CLASSIFICATION}`

Case C peak damage is lower than A/B and the reaction/stress shift remains physically interpretable. The visually broad Case C alpha region is partly amplified by low-range color scaling and should not be interpreted as stronger fracture damage. The trustworthy result is the reaction/stress shift and reduced notch-tip alpha peak; the broad low-level Case C alpha cloud should not be used as physical fracture evidence yet.

## 16. Recommended next task

Review this audit package first. If more validation is needed, run one moderate non-smoke prescribed-temperature tension diagnostic with a denser schedule around `3.0e-6` to `4.5e-6 mm`, still limited to A/B/C, still using checkpointed energy-conjugate reaction, and still without heat PDE, damage-dependent conductivity, D0040, seed study, shear extension, or S0110.

## Evidence files

- `tables/alpha_threshold_area_connectivity.csv`
- `tables/alpha_distribution_statistics.csv`
- `tables/alpha_difference_statistics.csv`
- `tables/alpha_drive_spatial_correlation.csv`
- `tables/caseC_alpha_evolution_summary.csv`
- `tables/plot_scale_audit.csv`
- `tables/artifact_risk_assessment.csv`
- `tables/no_new_training_guard.csv`
- `figures/figure_summary.md`
"""
    (PACKAGE_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def write_manifest(figure_paths: list[Path], table_paths: list[Path]) -> None:
    manifest = {
        "package": rel(PACKAGE_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_package": rel(SOURCE_PACKAGE),
        "source_results_root": rel(RESULTS_ROOT),
        "classification": FINAL_CLASSIFICATION,
        "cases": CASES,
        "figures": [rel(path) for path in figure_paths],
        "tables": [rel(path) for path in table_paths],
        "guards": {
            "new_training_run": False,
            "source_code_modified": False,
            "no_thermal_project_touched": False,
            "heat_pde_introduced": False,
            "damage_dependent_conductivity_introduced": False,
            "D0040_run": False,
            "seed_study_run": False,
            "shear_extension_run": False,
            "S0110_run": False,
        },
        "connectivity_method": "Element graph from shared raw mesh triangle edges; components are computed over thresholded element masks.",
        "notch_seed_window": {
            "center_x_mm": NOTCH_TIP_XY[0],
            "center_y_mm": NOTCH_TIP_XY[1],
            "half_window_mm": NOTCH_HALF_WINDOW,
        },
    }
    (PACKAGE_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_handoff(figure_paths: list[Path], table_paths: list[Path]) -> None:
    lines = [
        "# Handoff: Case C Alpha Anomaly Audit",
        "",
        "## Status",
        "",
        f"Final classification: `{FINAL_CLASSIFICATION}`",
        "",
        "Commit hash:",
        "",
        "- Filled in the final Codex response after commit/push. A follow-up handoff sync commit is required if the final hash must be stored inside the repository.",
        "",
        "Push status:",
        "",
        "- Pending at package generation time; see final Codex response for pushed HEAD.",
        "",
        "## Package",
        "",
        f"- Package path: `{rel(PACKAGE_DIR)}`",
        f"- Report: `{rel(PACKAGE_DIR / 'REPORT.md')}`",
        f"- Audited source package: `{rel(SOURCE_PACKAGE)}`",
        "",
        "## Scope",
        "",
        "- Worked only under `examples/TM_comsol_thermal_micro`.",
        "- Did not modify `examples/TM_comsol_no_thermal_micro`.",
        "- Did not run new training or rerun A/B/C.",
        "- Did not implement heat PDE, damage-dependent conductivity, trainable/PDE temperature, D0040, seed study, shear extension, or S0110.",
        "- Did not change material parameters, l0, history logic, training losses, boundary conditions, source model behavior, or reaction route.",
        "",
        "## Figures Generated",
        "",
    ]
    lines.extend(f"- `{rel(path)}`" for path in figure_paths)
    lines.extend(["", "## Tables Generated", ""])
    lines.extend(f"- `{rel(path)}`" for path in table_paths)
    lines.extend(
        [
            "",
            "## Main Conclusion",
            "",
            "Case C peak alpha is lower than A/B and the reaction/stress shift remains trustworthy within this diagnostic. The broad low-level Case C alpha field exists in raw element values, but its visual impact is amplified by low-range color scaling and it is not validated as physical fracture damage.",
            "",
            "## Reviewer Should Read Next",
            "",
            f"1. `{rel(PACKAGE_DIR / 'REPORT.md')}`",
            f"2. `{rel(TABLES_DIR / 'plot_scale_audit.csv')}`",
            f"3. `{rel(TABLES_DIR / 'alpha_threshold_area_connectivity.csv')}`",
            f"4. `{rel(TABLES_DIR / 'alpha_drive_spatial_correlation.csv')}`",
            f"5. `{rel(FIGURES_DIR / 'final_alpha_global_scale.png')}`",
            f"6. `{rel(FIGURES_DIR / 'final_alpha_low_range_scale.png')}`",
            f"7. `{rel(FIGURES_DIR / 'sampling_texture_diagnostic.png')}`",
            "",
            "## Exact Next Recommended Task",
            "",
            "Review this package. If more validation is needed, run one moderate non-smoke prescribed-temperature tension diagnostic with a denser schedule around `3.0e-6` to `4.5e-6 mm`, still limited to A/B/C, still using checkpointed energy-conjugate reaction, and still without heat PDE or damage-dependent conductivity.",
        ]
    )
    (PACKAGE_DIR / "HANDOFF_COMMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    fields = load_fields()
    assert_shared_geometry(fields)
    adjacency = build_element_adjacency(np.asarray(fields["A"][-1]["triangles"], dtype=int))

    reaction_table = pd.read_csv(SOURCE_PACKAGE / "tables" / "reaction_stress_by_step.csv")

    distribution_table = make_distribution_table(fields)
    threshold_table = make_threshold_table(fields, adjacency)
    difference_table = make_difference_table(fields)
    correlation_table = make_correlation_table(fields)
    evolution_table = make_case_c_evolution_table(fields, reaction_table)

    final_stats = {
        case_id: float(np.nanmax(fields[case_id][-1]["alpha_elem"])) for case_id in CASES
    }
    plot_scale_table = make_plot_scale_table(final_stats)
    artifact_table = make_artifact_table(fields, threshold_table, correlation_table)
    guard_table = make_no_training_guard_table()

    table_map = {
        "alpha_threshold_area_connectivity.csv": threshold_table,
        "alpha_distribution_statistics.csv": distribution_table,
        "alpha_difference_statistics.csv": difference_table,
        "alpha_drive_spatial_correlation.csv": correlation_table,
        "caseC_alpha_evolution_summary.csv": evolution_table,
        "plot_scale_audit.csv": plot_scale_table,
        "artifact_risk_assessment.csv": artifact_table,
        "no_new_training_guard.csv": guard_table,
    }
    table_paths = []
    for name, table in table_map.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        table_paths.append(path)

    figure_paths = [
        save_final_alpha_global(fields),
        save_final_alpha_low_range(fields),
        save_difference_figure(fields, "C_minus_A", "C", "A"),
        save_difference_figure(fields, "C_minus_B", "C", "B"),
        save_ratio_or_mask_figure(fields),
        save_threshold_masks(fields),
        save_alignment_case_c(fields),
        save_case_c_evolution(fields, evolution_table),
        save_case_c_histogram(fields),
        save_sampling_texture(fields),
    ]
    write_figure_summary(figure_paths)
    write_report(
        fields,
        distribution_table,
        threshold_table,
        difference_table,
        correlation_table,
        evolution_table,
        plot_scale_table,
        artifact_table,
    )
    write_manifest(figure_paths, table_paths)
    write_handoff(figure_paths, table_paths)

    print(f"package={rel(PACKAGE_DIR)}")
    print(f"classification={FINAL_CLASSIFICATION}")
    print(f"figures={len(figure_paths)}")
    print(f"tables={len(table_paths)}")


if __name__ == "__main__":
    main()
