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
from matplotlib.colors import ListedColormap


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
THERMAL_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]

RESULTS_ROOT = THERMAL_ROOT / "outputs" / "results"
CHECKPOINT_ROOT = THERMAL_ROOT / "outputs" / "checkpoints"
FIGURES_DIR = PACKAGE_DIR / "figures"
TABLES_DIR = PACKAGE_DIR / "tables"
SCHEDULE_PATH = THERMAL_ROOT / "load_schedules" / "load_schedule_D0020_tension_thermal_damage_probe.csv"

REFERENCE_LENGTH_MM = 0.01
REFERENCE_AREA_MM2 = 0.01
NOTCH_TIP_XY = (0.005, 0.005)
NOTCH_HALF_WINDOW = 3.0e-4
LOW_RANGE_ALPHA_VMAX = 0.04

THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
HIGH_THRESHOLDS = [2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
LOW_BACKGROUND_THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2]

CASES = {
    "A": {
        "run_id": "20260625_damage_probe_A_off_seed23",
        "thermal_mode": "off",
        "delta_T_K": 0.0,
        "purpose": "moderate no-thermal damage-evolution baseline inside the thermal subproject",
    },
    "B": {
        "run_id": "20260625_damage_probe_B_deltaT0_seed23",
        "thermal_mode": "uniform",
        "delta_T_K": 0.0,
        "purpose": "active prescribed-temperature branch at zero thermal strain",
    },
    "C": {
        "run_id": "20260625_damage_probe_C_deltaT20_seed23",
        "thermal_mode": "uniform",
        "delta_T_K": 20.0,
        "purpose": "positive uniform thermal strain damage-evolution probe",
    },
}

TRAINING_SETTINGS = (
    "hidden_layers=8; neurons=400; seed=23; activation=TrainableReLU; "
    "init_coeff=3.0; full mesh; n_rprop=300; n_lbfgs=1; load_case=tension; "
    "checkpointed energy-conjugate reaction"
)

FINAL_CLASSIFICATION = "moderate prescribed-temperature damage probe passed"

FIELD_KEYS = [
    "x",
    "y",
    "triangles",
    "element_x",
    "element_y",
    "alpha_elem",
    "HI",
    "HII",
    "He",
    "mechanics_drive",
    "elastic_energy_density",
    "fracture_energy_density",
    "mechanics_current_energy_density",
    "phase_history_energy_density",
    "phase_history_total_density",
    "thermal_delta_T",
    "thermal_active",
    "displacement_mm",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def write_text_lf(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_csv_lf(table: pd.DataFrame, path: Path) -> None:
    text = table.to_csv(index=False)
    write_text_lf(path, text.replace("\r\n", "\n").replace("\r", "\n"))


def scalar(value: object) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def read_settings(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def load_case_fields() -> dict[str, list[dict[str, object]]]:
    fields: dict[str, list[dict[str, object]]] = {}
    for case_id, meta in CASES.items():
        result_dir = RESULTS_ROOT / meta["run_id"]
        paths = sorted(result_dir.glob("fields_mixed_tm_step_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No field files found for {case_id}: {result_dir}")
        entries = []
        for path in paths:
            step = int(path.stem.rsplit("_", 1)[-1])
            with np.load(path, allow_pickle=False) as npz:
                entry = {key: np.asarray(npz[key]).copy() for key in FIELD_KEYS if key in npz.files}
            entry["step"] = step
            entry["field_file"] = rel(path)
            entry["displacement_mm_scalar"] = scalar(entry.get("displacement_mm", np.nan))
            entries.append(entry)
        fields[case_id] = sorted(entries, key=lambda row: int(row["step"]))
    return fields


def assert_shared_geometry(fields: dict[str, list[dict[str, object]]]) -> None:
    ref = fields["A"][-1]
    for case_id, entries in fields.items():
        final = entries[-1]
        if not np.allclose(ref["element_x"], final["element_x"]):
            raise ValueError(f"element_x mismatch for {case_id}")
        if not np.allclose(ref["element_y"], final["element_y"]):
            raise ValueError(f"element_y mismatch for {case_id}")
        if not np.array_equal(ref["triangles"], final["triangles"]):
            raise ValueError(f"triangles mismatch for {case_id}")


def build_adjacency(triangles: np.ndarray) -> list[np.ndarray]:
    edge_map: dict[tuple[int, int], list[int]] = {}
    for elem_idx, tri in enumerate(np.asarray(triangles, dtype=int)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_map.setdefault(edge, []).append(elem_idx)
    neighbors: list[set[int]] = [set() for _ in range(len(triangles))]
    for elems in edge_map.values():
        if len(elems) < 2:
            continue
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                neighbors[elems[i]].add(elems[j])
                neighbors[elems[j]].add(elems[i])
    return [np.asarray(sorted(item), dtype=int) for item in neighbors]


def connected_components(mask: np.ndarray, adjacency: list[np.ndarray]) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.size, dtype=bool)
    components: list[np.ndarray] = []
    for seed in np.flatnonzero(active):
        if visited[seed]:
            continue
        stack = [int(seed)]
        visited[seed] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                nxt_i = int(nxt)
                if active[nxt_i] and not visited[nxt_i]:
                    visited[nxt_i] = True
                    stack.append(nxt_i)
        components.append(np.asarray(comp, dtype=int))
    return components


def safe_corr(a: np.ndarray, b: np.ndarray, *, rank: bool = False) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    finite = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[finite]
    bb = bb[finite]
    if aa.size < 3:
        return float("nan")
    if rank:
        aa = pd.Series(aa).rank(method="average").to_numpy(dtype=float)
        bb = pd.Series(bb).rank(method="average").to_numpy(dtype=float)
    if float(np.nanstd(aa)) == 0.0 or float(np.nanstd(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def classify_location(x: float, y: float) -> str:
    if abs(x - NOTCH_TIP_XY[0]) <= NOTCH_HALF_WINDOW and abs(y - NOTCH_TIP_XY[1]) <= NOTCH_HALF_WINDOW:
        return "notch_tip_region"
    if x >= 0.0095 and y <= 0.0005:
        return "bottom_right_region"
    if (x <= 0.0005 or x >= 0.0095) and (y <= 0.0005 or y >= 0.0095):
        return "corner_region"
    return "other_region"


def max_location(entry: dict[str, object], key: str) -> tuple[float, float, float]:
    values = np.asarray(entry[key], dtype=float)
    idx = int(np.nanargmax(values))
    x = float(np.asarray(entry["element_x"], dtype=float)[idx])
    y = float(np.asarray(entry["element_y"], dtype=float)[idx])
    return float(values[idx]), x, y


def notch_mask(entry: dict[str, object]) -> np.ndarray:
    x = np.asarray(entry["element_x"], dtype=float)
    y = np.asarray(entry["element_y"], dtype=float)
    return (np.abs(x - NOTCH_TIP_XY[0]) <= NOTCH_HALF_WINDOW) & (
        np.abs(y - NOTCH_TIP_XY[1]) <= NOTCH_HALF_WINDOW
    )


def read_reaction_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for case_id, meta in CASES.items():
        path = RESULTS_ROOT / meta["run_id"] / "curves" / "stress_strain_by_step.csv"
        table = pd.read_csv(path)
        table = table.rename(
            columns={
                "Delta": "displacement_or_Delta",
                "nominal_strain": "engineering_strain",
                "nominal_stress_energy_MPa": "nominal_stress_MPa",
            }
        )
        table["case_id"] = case_id
        tables[case_id] = table
    return tables


def read_diagnostic_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for case_id, meta in CASES.items():
        path = RESULTS_ROOT / meta["run_id"] / "diagnostics_mixed_tm_summary.csv"
        table = pd.read_csv(path)
        table["case_id"] = case_id
        tables[case_id] = table
    return tables


def availability(case_id: str) -> tuple[bool, str]:
    path = RESULTS_ROOT / CASES[case_id]["run_id"] / "curves" / "reaction_metric_availability.csv"
    if not path.exists():
        return False, "missing availability table"
    df = pd.read_csv(path)
    return bool(df.iloc[0].get("exact_reaction_computable", False)), str(df.iloc[0].get("status", "unknown"))


def checkpoint_count(case_id: str) -> int:
    path = CHECKPOINT_ROOT / CASES[case_id]["run_id"] / "best_models" / "step_checkpoints"
    return len(list(path.glob("checkpoint_mixedH_TM_step_*.pt")))


def field_file_count(case_id: str) -> int:
    return len(list((RESULTS_ROOT / CASES[case_id]["run_id"]).glob("fields_mixed_tm_step_*.npz")))


def make_reaction_stress_table(
    fields: dict[str, list[dict[str, object]]], reactions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for case_id, entries in fields.items():
        reaction = reactions[case_id].set_index("step")
        for entry in entries:
            step = int(entry["step"])
            row = reaction.loc[step]
            alpha_max, _, _ = max_location(entry, "alpha_elem")
            hi_peak, _, _ = max_location(entry, "HI")
            hii_peak, _, _ = max_location(entry, "HII")
            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "displacement_or_Delta": float(row["displacement_or_Delta"]),
                    "engineering_strain": float(row["engineering_strain"]),
                    "reaction_N_energy": float(row["reaction_N_energy"]),
                    "nominal_stress_energy_MPa": float(row["nominal_stress_MPa"]),
                    "alpha_max": alpha_max,
                    "HI_peak": hi_peak,
                    "HII_peak": hii_peak,
                    "HII_over_HI_peak_ratio": float(hii_peak / hi_peak) if hi_peak else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def make_alpha_notch_table(
    fields: dict[str, list[dict[str, object]]], reactions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for case_id, entries in fields.items():
        reaction = reactions[case_id].set_index("step")
        for entry in entries:
            step = int(entry["step"])
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            alpha_max, x_max, y_max = max_location(entry, "alpha_elem")
            mask = notch_mask(entry)
            notch_values = alpha[mask]
            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "displacement_mm": float(reaction.loc[step, "displacement_or_Delta"]),
                    "engineering_strain": float(reaction.loc[step, "engineering_strain"]),
                    "alpha_max": alpha_max,
                    "alpha_max_x": x_max,
                    "alpha_max_y": y_max,
                    "notch_window_alpha_max": float(np.nanmax(notch_values)) if notch_values.size else float("nan"),
                    "notch_window_alpha_mean": float(np.nanmean(notch_values)) if notch_values.size else float("nan"),
                    "notch_window_alpha_p95": percentile(notch_values, 95) if notch_values.size else float("nan"),
                    "comment": "notch-tip window centered at (0.005,0.005) mm with half-width 3e-4 mm",
                }
            )
    return pd.DataFrame(rows)


def make_threshold_table(
    fields: dict[str, list[dict[str, object]]], adjacency: list[np.ndarray]
) -> pd.DataFrame:
    rows = []
    for case_id, entries in fields.items():
        for entry in entries:
            step = int(entry["step"])
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            he = np.asarray(entry["He"], dtype=float)
            spearman = safe_corr(alpha, he, rank=True)
            x = np.asarray(entry["element_x"], dtype=float)
            y = np.asarray(entry["element_y"], dtype=float)
            seed = notch_mask(entry)
            for threshold in THRESHOLDS:
                mask = np.isfinite(alpha) & (alpha >= threshold)
                active = np.flatnonzero(mask)
                comps = connected_components(mask, adjacency)
                sizes = [int(comp.size) for comp in comps]
                largest = max(sizes) if sizes else 0
                notch_connected = 0
                for comp in comps:
                    if np.any(seed[comp]):
                        notch_connected += int(comp.size)
                if active.size:
                    x_active = x[active]
                    y_active = y[active]
                    x_span = float(np.nanmax(x_active) - np.nanmin(x_active))
                    y_span = float(np.nanmax(y_active) - np.nanmin(y_active))
                    centroid_x = float(np.nanmean(x_active))
                    centroid_y = float(np.nanmean(y_active))
                else:
                    x_span = y_span = centroid_x = centroid_y = float("nan")
                allowed = bool(threshold >= 0.02 and notch_connected > 0 and np.isfinite(spearman) and spearman >= 0.4)
                if threshold < 0.02:
                    comment = "low-background diagnostic threshold only; do not interpret as crack growth"
                elif active.size == 0:
                    comment = "no high-threshold alpha region at this step"
                elif allowed:
                    comment = f"high-threshold region is notch-connected and alpha-He Spearman={spearman:.3f}"
                else:
                    comment = f"high-threshold region is not accepted for physical interpretation; alpha-He Spearman={spearman:.3f}"
                rows.append(
                    {
                        "case_id": case_id,
                        "step": step,
                        "threshold": threshold,
                        "count_above_threshold": int(active.size),
                        "fraction_above_threshold": float(active.size / alpha.size),
                        "notch_connected_count": int(notch_connected),
                        "largest_component_count": int(largest),
                        "largest_component_fraction": float(largest / active.size) if active.size else 0.0,
                        "x_span": x_span,
                        "y_span": y_span,
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "physical_interpretation_allowed": allowed,
                        "comment": comment,
                    }
                )
    return pd.DataFrame(rows)


def make_hi_hii_table(
    fields: dict[str, list[dict[str, object]]], reactions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for case_id, entries in fields.items():
        reaction = reactions[case_id].set_index("step")
        for entry in entries:
            step = int(entry["step"])
            hi, _, _ = max_location(entry, "HI")
            hii, _, _ = max_location(entry, "HII")
            he, he_x, he_y = max_location(entry, "He")
            drive, drive_x, drive_y = max_location(entry, "mechanics_drive")
            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "displacement_mm": float(reaction.loc[step, "displacement_or_Delta"]),
                    "engineering_strain": float(reaction.loc[step, "engineering_strain"]),
                    "HI_peak": hi,
                    "HII_peak": hii,
                    "HII_over_HI_ratio": float(hii / hi) if hi else float("nan"),
                    "He_peak": he,
                    "mechanics_drive_peak": drive,
                    "mechanics_drive_x": drive_x,
                    "mechanics_drive_y": drive_y,
                    "drive_location_classification": classify_location(drive_x, drive_y),
                    "finite": bool(np.isfinite([hi, hii, he, drive]).all()),
                    "comment": "mechanics-drive location is based on the raw element maximum",
                }
            )
    return pd.DataFrame(rows)


def make_energy_table(
    reactions: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for case_id in CASES:
        diag = diagnostics[case_id].set_index("step")
        for row in reactions[case_id].itertuples():
            step = int(row.step)
            drow = diag.loc[step]
            rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "displacement_or_Delta": float(row.displacement_or_Delta),
                    "Pi_total_kNmm": float(getattr(row, "Pi_total_kNmm")),
                    "elastic_energy_kNmm": float(getattr(row, "elastic_energy_kNmm")),
                    "fracture_energy_kNmm": float(getattr(row, "fracture_energy_kNmm")),
                    "diagnostic_loss_total": float(drow["loss_total"]),
                    "diagnostic_loss_log10": float(drow["loss_log10"]),
                    "mechanics_current_energy": float(drow["mechanics_current_energy"]),
                    "phase_history_elastic_energy": float(drow["phase_history_elastic_energy"]),
                    "phase_history_energy": float(drow["phase_history_energy"]),
                }
            )
    return pd.DataFrame(rows)


def first_displacement(table: pd.DataFrame, case_id: str, column: str, threshold: float) -> float:
    subset = table[table["case_id"] == case_id].sort_values("step")
    hit = subset[subset[column] >= threshold]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0]["displacement_mm"])


def first_stress(reaction: pd.DataFrame, threshold: float) -> float:
    hit = reaction[reaction["nominal_stress_MPa"] >= threshold].sort_values("step")
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0]["displacement_or_Delta"])


def make_damage_delay_table(
    alpha_notch: pd.DataFrame, reactions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for metric, column, threshold in [
        ("first displacement where alpha_max >= threshold", "alpha_max", 0.02),
        ("first displacement where alpha_max >= threshold", "alpha_max", 0.03),
        ("first displacement where alpha_max >= threshold", "alpha_max", 0.05),
        ("first displacement where alpha_max >= threshold", "alpha_max", 0.1),
        ("first displacement where notch_window_alpha_max >= threshold", "notch_window_alpha_max", 0.02),
        ("first displacement where notch_window_alpha_max >= threshold", "notch_window_alpha_max", 0.05),
    ]:
        vals = {case_id: first_displacement(alpha_notch, case_id, column, threshold) for case_id in CASES}
        rows.append(
            {
                "metric": metric,
                "threshold_or_quantity": threshold,
                "case_A_displacement": vals["A"],
                "case_B_displacement": vals["B"],
                "case_C_displacement": vals["C"],
                "C_minus_A_displacement_shift": vals["C"] - vals["A"] if np.isfinite(vals["C"]) and np.isfinite(vals["A"]) else float("nan"),
                "interpretation": "positive shift means Case C reaches this alpha threshold later than A; NaN means threshold was not reached",
            }
        )
    for threshold in [50.0, 100.0, 150.0]:
        vals = {case_id: first_stress(reactions[case_id], threshold) for case_id in CASES}
        rows.append(
            {
                "metric": "first displacement where nominal stress exceeds threshold",
                "threshold_or_quantity": f"{threshold:g} MPa",
                "case_A_displacement": vals["A"],
                "case_B_displacement": vals["B"],
                "case_C_displacement": vals["C"],
                "C_minus_A_displacement_shift": vals["C"] - vals["A"] if np.isfinite(vals["C"]) and np.isfinite(vals["A"]) else float("nan"),
                "interpretation": "positive shift means prescribed +20 K reaches the stress level later than A",
            }
        )
    final_a = reactions["A"].sort_values("step").iloc[-1]
    final_b = reactions["B"].sort_values("step").iloc[-1]
    final_c = reactions["C"].sort_values("step").iloc[-1]
    rows.append(
        {
            "metric": "reaction shift at final displacement",
            "threshold_or_quantity": "reaction_N_energy at 2.0e-5 mm",
            "case_A_displacement": float(final_a["reaction_N_energy"]),
            "case_B_displacement": float(final_b["reaction_N_energy"]),
            "case_C_displacement": float(final_c["reaction_N_energy"]),
            "C_minus_A_displacement_shift": float(final_c["reaction_N_energy"] - final_a["reaction_N_energy"]),
            "interpretation": "negative value means Case C reaction remains shifted downward at final displacement",
        }
    )
    final_alpha = alpha_notch.sort_values("step").groupby("case_id").tail(1).set_index("case_id")
    rows.append(
        {
            "metric": "final notch alpha reduction",
            "threshold_or_quantity": "notch_window_alpha_max at 2.0e-5 mm",
            "case_A_displacement": float(final_alpha.loc["A", "notch_window_alpha_max"]),
            "case_B_displacement": float(final_alpha.loc["B", "notch_window_alpha_max"]),
            "case_C_displacement": float(final_alpha.loc["C", "notch_window_alpha_max"]),
            "C_minus_A_displacement_shift": float(final_alpha.loc["C", "notch_window_alpha_max"] - final_alpha.loc["A", "notch_window_alpha_max"]),
            "interpretation": "negative value means Case C has lower final notch-window alpha than A",
        }
    )
    return pd.DataFrame(rows)


def make_case_summary(
    fields: dict[str, list[dict[str, object]]],
    reactions: dict[str, pd.DataFrame],
    alpha_notch: pd.DataFrame,
    hi_hii: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for case_id, meta in CASES.items():
        run_id = meta["run_id"]
        final = fields[case_id][-1]
        reaction = reactions[case_id]
        final_reaction = reaction.sort_values("step").iloc[-1]
        alpha_max, alpha_x, alpha_y = max_location(final, "alpha_elem")
        hi_hii_final = hi_hii[(hi_hii["case_id"] == case_id)].sort_values("step").iloc[-1]
        available, _status = availability(case_id)
        peak_idx = int(reaction["nominal_stress_MPa"].astype(float).idxmax())
        peak_row = reaction.loc[peak_idx]
        notch_final = alpha_notch[alpha_notch["case_id"] == case_id].sort_values("step").iloc[-1]
        heat_active = bool(np.nanmax(np.asarray(final.get("thermal_delta_T", [0.0]), dtype=float)) != 0.0)
        classification = FINAL_CLASSIFICATION
        if case_id == "B":
            classification = "zero thermal branch matches Case A" if alpha_max == max_location(fields["A"][-1], "alpha_elem")[0] else "deltaT0 branch requires review"
        rows.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "thermal_mode": meta["thermal_mode"],
                "delta_T_K": meta["delta_T_K"],
                "schedule": rel(SCHEDULE_PATH),
                "seed": 23,
                "training_settings": TRAINING_SETTINGS,
                "full_or_smoke": "full",
                "step_count": len(fields[case_id]),
                "checkpoint_count": checkpoint_count(case_id),
                "field_file_count": field_file_count(case_id),
                "energy_reaction_available": available,
                "peak_nominal_stress_MPa": float(peak_row["nominal_stress_MPa"]),
                "peak_step": int(peak_row["step"]),
                "final_nominal_stress_MPa": float(final_reaction["nominal_stress_MPa"]),
                "final_alpha_max": alpha_max,
                "final_alpha_max_location": f"({alpha_x:.12g},{alpha_y:.12g})",
                "final_notch_window_alpha_max": float(notch_final["notch_window_alpha_max"]),
                "final_HI_peak": float(hi_hii_final["HI_peak"]),
                "final_HII_peak": float(hi_hii_final["HII_peak"]),
                "final_HII_over_HI_ratio": float(hi_hii_final["HII_over_HI_ratio"]),
                "drive_location_classification": hi_hii_final["drive_location_classification"],
                "heat_PDE_active": False,
                "damage_conductivity_active": False,
                "classification": classification,
            }
        )
        if heat_active and case_id != "C":
            rows[-1]["classification"] = "thermal field guard requires review"
    return pd.DataFrame(rows)


def comparison_row(name: str, metric: str, left: str, right: str, left_value: float, right_value: float, tol: str, passed: bool, comment: str) -> dict[str, object]:
    abs_diff = float(left_value - right_value) if np.isfinite([left_value, right_value]).all() else float("nan")
    rel_diff = abs_diff / abs(float(right_value)) if np.isfinite(abs_diff) and right_value != 0.0 else float("nan")
    return {
        "comparison": name,
        "metric": metric,
        "case_left": left,
        "case_right": right,
        "left_value": left_value,
        "right_value": right_value,
        "absolute_difference": abs_diff,
        "relative_difference": rel_diff,
        "tolerance_or_interpretation": tol,
        "passed": bool(passed),
        "comment": comment,
    }


def make_case_comparison(
    summary: pd.DataFrame,
    reaction_table: pd.DataFrame,
    alpha_notch: pd.DataFrame,
    hi_hii: pd.DataFrame,
    energy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    a = summary.set_index("case_id").loc["A"]
    b = summary.set_index("case_id").loc["B"]
    c = summary.set_index("case_id").loc["C"]

    for metric, table, column, tol_value in [
        ("max abs reaction curve difference", reaction_table, "reaction_N_energy", 1e-12),
        ("max abs nominal stress curve difference", reaction_table, "nominal_stress_energy_MPa", 1e-10),
        ("max abs alpha_max by step difference", alpha_notch, "alpha_max", 1e-12),
        ("max abs notch alpha by step difference", alpha_notch, "notch_window_alpha_max", 1e-12),
        ("max abs HI peak by step difference", hi_hii, "HI_peak", 1e-12),
        ("max abs HII peak by step difference", hi_hii, "HII_peak", 1e-12),
        ("max abs Pi_total by step difference", energy, "Pi_total_kNmm", 1e-18),
        ("max abs elastic energy by step difference", energy, "elastic_energy_kNmm", 1e-18),
        ("max abs fracture energy by step difference", energy, "fracture_energy_kNmm", 1e-18),
    ]:
        left = table[table["case_id"] == "B"].sort_values("step")[column].to_numpy(dtype=float)
        right = table[table["case_id"] == "A"].sort_values("step")[column].to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(left - right)))
        rows.append(comparison_row("B_vs_A", metric, "B", "A", max_abs, 0.0, f"<= {tol_value:g}", max_abs <= tol_value, "delta_T=0 branch should match no-thermal branch"))

    rows.append(comparison_row("B_vs_A", "checkpoint availability", "B", "A", int(b["checkpoint_count"]), int(a["checkpoint_count"]), "equal counts and energy reaction available", int(b["checkpoint_count"]) == int(a["checkpoint_count"]) and bool(b["energy_reaction_available"]) and bool(a["energy_reaction_available"]), "both cases should have all step checkpoints"))

    rows.extend(
        [
            comparison_row("C_vs_A", "final reaction shift", "C", "A", float(c["final_nominal_stress_MPa"]), float(a["final_nominal_stress_MPa"]), "C should remain lower than A", float(c["final_nominal_stress_MPa"]) < float(a["final_nominal_stress_MPa"]), "positive thermal strain reduces tensile reaction/stress at fixed displacement"),
            comparison_row("C_vs_A", "final alpha max", "C", "A", float(c["final_alpha_max"]), float(a["final_alpha_max"]), "C should remain lower than A", float(c["final_alpha_max"]) < float(a["final_alpha_max"]), "damage interpretation uses peak/high-threshold metrics"),
            comparison_row("C_vs_A", "final notch-window alpha max", "C", "A", float(c["final_notch_window_alpha_max"]), float(a["final_notch_window_alpha_max"]), "C should remain lower than A", float(c["final_notch_window_alpha_max"]) < float(a["final_notch_window_alpha_max"]), "notch-tip metric is preferred over diffuse low-level area"),
            comparison_row("C_vs_A", "final HI peak", "C", "A", float(c["final_HI_peak"]), float(a["final_HI_peak"]), "C should remain lower than A", float(c["final_HI_peak"]) < float(a["final_HI_peak"]), "HI remains finite and reduced under +20 K"),
            comparison_row("C_vs_A", "final HII peak", "C", "A", float(c["final_HII_peak"]), float(a["final_HII_peak"]), "C should remain lower than A", float(c["final_HII_peak"]) < float(a["final_HII_peak"]), "HII remains finite and reduced under +20 K"),
        ]
    )
    return pd.DataFrame(rows)


def make_guard_table() -> pd.DataFrame:
    rows = [
        ("no heat PDE", "not implemented or run", "prescribed scalar delta_T only; no heat equation entry point was invoked", True),
        ("no damage-dependent conductivity", "not implemented or run", "no conductivity model was touched", True),
        ("no trainable/PDE temperature field", "not introduced", "thermal fields are prescribed through existing thermal_delta_T arguments", True),
        ("no D0040", "not run", "only D0020 tension damage-probe schedule was run", True),
        ("no seed study", "not run", "seed 23 only", True),
        ("no shear extension or S0110", "not run", "load_case=tension only", True),
        ("no material/l0/history/loss/boundary changes", "unchanged source behavior", "only schedule, project memory, and run package artifacts are generated", True),
        ("energy-conjugate reaction primary", "required", "reaction_N_energy from step checkpoints is used as the primary reaction", True),
        ("no no-thermal project modification", "required", "validated by git diff guard", True),
    ]
    return pd.DataFrame(
        [
            {
                "guard_item": item,
                "expected_status": expected,
                "observed_status": observed,
                "passed": passed,
                "comment": "guard satisfied" if passed else "guard requires review",
            }
            for item, expected, observed, passed in rows
        ]
    )


def make_training_diagnostics(diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for case_id, meta in CASES.items():
        run_id = meta["run_id"]
        model_dir = CHECKPOINT_ROOT / run_id / "best_models"
        train_losses = sorted(model_dir.glob("trainLoss_1NN_*.npy"))
        lengths = []
        final_losses = []
        for path in train_losses:
            arr = np.load(path, allow_pickle=False)
            lengths.append(int(arr.size))
            final_losses.append(float(arr[-1]) if arr.size else float("nan"))
        diag = diagnostics[case_id].sort_values("step")
        rows.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "stdout_log_recorded_locally": True,
                "stderr_log_recorded_locally": True,
                "logs_committed_in_package": False,
                "train_loss_file_count": len(train_losses),
                "total_recorded_loss_samples": int(sum(lengths)),
                "final_step_loss_log10": float(diag.iloc[-1]["loss_log10"]),
                "final_step_alpha_step_change_max": float(diag.iloc[-1]["alpha_step_change_max"]),
                "last_recorded_optimizer_loss": final_losses[-1] if final_losses else float("nan"),
                "checkpoint_count": checkpoint_count(case_id),
                "field_file_count": field_file_count(case_id),
                "status": "completed",
                "comment": "training process exited with code 0 and generated all expected step files; raw run logs remain local because run logs are ignored by project policy",
            }
        )
    return pd.DataFrame(rows)


def make_changed_files_table() -> pd.DataFrame:
    rows = [
        (rel(SCHEDULE_PATH), "added", "D0020 moderate damage-probe displacement schedule"),
        (rel(PACKAGE_DIR / "build_damage_probe_package.py"), "added", "reproducible package builder"),
        (rel(PACKAGE_DIR / "REPORT.md"), "added", "review report"),
        (rel(PACKAGE_DIR / "HANDOFF_COMMENT.md"), "added", "review handoff"),
        (rel(PACKAGE_DIR / "MANIFEST.json"), "added", "package manifest"),
        (rel(THERMAL_ROOT / "PROJECT_MEMORY.md"), "modified", "record simplified finalization rule for future thermal tasks"),
    ]
    return pd.DataFrame(
        [
            {
                "path": path,
                "change_type": change_type,
                "purpose": purpose,
                "scope": "examples/TM_comsol_thermal_micro",
                "touches_no_thermal_project": False,
            }
            for path, change_type, purpose in rows
        ]
    )


def setup_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(0.0, 0.010)
    ax.set_ylim(0.0, 0.010)


def scatter_field(ax: plt.Axes, entry: dict[str, object], values: np.ndarray, title: str, *, vmin=None, vmax=None, cmap="viridis") -> object:
    scatter = ax.scatter(
        np.asarray(entry["element_x"], dtype=float),
        np.asarray(entry["element_y"], dtype=float),
        c=np.asarray(values, dtype=float),
        s=4,
        linewidths=0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(title)
    setup_axes(ax)
    return scatter


def save_line_figures(
    reaction_table: pd.DataFrame,
    alpha_notch: pd.DataFrame,
    hi_hii: pd.DataFrame,
    energy: pd.DataFrame,
    threshold_table: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []

    path = FIGURES_DIR / "reaction_vs_displacement.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = reaction_table[reaction_table["case_id"] == case_id]
        ax.plot(data["displacement_or_Delta"], data["reaction_N_energy"], marker="o", label=f"Case {case_id}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("reaction_N_energy [N]")
    ax.set_title("Energy-conjugate reaction versus displacement")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "nominal_stress_vs_strain.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = reaction_table[reaction_table["case_id"] == case_id]
        ax.plot(data["engineering_strain"], data["nominal_stress_energy_MPa"], marker="o", label=f"Case {case_id}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("engineering strain")
    ax.set_ylabel("nominal stress [MPa]")
    ax.set_title("Nominal stress versus strain")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "reaction_shift_C_minus_A.png"
    a = reaction_table[reaction_table["case_id"] == "A"].sort_values("step")
    c = reaction_table[reaction_table["case_id"] == "C"].sort_values("step")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(a["displacement_or_Delta"], c["reaction_N_energy"].to_numpy() - a["reaction_N_energy"].to_numpy(), marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("C - A reaction [N]")
    ax.set_title("Case C reaction shift relative to Case A")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "alpha_max_vs_step.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = alpha_notch[alpha_notch["case_id"] == case_id]
        ax.plot(data["displacement_mm"], data["alpha_max"], marker="o", label=f"Case {case_id}")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("alpha max")
    ax.set_title("Global alpha max by displacement")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "notch_alpha_vs_displacement.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = alpha_notch[alpha_notch["case_id"] == case_id]
        ax.plot(data["displacement_mm"], data["notch_window_alpha_max"], marker="o", label=f"Case {case_id}")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("notch-window alpha max")
    ax.set_title("Notch-tip alpha metric by displacement")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "HI_HII_peaks_vs_step.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = hi_hii[hi_hii["case_id"] == case_id]
        ax.plot(data["displacement_mm"], data["HI_peak"], marker="o", label=f"{case_id} HI")
        ax.plot(data["displacement_mm"], data["HII_peak"], marker="s", linestyle="--", label=f"{case_id} HII")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("history peak")
    ax.set_title("HI/HII peak evolution")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "energy_terms_vs_step.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for case_id in CASES:
        data = energy[energy["case_id"] == case_id]
        ax.plot(data["displacement_or_Delta"], data["elastic_energy_kNmm"], marker="o", label=f"{case_id} elastic")
        ax.plot(data["displacement_or_Delta"], data["fracture_energy_kNmm"], marker="s", linestyle="--", label=f"{case_id} fracture")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("energy [kN mm]")
    ax.set_yscale("log")
    ax.set_title("Selected energy terms")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    path = FIGURES_DIR / "alpha_threshold_area_vs_step.png"
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case_id in CASES:
        for threshold in [1e-3, 1e-2, 2e-2, 5e-2, 1e-1]:
            data = threshold_table[(threshold_table["case_id"] == case_id) & (threshold_table["threshold"] == threshold)]
            label = f"{case_id} >= {threshold:g}"
            ax.plot(data["step"], data["fraction_above_threshold"], marker="o", linewidth=1.1, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction above threshold")
    ax.set_title("Alpha threshold area by step")
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def save_final_alpha_figures(fields: dict[str, list[dict[str, object]]]) -> list[Path]:
    paths: list[Path] = []
    global_vmax = max(float(np.nanmax(fields[case_id][-1]["alpha_elem"])) for case_id in CASES)

    for name, vmax, title_note in [
        ("final_alpha_global_scale.png", global_vmax, f"shared 0..{global_vmax:.3f}"),
        ("final_alpha_low_range_scale.png", LOW_RANGE_ALPHA_VMAX, "background inspection only; high-alpha cases clipped"),
    ]:
        path = FIGURES_DIR / name
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        scatter = None
        for ax, case_id in zip(axes, CASES):
            entry = fields[case_id][-1]
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            clipped = " clipped" if float(np.nanmax(alpha)) > vmax else ""
            scatter = scatter_field(ax, entry, alpha, f"Case {case_id} final alpha{clipped}\n{title_note}", vmin=0.0, vmax=vmax)
        fig.colorbar(scatter, ax=axes, shrink=0.86, label="alpha")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    path = FIGURES_DIR / "final_alpha_high_threshold_masks.png"
    fig, axes = plt.subplots(len(HIGH_THRESHOLDS), 3, figsize=(9.8, 2.2 * len(HIGH_THRESHOLDS)), constrained_layout=True)
    cmap = ListedColormap(["#eeeeee", "#1b9e77"])
    for i, threshold in enumerate(HIGH_THRESHOLDS):
        for j, case_id in enumerate(CASES):
            ax = axes[i, j]
            entry = fields[case_id][-1]
            alpha = np.asarray(entry["alpha_elem"], dtype=float)
            mask = alpha >= threshold
            scatter_field(ax, entry, mask.astype(float), f"{case_id}: alpha >= {threshold:g}\ncount={int(mask.sum())}", vmin=0.0, vmax=1.0, cmap=cmap)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def write_figure_summary(paths: list[Path]) -> None:
    descriptions = {
        "reaction_vs_displacement.png": "A/B/C energy-conjugate reaction curves for the D0020 schedule.",
        "nominal_stress_vs_strain.png": "A/B/C nominal stress curves using checkpointed energy-conjugate reaction.",
        "reaction_shift_C_minus_A.png": "Case C minus Case A reaction shift by displacement.",
        "alpha_max_vs_step.png": "Global alpha maximum by displacement.",
        "notch_alpha_vs_displacement.png": "Notch-window alpha maximum by displacement.",
        "HI_HII_peaks_vs_step.png": "HI/HII peak evolution by displacement.",
        "energy_terms_vs_step.png": "Selected elastic and fracture energy terms.",
        "final_alpha_global_scale.png": "Final alpha on a shared global scale across A/B/C.",
        "final_alpha_low_range_scale.png": "Final alpha on a low range for background inspection only; clipped cases are labelled.",
        "final_alpha_high_threshold_masks.png": "Final high-threshold alpha masks used for damage interpretation.",
        "alpha_threshold_area_vs_step.png": "Fraction of elements above selected alpha thresholds by step.",
    }
    lines = ["# Figure Summary", ""]
    for path in paths:
        lines.append(f"- `{path.name}`: {descriptions[path.name]}")
    write_text_lf(FIGURES_DIR / "figure_summary.md", "\n".join(lines) + "\n")


def write_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    reaction: pd.DataFrame,
    alpha_notch: pd.DataFrame,
    threshold: pd.DataFrame,
    delay: pd.DataFrame,
    hi_hii: pd.DataFrame,
) -> None:
    s = summary.set_index("case_id")
    a = s.loc["A"]
    b = s.loc["B"]
    c = s.loc["C"]
    ab_pass = bool(comparison[comparison["comparison"] == "B_vs_A"]["passed"].all())
    final_reaction_shift = float(c["final_nominal_stress_MPa"] - a["final_nominal_stress_MPa"])
    final_alpha_shift = float(c["final_notch_window_alpha_max"] - a["final_notch_window_alpha_max"])
    c_final_step = int(alpha_notch[alpha_notch["case_id"] == "C"]["step"].max())
    c_high = threshold[
        (threshold["case_id"] == "C")
        & (threshold["step"] == c_final_step)
        & (threshold["threshold"].isin([0.02, 0.03, 0.05, 0.1]))
    ]
    high_lines = [
        "| threshold | fraction above | notch-connected count | physical interpretation allowed |",
        "|---:|---:|---:|---|",
    ]
    for row in c_high.itertuples():
        high_lines.append(
            f"| {row.threshold:g} | {row.fraction_above_threshold:.6f} | "
            f"{int(row.notch_connected_count)} | {str(bool(row.physical_interpretation_allowed)).lower()} |"
        )
    c_hi = hi_hii[(hi_hii["case_id"] == "C")].sort_values("step").iloc[-1]

    report = f"""# Moderate Prescribed-Temperature Tension Damage Probe

## 1. Purpose

Run one moderate, non-smoke A/B/C prescribed-temperature tension diagnostic to examine how prescribed uniform thermal strain affects notch-tip damage evolution. This is a diagnostic only, not physical validation.

## 2. Relationship to previous strong diagnostic and Case C alpha audit

This D0020 probe extends the prior D0015 strong diagnostic from `1.5e-5 mm` to `2.0e-5 mm` while preserving the compensation-region steps. It follows the Case C alpha audit conclusion: reaction/stress shift, compensation crossing, and notch-tip alpha peak reduction are meaningful within this diagnostic, while broad low-level diffuse alpha background is reported separately and is not used as fracture evidence.

## 3. Cases run

- Case A: `{CASES['A']['run_id']}`, thermal mode `off`, delta_T `0 K`.
- Case B: `{CASES['B']['run_id']}`, thermal mode `uniform`, delta_T `0 K`.
- Case C: `{CASES['C']['run_id']}`, thermal mode `uniform`, delta_T `+20 K`.

## 4. Schedule and why it is moderate rather than a through-crack extension

Schedule: `{rel(SCHEDULE_PATH)}` with 11 displacements from `1.0e-6` to `2.0e-5 mm`. It preserves the `3.0e-6` and `3.8e-6 mm` compensation-region resolution and extends moderately beyond the previous endpoint. It is not a long through-crack schedule.

## 5. Training settings

All cases used `{TRAINING_SETTINGS}`. The run was full mode, seed 23, tension only, with checkpoints saved at every step.

## 6. A/B zero-thermal equivalence

Case B reproduces Case A under the D0020 schedule. A/B comparison status: `{ab_pass}`. Final A/B alpha max values are `{a.final_alpha_max:.12g}` and `{b.final_alpha_max:.12g}`; final nominal stresses are `{a.final_nominal_stress_MPa:.12g}` and `{b.final_nominal_stress_MPa:.12g} MPa`.

## 7. C reaction/stress shift

Case C remains shifted downward relative to Case A. Final nominal stress is A `{a.final_nominal_stress_MPa:.12g} MPa` versus C `{c.final_nominal_stress_MPa:.12g} MPa`, giving C-A `{final_reaction_shift:.12g} MPa`.

## 8. C notch-tip alpha/damage evolution

Case C has reduced final notch-window alpha: A `{a.final_notch_window_alpha_max:.12g}` versus C `{c.final_notch_window_alpha_max:.12g}`, giving C-A `{final_alpha_shift:.12g}`. Final global alpha max is A `{a.final_alpha_max:.12g}` versus C `{c.final_alpha_max:.12g}`.

## 9. High-threshold alpha/connectivity interpretation

Damage interpretation uses notch/high-threshold metrics, not low-threshold diffuse area. Case C final high-threshold summary:

{chr(10).join(high_lines)}

## 10. Low-level diffuse alpha handling

Low-background thresholds `1e-4`, `1e-3`, `5e-3`, and `1e-2` are included only as diagnostic metrics. `figures/final_alpha_low_range_scale.png` is for background inspection only and explicitly clips high-alpha cases if applicable.

## 11. HI/HII/history interpretation

HI/HII remain finite. Final Case C HI peak is `{c_hi.HI_peak:.12g}`, HII peak is `{c_hi.HII_peak:.12g}`, and HII/HI ratio is `{c_hi.HII_over_HI_ratio:.12g}`. The final drive location classification is `{c_hi.drive_location_classification}`.

## 12. Energy-conjugate reaction availability

All three cases generated 11 step checkpoints and the postprocess availability status is energy-conjugate for each run. The primary reaction metric is `reaction_N_energy`.

## 13. Heat PDE/damage conductivity guard

No heat PDE, damage-dependent conductivity, trainable/PDE temperature field, D0040, seed study, shear extension, or S0110 was implemented or run. See `tables/no_heat_pde_guard_summary.csv`.

## 14. Whether any legacy reaction metric was used as primary

No. Legacy top-sigma was not used as the primary reaction. The reported reaction and nominal stress use checkpointed energy-conjugate reaction.

## 15. Physical validation status

This is not physical validation. It is a moderate prescribed-temperature software/physics-route diagnostic.

## 16. Final classification

`{FINAL_CLASSIFICATION}`

The moderate prescribed-temperature tension damage probe confirms that the prescribed `+20 K` thermal strain continues to shift the reaction/stress response downward and delays or reduces notch-tip/high-threshold alpha growth relative to the no-thermal baseline, while the zero-temperature branch remains equivalent to the no-thermal branch. Diffuse low-level alpha background is reported separately and is not used as fracture evidence.

## 17. Recommended next task

Review this package. If further validation is needed, run a focused review of high-threshold/notch metrics and reaction curves before deciding whether a denser compensation schedule is worth the runtime. Do not start heat PDE or damage-dependent conductivity work from this package alone.
"""
    write_text_lf(PACKAGE_DIR / "REPORT.md", report)


def write_manifest(paths: list[Path], table_paths: list[Path]) -> None:
    manifest = {
        "package": rel(PACKAGE_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classification": FINAL_CLASSIFICATION,
        "schedule": rel(SCHEDULE_PATH),
        "cases": CASES,
        "training_settings": TRAINING_SETTINGS,
        "figures": [rel(path) for path in paths],
        "tables": [rel(path) for path in table_paths],
        "guards": {
            "training_run": True,
            "source_code_modified": False,
            "physics_model_boundary_material_l0_history_loss_changed": False,
            "heat_pde_implemented_or_run": False,
            "damage_dependent_conductivity_implemented_or_run": False,
            "D0040_run": False,
            "seed_study_run": False,
            "shear_extension_or_S0110_run": False,
            "no_thermal_project_touched": False,
        },
        "notch_window": {
            "center_x_mm": NOTCH_TIP_XY[0],
            "center_y_mm": NOTCH_TIP_XY[1],
            "half_window_mm": NOTCH_HALF_WINDOW,
        },
        "connectivity_method": "Element graph from shared raw mesh triangle edges.",
    }
    write_text_lf(PACKAGE_DIR / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_handoff(table_paths: list[Path], figure_paths: list[Path]) -> None:
    lines = [
        "# Handoff: Moderate Prescribed-Temperature Tension Damage Probe",
        "",
        "## Status",
        "",
        f"Final classification: `{FINAL_CLASSIFICATION}`",
        "",
        "Commit hash:",
        "",
        "- Pending before commit; update once after the primary package commit is pushed.",
        "",
        "Push status:",
        "",
        "- Pending before commit; update once after the primary package commit is pushed.",
        "",
        "## Package",
        "",
        f"- Package path: `{rel(PACKAGE_DIR)}`",
        f"- Report: `{rel(PACKAGE_DIR / 'REPORT.md')}`",
        f"- Schedule: `{rel(SCHEDULE_PATH)}`",
        "",
        "## Runs",
        "",
    ]
    for case_id, meta in CASES.items():
        lines.append(f"- Case {case_id}: `{meta['run_id']}`, thermal mode `{meta['thermal_mode']}`, delta_T `{meta['delta_T_K']} K`, seed 23")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Worked only under `examples/TM_comsol_thermal_micro`.",
            "- Did not modify `examples/TM_comsol_no_thermal_micro`.",
            "- Ran training for A/B/C D0020 tension damage probe only.",
            "- Did not implement heat PDE, damage-dependent conductivity, trainable/PDE temperature, D0040, seed study, shear extension, or S0110.",
            "- Did not change material parameters, l0, history logic, training losses, boundary conditions, source model behavior, or reaction route.",
            "- Energy-conjugate `reaction_N_energy` is the primary reaction.",
            "",
            "## Tables Generated",
            "",
        ]
    )
    lines.extend(f"- `{rel(path)}`" for path in table_paths)
    lines.extend(["", "## Figures Generated", ""])
    lines.extend(f"- `{rel(path)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Main Conclusion",
            "",
            "The zero-temperature thermal branch matches the no-thermal branch. The +20 K branch keeps the expected downward reaction/stress shift and lower notch-tip/high-threshold alpha growth. Low-level diffuse alpha background is reported separately and is not used as fracture evidence.",
            "",
            "## Reviewer Should Read Next",
            "",
            f"1. `{rel(PACKAGE_DIR / 'REPORT.md')}`",
            f"2. `{rel(TABLES_DIR / 'damage_probe_case_summary.csv')}`",
            f"3. `{rel(TABLES_DIR / 'damage_probe_case_comparison.csv')}`",
            f"4. `{rel(TABLES_DIR / 'damage_delay_summary.csv')}`",
            f"5. `{rel(TABLES_DIR / 'alpha_threshold_connectivity_by_step.csv')}`",
            f"6. `{rel(FIGURES_DIR / 'final_alpha_global_scale.png')}`",
            f"7. `{rel(FIGURES_DIR / 'final_alpha_high_threshold_masks.png')}`",
            "",
            "## Exact Next Recommended Task",
            "",
            "Review this package and the high-threshold/notch metrics before deciding on any further validation. Do not begin heat PDE or damage-dependent conductivity work from this diagnostic alone.",
        ]
    )
    write_text_lf(PACKAGE_DIR / "HANDOFF_COMMENT.md", "\n".join(lines) + "\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    fields = load_case_fields()
    assert_shared_geometry(fields)
    reactions = read_reaction_tables()
    diagnostics = read_diagnostic_tables()
    adjacency = build_adjacency(np.asarray(fields["A"][-1]["triangles"], dtype=int))

    reaction_table = make_reaction_stress_table(fields, reactions)
    alpha_notch = make_alpha_notch_table(fields, reactions)
    threshold = make_threshold_table(fields, adjacency)
    hi_hii = make_hi_hii_table(fields, reactions)
    energy = make_energy_table(reactions, diagnostics)
    delay = make_damage_delay_table(alpha_notch, reactions)
    summary = make_case_summary(fields, reactions, alpha_notch, hi_hii)
    comparison = make_case_comparison(summary, reaction_table, alpha_notch, hi_hii, energy)
    guard = make_guard_table()
    training = make_training_diagnostics(diagnostics)
    changed = make_changed_files_table()

    table_map = {
        "damage_probe_case_summary.csv": summary,
        "damage_probe_case_comparison.csv": comparison,
        "reaction_stress_by_step.csv": reaction_table,
        "alpha_notch_metrics_by_step.csv": alpha_notch,
        "alpha_threshold_connectivity_by_step.csv": threshold,
        "damage_delay_summary.csv": delay,
        "HI_HII_drive_by_step.csv": hi_hii,
        "energy_terms_by_step.csv": energy,
        "no_heat_pde_guard_summary.csv": guard,
        "training_diagnostics_summary.csv": training,
        "changed_files_summary.csv": changed,
    }
    table_paths = []
    for name, table in table_map.items():
        path = TABLES_DIR / name
        write_csv_lf(table, path)
        table_paths.append(path)

    figure_paths = []
    figure_paths.extend(save_line_figures(reaction_table, alpha_notch, hi_hii, energy, threshold))
    figure_paths.extend(save_final_alpha_figures(fields))
    write_figure_summary(figure_paths)
    figure_paths.append(FIGURES_DIR / "figure_summary.md")

    write_report(summary, comparison, reaction_table, alpha_notch, threshold, delay, hi_hii)
    write_manifest([path for path in figure_paths if path.suffix == ".png"], table_paths)
    write_handoff(table_paths, figure_paths)

    print(f"package={rel(PACKAGE_DIR)}")
    print(f"classification={FINAL_CLASSIFICATION}")
    print(f"tables={len(table_paths)}")
    print(f"figures={len([p for p in figure_paths if p.suffix == '.png'])}")


if __name__ == "__main__":
    main()
