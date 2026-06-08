import argparse
import csv
import re
from pathlib import Path

import numpy as np


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
CORNER_WINDOW_MM = 5.0e-4
BOUNDARY_BAND_MM = 5.0e-4


def _safe_array(npz, key, fallback=None):
    if key in npz.files:
        return np.asarray(npz[key], dtype=float)
    if fallback is not None and fallback in npz.files:
        return np.asarray(npz[fallback], dtype=float)
    return None


def _safe_float(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _read_diagnostics(path):
    diag = {}
    if not path.exists():
        return diag
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row.get("step", len(diag))))
            diag[step] = row
    return diag


def _read_displacements(path):
    values = {}
    if not path.exists():
        return values
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "step" in reader.fieldnames:
            for row in reader:
                values[int(float(row["step"]))] = _safe_float(
                    row.get("displacement_mm") or row.get("Delta")
                )
            return values
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1 and raw.size >= 2:
        values[int(raw[0])] = float(raw[1])
    elif raw.ndim == 2:
        for row in raw:
            values[int(row[0])] = float(row[1])
    return values


def _mask_regions(x, y):
    notch_tip = (
        (x >= NOTCH_TIP_X_MM - TIP_HALF_WINDOW_MM)
        & (x <= NOTCH_TIP_X_MM + TIP_HALF_WINDOW_MM)
        & (np.abs(y - NOTCH_CENTER_Y_MM) <= TIP_HALF_WINDOW_MM)
    )
    bottom_right = (
        (x >= SPECIMEN_SIZE_MM - BOTTOM_RIGHT_WINDOW_MM)
        & (x <= SPECIMEN_SIZE_MM)
        & (y >= 0.0)
        & (y <= BOTTOM_RIGHT_WINDOW_MM)
    )
    corner = (
        ((x <= CORNER_WINDOW_MM) | (x >= SPECIMEN_SIZE_MM - CORNER_WINDOW_MM))
        & ((y <= CORNER_WINDOW_MM) | (y >= SPECIMEN_SIZE_MM - CORNER_WINDOW_MM))
    )
    boundary = (
        (x <= BOUNDARY_BAND_MM)
        | (x >= SPECIMEN_SIZE_MM - BOUNDARY_BAND_MM)
        | (y <= BOUNDARY_BAND_MM)
        | (y >= SPECIMEN_SIZE_MM - BOUNDARY_BAND_MM)
    )
    bulk = (~notch_tip) & (~bottom_right) & (~corner) & (~boundary)
    return {
        "notch_tip": notch_tip,
        "bottom_right": bottom_right,
        "bulk": bulk,
    }


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "max": np.nan,
            "min": np.nan,
            "p95": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _region_stat(row, name, values, mask, prefix):
    subset = values[mask] if values is not None and mask.size == values.size else np.array([])
    s = _stats(subset)
    row[f"{prefix}_{name}_mean"] = s["mean"]
    row[f"{prefix}_{name}_max"] = s["max"]
    row[f"{prefix}_{name}_p95"] = s["p95"]


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return np.nan
    return float(num / den)


def _max_location(values, x, y):
    if values is None or values.size == 0:
        return np.nan, np.nan, np.nan
    idx = int(np.nanargmax(values))
    return float(np.nanmax(values)), float(x[idx]), float(y[idx])


def _alpha_element(npz):
    alpha_elem = _safe_array(npz, "alpha_elem")
    if alpha_elem is not None:
        return alpha_elem
    alpha = _safe_array(npz, "alpha")
    triangles = _safe_array(npz, "triangles")
    if alpha is None or triangles is None or triangles.size == 0:
        return alpha
    triangles = triangles.astype(int)
    return (alpha[triangles[:, 0]] + alpha[triangles[:, 1]] + alpha[triangles[:, 2]]) / 3.0


def _step_from_path(path):
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def analyze(run_dir, out_csv, events_csv, summary_md, case_name):
    run_dir = Path(run_dir)
    diag = _read_diagnostics(run_dir / "diagnostics_mixed_tm_summary.csv")
    displacements = _read_displacements(run_dir / "displacement_list.csv")
    npz_paths = sorted(run_dir.glob("fields_mixed_tm_step_*.npz"), key=_step_from_path)
    rows = []

    for path in npz_paths:
        step = _step_from_path(path)
        with np.load(path) as npz:
            x = _safe_array(npz, "element_x")
            y = _safe_array(npz, "element_y")
            alpha_elem = _alpha_element(npz)
            he_current = _safe_array(npz, "He_current", "He")
            he_history = _safe_array(npz, "He_history", "He")
            mechanics_drive = _safe_array(npz, "mechanics_drive", "He")
            if x is None or y is None:
                raise ValueError(f"{path} does not contain element_x/element_y")
            masks = _mask_regions(x, y)
            row = {
                "case": case_name,
                "step": step,
                "Delta": _safe_float(diag.get(step, {}).get("Delta")),
            }
            if not np.isfinite(row["Delta"]):
                row["Delta"] = displacements.get(step, np.nan)

            a_stats = _stats(alpha_elem)
            row.update(
                {
                    "alpha_min": a_stats["min"],
                    "alpha_mean": a_stats["mean"],
                    "alpha_std": a_stats["std"],
                    "alpha_max": a_stats["max"],
                    "alpha_gt_0p5_area_fraction": float(np.mean(alpha_elem > 0.5))
                    if alpha_elem is not None and alpha_elem.size
                    else np.nan,
                }
            )
            for name, values in (
                ("He_current", he_current),
                ("He_history", he_history),
                ("mechanics_drive", mechanics_drive),
            ):
                s = _stats(values)
                row[f"{name}_mean"] = s["mean"]
                row[f"{name}_std"] = s["std"]
                row[f"{name}_max"] = s["max"]
                max_val, max_x, max_y = _max_location(values, x, y)
                row[f"max_{name}"] = max_val
                row[f"max_{name}_x"] = max_x
                row[f"max_{name}_y"] = max_y

            for prefix, mask in masks.items():
                _region_stat(row, "alpha", alpha_elem, mask, prefix)
                _region_stat(row, "He_current", he_current, mask, prefix)
                _region_stat(row, "mechanics_drive", mechanics_drive, mask, prefix)

            row["bulk_He_current_p95_over_notch_tip_He_current_max"] = _ratio(
                row["bulk_He_current_p95"], row["notch_tip_He_current_max"]
            )
            row["bottom_right_He_current_max_over_notch_tip_He_current_max"] = _ratio(
                row["bottom_right_He_current_max"], row["notch_tip_He_current_max"]
            )
            row["bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max"] = _ratio(
                row["bulk_mechanics_drive_p95"], row["notch_tip_mechanics_drive_max"]
            )
            row["bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max"] = _ratio(
                row["bottom_right_mechanics_drive_max"], row["notch_tip_mechanics_drive_max"]
            )
            row["bulk_alpha_mean_over_notch_tip_alpha_max"] = _ratio(
                row["bulk_alpha_mean"], row["notch_tip_alpha_max"]
            )
            row["bottom_right_alpha_max_over_notch_tip_alpha_max"] = _ratio(
                row["bottom_right_alpha_max"], row["notch_tip_alpha_max"]
            )
            diag_row = diag.get(step, {})
            for key in (
                "elastic_energy",
                "fracture_energy",
                "loss_total",
                "loss_log10",
                "reaction_N_tm_eff",
                "macro_stress",
                "macro_strain",
            ):
                row[key] = _safe_float(diag_row.get(key))
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case",
        "step",
        "Delta",
        "alpha_min",
        "alpha_mean",
        "alpha_std",
        "alpha_max",
        "alpha_gt_0p5_area_fraction",
        "He_current_mean",
        "He_current_std",
        "He_current_max",
        "He_history_mean",
        "He_history_std",
        "He_history_max",
        "mechanics_drive_mean",
        "mechanics_drive_std",
        "mechanics_drive_max",
        "max_He_current_x",
        "max_He_current_y",
        "max_He_history_x",
        "max_He_history_y",
        "max_mechanics_drive_x",
        "max_mechanics_drive_y",
        "notch_tip_alpha_max",
        "notch_tip_alpha_mean",
        "bulk_alpha_mean",
        "bulk_alpha_p95",
        "bottom_right_alpha_max",
        "bottom_right_alpha_mean",
        "notch_tip_He_current_max",
        "notch_tip_He_current_mean",
        "bulk_He_current_p95",
        "bulk_He_current_mean",
        "bottom_right_He_current_max",
        "bottom_right_He_current_mean",
        "notch_tip_mechanics_drive_max",
        "notch_tip_mechanics_drive_mean",
        "bulk_mechanics_drive_p95",
        "bulk_mechanics_drive_mean",
        "bottom_right_mechanics_drive_max",
        "bottom_right_mechanics_drive_mean",
        "bulk_He_current_p95_over_notch_tip_He_current_max",
        "bottom_right_He_current_max_over_notch_tip_He_current_max",
        "bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max",
        "bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max",
        "bulk_alpha_mean_over_notch_tip_alpha_max",
        "bottom_right_alpha_max_over_notch_tip_alpha_max",
        "reaction_N_tm_eff",
        "elastic_energy",
        "fracture_energy",
        "loss_total",
        "loss_log10",
    ]
    ordered = [c for c in preferred if c in fieldnames] + [c for c in fieldnames if c not in preferred]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    event_defs = [
        ("first_alpha_mean_gt_0p05", "alpha_mean", 0.05),
        ("first_alpha_mean_gt_0p1", "alpha_mean", 0.1),
        ("first_alpha_mean_gt_0p2", "alpha_mean", 0.2),
        ("first_alpha_mean_gt_0p4", "alpha_mean", 0.4),
        ("first_bulk_He_ratio_gt_0p25", "bulk_He_current_p95_over_notch_tip_He_current_max", 0.25),
        ("first_bulk_He_ratio_gt_0p5", "bulk_He_current_p95_over_notch_tip_He_current_max", 0.5),
        ("first_bulk_drive_ratio_gt_0p25", "bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max", 0.25),
        ("first_bulk_drive_ratio_gt_0p5", "bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max", 0.5),
        ("first_bottom_He_ratio_gt_0p5", "bottom_right_He_current_max_over_notch_tip_He_current_max", 0.5),
        ("first_bottom_drive_ratio_gt_0p5", "bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max", 0.5),
    ]
    events = []
    for event, key, threshold in event_defs:
        hit = None
        for row in rows:
            val = row.get(key, np.nan)
            if np.isfinite(val) and val > threshold:
                hit = row
                break
        events.append(
            {
                "case": case_name,
                "event": event,
                "metric": key,
                "threshold": threshold,
                "step": hit["step"] if hit else "",
                "Delta": hit["Delta"] if hit else "",
                "value": hit[key] if hit else "",
            }
        )
    with events_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "event", "metric", "threshold", "step", "Delta", "value"])
        writer.writeheader()
        writer.writerows(events)

    event_by_name = {e["event"]: e for e in events}
    alpha_event = event_by_name["first_alpha_mean_gt_0p05"]["step"]
    he_event = event_by_name["first_bulk_He_ratio_gt_0p25"]["step"]
    drive_event = event_by_name["first_bulk_drive_ratio_gt_0p25"]["step"]
    if alpha_event == "" and he_event == "" and drive_event == "":
        causal_label = "insufficient event crossings"
    elif alpha_event != "" and (he_event == "" or int(alpha_event) < int(he_event)) and (
        drive_event == "" or int(alpha_event) < int(drive_event)
    ):
        causal_label = "A: alpha broadens before bulk drive ratios"
    elif he_event != "" and (alpha_event == "" or int(he_event) < int(alpha_event)):
        causal_label = "B: He_current broadens before alpha threshold"
    elif drive_event != "" and (alpha_event == "" or int(drive_event) < int(alpha_event)):
        causal_label = "B: mechanics_drive broadens before alpha threshold"
    elif alpha_event != "" and (he_event == alpha_event or drive_event == alpha_event):
        causal_label = "C: alpha and drive broaden in the same step"
    else:
        causal_label = "E: mixed or unclear ordering"

    final = rows[-1] if rows else {}
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write(f"# Drive Broadening Summary: {case_name}\n\n")
        handle.write(f"Run directory: `{run_dir}`\n\n")
        handle.write(f"Steps analyzed: {len(rows)}\n\n")
        handle.write("## Event Ordering\n\n")
        handle.write(f"Ordering label: **{causal_label}**\n\n")
        handle.write("| event | step | Delta | value |\n|---|---:|---:|---:|\n")
        for e in events:
            handle.write(f"| {e['event']} | {e['step']} | {e['Delta']} | {e['value']} |\n")
        handle.write("\n## Final Step Snapshot\n\n")
        for key in (
            "step",
            "Delta",
            "alpha_mean",
            "alpha_std",
            "alpha_max",
            "bulk_He_current_p95_over_notch_tip_He_current_max",
            "bottom_right_He_current_max_over_notch_tip_He_current_max",
            "bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max",
            "bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max",
            "reaction_N_tm_eff",
        ):
            handle.write(f"- {key}: {final.get(key, np.nan)}\n")
        handle.write("\nThis diagnostic does not claim physical validation.\n")

    return rows, events, causal_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--events-out", default=None)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--case", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out)
    events_out = Path(args.events_out) if args.events_out else out.with_name(out.stem.replace("stepwise_summary", "broadening_events") + ".csv")
    case = args.case or run_dir.name
    analyze(run_dir, out, events_out, Path(args.summary), case)


if __name__ == "__main__":
    main()
