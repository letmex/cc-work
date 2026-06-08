import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
BOUNDARY_TOL_MM = 1.0e-9
RATIO_VALID_NOTCH_HE_THRESHOLD = 1.0e-8


def _safe_array(npz, key):
    if key not in npz.files:
        return None
    return np.asarray(npz[key], dtype=float)


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": np.nan, "mean": np.nan, "max": np.nan, "std": np.nan, "p95": np.nan}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return np.nan
    return float(num / den)


def _region_masks(x, y):
    notch = (
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
    boundary = (
        (x <= 5.0e-4)
        | (x >= SPECIMEN_SIZE_MM - 5.0e-4)
        | (y <= 5.0e-4)
        | (y >= SPECIMEN_SIZE_MM - 5.0e-4)
    )
    bulk = (~notch) & (~bottom_right) & (~boundary)
    return {"notch_tip": notch, "bottom_right": bottom_right, "bulk": bulk}


def _nearest_node_pair_rows(x, y, u, v):
    upper = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM + 5.0e-5)) <= BOUNDARY_TOL_MM)
    )[0]
    lower = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM - 5.0e-5)) <= BOUNDARY_TOL_MM)
    )[0]
    rows = []
    for idx in upper:
        if lower.size == 0:
            break
        j = lower[int(np.argmin(np.abs(x[lower] - x[idx])))]
        rows.append(
            {
                "x": float(x[idx]),
                "upper_node": int(idx),
                "lower_node": int(j),
                "gap_y": float(y[idx] - y[j]),
                "u_jump_upper_minus_lower": float(u[idx] - u[j]),
                "v_jump_upper_minus_lower": float(v[idx] - v[j]),
                "upper_u": float(u[idx]),
                "lower_u": float(u[j]),
                "upper_v": float(v[idx]),
                "lower_v": float(v[j]),
            }
        )
    return rows


def _field_summary(npz_path, case):
    with np.load(npz_path) as npz:
        x_elem = _safe_array(npz, "element_x")
        y_elem = _safe_array(npz, "element_y")
        x = _safe_array(npz, "x")
        y = _safe_array(npz, "y")
        u = _safe_array(npz, "u")
        v = _safe_array(npz, "v")
        alpha = _safe_array(npz, "alpha_elem")
        he = _safe_array(npz, "He_current")
        mechanics = _safe_array(npz, "mechanics_drive")
        eps_xx = _safe_array(npz, "eps_xx")
        eps_yy = _safe_array(npz, "eps_yy")
        eps_xy = _safe_array(npz, "eps_xy")
        if any(a is None for a in (x_elem, y_elem, x, y, u, v, alpha, he, mechanics)):
            raise ValueError(f"{npz_path} is missing required fields")

        masks = _region_masks(x_elem, y_elem)
        he_idx = int(np.nanargmax(he))
        mech_idx = int(np.nanargmax(mechanics))
        notch_he = _stats(he[masks["notch_tip"]])["max"]
        bulk_he_p95 = _stats(he[masks["bulk"]])["p95"]
        bottom_he = _stats(he[masks["bottom_right"]])["max"]
        notch_mech = _stats(mechanics[masks["notch_tip"]])["max"]
        bulk_mech_p95 = _stats(mechanics[masks["bulk"]])["p95"]
        bottom_mech = _stats(mechanics[masks["bottom_right"]])["max"]
        strain_grad_proxy = np.sqrt(
            np.gradient(eps_xx.astype(float)) ** 2
            + np.gradient(eps_yy.astype(float)) ** 2
            + np.gradient(eps_xy.astype(float)) ** 2
        )
        top_mask = np.abs(y - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL_MM
        bottom_mask = np.abs(y) <= BOUNDARY_TOL_MM
        notch_pairs = _nearest_node_pair_rows(x, y, u, v)
        u_jumps = np.array([r["u_jump_upper_minus_lower"] for r in notch_pairs], dtype=float)
        v_jumps = np.array([r["v_jump_upper_minus_lower"] for r in notch_pairs], dtype=float)
        displacement = float(npz["displacement_mm"]) if "displacement_mm" in npz.files else np.nan
        row = {
            "case": case,
            "npz": str(npz_path),
            "step": _step_from_path(npz_path),
            "Delta": displacement,
            "alpha_min": _stats(alpha)["min"],
            "alpha_mean": _stats(alpha)["mean"],
            "alpha_std": _stats(alpha)["std"],
            "alpha_max": _stats(alpha)["max"],
            "He_current_mean": _stats(he)["mean"],
            "He_current_std": _stats(he)["std"],
            "He_current_max": _stats(he)["max"],
            "max_He_current_x": float(x_elem[he_idx]),
            "max_He_current_y": float(y_elem[he_idx]),
            "mechanics_drive_max": _stats(mechanics)["max"],
            "max_mechanics_drive_x": float(x_elem[mech_idx]),
            "max_mechanics_drive_y": float(y_elem[mech_idx]),
            "notch_tip_He_current_max": notch_he,
            "bulk_He_current_p95": bulk_he_p95,
            "bottom_right_He_current_max": bottom_he,
            "bulk_to_notch_He_current": _ratio(bulk_he_p95, notch_he),
            "bottom_to_notch_He_current": _ratio(bottom_he, notch_he),
            "ratio_valid": bool(np.isfinite(notch_he) and notch_he > RATIO_VALID_NOTCH_HE_THRESHOLD),
            "notch_tip_mechanics_drive_max": notch_mech,
            "bulk_mechanics_drive_p95": bulk_mech_p95,
            "bottom_right_mechanics_drive_max": bottom_mech,
            "bulk_to_notch_mechanics_drive": _ratio(bulk_mech_p95, notch_mech),
            "bottom_to_notch_mechanics_drive": _ratio(bottom_mech, notch_mech),
            "top_u_abs_max": float(np.max(np.abs(u[top_mask]))) if np.any(top_mask) else np.nan,
            "top_v_error_max": float(np.max(np.abs(v[top_mask] - displacement))) if np.any(top_mask) else np.nan,
            "bottom_u_abs_max": float(np.max(np.abs(u[bottom_mask]))) if np.any(bottom_mask) else np.nan,
            "bottom_v_abs_max": float(np.max(np.abs(v[bottom_mask]))) if np.any(bottom_mask) else np.nan,
            "notch_lip_pairs": len(notch_pairs),
            "notch_lip_u_jump_abs_max": float(np.max(np.abs(u_jumps))) if u_jumps.size else np.nan,
            "notch_lip_v_jump_abs_max": float(np.max(np.abs(v_jumps))) if v_jumps.size else np.nan,
            "notch_lip_u_jump_std": float(np.std(u_jumps)) if u_jumps.size else np.nan,
            "notch_lip_v_jump_std": float(np.std(v_jumps)) if v_jumps.size else np.nan,
            "eps_xx_std": _stats(eps_xx)["std"] if eps_xx is not None else np.nan,
            "eps_yy_std": _stats(eps_yy)["std"] if eps_yy is not None else np.nan,
            "eps_xy_std": _stats(eps_xy)["std"] if eps_xy is not None else np.nan,
            "strain_grad_proxy_p95": _stats(strain_grad_proxy)["p95"],
            "strain_grad_proxy_max": _stats(strain_grad_proxy)["max"],
        }
        return row, notch_pairs


def _step_from_path(path):
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_command(cmd, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(str(c) for c in cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log.write(proc.stdout)
        log.write(f"\nRETURN_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with {proc.returncode}: {' '.join(str(c) for c in cmd)}")


def _latest_result_dir(suffix):
    candidates = [p for p in (ROOT / "results").glob(f"*{suffix}") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_budget_sweep(args):
    if args.skip_budget_sweep:
        return []
    rows = []
    for rprop in args.rprop_budgets:
        suffix = f"step0_budget_rprop{rprop}"
        cmd = [
            sys.executable,
            "main.py",
            "8",
            "400",
            "2",
            "TrainableReLU",
            "3.0",
            "--full",
            "--pff-model",
            "AT2",
            "--mixed-mechanics-mode",
            "history",
            "--alpha-init-intact",
            "--load-schedule-file",
            "load_schedule_D0020_extended.csv",
            "--n-rprop",
            str(rprop),
            "--n-lbfgs",
            "0",
            "--max-steps",
            "1",
            "--run-suffix",
            suffix,
        ]
        log_path = args.out_dir / "logs" / f"{suffix}.log"
        try:
            _run_command(cmd, ROOT, log_path)
        except RuntimeError as exc:
            rows.append(
                {
                    "case": f"budget_rprop_{rprop}",
                    "rprop_budget": rprop,
                    "status": "failed",
                    "failure": str(exc),
                    "log": str(log_path),
                }
            )
            continue
        run_dir = _latest_result_dir(suffix)
        if run_dir is None:
            raise FileNotFoundError(f"Could not locate result directory for suffix {suffix}")
        npz = run_dir / "fields_mixed_tm_step_0000.npz"
        row, _pairs = _field_summary(npz, f"budget_rprop_{rprop}")
        row["rprop_budget"] = rprop
        row["status"] = "ok"
        row["failure"] = ""
        row["result_dir"] = str(run_dir)
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--topufree-run", type=Path, required=True)
    parser.add_argument("--old-history-run", type=Path, required=True)
    parser.add_argument("--rprop-budgets", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--skip-budget-sweep", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = args.out_dir / "tables"
    rows = []
    pair_rows = []
    for case, run_dir in (
        ("alpha_intact_history_topufixed_step0", args.fixed_run),
        ("alpha_intact_history_topufree_step0", args.topufree_run),
        ("old_history_default_alpha_step0", args.old_history_run),
    ):
        row, pairs = _field_summary(run_dir / "fields_mixed_tm_step_0000.npz", case)
        row["result_dir"] = str(run_dir)
        rows.append(row)
        for pair in pairs:
            pair_rows.append({"case": case, **pair})
    _write_rows(tables / "step0_field_summary.csv", rows)
    _write_rows(tables / "notch_lip_node_pairs.csv", pair_rows)

    budget_rows = run_budget_sweep(args)
    if budget_rows:
        _write_rows(tables / "optimizer_budget_step0_summary.csv", budget_rows)

    summary_path = args.out_dir / "reports" / "step0_root_cause_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Step-0 Root-Cause Diagnostic Summary\n\n")
        handle.write("This diagnostic is minimal and does not change `l0`, material parameters, TM split, phase-field notch seeding, alpha=1 seeding, thermal field, or history update logic.\n\n")
        handle.write("## Saved Full-Run Step-0 Fields\n\n")
        handle.write("| case | alpha_mean | notch_He | bulk_He_p95 | bulk/notch | bottom/notch | max_He_x | max_He_y | notch_lip_u_jump_abs_max | notch_lip_v_jump_abs_max |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['case']} | {row['alpha_mean']} | {row['notch_tip_He_current_max']} | "
                f"{row['bulk_He_current_p95']} | {row['bulk_to_notch_He_current']} | "
                f"{row['bottom_to_notch_He_current']} | {row['max_He_current_x']} | "
                f"{row['max_He_current_y']} | {row['notch_lip_u_jump_abs_max']} | "
                f"{row['notch_lip_v_jump_abs_max']} |\n"
            )
        if budget_rows:
            handle.write("\n## Optimizer Budget Step-0 Sweep\n\n")
            handle.write("| rprop_budget | alpha_mean | notch_He | bulk_He_p95 | bulk/notch | bottom/notch | max_He_x | max_He_y |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in budget_rows:
                handle.write(
                    f"| {row['rprop_budget']} | {row['alpha_mean']} | {row['notch_tip_He_current_max']} | "
                    f"{row['bulk_He_current_p95']} | {row['bulk_to_notch_He_current']} | "
                    f"{row['bottom_to_notch_He_current']} | {row['max_He_current_x']} | "
                    f"{row['max_He_current_y']} |\n"
                )
        handle.write("\n## Interpretation Bounds\n\n")
        handle.write("This script reports evidence for ansatz and optimizer-path diagnostics only. It does not claim physical validation.\n")
    print(f"wrote {tables / 'step0_field_summary.csv'}")
    if budget_rows:
        print(f"wrote {tables / 'optimizer_budget_step0_summary.csv'}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
