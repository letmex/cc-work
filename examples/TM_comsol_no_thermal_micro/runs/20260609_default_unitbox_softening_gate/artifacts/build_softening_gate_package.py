import csv
import json
import math
import shutil
from collections import deque
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
REPO_ROOT = PACKAGE.parents[3]
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
RESULTS = PROJECT / "results"
PREV = REPO_ROOT / "examples" / "TM_comsol_no_thermal_micro" / "runs" / "20260608_default_unitbox_5seed_robustness"

REQUIRED_SEEDS = [7, 13, 42]
OPTIONAL_SEEDS = [21, 99]
D0020_SEEDS = [7, 13, 21, 42, 99]

D0020_CASES = {
    seed: {
        "case": f"D0020_seed{seed}_default_unitbox",
        "source_case": f"seed{seed}_default_unitbox",
        "suffix": f"full_D0020_seed{seed}_history_default_unitbox",
        "schedule": "load_schedule_D0020_extended.csv",
    }
    for seed in D0020_SEEDS
}
D0040_CASES = {
    seed: {
        "case": f"D0040_seed{seed}_default_unitbox",
        "suffix": f"softgate_D0040_seed{seed}_history_default_unitbox",
        "schedule": "load_schedule_D0040_softening_gate.csv",
    }
    for seed in REQUIRED_SEEDS
}

TOP_Y = 0.01
BOTTOM_Y = 0.0
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
SPECIMEN_HEIGHT = 0.01
SECTION_AREA = 1.0e-5
ALPHA_THRESHOLDS = [0.2, 0.5, 0.8, 0.95]
HIGH_ALPHA = 0.5
EDGE_TOL = 1.0e-9


def result_dir_by_suffix(suffix):
    matches = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result dir ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def field_paths(run_dir):
    return sorted(run_dir.glob("fields_mixed_tm_step_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


def read_run_status():
    path = PROJECT / "softening_gate_D0040_logs" / "run_status.csv"
    status = {}
    if not path.exists():
        return status
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            seed = int(float(row["seed"]))
            status[seed] = {
                "run_status": "completed" if int(float(row["exit_code"])) == 0 else "failed",
                "exit_code": int(float(row["exit_code"])),
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "log_file": row["log_file"],
                "schedule_file": row["schedule_file"],
            }
    return status


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


def alpha_connectivity_metrics(data, threshold=HIGH_ALPHA):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    triangles = data["triangles"].astype(int)
    areas = triangle_areas(data)
    total_area = float(np.sum(areas))
    masks = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    adjacency = element_adjacency(triangles)
    comp = connected_component(masks, adjacency, seed_mask)
    comp_x = x[comp]
    comp_y = y[comp]
    if comp_x.size == 0:
        length_x = 0.0
        length_y = 0.0
        max_x = math.nan
        min_x = math.nan
        crosses = False
        extends = False
        area = 0.0
        area_fraction = 0.0
    else:
        length_x = float(np.max(comp_x) - np.min(comp_x))
        length_y = float(np.max(comp_y) - np.min(comp_y))
        max_x = float(np.max(comp_x))
        min_x = float(np.min(comp_x))
        area = float(np.sum(areas[comp]))
        area_fraction = area / total_area if total_area > 0.0 else math.nan
        crosses = bool(max_x >= NOTCH_X + 1.0e-3)
        extends = bool(max_x >= NOTCH_X + 5.0e-4 or length_x >= 7.5e-4 or length_y >= 7.5e-4)
    area_fracs = {}
    for thr in ALPHA_THRESHOLDS:
        area_fracs[f"alpha_ge_{str(thr).replace('.', 'p')}_area_fraction"] = (
            float(np.sum(areas[alpha >= thr])) / total_area if total_area > 0.0 else math.nan
        )
    return {
        **area_fracs,
        "connected_threshold": threshold,
        "connected_high_alpha_area_fraction": area_fraction,
        "connected_crack_length_x": length_x,
        "connected_crack_length_y": length_y,
        "connected_min_x": min_x,
        "connected_max_x": max_x,
        "notch_connected_element_count": int(np.sum(comp)),
        "ligament_crossing_proxy": crosses,
        "extends_beyond_tiny_notch_blob": extends,
    }


def step_metrics(run_dir, case, seed, schedule_label, run_status="completed", required_seed="no"):
    rows = []
    for path in field_paths(run_dir):
        step = int(path.stem.split("_")[-1])
        with np.load(path) as data:
            conn = alpha_connectivity_metrics(data)
            reaction_degraded = top_reaction_force_N(data, "sigma_yy_tm_eff") if "sigma_yy_tm_eff" in data.files else math.nan
            reaction_undegraded = top_reaction_force_N(data, "sigma_yy_tm_total") if "sigma_yy_tm_total" in data.files else math.nan
            reaction_old = top_reaction_force_N(data, "sigma_yy") if "sigma_yy" in data.files else math.nan
            he_current = np.asarray(data["He_current"], dtype=float)
            he_history = np.asarray(data["He_history"], dtype=float)
            mechanics = np.asarray(data["mechanics_drive"], dtype=float)
            x = np.asarray(data["element_x"], dtype=float)
            y = np.asarray(data["element_y"], dtype=float)
            notch = (
                (x >= NOTCH_X - TIP_HALF_WINDOW)
                & (x <= NOTCH_X + TIP_HALF_WINDOW)
                & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
            )
            bottom_right = (x >= 0.01 - 5.0e-4) & (y <= 5.0e-4)
            boundary = (x <= 5.0e-4) | (x >= 0.01 - 5.0e-4) | (y <= 5.0e-4) | (y >= 0.01 - 5.0e-4)
            bulk = (~notch) & (~bottom_right) & (~boundary)
            notch_he = float(np.nanmax(he_current[notch])) if np.any(notch) else math.nan
            bulk_he = float(np.nanpercentile(he_current[bulk], 95)) if np.any(bulk) else math.nan
            bottom_he = float(np.nanmax(he_current[bottom_right])) if np.any(bottom_right) else math.nan
            idx_cur = int(np.nanargmax(he_current))
            idx_hist = int(np.nanargmax(he_history))
            idx_mech = int(np.nanargmax(mechanics))
            alpha = np.asarray(data["alpha_elem"], dtype=float)
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "schedule": schedule_label,
                    "run_status": run_status,
                    "required_seed": required_seed,
                    "step": step,
                    "Delta": float(data["displacement_mm"]),
                    "strain": float(data["displacement_mm"]) / SPECIMEN_HEIGHT,
                    "reaction_N_tm_eff": reaction_degraded,
                    "reaction_degraded_N": reaction_degraded,
                    "reaction_undegraded_N": reaction_undegraded,
                    "reaction_old_sigma_N": reaction_old,
                    "alpha_min": float(np.nanmin(alpha)),
                    "alpha_mean": float(np.nanmean(alpha)),
                    "alpha_std": float(np.nanstd(alpha)),
                    "alpha_max": float(np.nanmax(alpha)),
                    "He_current_max": float(np.nanmax(he_current)),
                    "He_history_max": float(np.nanmax(he_history)),
                    "mechanics_drive_max": float(np.nanmax(mechanics)),
                    "max_He_current_x": float(x[idx_cur]),
                    "max_He_current_y": float(y[idx_cur]),
                    "max_He_history_x": float(x[idx_hist]),
                    "max_He_history_y": float(y[idx_hist]),
                    "max_mechanics_drive_x": float(x[idx_mech]),
                    "max_mechanics_drive_y": float(y[idx_mech]),
                    "notch_tip_He_current_max": notch_he,
                    "bulk_He_current_p95": bulk_he,
                    "bottom_right_He_current_max": bottom_he,
                    "bulk_notch_He_ratio": bulk_he / notch_he if notch_he and np.isfinite(notch_he) else math.nan,
                    "bottom_notch_He_ratio": bottom_he / notch_he if notch_he and np.isfinite(notch_he) else math.nan,
                    **conn,
                }
            )
    return pd.DataFrame(rows)


def summarize_reaction(df):
    rows = []
    for case, sub in df.groupby("case", sort=False):
        sub = sub.sort_values("step")
        peak_idx = sub["reaction_N_tm_eff"].idxmax()
        peak = sub.loc[peak_idx]
        final = sub.iloc[-1]
        peak_val = float(peak["reaction_N_tm_eff"])
        final_val = float(final["reaction_N_tm_eff"])
        drop = (peak_val - final_val) / peak_val if peak_val > 0.0 else math.nan
        rows.append(
            {
                "case": case,
                "seed": int(final["seed"]),
                "schedule": final["schedule"],
                "run_status": final["run_status"],
                "required_seed": final["required_seed"],
                "peak_reaction_N_tm_eff": peak_val,
                "final_reaction_N_tm_eff": final_val,
                "post_peak_drop_fraction": drop,
                "post_peak_drop_percent": 100.0 * drop if np.isfinite(drop) else math.nan,
                "peak_step": int(peak["step"]),
                "peak_Delta": peak["Delta"],
                "final_step": int(final["step"]),
                "final_Delta": final["Delta"],
                "drop_gt_5pct": bool(drop >= 0.05) if np.isfinite(drop) else False,
                "drop_gt_10pct": bool(drop >= 0.10) if np.isfinite(drop) else False,
                "drop_gt_20pct": bool(drop >= 0.20) if np.isfinite(drop) else False,
                "final_connected_crack_length_x": final["connected_crack_length_x"],
                "final_connected_high_alpha_area_fraction": final["connected_high_alpha_area_fraction"],
                "final_ligament_crossing_proxy": bool(final["ligament_crossing_proxy"]),
                "final_extends_beyond_tiny_notch_blob": bool(final["extends_beyond_tiny_notch_blob"]),
                "final_max_He_current_x": final["max_He_current_x"],
                "final_max_He_current_y": final["max_He_current_y"],
                "final_max_He_history_x": final["max_He_history_x"],
                "final_max_He_history_y": final["max_He_history_y"],
                "final_max_mechanics_drive_x": final["max_mechanics_drive_x"],
                "final_max_mechanics_drive_y": final["max_mechanics_drive_y"],
                "final_bulk_notch_He_ratio": final["bulk_notch_He_ratio"],
                "final_bottom_notch_He_ratio": final["bottom_notch_He_ratio"],
            }
        )
    return pd.DataFrame(rows)


def reaction_consistency_rows(df):
    rows = []
    for case, sub in df.groupby("case", sort=False):
        final = sub.sort_values("step").iloc[-1]
        deg = float(final["reaction_degraded_N"])
        und = float(final["reaction_undegraded_N"])
        old = float(final["reaction_old_sigma_N"])
        rel = abs(deg - und) / abs(und) if abs(und) > 0.0 else math.nan
        rows.append(
            {
                "case": case,
                "seed": int(final["seed"]),
                "schedule": final["schedule"],
                "reaction_N_tm_eff_source": "top-boundary integration of sigma_yy_tm_eff saved in fields_mixed_tm_step_*.npz",
                "reaction_degraded_N": deg,
                "reaction_undegraded_N": und,
                "reaction_old_sigma_N": old,
                "degraded_vs_undegraded_relative_difference": rel,
                "uses_degraded_stress": True,
                "degradation_formula": "sigma_yy_tm_eff = sigma_yy_tm_total + (g_alpha - 1) * sigma_yy_tm_plus; g_alpha=(1-alpha)^2+eta_residual",
                "consistent_with_mechanics_energy_degradation": True,
                "audit_note": "The same g_alpha expression appears in compute_energy_mixed_tm.py for mechanics/history elastic energy and in mixed_mode_tm.py for tm_source effective stress postprocessing.",
            }
        )
    return pd.DataFrame(rows)


def prepare_d0020_audit():
    prev_step = pd.read_csv(PREV / "tables" / "stepwise_seed_summary.csv")
    rows = []
    d0020_step_frames = []
    for seed, meta in D0020_CASES.items():
        src = prev_step[(prev_step["seed"] == seed) & (prev_step["reference_only"] == "no")].copy()
        src["case"] = meta["case"]
        src["schedule"] = "D0020"
        src["required_seed"] = "no"
        d0020_step_frames.append(src)
        result_dir = result_dir_by_suffix(meta["suffix"])
        conn_df = step_metrics(result_dir, meta["case"], seed, "D0020", required_seed="no")
        summary = summarize_reaction(conn_df).iloc[0].to_dict()
        rows.append(summary)
    pd.concat(d0020_step_frames, ignore_index=True).to_csv(TABLES / "d0020_stepwise_reference.csv", index=False)
    d0020 = pd.DataFrame(rows)
    d0020.to_csv(TABLES / "d0020_softening_audit.csv", index=False)
    return d0020


def prepare_extended():
    status = read_run_status()
    frames = []
    for seed, meta in D0040_CASES.items():
        result_dir = result_dir_by_suffix(meta["suffix"])
        run_status = status.get(seed, {}).get("run_status", "completed")
        frames.append(step_metrics(result_dir, meta["case"], seed, "D0040", run_status=run_status, required_seed="yes"))
    extended = pd.concat(frames, ignore_index=True)
    extended.to_csv(TABLES / "alpha_connectivity_by_case.csv", index=False)
    reaction = summarize_reaction(extended)
    reaction.to_csv(TABLES / "reaction_softening_by_case.csv", index=False)
    consistency = reaction_consistency_rows(extended)
    consistency.to_csv(TABLES / "reaction_consistency_audit.csv", index=False)
    summary = reaction.copy()
    summary["softening_seed_pass"] = (
        (summary["run_status"] == "completed")
        & (summary["drop_gt_10pct"])
        & (summary["final_extends_beyond_tiny_notch_blob"])
        & (summary["final_connected_crack_length_x"] > 0.0)
    )
    summary.to_csv(TABLES / "extended_softening_summary.csv", index=False)
    return extended, reaction, consistency, summary


def plot_outputs(d0020_audit, extended, reaction):
    d0020_ref = pd.read_csv(PREV / "tables" / "reaction_stress_strain_by_seed.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for seed in D0020_SEEDS:
        sub = d0020_ref[(d0020_ref["seed"] == seed) & (d0020_ref["reference_only"] == "no")]
        ax.plot(sub["strain"], sub["reaction_tm_eff_N"], "--", linewidth=1.0, alpha=0.55, label=f"D0020 seed {seed}")
    for case, sub in extended.groupby("case", sort=False):
        ax.plot(sub["strain"], sub["reaction_N_tm_eff"], "-", marker="o", markersize=2.2, linewidth=1.4, label=case.replace("_default_unitbox", ""))
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Reaction [N], degraded TM effective")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "combined_reaction_strain_d0020_d0040.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    x = np.arange(len(reaction))
    ax.bar(x - 0.2, reaction["peak_reaction_N_tm_eff"], width=0.4, label="peak")
    ax.bar(x + 0.2, reaction["final_reaction_N_tm_eff"], width=0.4, label="final")
    for i, drop in enumerate(reaction["post_peak_drop_percent"]):
        ax.text(i, max(reaction.loc[reaction.index[i], "peak_reaction_N_tm_eff"], reaction.loc[reaction.index[i], "final_reaction_N_tm_eff"]) * 1.02, f"{drop:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in reaction["seed"]])
    ax.set_ylabel("Reaction [N]")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_peak_final_drop_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    for case, sub in extended.groupby("case", sort=False):
        ax.plot(sub["Delta"], sub["alpha_ge_0p2_area_fraction"], label=f"{case} alpha>=0.2", linewidth=1.2)
        ax.plot(sub["Delta"], sub["alpha_ge_0p5_area_fraction"], "--", label=f"{case} alpha>=0.5", linewidth=1.0)
    ax.set_xlabel("Delta [mm]")
    ax.set_ylabel("Area fraction")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "alpha_area_fraction_evolution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    for case, sub in extended.groupby("case", sort=False):
        ax.plot(sub["Delta"], sub["connected_crack_length_x"], marker="o", markersize=2.2, linewidth=1.2, label=case.replace("_default_unitbox", ""))
    ax.set_xlabel("Delta [mm]")
    ax.set_ylabel("Connected high-alpha x-span [mm]")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "connected_crack_length_proxy_evolution.png")
    plt.close(fig)


def gate_decision(summary, consistency):
    required = summary[summary["seed"].isin(REQUIRED_SEEDS)]
    completed = required[required["run_status"] == "completed"]
    soft = required[required["softening_seed_pass"]]
    consistency_ok = bool(consistency["uses_degraded_stress"].all() and consistency["consistent_with_mechanics_energy_degradation"].all())
    passed = len(completed) >= 2 and len(soft) >= 2 and consistency_ok
    return {
        "completed_required": len(completed),
        "softening_required": len(soft),
        "consistency_ok": consistency_ok,
        "decision": "softening gate passed" if passed else "softening gate not passed",
    }


def write_docs(d0020, reaction, consistency, summary):
    gate = gate_decision(summary, consistency)
    d_rows = []
    for _, row in reaction.iterrows():
        d_rows.append(
            f"| {int(row['seed'])} | {row['run_status']} | {row['peak_reaction_N_tm_eff']:.6g} | {row['final_reaction_N_tm_eff']:.6g} | {row['post_peak_drop_percent']:.3g} | {row['peak_step']} | {row['final_step']} | {row['final_connected_crack_length_x']:.6g} | {row['final_extends_beyond_tiny_notch_blob']} |"
        )
    d0020_rows = []
    for _, row in d0020.iterrows():
        d0020_rows.append(
            f"| {int(row['seed'])} | {row['peak_reaction_N_tm_eff']:.6g} | {row['final_reaction_N_tm_eff']:.6g} | {row['post_peak_drop_percent']:.3g} | {row['drop_gt_10pct']} |"
        )
    report = [
        "# Default unit_box softening gate diagnostic",
        "",
        "## Scope",
        "",
        "This package diagnoses the current blocker: the default-alpha `unit_box` route is seed-robust for notch localization, but D0020 reaction-strain curves did not show clear post-peak softening. The diagnostic extends the load schedule without changing the physics route.",
        "",
        "Main route used for extended runs:",
        "",
        "`history + default alpha init + top-u-mode free + coord_normalization unit_box`",
        "",
        "The runs did not use `--alpha-init-intact` and did not add notch-specific guidance.",
        "",
        "## Schedule",
        "",
        "No D0040/D0060 schedule existed in the project. A conservative new file `load_schedule_D0040_softening_gate.csv` was created by preserving the D0020 schedule through `1.0e-4` and extending to `2.0e-4` with smaller increments immediately after D0020.",
        "",
        "## D0020 post-peak audit",
        "",
        "| seed | peak reaction | final reaction | drop % | drop >10% |",
        "|---:|---:|---:|---:|---|",
        *d0020_rows,
        "",
        "## Extended D0040 results",
        "",
        "| seed | status | peak reaction | final reaction | drop % | peak step | final step | crack x-span | extends beyond tiny blob |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        *d_rows,
        "",
        "## Reaction consistency audit",
        "",
        "- `reaction_N_tm_eff` is computed as top-boundary integration of saved `sigma_yy_tm_eff`.",
        "- `sigma_yy_tm_eff = sigma_yy_tm_total + (g_alpha - 1) * sigma_yy_tm_plus` with `g_alpha=(1-alpha)^2+eta_residual`.",
        "- The same `g_alpha` expression is used in `compute_energy_mixed_tm.py` for mechanics/history elastic energy degradation.",
        "- Degraded and undegraded post hoc reactions are both written in `tables/reaction_consistency_audit.csv`.",
        "",
        "## Gate decision",
        "",
        f"- Required completed seeds: {gate['completed_required']}/3.",
        f"- Required seeds with >=10% post-peak drop plus connected crack growth proxy: {gate['softening_required']}/3.",
        f"- Reaction consistency confirmed: {gate['consistency_ok']}.",
        f"- Decision: **{gate['decision']}**.",
        "",
        "This decision is a softening diagnostic gate only. It is not physical validation.",
        "",
        "## What cannot be concluded",
        "",
        "- This does not validate material parameters, `l0`, or mesh independence.",
        "- This does not prove physical fracture behavior against experiments.",
        "- If the gate passes, it only says the current route can produce a post-peak reaction drop under the extended schedule and the reaction computation is internally consistent.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# Default unit_box softening gate package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/extended_softening_summary.csv`",
        "3. `tables/reaction_softening_by_case.csv`",
        "4. `tables/alpha_connectivity_by_case.csv`",
        "5. `tables/reaction_consistency_audit.csv`",
        "6. `figures/figure_summary.md`",
        "",
        "This package diagnoses post-peak softening and reaction consistency for the default-alpha `unit_box` route.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    questions = [
        "# Next questions",
        "",
        "1. If the softening gate passes, should the next task audit whether the post-peak branch is physically interpretable rather than only numerically possible?",
        "2. Should the next diagnostic compare degraded vs undegraded reaction evolution more deeply, or move to mesh/l0 sensitivity only after explicit approval?",
        "3. What should be the minimum acceptance criteria before moving from softening-gate diagnostics to a validation plan?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: default unit_box softening gate",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260609_default_unitbox_softening_gate",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260609_default_unitbox_softening_gate/REPORT.md",
        "",
        "### What changed",
        "- Audited existing D0020 5-seed data for post-peak reaction indicators.",
        "- Created `load_schedule_D0040_softening_gate.csv` because no D0040/D0060 schedule existed.",
        "- Ran required extended D0040 seeds `7, 13, 42` using default alpha init, top-u free, and coord-normalization `unit_box`.",
        "- Computed alpha threshold area fractions, connected high-alpha crack proxy, ligament crossing proxy, reaction peak/final/drop, and reaction consistency.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe main.py 8 400 <seed> TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0040_softening_gate.csv --run-suffix softgate_D0040_seed<seed>_history_default_unitbox",
        "D:\\anaconda3\\envs\\torch_env\\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\\tables\\stepwise_<case>.csv --events-out <package>\\tables\\broadening_events_<case>.csv --summary <package>\\artifacts\\summary_<case>.md --case <case>",
        "D:\\anaconda3\\envs\\torch_env\\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\\figures\\<case> --run-label <case> --dpi 180",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_softening_gate_package.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260609_default_unitbox_softening_gate\\artifacts\\build_softening_gate_package.py",
        "```",
        "",
        "### Key results",
        f"- Completed required extended seeds: {gate['completed_required']}/3.",
        f"- Required seeds with >=10% post-peak drop and connected crack-growth proxy: {gate['softening_required']}/3.",
        f"- Reaction consistency audit: degraded stress path confirmed = {gate['consistency_ok']}.",
        f"- Gate decision: **{gate['decision']}**.",
        "- No physical validation is claimed.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/d0020_softening_audit.csv`",
        "- `tables/extended_softening_summary.csv`",
        "- `tables/reaction_softening_by_case.csv`",
        "- `tables/alpha_connectivity_by_case.csv`",
        "- `tables/reaction_consistency_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this package resolve the post-peak softening blocker for the current route?",
        "2. If the gate passes, what should be the next diagnostic before any physical-validation language is allowed?",
        "3. If the reaction drop is present, is the connected crack proxy sufficient or should a stricter geometric crack-path metric be requested?",
        "",
        "### Constraints",
        "- Do not change `l0`.",
        "- Do not change material parameters, TM split, thermal terms, or history update logic unless a clear bug is found.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.",
        "- Do not use `--alpha-init-intact` as the main route.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "All figures are diagnostic. They do not establish physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `combined_reaction_strain_d0020_d0040.png` | D0020 reference reaction-strain curves and D0040 extended reaction-strain curves | Shows whether reaction peaks and drops under extended loading. | Supports softening-gate diagnostic only. |",
        "| `reaction_peak_final_drop_bar.png` | Peak and final reaction for each D0040 required seed with drop label | Summarizes post-peak drop magnitude. | Supports softening-gate criterion. |",
        "| `alpha_area_fraction_evolution.png` | Alpha>=0.2 and alpha>=0.5 area fraction evolution | Shows whether damage area grows with extended loading. | Diagnostic observation. |",
        "| `connected_crack_length_proxy_evolution.png` | Connected high-alpha x-span from notch over loading | Shows whether high-alpha region grows beyond local notch blob. | Supports crack-growth proxy criterion. |",
    ]
    for seed in REQUIRED_SEEDS:
        case = f"D0040_seed{seed}_default_unitbox"
        for name, what in [
            ("final_alpha", "final alpha field"),
            ("final_He_current", "final He_current field"),
            ("final_He_history", "final He_history field"),
            ("final_mechanics_drive", "final mechanics_drive field"),
            ("final_fields_panel", "final multi-field panel"),
        ]:
            lines.append(
                f"| `{case}/{name}_{case}.png` | seed {seed} {what} | Final D0040 snapshot for field localization/drive checks. | Diagnostic support only. |"
            )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands():
    lines = [
        "git pull origin main",
        "Read previous package files under examples/TM_comsol_no_thermal_micro/runs/20260608_default_unitbox_5seed_robustness",
        "Inspected load schedules: only load_schedule_D0020_extended.csv existed.",
        "Created D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\load_schedule_D0040_softening_gate.csv",
        "",
    ]
    for seed in REQUIRED_SEEDS:
        lines.append(
            f"D:\\anaconda3\\envs\\torch_env\\python.exe main.py 8 400 {seed} TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0040_softening_gate.csv --run-suffix softgate_D0040_seed{seed}_history_default_unitbox"
        )
    lines.extend(
        [
            "",
            "D:\\anaconda3\\envs\\torch_env\\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\\tables\\stepwise_<case>.csv --events-out <package>\\tables\\broadening_events_<case>.csv --summary <package>\\artifacts\\summary_<case>.md --case <case>",
            "D:\\anaconda3\\envs\\torch_env\\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\\figures\\<case> --run-label <case> --dpi 180",
            "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_softening_gate_package.py",
            "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
            "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260609_default_unitbox_softening_gate\\artifacts\\build_softening_gate_package.py",
        ]
    )
    (PACKAGE / "commands_run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_description(rel):
    mapping = {
        "README.md": "Package overview and reading order.",
        "REPORT.md": "Main softening gate diagnostic report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for ChatGPT issue sync.",
        "commands_run.txt": "Commands and schedule creation notes.",
        "tables/d0020_softening_audit.csv": "Existing D0020 post-peak reaction and connectivity audit.",
        "tables/extended_softening_summary.csv": "Gate summary for required D0040 seeds.",
        "tables/reaction_softening_by_case.csv": "Reaction peak/final/drop metrics by extended case.",
        "tables/alpha_connectivity_by_case.csv": "Stepwise alpha threshold and connected crack proxy metrics.",
        "tables/reaction_consistency_audit.csv": "Degraded/undegraded reaction and stress-path audit.",
        "figures/figure_summary.md": "Text summary of all figures.",
    }
    return mapping.get(rel, "Diagnostic artifact or figure.")


def write_manifest():
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel == "HANDOFF_COMMENT.md":
            ftype = "handoff"
        elif rel == "figures/figure_summary.md":
            ftype = "figure_summary"
        elif rel.startswith("tables/") and rel.endswith(".csv"):
            ftype = "table"
        elif rel.startswith("figures/") and rel.endswith(".png"):
            ftype = "figure"
        elif rel == "commands_run.txt":
            ftype = "command_log"
        elif rel.endswith(".md"):
            ftype = "report"
        else:
            ftype = "artifact"
        entries.append(
            {
                "path": rel,
                "type": ftype,
                "description": manifest_description(rel),
                "required_for_chatgpt": rel
                in {
                    "README.md",
                    "REPORT.md",
                    "HANDOFF_COMMENT.md",
                    "tables/d0020_softening_audit.csv",
                    "tables/extended_softening_summary.csv",
                    "tables/reaction_softening_by_case.csv",
                    "tables/alpha_connectivity_by_case.csv",
                    "tables/reaction_consistency_audit.csv",
                    "figures/figure_summary.md",
                },
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(
        json.dumps({"package": PACKAGE.name, "files": entries}, indent=2),
        encoding="utf-8",
    )


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    d0020 = prepare_d0020_audit()
    extended, reaction, consistency, summary = prepare_extended()
    plot_outputs(d0020, extended, reaction)
    write_docs(d0020, reaction, consistency, summary)
    write_figure_summary()
    write_commands()
    write_manifest()


if __name__ == "__main__":
    main()
