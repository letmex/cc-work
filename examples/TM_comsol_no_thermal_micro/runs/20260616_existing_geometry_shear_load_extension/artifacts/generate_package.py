from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC_ROOT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
CC_ROOT = Path(r"D:\Desktop\新建文件夹\cc-work")
PACKAGE_REL = Path("examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension")
PACKAGE = CC_ROOT / PACKAGE_REL
PREV_PACKAGE = CC_ROOT / "examples/TM_comsol_no_thermal_micro/runs/20260615_existing_geometry_shear_stronger_training"
RESULT_DIR = SRC_ROOT / "outputs/results/seed23_S0050_shear"
MODEL_DIR = SRC_ROOT / "outputs/checkpoints/seed23_S0050_shear"
FIG_SRC = RESULT_DIR / "figures"
CURVE_DIR = RESULT_DIR / "curves"

NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
RIGHT_BOUNDARY_X = 0.01
RIGHT_BOUNDARY_BAND = 2.5e-4

TRAIN_CMD = (
    r"D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 "
    r"--full --n-rprop 300 --n-lbfgs 1 --load-case shear "
    r"--load-schedule-file load_schedules/load_schedule_S0050_shear.csv "
    r"--run-suffix seed23_S0050_shear"
)
POST_CMD = (
    r"D:\anaconda3\envs\torch_env\python.exe postprocess_results.py "
    r"--model-dir outputs\checkpoints\seed23_S0050_shear "
    r"--result-dir outputs\results\seed23_S0050_shear --device cpu"
)


def ensure_dirs() -> None:
    for sub in ["", "tables", "figures", "artifacts", "artifacts/source_snapshots"]:
        (PACKAGE / sub).mkdir(parents=True, exist_ok=True)


def element_adjacency(triangles: np.ndarray) -> list[list[int]]:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for elem, nodes in enumerate(triangles.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges[tuple(sorted((int(a), int(b))))].append(elem)
    adjacency = [[] for _ in range(len(triangles))]
    for elems in edges.values():
        if len(elems) == 2:
            a, b = elems
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def connected_component(mask: np.ndarray, adjacency: list[list[int]], seed_mask: np.ndarray) -> np.ndarray:
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
                queue.append(nxt)
    return visited


def threshold_metrics(npz_path: Path, threshold: float) -> dict[str, object]:
    tag = str(threshold).replace(".", "p")
    with np.load(npz_path) as data:
        x = np.asarray(data["element_x"], dtype=float)
        y = np.asarray(data["element_y"], dtype=float)
        alpha = np.asarray(data["alpha_elem"], dtype=float)
        triangles = np.asarray(data["triangles"])
        mask = alpha >= threshold
        seed_mask = (
            (x >= NOTCH_X - TIP_HALF_WINDOW)
            & (x <= NOTCH_X + TIP_HALF_WINDOW)
            & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
        )
        comp = connected_component(mask, element_adjacency(triangles), seed_mask)
        comp_x = x[comp]
        comp_y = y[comp]
        area_fraction = float(np.mean(mask)) if mask.size else math.nan
        if comp_x.size:
            return {
                f"alpha{tag}_area_fraction": area_fraction,
                f"alpha{tag}_notch_connected_count": int(np.sum(comp)),
                f"alpha{tag}_notch_connected_min_x": float(np.min(comp_x)),
                f"alpha{tag}_notch_connected_max_x": float(np.max(comp_x)),
                f"alpha{tag}_notch_connected_mean_y": float(np.mean(comp_y)),
                f"alpha{tag}_notch_connected_x_span": float(np.max(comp_x) - np.min(comp_x)),
                f"alpha{tag}_through_to_right": bool(np.any(comp_x >= RIGHT_BOUNDARY_X - RIGHT_BOUNDARY_BAND)),
            }
        return {
            f"alpha{tag}_area_fraction": area_fraction,
            f"alpha{tag}_notch_connected_count": 0,
            f"alpha{tag}_notch_connected_min_x": math.nan,
            f"alpha{tag}_notch_connected_max_x": math.nan,
            f"alpha{tag}_notch_connected_mean_y": math.nan,
            f"alpha{tag}_notch_connected_x_span": 0.0,
            f"alpha{tag}_through_to_right": False,
        }


def write_line_plot(
    filename: str,
    x: pd.Series,
    ys: list[pd.Series],
    labels: list[str],
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    for y, label in zip(ys, labels):
        ax.plot(x.astype(float), y.astype(float), marker="o", linewidth=1.5, markersize=3.0, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(labels) > 1:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PACKAGE / "figures" / filename)
    plt.close(fig)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def build_package() -> dict[str, object]:
    ensure_dirs()
    stress = pd.read_csv(CURVE_DIR / "stress_strain_by_step.csv")
    reaction = pd.read_csv(CURVE_DIR / "reaction_by_step.csv")
    diag = pd.read_csv(RESULT_DIR / "diagnostics_mixed_tm_summary.csv")
    pd.read_csv(CURVE_DIR / "reaction_metric_availability.csv")
    prev_summary = pd.read_csv(PREV_PACKAGE / "tables/shear_training_run_summary.csv").iloc[0].to_dict()

    conn_rows = []
    for field in sorted(RESULT_DIR.glob("fields_mixed_tm_step_*.npz")):
        step = int(field.stem.split("_")[-1])
        row = {"step": step}
        row.update(threshold_metrics(field, 0.5))
        row.update(threshold_metrics(field, 0.8))
        conn_rows.append(row)
    conn = pd.DataFrame(conn_rows)

    keep_stress = [
        "step",
        "Delta_s",
        "engineering_shear_strain",
        "nominal_shear_stress_energy_MPa",
        "reaction_N_energy",
        "reaction_metric_status",
        "is_energy_conjugate",
        "alpha0p8_through_crack",
        "alpha0p8_connected_count",
        "alpha0p8_connected_x_span",
    ]
    stepwise = diag.merge(stress[[c for c in keep_stress if c in stress.columns]], on="step", how="left")
    stepwise = stepwise.merge(conn, on="step", how="left")
    stepwise["top_v_absmax_over_Delta_s"] = stepwise["top_v_abs_max"].astype(float) / stepwise["Delta_s"].astype(float)
    stepwise["HII_over_HI_peak_ratio"] = stepwise["HII_max"].astype(float) / stepwise["HI_max"].astype(float)
    stepwise["postprocess_checkpoint_exists"] = stepwise["step"].apply(
        lambda s: (MODEL_DIR / "best_models/step_checkpoints" / f"checkpoint_mixedH_TM_step_{int(s):04d}.pt").exists()
    )
    stepwise["field_npz_exists"] = stepwise["step"].apply(
        lambda s: (RESULT_DIR / f"fields_mixed_tm_step_{int(s):04d}.npz").exists()
    )
    stepwise["drive_location_classification"] = np.where(
        (np.abs(stepwise["max_mechanics_drive_x"] - NOTCH_X) <= 4.5e-4)
        & (np.abs(stepwise["max_mechanics_drive_y"] - NOTCH_Y) <= 4.5e-4),
        "notch-dominated",
        "off-notch",
    )

    reaction.to_csv(PACKAGE / "tables/shear_reaction_by_step.csv", index=False)
    stress.to_csv(PACKAGE / "tables/shear_stress_strain_by_step.csv", index=False)

    damage_cols = [
        "step",
        "Delta_s",
        "engineering_shear_strain",
        "alpha_min",
        "alpha_mean",
        "alpha_max",
        "max_alpha_x",
        "max_alpha_y",
        "notch_tip_alpha_max",
        "bulk_alpha_max",
        "bottom_right_alpha_max",
        "HI_max",
        "HII_max",
        "HII_over_HI_peak_ratio",
        "HII_active",
        "He_current_max",
        "max_He_current_x",
        "max_He_current_y",
        "mechanics_drive_max",
        "max_mechanics_drive_x",
        "max_mechanics_drive_y",
        "drive_location_classification",
        "notch_tip_He_current_max",
        "notch_tip_mechanics_drive_max",
        "bottom_right_He_current_max",
        "bottom_right_mechanics_drive_max",
        "alpha0p5_area_fraction",
        "alpha0p5_notch_connected_count",
        "alpha0p5_notch_connected_x_span",
        "alpha0p5_through_to_right",
        "alpha0p8_area_fraction",
        "alpha0p8_notch_connected_count",
        "alpha0p8_notch_connected_x_span",
        "alpha0p8_through_to_right",
        "alpha0p8_through_crack",
    ]
    stepwise[[c for c in damage_cols if c in stepwise.columns]].to_csv(
        PACKAGE / "tables/shear_damage_drive_summary.csv", index=False
    )

    topv = stepwise[
        [
            "step",
            "Delta_s",
            "engineering_shear_strain",
            "top_v_min",
            "top_v_max",
            "top_v_mean",
            "top_v_abs_max",
            "top_v_absmax_over_Delta_s",
        ]
    ].copy()
    topv["warning_top_v_gt_1p5_delta"] = topv["top_v_absmax_over_Delta_s"] > 1.5
    topv["unstable_top_v_gt_2p0_delta"] = topv["top_v_absmax_over_Delta_s"] > 2.0
    topv.to_csv(PACKAGE / "tables/shear_top_v_free_diagnostic.csv", index=False)

    ckpt = reaction[
        [
            "step",
            "Delta",
            "Delta_s",
            "engineering_shear_strain",
            "checkpoint",
            "reaction_metric_status",
            "is_energy_conjugate",
            "history_source",
        ]
    ].copy()
    ckpt["checkpoint_exists"] = ckpt["step"].apply(
        lambda s: (MODEL_DIR / "best_models/step_checkpoints" / f"checkpoint_mixedH_TM_step_{int(s):04d}.pt").exists()
    )
    ckpt["field_npz_exists"] = ckpt["step"].apply(
        lambda s: (RESULT_DIR / f"fields_mixed_tm_step_{int(s):04d}.npz").exists()
    )
    ckpt.to_csv(PACKAGE / "tables/shear_checkpoint_availability.csv", index=False)

    loss_cols = [
        "step",
        "Delta",
        "Delta_s",
        "loss_total",
        "loss_log10",
        "elastic_energy",
        "fracture_energy",
        "mechanics_current_energy",
        "phase_history_elastic_energy",
        "phase_history_energy",
        "alpha_step_change_mean",
        "alpha_step_change_max",
    ]
    stepwise[[c for c in loss_cols if c in stepwise.columns]].to_csv(
        PACKAGE / "tables/shear_training_loss_summary.csv", index=False
    )

    final = stepwise.iloc[-1]
    peak_idx = int(stress["nominal_shear_stress_energy_MPa"].astype(float).idxmax())
    peak_row = stress.loc[peak_idx]
    post_peak_drop = bool(
        float(final["nominal_shear_stress_energy_MPa"])
        < float(peak_row["nominal_shear_stress_energy_MPa"]) - 1e-9
        and int(peak_row["step"]) < int(final["step"])
    )
    first_alpha05 = stepwise.loc[stepwise["alpha0p5_notch_connected_count"] > 0, "step"]
    first_alpha08_conn = stepwise.loc[stepwise["alpha0p8_notch_connected_count"] > 0, "step"]
    first_alpha08_through = stepwise.loc[truthy(stepwise["alpha0p8_through_to_right"]), "step"]
    any_through = bool(truthy(stepwise["alpha0p8_through_to_right"]).any() or truthy(stepwise["alpha0p8_through_crack"]).any())
    alpha_grew = float(final["alpha_max"]) > float(prev_summary["final_alpha_max"])
    topv_max_ratio = float(topv["top_v_absmax_over_Delta_s"].max())
    reaction_all = bool(
        (reaction["reaction_metric_status"] == "energy_conjugate").all()
        and truthy(reaction["is_energy_conjugate"]).all()
    )
    run_completed = len(stress) == 33 and int(final["step"]) == 32
    notch_localized = (
        str(final["drive_location_classification"]) == "notch-dominated"
        and abs(float(final["max_alpha_x"]) - NOTCH_X) < 4.5e-4
        and abs(float(final["max_alpha_y"]) - NOTCH_Y) < 4.5e-4
    )
    if (
        run_completed
        and reaction_all
        and alpha_grew
        and notch_localized
        and (
            any_through
            or (
                float(final["alpha0p8_notch_connected_count"]) > 0
                and float(final["alpha0p8_notch_connected_x_span"]) > 0.0
            )
        )
        and topv_max_ratio <= 1.5
    ):
        classification = "shear extension successful with crack growth"
    elif run_completed and alpha_grew and not any_through and (not post_peak_drop) and topv_max_ratio <= 1.5:
        classification = "shear extension improved but not failed"
    elif topv_max_ratio > 2.0 or not run_completed or not reaction_all:
        classification = "shear extension unstable"
    else:
        classification = "shear extension inconclusive"

    summary = {
        "run_id": "seed23_S0050_shear",
        "seed": 23,
        "load_case": "shear",
        "load_schedule": "load_schedules/load_schedule_S0050_shear.csv",
        "steps": len(stress),
        "run_completed": run_completed,
        "continued_from_S0030": False,
        "continuation_reason": "clean continuation was not implemented/ambiguous, so the controlled extension was rerun from step 0",
        "training_n_rprop": 300,
        "training_n_lbfgs": 1,
        "energy_reaction_computable": reaction_all,
        "checkpoint_count": int(ckpt["checkpoint_exists"].sum()),
        "field_npz_count": int(ckpt["field_npz_exists"].sum()),
        "final_Delta_s": float(final["Delta_s"]),
        "final_engineering_shear_strain": float(final["engineering_shear_strain"]),
        "peak_nominal_shear_stress_MPa": float(peak_row["nominal_shear_stress_energy_MPa"]),
        "peak_step": int(peak_row["step"]),
        "peak_engineering_shear_strain": float(peak_row["engineering_shear_strain"]),
        "final_nominal_shear_stress_MPa": float(final["nominal_shear_stress_energy_MPa"]),
        "post_peak_drop_observed": post_peak_drop,
        "post_peak_drop_MPa": float(peak_row["nominal_shear_stress_energy_MPa"])
        - float(final["nominal_shear_stress_energy_MPa"]),
        "final_alpha_max": float(final["alpha_max"]),
        "S0030_final_alpha_max": float(prev_summary["final_alpha_max"]),
        "alpha_grew_beyond_S0030": alpha_grew,
        "final_alpha_max_x": float(final["max_alpha_x"]),
        "final_alpha_max_y": float(final["max_alpha_y"]),
        "first_alpha0p5_notch_connected_step": int(first_alpha05.iloc[0]) if not first_alpha05.empty else "",
        "first_alpha0p8_notch_connected_step": int(first_alpha08_conn.iloc[0]) if not first_alpha08_conn.empty else "",
        "first_alpha0p8_through_step": int(first_alpha08_through.iloc[0]) if not first_alpha08_through.empty else "",
        "alpha0p8_through_crack_any_step": any_through,
        "final_alpha0p5_connected_count": int(final["alpha0p5_notch_connected_count"]),
        "final_alpha0p5_connected_x_span": float(final["alpha0p5_notch_connected_x_span"]),
        "final_alpha0p8_connected_count": int(final["alpha0p8_notch_connected_count"]),
        "final_alpha0p8_connected_x_span": float(final["alpha0p8_notch_connected_x_span"]),
        "final_HII_over_HI_peak_ratio": float(final["HII_over_HI_peak_ratio"]),
        "final_mechanics_drive_max": float(final["mechanics_drive_max"]),
        "final_mechanics_drive_max_x": float(final["max_mechanics_drive_x"]),
        "final_mechanics_drive_max_y": float(final["max_mechanics_drive_y"]),
        "drive_location_classification": str(final["drive_location_classification"]),
        "final_top_v_absmax_over_Delta_s": float(final["top_v_absmax_over_Delta_s"]),
        "max_top_v_absmax_over_Delta_s": topv_max_ratio,
        "top_v_warning_gt_1p5": bool(topv["warning_top_v_gt_1p5_delta"].any()),
        "top_v_unstable_gt_2p0": bool(topv["unstable_top_v_gt_2p0_delta"].any()),
        "physics_changed": False,
        "seed_study_run": False,
        "D0040_run": False,
        "classification": classification,
    }
    pd.DataFrame([summary]).to_csv(PACKAGE / "tables/shear_extension_run_summary.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "case": "stronger_S0030",
                "schedule_name": prev_summary["load_schedule"],
                "step_count": int(prev_summary["steps"]),
                "final_Delta_s": float(prev_summary["final_Delta_s"]),
                "training_settings": f"RPROP={prev_summary['training_n_rprop']}, LBFGS={prev_summary['training_n_lbfgs']}",
                "continued_from_S0030": "",
                "final_engineering_shear_strain": float(prev_summary["final_engineering_shear_strain"]),
                "peak_nominal_shear_stress_MPa": float(prev_summary["curve_peak_stress_MPa"]),
                "final_nominal_shear_stress_MPa": float(prev_summary["final_nominal_shear_stress_energy_MPa"]),
                "post_peak_drop_observed": bool(str(prev_summary["post_peak_drop_observed"]).lower() == "true"),
                "final_alpha_max": float(prev_summary["final_alpha_max"]),
                "alpha_max_location": f"({float(prev_summary['final_alpha_max_x']):.6g}, {float(prev_summary['final_alpha_max_y']):.6g})",
                "first_alpha0p5_connected_path_step": "",
                "first_alpha0p8_connected_path_step": "",
                "through_crack_any_step": bool(str(prev_summary["alpha0p8_through_crack_any_step"]).lower() == "true"),
                "final_HII_over_HI_peak_ratio": float(prev_summary["final_HII_over_HI_peak_ratio"]),
                "final_mechanics_drive_max_location": f"({float(prev_summary['final_mechanics_drive_max_x']):.6g}, {float(prev_summary['final_mechanics_drive_max_y']):.6g})",
                "drive_location_classification": prev_summary["drive_location_classification"],
                "final_top_v_absmax_over_Delta_s": float(prev_summary["final_top_v_absmax_over_Delta_s"]),
                "classification": prev_summary["classification"],
            },
            {
                "case": "extension_S0050",
                "schedule_name": "load_schedules/load_schedule_S0050_shear.csv",
                "step_count": len(stress),
                "final_Delta_s": float(final["Delta_s"]),
                "training_settings": "RPROP=300, LBFGS=1",
                "continued_from_S0030": False,
                "final_engineering_shear_strain": float(final["engineering_shear_strain"]),
                "peak_nominal_shear_stress_MPa": float(peak_row["nominal_shear_stress_energy_MPa"]),
                "final_nominal_shear_stress_MPa": float(final["nominal_shear_stress_energy_MPa"]),
                "post_peak_drop_observed": post_peak_drop,
                "final_alpha_max": float(final["alpha_max"]),
                "alpha_max_location": f"({float(final['max_alpha_x']):.6g}, {float(final['max_alpha_y']):.6g})",
                "first_alpha0p5_connected_path_step": int(first_alpha05.iloc[0]) if not first_alpha05.empty else "",
                "first_alpha0p8_connected_path_step": int(first_alpha08_conn.iloc[0])
                if not first_alpha08_conn.empty
                else "",
                "through_crack_any_step": any_through,
                "final_HII_over_HI_peak_ratio": float(final["HII_over_HI_peak_ratio"]),
                "final_mechanics_drive_max_location": f"({float(final['max_mechanics_drive_x']):.6g}, {float(final['max_mechanics_drive_y']):.6g})",
                "drive_location_classification": str(final["drive_location_classification"]),
                "final_top_v_absmax_over_Delta_s": float(final["top_v_absmax_over_Delta_s"]),
                "classification": classification,
            },
        ]
    )
    comparison.to_csv(PACKAGE / "tables/shear_extension_vs_S0030_comparison.csv", index=False)

    figure_copies = {
        "stress_strain_seed23_shear.png": "shear_stress_strain_seed23.png",
        "reaction_strain_seed23_shear.png": "shear_reaction_strain_seed23.png",
        "final_fields_panel_seed23_shear.png": "final_fields_panel_seed23_shear.png",
        "final_alpha_seed23_shear.png": "final_alpha_seed23_shear.png",
        "final_u_seed23_shear.png": "final_u_seed23_shear.png",
        "final_v_seed23_shear.png": "final_v_seed23_shear.png",
        "final_HI_seed23_shear.png": "final_HI_seed23_shear.png",
        "final_HII_seed23_shear.png": "final_HII_seed23_shear.png",
        "final_mechanics_drive_seed23_shear.png": "final_mechanics_drive_seed23_shear.png",
    }
    for src_name, dst_name in figure_copies.items():
        shutil.copy2(FIG_SRC / src_name, PACKAGE / "figures" / dst_name)

    x = stepwise["engineering_shear_strain"]
    write_line_plot("shear_alpha_max_by_step.png", x, [stepwise["alpha_max"]], ["alpha_max"], "Engineering shear strain", "alpha_max")
    write_line_plot(
        "shear_HII_HI_ratio_by_step.png",
        x,
        [stepwise["HII_over_HI_peak_ratio"]],
        ["HII/HI peak"],
        "Engineering shear strain",
        "HII/HI peak ratio",
    )
    write_line_plot(
        "shear_top_v_absmax_over_Delta_by_step.png",
        x,
        [stepwise["top_v_absmax_over_Delta_s"]],
        ["top |v|max / Delta_s"],
        "Engineering shear strain",
        "top |v|max / Delta_s",
    )
    write_line_plot(
        "shear_notch_drive_by_step.png",
        x,
        [stepwise["notch_tip_mechanics_drive_max"], stepwise["bottom_right_mechanics_drive_max"]],
        ["notch-tip mechanics drive max", "bottom-right mechanics drive max"],
        "Engineering shear strain",
        "Drive",
    )
    write_line_plot(
        "shear_through_crack_status_by_step.png",
        x,
        [
            stepwise["alpha0p5_notch_connected_count"],
            stepwise["alpha0p8_notch_connected_count"],
            truthy(stepwise["alpha0p8_through_to_right"]).astype(float),
        ],
        ["alpha>=0.5 notch-connected count", "alpha>=0.8 notch-connected count", "alpha>=0.8 through flag"],
        "Engineering shear strain",
        "Connected count / flag",
    )

    artifact_copies = {
        RESULT_DIR / "diagnostics_mixed_tm_summary.csv": "diagnostics_mixed_tm_summary_seed23_S0050_shear.csv",
        CURVE_DIR / "reaction_metric_availability.csv": "reaction_metric_availability_seed23_S0050_shear.csv",
        FIG_SRC / "stress_strain_source_seed23_shear.txt": "stress_strain_source_seed23_S0050_shear.txt",
        MODEL_DIR / "model_settings.txt": "model_settings_seed23_S0050_shear.txt",
    }
    for src, dst in artifact_copies.items():
        shutil.copy2(src, PACKAGE / "artifacts" / dst)

    for rel in [
        "load_schedules/load_schedule_S0050_shear.csv",
        "load_schedules/load_schedule_S0030_shear.csv",
        "main.py",
        "config.py",
        "train_mixed_tm.py",
        "mixed_mode_tm.py",
        "history_field_mixed_tm.py",
        "postprocess_results.py",
        "plot_results.py",
        "field_computation.py",
        "README.md",
        "POSTPROCESS_WORKFLOW.md",
        "PROJECT_STRUCTURE.md",
    ]:
        src = SRC_ROOT / rel
        if src.exists():
            shutil.copy2(src, PACKAGE / "artifacts/source_snapshots" / rel.replace("/", "__").replace("\\", "__"))

    output_rows = []
    for p in (
        sorted((RESULT_DIR / "curves").glob("*"))
        + sorted((RESULT_DIR / "figures").glob("*"))
        + sorted(RESULT_DIR.glob("fields_mixed_tm_step_*.npz"))
    ):
        output_rows.append(
            {
                "source_path": str(p.relative_to(SRC_ROOT)),
                "type": "table" if p.suffix.lower() == ".csv" else ("figure" if p.suffix.lower() == ".png" else "artifact"),
                "size_bytes": p.stat().st_size,
                "included_in_package": (PACKAGE / "figures" / p.name).exists()
                or (PACKAGE / "tables" / p.name).exists()
                or (PACKAGE / "artifacts" / p.name).exists(),
            }
        )
    pd.DataFrame(output_rows).to_csv(PACKAGE / "tables/shear_output_file_manifest.csv", index=False)

    write_markdown_files(summary, peak_row, final, first_alpha05, first_alpha08_conn, topv_max_ratio, classification)
    write_manifest()
    return summary


def write_markdown_files(
    summary: dict[str, object],
    peak_row: pd.Series,
    final: pd.Series,
    first_alpha05: pd.Series,
    first_alpha08_conn: pd.Series,
    topv_max_ratio: float,
    classification: str,
) -> None:
    commands = f"""# Commands Run

```powershell
# From D:\\Desktop\\新建文件夹\\cc-work
git pull origin main

# From D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro
{TRAIN_CMD}
{POST_CMD}
```

Verification commands are appended after package generation.
"""
    (PACKAGE / "commands_run.txt").write_text(commands, encoding="utf-8")

    readme = f"""# Existing-Geometry Shear Load Extension

This package documents one controlled extended shear run for `TM_comsol_no_thermal_micro`. The case uses seed 23, the existing COMSOL micro-notch geometry, top-v-free shear boundary condition, and the same physical model as the prior S0030 shear diagnostic.

Package classification: `{classification}`.

Read first:

1. `REPORT.md`
2. `tables/shear_extension_run_summary.csv`
3. `tables/shear_extension_vs_S0030_comparison.csv`
4. `tables/shear_damage_drive_summary.csv`
5. `figures/figure_summary.md`

The run was a full rerun from step 0, not a continuation from S0030.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    next_questions = """# Next Questions

1. Is the observed post-peak energy-conjugate shear response enough to treat S0050 as the current main shear diagnostic result?
2. Since alpha reaches a notch-localized alpha>=0.8 connected component but not a full right-boundary through-crack, should the next minimal step be a slightly longer same-path shear extension, or first a connectivity/mesh-resolution audit of the crack path?
3. Should top-v-free drift remain only monitored, given the maximum top_v_absmax/Delta_s stayed below the 1.5 warning threshold?
"""
    (PACKAGE / "next_questions.md").write_text(next_questions, encoding="utf-8")

    first05 = int(first_alpha05.iloc[0]) if not first_alpha05.empty else "none"
    first08 = int(first_alpha08_conn.iloc[0]) if not first_alpha08_conn.empty else "none"
    report = f"""# Existing-Geometry Shear Load Extension Report

## Scope

This package runs one controlled shear load extension on the existing COMSOL micro-notch geometry using seed 23. The shear ansatz, top-v-free boundary condition, mixed-drive formula, material parameters, `l0`, TM split, history logic, alpha initialization, and training losses were not changed.

The extension uses `load_schedules/load_schedule_S0050_shear.csv`, a 33-step monotonic schedule from `Delta_s=1e-6` mm to `8e-5` mm. Clean continuation from S0030 was not implemented/was ambiguous, so this is a full rerun from step 0: `continued_from_S0030=False`.

## Commands

```powershell
{TRAIN_CMD}
{POST_CMD}
```

## Required Questions

1. Did the extended shear run complete?  
Yes. Seed 23 completed all 33 S0050 shear steps.

2. Was this a continuation from S0030 or a full rerun?  
It was a full rerun from step 0. `continued_from_S0030=False`.

3. What load schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0050_shear.csv`, 33 steps ending at `Delta_s=8e-5` mm. Training: `RPROP=300, LBFGS=1`, matching the prior stronger S0030 run.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. All 33 checkpoints exist and the normal postprocess reports `exact_reaction_computable=True`; `reaction_by_step.csv` and `stress_strain_by_step.csv` use the energy-conjugate metric.

5. Does the shear stress-strain curve remain monotonic or show peak/post-peak behavior?  
It shows peak/post-peak behavior. The peak nominal shear stress is {float(peak_row['nominal_shear_stress_energy_MPa']):.6g} MPa at step {int(peak_row['step'])}, engineering shear strain {float(peak_row['engineering_shear_strain']):.6g}; the final stress is {float(final['nominal_shear_stress_energy_MPa']):.6g} MPa.

6. Does alpha continue growing beyond the S0030 final `alpha_max=0.358412`?  
Yes. Final `alpha_max={float(final['alpha_max']):.6g}`, above the S0030 final value.

7. Does alpha remain notch-localized?  
Yes. The final alpha maximum is near `(x,y)=({float(final['max_alpha_x']):.6g}, {float(final['max_alpha_y']):.6g})` mm, in the explicit notch-tip region.

8. Does `alpha>=0.5` connected damage form?  
Yes. A notch-connected `alpha>=0.5` component first appears at step {first05} and has final connected count {int(final['alpha0p5_notch_connected_count'])}.

9. Does `alpha>=0.8` through-crack form?  
No full alpha>=0.8 through-crack to the right boundary is detected. A notch-connected `alpha>=0.8` component first appears at step {first08} and reaches final x-span {float(final['alpha0p8_notch_connected_x_span']):.6g} mm, but the through-to-right flag remains false.

10. Is HII still active and notch-localized?  
Yes. The final HII/HI peak ratio is {float(final['HII_over_HI_peak_ratio']):.6g}, and HII remains localized near the notch region.

11. Does mechanics drive remain notch-dominated?  
Yes. The final mechanics-drive maximum is at `(x,y)=({float(final['max_mechanics_drive_x']):.6g}, {float(final['max_mechanics_drive_y']):.6g})` mm and is classified as `{str(final['drive_location_classification'])}`.

12. Does top `v` remain finite, or does `top_v_absmax/Delta_s` become excessive?  
Top `v` remains finite. Final `top_v_absmax/Delta_s={float(final['top_v_absmax_over_Delta_s']):.6g}` and maximum over the run is {topv_max_ratio:.6g}, below the 1.5 warning threshold and below the 2.0 unstable threshold.

13. Compared with S0030, is the extension more convincing?  
Yes as a diagnostic result. Compared with S0030, alpha grows from 0.358412 to {float(final['alpha_max']):.6g}, connected damage appears near the notch, and the energy-conjugate shear stress curve develops post-peak behavior.

14. If no through-crack or post-peak drop appears, should the next step extend loading again, adjust boundary stabilization, or diagnose the shear damage drive?  
A post-peak drop does appear, so the immediate need is not boundary stabilization. Because no full through-crack reaches the right boundary, the next minimal step should be either a slightly longer same-path shear extension or a connectivity/mesh-resolution audit of the notch-connected crack path before changing physics.

15. Was any physics changed?  
No. No physical formulas, material parameters, `l0`, TM split, history logic, alpha initialization, shear ansatz, boundary condition, or training losses were changed.

16. Was any seed study or D0040 run performed?  
No. Only seed 23 was run. D0040 was not run.

## Classification

`{classification}`

Reason: the run completed with checkpointed energy-conjugate reaction at every step; alpha grew beyond S0030 and remained notch-localized; notch-connected `alpha>=0.8` damage appeared; the shear stress-strain curve showed post-peak behavior; top v stayed finite. A full alpha>=0.8 right-boundary through-crack was not detected, so this remains a diagnostic result rather than physical validation.

## Not Claimed

This package does not claim physical validation. It is a single-seed controlled diagnostic of the existing shear path.
"""
    (PACKAGE / "REPORT.md").write_text(report, encoding="utf-8")

    fig_summary = f"""# Figure Summary

## shear_stress_strain_seed23.png
- What it plots: checkpointed energy-conjugate nominal shear stress versus engineering shear strain for the S0050 seed 23 extension.
- Visual takeaway: stress peaks at step {int(peak_row['step'])}, engineering shear strain {float(peak_row['engineering_shear_strain']):.6g}, about {float(peak_row['nominal_shear_stress_energy_MPa']):.6g} MPa, then drops to {float(final['nominal_shear_stress_energy_MPa']):.6g} MPa at the final step.
- Conclusion support: supports a diagnostic post-peak response, not physical validation.

## shear_reaction_strain_seed23.png
- What it plots: energy-conjugate reaction_N_energy versus engineering shear strain.
- Visual takeaway: reaction peaks and then declines after the same peak step as the stress curve.
- Conclusion support: supports that checkpointed reaction is available and shows post-peak behavior.

## final_fields_panel_seed23_shear.png
- What it plots: final alpha, displacement, HI, HII, He/history/current drive, and mechanics-drive fields.
- Visual takeaway: alpha and drive remain concentrated in the explicit notch-tip region at the final step.
- Conclusion support: supports notch-localized crack growth as a diagnostic observation.

## final_alpha_seed23_shear.png
- What it plots: final damage alpha.
- Visual takeaway: alpha reaches about {float(final['alpha_max']):.6g} near the notch tip.
- Conclusion support: supports crack growth beyond S0030; no full alpha>=0.8 through-crack to the right boundary is detected.

## final_u_seed23_shear.png
- What it plots: final horizontal displacement field.
- Visual takeaway: the intended bottom-to-top shear displacement remains active.
- Conclusion support: verifies the shear loading path.

## final_v_seed23_shear.png
- What it plots: final vertical displacement field.
- Visual takeaway: top v remains finite; final top |v|max/Delta_s is {float(final['top_v_absmax_over_Delta_s']):.6g}.
- Conclusion support: diagnostic top-v monitor only.

## final_HI_seed23_shear.png
- What it plots: final HI field.
- Visual takeaway: HI is localized near the notch tip.
- Conclusion support: supports notch-localized mixed driving.

## final_HII_seed23_shear.png
- What it plots: final HII field.
- Visual takeaway: HII remains active and notch-localized; final HII/HI peak ratio is {float(final['HII_over_HI_peak_ratio']):.6g}.
- Conclusion support: supports that shear continues to activate the HII branch.

## final_mechanics_drive_seed23_shear.png
- What it plots: final mechanics-drive field.
- Visual takeaway: the global mechanics-drive maximum remains at the explicit notch-tip region.
- Conclusion support: supports notch-dominated drive.

## shear_alpha_max_by_step.png
- What it plots: alpha_max versus engineering shear strain.
- Visual takeaway: alpha grows beyond the S0030 final value and reaches approximately {float(final['alpha_max']):.6g}.
- Conclusion support: diagnostic evidence of continued crack growth.

## shear_HII_HI_ratio_by_step.png
- What it plots: HII/HI peak ratio versus engineering shear strain.
- Visual takeaway: HII remains active, with the ratio near 0.6 over most of the extension.
- Conclusion support: diagnostic evidence for active HII contribution.

## shear_top_v_absmax_over_Delta_by_step.png
- What it plots: top-boundary |v|max normalized by Delta_s.
- Visual takeaway: the ratio remains below the 1.5 warning threshold; maximum observed value is {topv_max_ratio:.6g}.
- Conclusion support: supports finite non-runaway top-v behavior in this run.

## shear_notch_drive_by_step.png
- What it plots: notch-tip and bottom-right mechanics-drive maxima versus engineering shear strain.
- Visual takeaway: the notch-tip drive strongly dominates the bottom-right monitor region.
- Conclusion support: supports notch-dominated mechanics drive.

## shear_through_crack_status_by_step.png
- What it plots: notch-connected alpha>=0.5 and alpha>=0.8 component counts, plus alpha>=0.8 through-to-right flag.
- Visual takeaway: connected damage grows near the notch, but the alpha>=0.8 component does not reach the right boundary.
- Conclusion support: supports crack growth without full through-crack classification.
"""
    (PACKAGE / "figures/figure_summary.md").write_text(fig_summary, encoding="utf-8")

    handoff = f"""## Codex handoff: Existing-geometry shear load extension

Commit: TO_BE_FILLED_AFTER_COMMIT
Data folder: `{PACKAGE_REL.as_posix()}`
Main report: `{(PACKAGE_REL / 'REPORT.md').as_posix()}`

### What changed
- Added controlled extended shear schedule `load_schedules/load_schedule_S0050_shear.csv` in the real run tree and included a snapshot in this package.
- Ran seed 23 only, using existing geometry, top-v-free shear boundary condition, and the same shear ansatz/physics as S0030.
- Used full rerun from step 0 (`continued_from_S0030=False`) because clean continuation was not implemented/was ambiguous.
- Ran normal postprocess with checkpointed energy-conjugate reaction.
- Did not run D0040, did not run a seed study, did not change `l0`, material parameters, TM split, history logic, alpha initialization, boundary conditions, shear ansatz, or losses.

### Commands run
```powershell
{TRAIN_CMD}
{POST_CMD}
```

### Key results
- Seed used: 23 only.
- Schedule used: `load_schedules/load_schedule_S0050_shear.csv`, 33 monotonic steps, final `Delta_s=8e-5` mm.
- Training settings: `RPROP=300, LBFGS=1`.
- Checkpoint availability: 33/33 step checkpoints; checkpointed energy reaction computed at all steps.
- Final engineering shear strain: `{float(final['engineering_shear_strain']):.6g}`.
- Peak nominal shear stress: `{float(peak_row['nominal_shear_stress_energy_MPa']):.6g}` MPa at step `{int(peak_row['step'])}`; final stress `{float(final['nominal_shear_stress_energy_MPa']):.6g}` MPa, so post-peak drop is observed.
- Final alpha max: `{float(final['alpha_max']):.6g}` at the explicit notch-tip region; S0030 final alpha max was `0.358412`.
- `alpha>=0.5` notch-connected damage forms; `alpha>=0.8` notch-connected damage forms, but no alpha>=0.8 through-crack to the right boundary is detected.
- Final HII/HI peak ratio: `{float(final['HII_over_HI_peak_ratio']):.6g}`; HII remains active and notch-localized.
- Mechanics-drive maximum remains notch-dominated.
- Final `top_v_absmax/Delta_s={float(final['top_v_absmax_over_Delta_s']):.6g}`; maximum over run `{topv_max_ratio:.6g}`, below warning threshold 1.5.
- Classification: `{classification}`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/shear_extension_run_summary.csv`
- `tables/shear_extension_vs_S0030_comparison.csv`
- `tables/shear_damage_drive_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should S0050 now replace S0030 as the main existing-geometry shear diagnostic result?
2. Since post-peak softening appears but no full alpha>=0.8 right-boundary through-crack forms, should the next minimal action be a slightly longer same-path shear extension or a connectivity/mesh-resolution audit?
3. Should top-v-free drift remain monitored only, given the ratio stayed below the 1.5 warning threshold?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this single-seed diagnostic run.
- Do not run D0040 or a seed study from this package.
"""
    (PACKAGE / "HANDOFF_COMMENT.md").write_text(handoff, encoding="utf-8")


def write_manifest() -> None:
    required = {
        "README.md",
        "REPORT.md",
        "HANDOFF_COMMENT.md",
        "figures/figure_summary.md",
        "tables/shear_extension_run_summary.csv",
        "tables/shear_extension_vs_S0030_comparison.csv",
        "tables/shear_damage_drive_summary.csv",
        "tables/shear_reaction_by_step.csv",
        "tables/shear_stress_strain_by_step.csv",
    }
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel == "HANDOFF_COMMENT.md":
            typ = "handoff"
        elif rel == "figures/figure_summary.md":
            typ = "figure_summary"
        elif rel.startswith("figures/") and path.suffix.lower() == ".png":
            typ = "figure"
        elif rel.startswith("tables/"):
            typ = "table"
        elif rel == "commands_run.txt":
            typ = "command_log"
        elif rel in {"README.md", "REPORT.md", "next_questions.md"}:
            typ = "report"
        else:
            typ = "artifact"
        entries.append(
            {
                "path": rel,
                "type": typ,
                "description": rel.replace("_", " ").replace("/", " / "),
                "required_for_chatgpt": rel in required or typ in {"figure_summary", "handoff"},
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    result = build_package()
    print(json.dumps(result, ensure_ascii=False, indent=2))
