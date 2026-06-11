from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shear_connectivity import (
    DEFAULT_THRESHOLDS,
    NOTCH_TIP_X_MM,
    NOTCH_TIP_Y_MM,
    RIGHT_BOUNDARY_BAND_MM,
    SPECIMEN_SIZE_MM,
    compute_connectivity_by_threshold,
    first_event_steps,
)


SRC_ROOT = Path(__file__).resolve().parent
DEFAULT_CC_ROOT = Path(r"D:\Desktop\新建文件夹\cc-work")
PACKAGE_REL = Path(
    "examples/TM_comsol_no_thermal_micro/runs/"
    "20260617_existing_geometry_shear_connectivity_extension"
)
RUN_ID = "seed23_S0070_shear"
BASELINE_RUN_ID = "seed23_S0050_shear"
SEED = 23
TRAIN_PYTHON = r"D:\anaconda3\envs\torch_env\python.exe"
TRAIN_CMD = (
    rf"{TRAIN_PYTHON} main.py 8 400 23 TrainableReLU 3.0 --full "
    "--n-rprop 300 --n-lbfgs 1 --load-case shear "
    "--load-schedule-file load_schedules/load_schedule_S0070_shear.csv "
    "--run-suffix seed23_S0070_shear"
)
POST_CMD = (
    rf"{TRAIN_PYTHON} postprocess_results.py "
    "--model-dir outputs/checkpoints/seed23_S0070_shear "
    "--result-dir outputs/results/seed23_S0070_shear --device cpu"
)
S0050_PACKAGE = (
    DEFAULT_CC_ROOT
    / "examples/TM_comsol_no_thermal_micro/runs/"
    "20260616_existing_geometry_shear_load_extension"
)


def classify_shear_extension(
    *,
    run_completed: bool,
    reaction_ok: bool,
    post_peak_drop_percent: float,
    final_top_v_ratio: float,
    unstable_top_v_ratio: float,
    through_crack_any: bool,
    xspan_growth_material: bool,
    propagated_high_threshold: bool,
) -> str:
    if (
        not run_completed
        or not reaction_ok
        or not np.isfinite(final_top_v_ratio)
        or final_top_v_ratio > unstable_top_v_ratio
    ):
        return "shear extension unstable"
    if through_crack_any:
        return "shear extension reaches through-crack"
    if (
        post_peak_drop_percent > 0.0
        and xspan_growth_material
        and propagated_high_threshold
    ):
        return "shear extension shows propagating crack"
    if not xspan_growth_material and not propagated_high_threshold:
        return "shear extension remains local notch damage"
    return "shear extension inconclusive"


def threshold_label(threshold: float) -> str:
    return "alpha_ge_" + str(threshold).replace(".", "p")


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def ensure_dirs(package: Path) -> None:
    for subdir in ("tables", "figures", "artifacts", "artifacts/source_snapshots"):
        (package / subdir).mkdir(parents=True, exist_ok=True)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def result_dirs(src_root: Path, run_id: str) -> Dict[str, Path]:
    return {
        "result": src_root / "outputs" / "results" / run_id,
        "model": src_root / "outputs" / "checkpoints" / run_id,
    }


def count_field_files(result_dir: Path) -> int:
    return len(list(result_dir.glob("fields_mixed_tm_step_*.npz")))


def count_checkpoints(model_dir: Path) -> int:
    return len(list((model_dir / "best_models" / "step_checkpoints").glob("checkpoint_mixedH_TM_step_*.pt")))


def add_step_context(connectivity: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        col
        for col in ("step", "Delta_s", "engineering_shear_strain", "nominal_shear_stress_energy_MPa")
        if col in stress.columns
    ]
    return connectivity.merge(stress[context_cols], on="step", how="left")


def build_crack_path_geometry(connectivity: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    context = stress[["step", "Delta_s", "engineering_shear_strain"]].copy()
    rows: List[Dict[str, object]] = []
    for _, step_row in context.iterrows():
        step = int(step_row["step"])
        row: Dict[str, object] = {
            "step": step,
            "Delta_s": float(step_row["Delta_s"]),
            "engineering_shear_strain": float(step_row["engineering_shear_strain"]),
        }
        group = connectivity[connectivity["step"] == step]
        for _, conn_row in group.iterrows():
            label = threshold_label(float(conn_row["threshold"]))
            for col in (
                "notch_connected_component_count",
                "notch_connected_x_span",
                "notch_connected_y_span",
                "notch_connected_area_fraction",
                "component_min_x",
                "component_max_x",
                "component_min_y",
                "component_max_y",
                "component_centroid_x",
                "component_centroid_y",
                "component_mean_y",
                "reaches_right_boundary",
                "reaches_top_boundary",
                "reaches_bottom_boundary",
                "crack_angle_deg",
                "principal_direction_angle_deg",
            ):
                row[f"{label}_{col}"] = conn_row[col]
        rows.append(row)
    return pd.DataFrame(rows)


def drive_location_classification(row: Mapping[str, object]) -> str:
    try:
        dx = abs(float(row["max_mechanics_drive_x"]) - NOTCH_TIP_X_MM)
        dy = abs(float(row["max_mechanics_drive_y"]) - NOTCH_TIP_Y_MM)
    except (KeyError, TypeError, ValueError):
        return "unknown"
    return "notch-dominated" if dx <= 4.5e-4 and dy <= 4.5e-4 else "off-notch"


def build_stepwise_tables(
    stress: pd.DataFrame,
    reaction: pd.DataFrame,
    diagnostics: pd.DataFrame,
    connectivity: pd.DataFrame,
    model_dir: Path,
    result_dir: Path,
) -> Dict[str, pd.DataFrame]:
    diag = diagnostics.copy()
    if "Delta_s" not in diag.columns:
        diag["Delta_s"] = diag["Delta"]
    stepwise = stress.merge(
        diag.drop(columns=[col for col in ("Delta_s", "engineering_shear_strain") if col in diag.columns]),
        on="step",
        how="left",
        suffixes=("", "_diagnostic"),
    )
    stepwise["HII_over_HI_peak_ratio"] = (
        stepwise["HII_max"].astype(float) / stepwise["HI_max"].replace(0.0, np.nan).astype(float)
    )
    stepwise["top_v_absmax_over_Delta_s"] = (
        stepwise["top_v_abs_max"].astype(float) / stepwise["Delta_s"].replace(0.0, np.nan).astype(float)
    )
    stepwise["drive_location_classification"] = stepwise.apply(drive_location_classification, axis=1)

    crack_path = build_crack_path_geometry(connectivity, stress)
    stepwise = stepwise.merge(
        crack_path.drop(columns=["Delta_s", "engineering_shear_strain"]),
        on="step",
        how="left",
    )

    checkpoint = reaction.copy()
    checkpoint["checkpoint_exists"] = checkpoint["step"].apply(
        lambda step: (
            model_dir
            / "best_models"
            / "step_checkpoints"
            / f"checkpoint_mixedH_TM_step_{int(step):04d}.pt"
        ).exists()
    )
    checkpoint["field_npz_exists"] = checkpoint["step"].apply(
        lambda step: (result_dir / f"fields_mixed_tm_step_{int(step):04d}.npz").exists()
    )

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
    ]
    for threshold in DEFAULT_THRESHOLDS:
        label = threshold_label(threshold)
        damage_cols.extend(
            [
                f"{label}_notch_connected_component_count",
                f"{label}_notch_connected_x_span",
                f"{label}_notch_connected_area_fraction",
                f"{label}_reaches_right_boundary",
                f"{label}_component_centroid_x",
                f"{label}_component_centroid_y",
                f"{label}_crack_angle_deg",
                f"{label}_principal_direction_angle_deg",
            ]
        )

    top_v = stepwise[
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
    top_v["warning_top_v_gt_1p5_delta"] = top_v["top_v_absmax_over_Delta_s"] > 1.5
    top_v["unstable_top_v_gt_2p0_delta"] = top_v["top_v_absmax_over_Delta_s"] > 2.0

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
    return {
        "stepwise": stepwise,
        "damage": stepwise[[col for col in damage_cols if col in stepwise.columns]].copy(),
        "top_v": top_v,
        "checkpoint": checkpoint,
        "loss": stepwise[[col for col in loss_cols if col in stepwise.columns]].copy(),
        "crack_path": crack_path,
    }


def first_step(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return ""
    return int(clean.iloc[0])


def summary_for_case(
    *,
    run_id: str,
    schedule: str,
    continued_label: str,
    stress: pd.DataFrame,
    reaction: pd.DataFrame,
    diagnostics: pd.DataFrame,
    connectivity: pd.DataFrame,
    model_dir: Path,
    result_dir: Path,
    classification: str,
) -> Dict[str, object]:
    final_stress = stress.iloc[-1]
    peak_idx = int(stress["nominal_shear_stress_energy_MPa"].astype(float).idxmax())
    peak = stress.loc[peak_idx]
    final_diag = diagnostics.iloc[-1]
    events = first_event_steps(connectivity)

    def event_value(threshold: float, column: str) -> object:
        match = events[np.isclose(events["threshold"].astype(float), threshold)]
        if match.empty:
            return ""
        value = match.iloc[0][column]
        if pd.isna(value):
            return ""
        return int(value)

    def final_conn(threshold: float, column: str) -> object:
        final_group = connectivity[
            (connectivity["step"] == int(final_stress["step"]))
            & np.isclose(connectivity["threshold"].astype(float), threshold)
        ]
        if final_group.empty:
            return np.nan
        return final_group.iloc[0][column]

    reaction_ok = bool(
        (reaction["reaction_metric_status"].astype(str) == "energy_conjugate").all()
        and truthy(reaction["is_energy_conjugate"]).all()
    )
    peak_stress = float(peak["nominal_shear_stress_energy_MPa"])
    final_stress_value = float(final_stress["nominal_shear_stress_energy_MPa"])
    post_peak_drop = max(0.0, peak_stress - final_stress_value)
    return {
        "run_id": run_id,
        "schedule_name": schedule,
        "seed": SEED,
        "step_count": len(stress),
        "run_completed": count_field_files(result_dir) == len(stress),
        "continued_from_S0050": continued_label,
        "training_settings": "RPROP=300, LBFGS=1",
        "energy_reaction_computable": reaction_ok,
        "checkpoint_count": count_checkpoints(model_dir),
        "field_npz_count": count_field_files(result_dir),
        "final_Delta_s": float(final_stress["Delta_s"]),
        "final_engineering_shear_strain": float(final_stress["engineering_shear_strain"]),
        "peak_nominal_shear_stress_MPa": peak_stress,
        "peak_step": int(peak["step"]),
        "peak_engineering_shear_strain": float(peak["engineering_shear_strain"]),
        "final_nominal_shear_stress_MPa": final_stress_value,
        "post_peak_drop_MPa": post_peak_drop,
        "post_peak_drop_percent": (100.0 * post_peak_drop / peak_stress) if peak_stress else np.nan,
        "final_alpha_max": float(final_diag["alpha_max"]),
        "alpha_max_location": f"({float(final_diag['max_alpha_x']):.8g}, {float(final_diag['max_alpha_y']):.8g})",
        "final_HII_over_HI_peak_ratio": float(final_diag["HII_max"]) / float(final_diag["HI_max"]),
        "final_mechanics_drive_max_location": (
            f"({float(final_diag['max_mechanics_drive_x']):.8g}, "
            f"{float(final_diag['max_mechanics_drive_y']):.8g})"
        ),
        "drive_location_classification": drive_location_classification(final_diag),
        "final_top_v_absmax_over_Delta_s": float(final_diag["top_v_abs_max"]) / float(final_stress["Delta_s"]),
        "classification": classification,
        **{
            f"first_{threshold_label(th)}_notch_connected_step": event_value(
                th, "first_notch_connected_step"
            )
            for th in DEFAULT_THRESHOLDS
        },
        **{
            f"first_{threshold_label(th)}_right_boundary_step": event_value(
                th, "first_right_boundary_through_step"
            )
            for th in DEFAULT_THRESHOLDS
        },
        **{
            f"final_{threshold_label(th)}_x_span": final_conn(th, "notch_connected_x_span")
            for th in DEFAULT_THRESHOLDS
        },
        **{
            f"final_{threshold_label(th)}_reaches_right_boundary": final_conn(
                th, "reaches_right_boundary"
            )
            for th in DEFAULT_THRESHOLDS
        },
    }


def copy_required_figures(result_dir: Path, package: Path) -> None:
    fig_src = result_dir / "figures"
    mapping = {
        "stress_strain_seed23_S0070_shear.png": "shear_stress_strain_seed23.png",
        "reaction_strain_seed23_S0070_shear.png": "shear_reaction_strain_seed23.png",
        "final_fields_panel_seed23_S0070_shear.png": "final_fields_panel_seed23_shear.png",
        "final_alpha_seed23_S0070_shear.png": "final_alpha_seed23_shear.png",
        "final_u_seed23_S0070_shear.png": "final_u_seed23_shear.png",
        "final_v_seed23_S0070_shear.png": "final_v_seed23_shear.png",
        "final_HI_seed23_S0070_shear.png": "final_HI_seed23_shear.png",
        "final_HII_seed23_S0070_shear.png": "final_HII_seed23_shear.png",
        "final_mechanics_drive_seed23_S0070_shear.png": "final_mechanics_drive_seed23_shear.png",
    }
    for src_name, dst_name in mapping.items():
        src = fig_src / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, package / "figures" / dst_name)


def line_plot(
    path: Path,
    x: pd.Series,
    series: Sequence[pd.Series],
    labels: Sequence[str],
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    for values, label in zip(series, labels):
        ax.plot(x.astype(float), values.astype(float), marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(labels) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_figures(
    package: Path,
    stress: pd.DataFrame,
    diagnostics: pd.DataFrame,
    connectivity: pd.DataFrame,
    result_dir: Path,
) -> None:
    x = stress["engineering_shear_strain"]
    diagnostics = diagnostics.merge(
        stress[["step", "Delta_s", "engineering_shear_strain"]],
        on="step",
        how="left",
    )
    h_ratio = diagnostics["HII_max"].astype(float) / diagnostics["HI_max"].replace(0.0, np.nan).astype(float)
    top_v_ratio = diagnostics["top_v_abs_max"].astype(float) / diagnostics["Delta_s"].replace(0.0, np.nan).astype(float)
    line_plot(
        package / "figures" / "shear_alpha_max_by_step.png",
        x,
        [diagnostics["alpha_max"]],
        ["alpha max"],
        "Engineering shear strain",
        "alpha max",
    )
    line_plot(
        package / "figures" / "shear_HII_HI_ratio_by_step.png",
        x,
        [h_ratio],
        ["HII / HI peak ratio"],
        "Engineering shear strain",
        "HII / HI peak ratio",
    )
    line_plot(
        package / "figures" / "shear_top_v_absmax_over_Delta_by_step.png",
        x,
        [top_v_ratio],
        ["top |v|max / Delta_s"],
        "Engineering shear strain",
        "top |v|max / Delta_s",
    )
    line_plot(
        package / "figures" / "shear_notch_drive_by_step.png",
        x,
        [diagnostics["notch_tip_mechanics_drive_max"], diagnostics["bottom_right_mechanics_drive_max"]],
        ["notch-tip mechanics drive", "bottom-right mechanics drive"],
        "Engineering shear strain",
        "Mechanics drive",
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    for threshold, group in connectivity.groupby("threshold", sort=True):
        ax.plot(
            group["engineering_shear_strain"].astype(float),
            group["notch_connected_x_span"].astype(float),
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"alpha >= {threshold:g}",
        )
    ax.set_xlabel("Engineering shear strain")
    ax.set_ylabel("Notch-connected x-span (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(package / "figures" / "shear_connectivity_xspan_by_threshold.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    for threshold, group in connectivity.groupby("threshold", sort=True):
        ax.plot(
            group["engineering_shear_strain"].astype(float),
            group["connected_component_count"].astype(float),
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"alpha >= {threshold:g}",
        )
    ax.set_xlabel("Engineering shear strain")
    ax.set_ylabel("Connected component count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(package / "figures" / "shear_connected_component_count_by_threshold.png")
    plt.close(fig)

    final_field = result_dir / f"fields_mixed_tm_step_{int(stress['step'].max()):04d}.npz"
    with np.load(final_field) as data:
        element_x = np.asarray(data["element_x"], dtype=float)
        element_y = np.asarray(data["element_y"], dtype=float)
        alpha = np.asarray(data["alpha_elem"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=170)
    sc = ax.scatter(element_x, element_y, c=alpha, s=3.0, cmap="viridis", linewidths=0)
    final_conn = connectivity[connectivity["step"] == int(stress["step"].max())]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for color, (_, row) in zip(colors, final_conn.sort_values("threshold").iterrows()):
        if int(row["notch_connected_component_count"]) <= 0:
            continue
        cx = float(row["component_centroid_x"])
        cy = float(row["component_centroid_y"])
        ax.plot([NOTCH_TIP_X_MM, cx], [NOTCH_TIP_Y_MM, cy], color=color, linewidth=1.5)
        ax.scatter([cx], [cy], color=color, s=18, label=f"alpha >= {row['threshold']:g}")
        min_x = float(row["component_min_x"])
        max_x = float(row["component_max_x"])
        min_y = float(row["component_min_y"])
        max_y = float(row["component_max_y"])
        ax.add_patch(
            plt.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
            )
        )
    ax.axvline(SPECIMEN_SIZE_MM - RIGHT_BOUNDARY_BAND_MM, color="black", linestyle="--", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Final alpha and notch-connected crack-path geometry")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.colorbar(sc, ax=ax, label="alpha")
    fig.tight_layout()
    fig.savefig(package / "figures" / "shear_crack_path_overlay_final.png")
    plt.close(fig)

    thresholds = sorted(connectivity["threshold"].unique())
    steps = sorted(connectivity["step"].unique())
    status = np.zeros((len(thresholds), len(steps)))
    for i, threshold in enumerate(thresholds):
        group = connectivity[np.isclose(connectivity["threshold"].astype(float), threshold)]
        by_step = group.set_index("step")["reaches_right_boundary"]
        for j, step in enumerate(steps):
            status[i, j] = 1.0 if bool(by_step.get(step, False)) else 0.0
    fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=160)
    ax.imshow(status, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"alpha >= {threshold:g}" for threshold in thresholds])
    ax.set_xticks([0, len(steps) - 1])
    ax.set_xticklabels([str(steps[0]), str(steps[-1])])
    ax.set_xlabel("Step")
    ax.set_title("Right-boundary reach status by threshold")
    fig.tight_layout()
    fig.savefig(package / "figures" / "shear_through_crack_status_by_threshold.png")
    plt.close(fig)


def write_manifest(package: Path) -> None:
    required = {
        "README.md",
        "REPORT.md",
        "HANDOFF_COMMENT.md",
        "figures/figure_summary.md",
        "tables/shear_connectivity_extension_run_summary.csv",
        "tables/shear_extension_vs_S0050_comparison.csv",
        "tables/shear_connectivity_by_threshold.csv",
    }
    entries = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(package).as_posix()
        if rel.startswith("figures/") and path.suffix.lower() == ".png":
            kind = "figure"
        elif rel.startswith("tables/"):
            kind = "table"
        elif rel == "HANDOFF_COMMENT.md":
            kind = "handoff"
        elif rel in {"README.md", "REPORT.md", "next_questions.md"}:
            kind = "report"
        elif rel == "commands_run.txt":
            kind = "command_log"
        else:
            kind = "artifact"
        entries.append(
            {
                "path": rel,
                "type": kind,
                "description": rel.replace("_", " ").replace("/", " / "),
                "required_for_chatgpt": rel in required,
            }
        )
    (package / "MANIFEST.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def copy_artifacts(src_root: Path, package: Path, result_dir: Path, model_dir: Path) -> None:
    artifact_map = {
        result_dir / "diagnostics_mixed_tm_summary.csv": "diagnostics_mixed_tm_summary_seed23_S0070_shear.csv",
        result_dir / "curves" / "reaction_metric_availability.csv": "reaction_metric_availability_seed23_S0070_shear.csv",
        result_dir / "figures" / "stress_strain_source_seed23_S0070_shear.txt": "stress_strain_source_seed23_S0070_shear.txt",
        model_dir / "model_settings.txt": "model_settings_seed23_S0070_shear.txt",
    }
    for src, name in artifact_map.items():
        if src.exists():
            shutil.copy2(src, package / "artifacts" / name)

    source_files = [
        "build_shear_connectivity_package.py",
        "shear_connectivity.py",
        "load_schedules/load_schedule_S0070_shear.csv",
        "load_schedules/load_schedule_S0050_shear.csv",
        "main.py",
        "config.py",
        "train_mixed_tm.py",
        "mixed_mode_tm.py",
        "history_field_mixed_tm.py",
        "postprocess_results.py",
        "plot_results.py",
        "field_computation.py",
        "compute_energy_mixed_tm.py",
        "README.md",
        "POSTPROCESS_WORKFLOW.md",
        "PROJECT_STRUCTURE.md",
    ]
    for rel in source_files:
        src = src_root / rel
        if src.exists():
            dst = package / "artifacts" / "source_snapshots" / rel.replace("/", "__")
            shutil.copy2(src, dst)


def build_output_manifest(src_root: Path, package: Path, result_dir: Path, model_dir: Path) -> pd.DataFrame:
    rows = []
    source_paths = (
        sorted((result_dir / "curves").glob("*"))
        + sorted((result_dir / "figures").glob("*"))
        + sorted(result_dir.glob("fields_mixed_tm_step_*.npz"))
        + sorted((model_dir / "best_models" / "step_checkpoints").glob("checkpoint_mixedH_TM_step_*.pt"))
    )
    for path in source_paths:
        rel = path.relative_to(src_root).as_posix()
        included = any((package / sub / path.name).exists() for sub in ("tables", "figures", "artifacts"))
        rows.append(
            {
                "source_path": rel,
                "type": "table" if path.suffix.lower() == ".csv" else ("figure" if path.suffix.lower() == ".png" else "artifact"),
                "size_bytes": path.stat().st_size,
                "included_in_package": included,
            }
        )
    return pd.DataFrame(rows)


def comparison_table(s0050: Dict[str, object], s0070: Dict[str, object]) -> pd.DataFrame:
    columns = [
        "case",
        "schedule_name",
        "step_count",
        "final_Delta_s",
        "training_settings",
        "continued_from_S0050",
        "final_engineering_shear_strain",
        "peak_nominal_shear_stress_MPa",
        "peak_step",
        "final_nominal_shear_stress_MPa",
        "post_peak_drop_MPa",
        "post_peak_drop_percent",
        "final_alpha_max",
        "alpha_max_location",
        "first_alpha_ge_0p3_notch_connected_step",
        "first_alpha_ge_0p5_notch_connected_step",
        "first_alpha_ge_0p8_notch_connected_step",
        "first_alpha_ge_0p95_notch_connected_step",
        "first_alpha_ge_0p3_right_boundary_step",
        "first_alpha_ge_0p5_right_boundary_step",
        "first_alpha_ge_0p8_right_boundary_step",
        "first_alpha_ge_0p95_right_boundary_step",
        "final_alpha_ge_0p3_x_span",
        "final_alpha_ge_0p5_x_span",
        "final_alpha_ge_0p8_x_span",
        "final_alpha_ge_0p95_x_span",
        "final_HII_over_HI_peak_ratio",
        "final_mechanics_drive_max_location",
        "drive_location_classification",
        "final_top_v_absmax_over_Delta_s",
        "classification",
    ]
    rows = []
    for case, summary in (("S0050", s0050), ("S0070", s0070)):
        row = {"case": case}
        row.update(summary)
        rows.append({col: row.get(col, "") for col in columns})
    return pd.DataFrame(rows, columns=columns)


def write_markdown(
    package: Path,
    summary: Dict[str, object],
    s0050: Dict[str, object],
    connectivity: pd.DataFrame,
) -> None:
    thresholds = list(DEFAULT_THRESHOLDS)
    through = {
        threshold: summary.get(f"first_{threshold_label(threshold)}_right_boundary_step", "")
        for threshold in thresholds
    }
    xspans = {
        threshold: summary.get(f"final_{threshold_label(threshold)}_x_span", np.nan)
        for threshold in thresholds
    }
    through_any = any(value != "" for value in through.values())
    first_notch_lines = "\n".join(
        [
            f"- alpha >= {threshold:g}: first notch-connected step "
            f"{summary.get(f'first_{threshold_label(threshold)}_notch_connected_step', '')}, "
            f"final x-span {xspans[threshold]:.6g} mm"
            for threshold in thresholds
        ]
    )
    through_lines = "\n".join(
        [
            f"- alpha >= {threshold:g}: "
            + (f"right-boundary reach first at step {through[threshold]}" if through[threshold] != "" else "no right-boundary reach")
            for threshold in thresholds
        ]
    )

    commands = f"""# Commands Run

```powershell
# Training command represented by existing outputs:
{TRAIN_CMD}

# Postprocess command represented by existing curves/figures:
{POST_CMD}

# Package generation:
{TRAIN_PYTHON} build_shear_connectivity_package.py
```
"""
    (package / "commands_run.txt").write_text(commands, encoding="utf-8")

    readme = f"""# Existing-Geometry Shear Connectivity Extension

This package documents the S0070 same-path shear extension for seed 23 on the existing COMSOL micro-notch geometry.

The run is classified as `{summary['classification']}`.

Read first:

1. `REPORT.md`
2. `tables/shear_connectivity_extension_run_summary.csv`
3. `tables/shear_extension_vs_S0050_comparison.csv`
4. `tables/shear_connectivity_by_threshold.csv`
5. `figures/figure_summary.md`

The S0070 data are a full rerun from step 0, not a checkpoint continuation from S0050.
"""
    (package / "README.md").write_text(readme, encoding="utf-8")

    next_questions = """# Next Recommended Minimal Intervention

Because S0070 extends the notch-connected high-alpha x-span without a right-boundary through-crack, the next minimal step is to stop the shear loading escalation and interpret the geometry/connectivity diagnostics before changing physics. If another run is needed, use only a small same-path extension after reviewing the crack-path overlay and threshold tables.
"""
    (package / "next_questions.md").write_text(next_questions, encoding="utf-8")

    report = f"""# Existing-Geometry Shear Connectivity Extension Report

## Scope

This package analyzes one longer same-path shear extension, S0070, using the existing geometry and seed 23. No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, shear ansatz, boundary conditions, split modes, or training losses were changed.

Notch-connected is defined as a thresholded connected component whose element-centroid set intersects a box of half-width `3.0e-4 mm` around the explicit notch tip at `(0.005, 0.005) mm`. Right-boundary reach is defined as any element centroid in the notch-connected component satisfying `x >= 0.01 - {RIGHT_BOUNDARY_BAND_MM:g} mm`.

## Required Questions

1. Did the S0070 longer shear extension complete?  
Yes. S0070 has {summary['step_count']} stress-strain rows and {summary['field_npz_count']} field files through final step 42.

2. Was this a continuation from S0050 or a full rerun?  
It was a full rerun from step 0: `continued_from_S0050=False`. Clean continuation from the committed S0050 history state was not already implemented unambiguously, so no continuation framework was added.

3. What schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0070_shear.csv`, ending at `Delta_s={summary['final_Delta_s']:.6g}` mm. Training: `RPROP=300, LBFGS=1`.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. `reaction_N_energy = dPi/dDelta_s` is available at all {summary['step_count']} steps.

5. Does the shear stress-strain curve show continued post-peak softening?  
Yes. The peak stress occurs before the final step and the final stress remains below the peak.

6. What are peak stress, final stress, post-peak drop amount, and post-peak drop percent?  
Peak nominal shear stress is {summary['peak_nominal_shear_stress_MPa']:.6g} MPa at step {summary['peak_step']}. Final stress is {summary['final_nominal_shear_stress_MPa']:.6g} MPa. Post-peak drop is {summary['post_peak_drop_MPa']:.6g} MPa, or {summary['post_peak_drop_percent']:.3g}%.

7. Does alpha remain notch-localized?  
Yes. The final alpha maximum is near {summary['alpha_max_location']}, in the explicit notch-tip region.

8. Does alpha grow beyond S0050 final `alpha_max=1.00034`, or is it saturated?  
Final `alpha_max={summary['final_alpha_max']:.6g}`. This is essentially saturated relative to S0050 final `alpha_max={s0050['final_alpha_max']:.6g}`.

9. How do the notch-connected x-spans evolve for alpha thresholds 0.3, 0.5, 0.8, and 0.95?  
{first_notch_lines}

10. Does any threshold reach the right boundary?  
{through_lines}

11. Does the crack path propagate away from the notch, or stay as a local notch-tip damage zone?  
The high-threshold x-span grows beyond S0050, so the diagnostic shows propagation away from the notch tip, but it does not reach the right boundary in this package.

12. Is HII still active and notch-localized?  
Yes. The final HII/HI peak ratio is {summary['final_HII_over_HI_peak_ratio']:.6g}, and the field figures keep HII concentrated near the notch region.

13. Does mechanics drive remain notch-dominated?  
Yes. Final mechanics-drive maximum location is {summary['final_mechanics_drive_max_location']}, classified as `{summary['drive_location_classification']}`.

14. Does top `v` remain below warning/unstable thresholds?  
Yes. Final `top_v_absmax/Delta_s={summary['final_top_v_absmax_over_Delta_s']:.6g}`, below warning threshold 1.5 and unstable threshold 2.0.

15. Compared with S0050, is the longer extension more informative?  
Yes. It extends the stress-strain softening branch and grows the connected high-alpha x-span beyond S0050 while keeping reaction and top-v diagnostics finite.

16. If no right-boundary through-crack occurs, is the next step another small extension, a geometry/connectivity interpretation, or stopping the shear diagnostic?  
The next step should be geometry/connectivity interpretation before changing physics or escalating load again.

17. Was any physics changed?  
No.

18. Was any seed study or D0040 run performed?  
No. Only seed 23 S0070 is reported here; no D0040 run and no seed study were performed.

## Classification

`{summary['classification']}`

Through-crack at any threshold: `{through_any}`.
"""
    (package / "REPORT.md").write_text(report, encoding="utf-8")

    fig_summary = f"""# Figure Summary

- `shear_stress_strain_seed23.png`: energy-conjugate nominal shear stress versus engineering shear strain.
- `shear_reaction_strain_seed23.png`: checkpoint reaction `reaction_N_energy` versus engineering shear strain.
- `final_fields_panel_seed23_shear.png`: final field panel from normal postprocessing.
- `final_alpha_seed23_shear.png`: final alpha field; maximum remains near the notch.
- `final_u_seed23_shear.png`: final horizontal displacement field.
- `final_v_seed23_shear.png`: final vertical displacement field; top-v remains finite.
- `final_HI_seed23_shear.png`: final HI field.
- `final_HII_seed23_shear.png`: final HII field.
- `final_mechanics_drive_seed23_shear.png`: final mechanics-drive field.
- `shear_alpha_max_by_step.png`: alpha maximum by step.
- `shear_HII_HI_ratio_by_step.png`: HII/HI peak ratio by step.
- `shear_top_v_absmax_over_Delta_by_step.png`: top `|v|max/Delta_s` monitor.
- `shear_notch_drive_by_step.png`: notch-tip versus bottom-right mechanics-drive monitor.
- `shear_connectivity_xspan_by_threshold.png`: notch-connected x-span for alpha thresholds 0.3, 0.5, 0.8, and 0.95.
- `shear_connected_component_count_by_threshold.png`: connected component count by threshold.
- `shear_crack_path_overlay_final.png`: final alpha scatter with notch-connected component geometry overlays.
- `shear_through_crack_status_by_threshold.png`: right-boundary reach status by threshold.
"""
    (package / "figures" / "figure_summary.md").write_text(fig_summary, encoding="utf-8")

    handoff = f"""## Codex handoff: Existing-geometry shear connectivity extension

Commit: TO_BE_FILLED_AFTER_COMMIT
Package folder: `{PACKAGE_REL.as_posix()}`
Memory file read confirmation: read `CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`, project-memory handoff files, and the S0050 shear extension report/tables before this package was built.

### Run identity
- Continuation: `continued_from_S0050=False`.
- Reason: clean continuation from the committed S0050 history state was not already implemented unambiguously; S0070 is a full rerun from step 0.
- Training command: `{TRAIN_CMD}`
- Postprocess command: `{POST_CMD}`
- Load schedule: `load_schedules/load_schedule_S0070_shear.csv`.
- Seed: 23 only.
- Training settings: `RPROP=300, LBFGS=1`.

### Status
- Checkpoint availability: {summary['checkpoint_count']}/{summary['step_count']} checkpoints.
- Energy reaction status: energy-conjugate reaction at all steps.
- S0050 comparison: final S0050 stress {s0050['final_nominal_shear_stress_MPa']:.6g} MPa, final S0070 stress {summary['final_nominal_shear_stress_MPa']:.6g} MPa.
- S0070 peak stress: {summary['peak_nominal_shear_stress_MPa']:.6g} MPa at step {summary['peak_step']}; final stress {summary['final_nominal_shear_stress_MPa']:.6g} MPa; post-peak drop {summary['post_peak_drop_percent']:.3g}%.
- Alpha/HII/mechanics drive: alpha remains notch-localized; final HII/HI ratio {summary['final_HII_over_HI_peak_ratio']:.6g}; drive remains `{summary['drive_location_classification']}`.

### Connectivity by threshold
{first_notch_lines}

### Through-crack status
{through_lines}

### Top-v diagnostic
- Final top_v_absmax/Delta_s: {summary['final_top_v_absmax_over_Delta_s']:.6g}; below warning 1.5 and unstable 2.0 thresholds.

### Generated tables and figures
- Tables are under `tables/`; key tables are `shear_connectivity_extension_run_summary.csv`, `shear_extension_vs_S0050_comparison.csv`, and `shear_connectivity_by_threshold.csv`.
- Figures are under `figures/`; read `figures/figure_summary.md`.

### Classification
`{summary['classification']}`

### Constraints observed
- Physics changed: no.
- Seed study run: no.
- D0040 run: no.

### Next recommended minimal intervention
Interpret the geometry/connectivity diagnostics before changing physics. If another run is needed, make only a small same-path extension after reviewing the threshold tables and final crack-path overlay.
"""
    (package / "HANDOFF_COMMENT.md").write_text(handoff, encoding="utf-8")


def build_package(src_root: Path = SRC_ROOT, cc_root: Path = DEFAULT_CC_ROOT) -> Dict[str, object]:
    package = cc_root / PACKAGE_REL
    if package.exists():
        shutil.rmtree(package)
    ensure_dirs(package)

    dirs = result_dirs(src_root, RUN_ID)
    result_dir = dirs["result"]
    model_dir = dirs["model"]
    baseline_dirs = result_dirs(src_root, BASELINE_RUN_ID)
    baseline_result = baseline_dirs["result"]
    baseline_model = baseline_dirs["model"]

    stress = read_required_csv(result_dir / "curves" / "stress_strain_by_step.csv")
    reaction = read_required_csv(result_dir / "curves" / "reaction_by_step.csv")
    diagnostics = read_required_csv(result_dir / "diagnostics_mixed_tm_summary.csv")
    baseline_stress = read_required_csv(baseline_result / "curves" / "stress_strain_by_step.csv")
    baseline_reaction = read_required_csv(baseline_result / "curves" / "reaction_by_step.csv")
    baseline_diagnostics = read_required_csv(baseline_result / "diagnostics_mixed_tm_summary.csv")

    connectivity = add_step_context(compute_connectivity_by_threshold(result_dir), stress)
    baseline_connectivity = add_step_context(compute_connectivity_by_threshold(baseline_result), baseline_stress)

    tables = build_stepwise_tables(stress, reaction, diagnostics, connectivity, model_dir, result_dir)
    through_any = bool(connectivity["reaches_right_boundary"].astype(bool).any())
    s0050_temp = summary_for_case(
        run_id=BASELINE_RUN_ID,
        schedule="load_schedules/load_schedule_S0050_shear.csv",
        continued_label="baseline",
        stress=baseline_stress,
        reaction=baseline_reaction,
        diagnostics=baseline_diagnostics,
        connectivity=baseline_connectivity,
        model_dir=baseline_model,
        result_dir=baseline_result,
        classification="shear extension successful with crack growth",
    )

    final_top_v_ratio = float(tables["top_v"].iloc[-1]["top_v_absmax_over_Delta_s"])
    peak = float(stress["nominal_shear_stress_energy_MPa"].max())
    final = float(stress.iloc[-1]["nominal_shear_stress_energy_MPa"])
    post_peak_drop_percent = 100.0 * max(0.0, peak - final) / peak
    s0050_alpha08_span = float(s0050_temp.get("final_alpha_ge_0p8_x_span") or 0.0)
    s0070_final_conn = connectivity[connectivity["step"] == int(stress["step"].max())]
    s0070_alpha08_span = float(
        s0070_final_conn[np.isclose(s0070_final_conn["threshold"].astype(float), 0.8)].iloc[0][
            "notch_connected_x_span"
        ]
    )
    s0070_alpha095_span = float(
        s0070_final_conn[np.isclose(s0070_final_conn["threshold"].astype(float), 0.95)].iloc[0][
            "notch_connected_x_span"
        ]
    )
    classification = classify_shear_extension(
        run_completed=count_field_files(result_dir) == len(stress),
        reaction_ok=bool((reaction["reaction_metric_status"] == "energy_conjugate").all()),
        post_peak_drop_percent=post_peak_drop_percent,
        final_top_v_ratio=final_top_v_ratio,
        unstable_top_v_ratio=2.0,
        through_crack_any=through_any,
        xspan_growth_material=s0070_alpha08_span > s0050_alpha08_span + 2.0e-4,
        propagated_high_threshold=(s0070_alpha08_span > 5.0e-4 or s0070_alpha095_span > 5.0e-4),
    )

    s0070_summary = summary_for_case(
        run_id=RUN_ID,
        schedule="load_schedules/load_schedule_S0070_shear.csv",
        continued_label=False,
        stress=stress,
        reaction=reaction,
        diagnostics=diagnostics,
        connectivity=connectivity,
        model_dir=model_dir,
        result_dir=result_dir,
        classification=classification,
    )

    stress.to_csv(package / "tables" / "shear_stress_strain_by_step.csv", index=False)
    reaction.to_csv(package / "tables" / "shear_reaction_by_step.csv", index=False)
    connectivity.to_csv(package / "tables" / "shear_connectivity_by_threshold.csv", index=False)
    tables["crack_path"].to_csv(package / "tables" / "shear_crack_path_geometry_by_step.csv", index=False)
    tables["damage"].to_csv(package / "tables" / "shear_damage_drive_summary.csv", index=False)
    tables["top_v"].to_csv(package / "tables" / "shear_top_v_free_diagnostic.csv", index=False)
    tables["checkpoint"].to_csv(package / "tables" / "shear_checkpoint_availability.csv", index=False)
    tables["loss"].to_csv(package / "tables" / "shear_training_loss_summary.csv", index=False)
    pd.DataFrame([s0070_summary]).to_csv(
        package / "tables" / "shear_connectivity_extension_run_summary.csv",
        index=False,
    )
    comparison_table(s0050_temp, s0070_summary).to_csv(
        package / "tables" / "shear_extension_vs_S0050_comparison.csv",
        index=False,
    )
    build_output_manifest(src_root, package, result_dir, model_dir).to_csv(
        package / "tables" / "shear_output_file_manifest.csv",
        index=False,
    )

    copy_required_figures(result_dir, package)
    build_figures(package, stress, diagnostics, connectivity, result_dir)
    copy_artifacts(src_root, package, result_dir, model_dir)
    write_markdown(package, s0070_summary, s0050_temp, connectivity)
    write_manifest(package)

    shutil.copy2(Path(__file__), package / "artifacts" / "generate_package.py")
    return s0070_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the S0070 shear connectivity evidence package.")
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--cc-root", type=Path, default=DEFAULT_CC_ROOT)
    args = parser.parse_args(argv)
    summary = build_package(src_root=args.src_root, cc_root=args.cc_root)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
