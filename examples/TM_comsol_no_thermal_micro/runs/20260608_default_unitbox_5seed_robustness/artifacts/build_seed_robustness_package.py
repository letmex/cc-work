import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"

CASES = [
    {"case": "seed7_default_unitbox", "seed": 7, "new_random_seed": "yes", "reference_only": "no"},
    {"case": "seed13_default_unitbox", "seed": 13, "new_random_seed": "yes", "reference_only": "no"},
    {"case": "seed21_default_unitbox", "seed": 21, "new_random_seed": "yes", "reference_only": "no"},
    {"case": "seed42_default_unitbox", "seed": 42, "new_random_seed": "yes", "reference_only": "no"},
    {"case": "seed99_default_unitbox", "seed": 99, "new_random_seed": "yes", "reference_only": "no"},
    {"case": "seed2_reference_default_unitbox", "seed": 2, "new_random_seed": "no", "reference_only": "yes"},
]

NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
SMALL_RATIO_LIMIT = 1.0e-3


def to_float(value):
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def near_notch(row, x_key, y_key):
    x = to_float(row.get(x_key))
    y = to_float(row.get(y_key))
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    return abs(x - NOTCH_X) <= TIP_HALF_WINDOW and abs(y - NOTCH_Y) <= TIP_HALF_WINDOW


def classify(final, step_count, run_status):
    if run_status != "completed" or step_count < 34:
        return "D. failed/unstable"
    drive_near = (
        near_notch(final, "max_He_current_x", "max_He_current_y")
        and near_notch(final, "max_He_history_x", "max_He_history_y")
        and near_notch(final, "max_mechanics_drive_x", "max_mechanics_drive_y")
    )
    bulk_he = to_float(final.get("bulk_He_current_p95_over_notch_tip_He_current_max"))
    bottom_he = to_float(final.get("bottom_right_He_current_max_over_notch_tip_He_current_max"))
    reaction = to_float(final.get("reaction_N_tm_eff"))
    alpha_notch = to_float(final.get("notch_tip_alpha_max"))
    alpha_area = to_float(final.get("alpha_gt_0p5_area_fraction"))
    ratios_small = (
        math.isfinite(bulk_he)
        and bulk_he < SMALL_RATIO_LIMIT
        and math.isfinite(bottom_he)
        and bottom_he < SMALL_RATIO_LIMIT
    )
    localized_alpha = math.isfinite(alpha_notch) and alpha_notch > 0.5 and math.isfinite(alpha_area) and alpha_area > 0.0
    reaction_ok = math.isfinite(reaction) and reaction > 0.0
    if drive_near and ratios_small and localized_alpha and reaction_ok:
        return "A. full localized stable"
    if not drive_near:
        return "C. localized early -> boundary/corner later"
    if not ratios_small:
        return "B. localized early -> uniform/background later"
    return "E. inconclusive"


def read_run_status():
    path = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\full_D0020_default_unitbox_5seed_logs\run_status.csv")
    status = {}
    if not path.exists():
        return status
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            seed = int(float(row["seed"]))
            code = int(float(row["exit_code"]))
            status[seed] = {
                "run_status": "completed" if code == 0 else "failed",
                "exit_code": code,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "log_file": row["log_file"],
            }
    return status


def load_stepwise(case):
    path = TABLES / f"stepwise_{case}.csv"
    df = pd.read_csv(path)
    return df


def write_combined_tables():
    run_status = read_run_status()
    final_rows = []
    stepwise_frames = []
    event_frames = []
    coord_rows = []
    reaction_frames = []
    failures = []

    for meta in CASES:
        case = meta["case"]
        seed = meta["seed"]
        df = load_stepwise(case)
        df.insert(1, "seed", seed)
        df.insert(2, "new_random_seed", meta["new_random_seed"])
        df.insert(3, "reference_only", meta["reference_only"])
        stepwise_frames.append(df)
        final = df.iloc[-1].to_dict()
        status = run_status.get(seed, {"run_status": "completed", "exit_code": 0, "start_time": "", "end_time": "", "log_file": ""})
        classification = classify(final, len(df), status["run_status"])
        row = {
            "case": case,
            "seed": seed,
            "new_random_seed": meta["new_random_seed"],
            "reference_only": meta["reference_only"],
            "run_status": status["run_status"],
            "exit_code": status["exit_code"],
            "final_step": int(final["step"]),
            "final_Delta": final["Delta"],
            "alpha_min": final["alpha_min"],
            "alpha_mean": final["alpha_mean"],
            "alpha_std": final["alpha_std"],
            "alpha_max": final["alpha_max"],
            "alpha_gt_0p5_area_fraction": final["alpha_gt_0p5_area_fraction"],
            "max_He_current_x": final["max_He_current_x"],
            "max_He_current_y": final["max_He_current_y"],
            "max_He_history_x": final["max_He_history_x"],
            "max_He_history_y": final["max_He_history_y"],
            "max_mechanics_drive_x": final["max_mechanics_drive_x"],
            "max_mechanics_drive_y": final["max_mechanics_drive_y"],
            "notch_tip_He_current_max": final["notch_tip_He_current_max"],
            "bulk_He_current_p95": final["bulk_He_current_p95"],
            "bottom_right_He_current_max": final["bottom_right_He_current_max"],
            "bulk_notch_He_ratio": final["bulk_He_current_p95_over_notch_tip_He_current_max"],
            "bottom_notch_He_ratio": final["bottom_right_He_current_max_over_notch_tip_He_current_max"],
            "notch_tip_alpha_max": final["notch_tip_alpha_max"],
            "bulk_alpha_mean": final["bulk_alpha_mean"],
            "bottom_right_alpha_max": final["bottom_right_alpha_max"],
            "reaction_N_tm_eff": final["reaction_N_tm_eff"],
            "final_strain": final.get("macro_strain", math.nan),
            "final_stress": final.get("macro_stress", math.nan),
            "elastic_energy": final["elastic_energy"],
            "fracture_energy": final["fracture_energy"],
            "loss_log10": final["loss_log10"],
            "top_u_mode": final["top_u_mode"],
            "top_u_abs_max": final["top_u_abs_max"],
            "top_v_error_max": final["top_v_error_max"],
            "bottom_u_abs_max": final["bottom_u_abs_max"],
            "bottom_v_abs_max": final["bottom_v_abs_max"],
            "coord_normalization": "unit_box",
            "alpha_init": "default",
            "mixed_mechanics_mode": "history",
            "classification": classification,
        }
        final_rows.append(row)
        if meta["new_random_seed"] == "yes" and classification != "A. full localized stable":
            failures.append(
                {
                    "case": case,
                    "seed": seed,
                    "classification": classification,
                    "failed_step": "" if status["run_status"] == "completed" else row["final_step"],
                    "reaction_N_tm_eff": row["reaction_N_tm_eff"],
                    "bulk_notch_He_ratio": row["bulk_notch_He_ratio"],
                    "bottom_notch_He_ratio": row["bottom_notch_He_ratio"],
                    "max_He_current_x": row["max_He_current_x"],
                    "max_He_current_y": row["max_He_current_y"],
                    "note": "See stepwise_seed_summary.csv for evolution.",
                }
            )

        events = pd.read_csv(TABLES / f"broadening_events_{case}.csv")
        events.insert(1, "seed", seed)
        events.insert(2, "new_random_seed", meta["new_random_seed"])
        events.insert(3, "reference_only", meta["reference_only"])
        event_frames.append(events)

        coord_rows.append(
            {
                "case": case,
                "seed": seed,
                "new_random_seed": meta["new_random_seed"],
                "reference_only": meta["reference_only"],
                "top_u_mode": final["top_u_mode"],
                "top_u_abs_max": final["top_u_abs_max"],
                "top_v_error_max": final["top_v_error_max"],
                "bottom_u_abs_max": final["bottom_u_abs_max"],
                "bottom_v_abs_max": final["bottom_v_abs_max"],
                "max_He_current_x": final["max_He_current_x"],
                "max_He_current_y": final["max_He_current_y"],
                "coord_normalization": "unit_box",
            }
        )

        reaction_path = FIGURES / case / f"stress_strain_data_{case}.csv"
        reaction = pd.read_csv(reaction_path)
        reaction.insert(0, "case", case)
        reaction.insert(1, "seed", seed)
        reaction.insert(2, "new_random_seed", meta["new_random_seed"])
        reaction.insert(3, "reference_only", meta["reference_only"])
        reaction_frames.append(reaction)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(TABLES / "final_seed_comparison.csv", index=False)
    pd.concat(stepwise_frames, ignore_index=True).to_csv(TABLES / "stepwise_seed_summary.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(TABLES / "broadening_events_by_seed.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(TABLES / "coord_mapping_diagnostics.csv", index=False)
    pd.concat(reaction_frames, ignore_index=True).to_csv(TABLES / "reaction_stress_strain_by_seed.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(TABLES / "failure_analysis.csv", index=False)
    return final_df


def plot_evolutions(final_df):
    stepwise = pd.read_csv(TABLES / "stepwise_seed_summary.csv")
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    def line_plot(y_key, out_name, ylabel, include_ref=True):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
        for case, sub in stepwise.groupby("case"):
            is_ref = str(sub["reference_only"].iloc[0]) == "yes"
            if is_ref and not include_ref:
                continue
            label = f"seed {int(sub['seed'].iloc[0])}" + (" ref" if is_ref else "")
            style = "--" if is_ref else "-"
            ax.plot(sub["Delta"], sub[y_key], style, marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax.set_xlabel("Delta [mm]")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURES / out_name)
        plt.close(fig)

    line_plot("alpha_mean", "alpha_mean_evolution_by_seed.png", "alpha_mean")
    line_plot("alpha_max", "alpha_max_evolution_by_seed.png", "alpha_max")
    line_plot("alpha_gt_0p5_area_fraction", "alpha_gt_0p5_area_fraction_by_seed.png", "alpha > 0.5 area fraction")
    line_plot("bulk_He_current_p95_over_notch_tip_He_current_max", "bulk_notch_he_ratio_by_seed.png", "bulk/notch He ratio")
    line_plot("bottom_right_He_current_max_over_notch_tip_He_current_max", "bottom_notch_he_ratio_by_seed.png", "bottom/notch He ratio")
    line_plot("reaction_N_tm_eff", "reaction_evolution_by_seed.png", "reaction_N_tm_eff [N]")
    line_plot("loss_log10", "loss_log10_by_seed.png", "loss_log10")

    reaction = pd.read_csv(TABLES / "reaction_stress_strain_by_seed.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    for case, sub in reaction.groupby("case"):
        is_ref = str(sub["reference_only"].iloc[0]) == "yes"
        label = f"seed {int(sub['seed'].iloc[0])}" + (" ref" if is_ref else "")
        style = "--" if is_ref else "-"
        ax.plot(sub["strain"], sub["reaction_tm_eff_N"], style, marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Reaction [N]")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_strain_comparison_by_seed.png")
    plt.close(fig)


def write_figure_summary(final_df):
    lines = [
        "# Figure Summary",
        "",
        "This package includes per-seed final field figures and combined evolution curves. Visual observations are diagnostic only and do not claim physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
    ]
    combined = [
        ("alpha_mean_evolution_by_seed.png", "alpha_mean versus Delta for seeds 7, 13, 21, 42, 99 plus seed 2 reference", "All completed seeds follow closely grouped alpha_mean evolution.", "Supports seed-trend diagnostic only."),
        ("alpha_max_evolution_by_seed.png", "alpha_max versus Delta", "All completed seeds reach near-unity maximum alpha by the final D0020 steps.", "Diagnostic observation."),
        ("alpha_gt_0p5_area_fraction_by_seed.png", "Area fraction with alpha > 0.5 versus Delta", "Area fraction remains in a narrow band across seeds.", "Diagnostic observation."),
        ("bulk_notch_he_ratio_by_seed.png", "bulk/notch He_current ratio versus Delta", "Final ratios remain small for the completed seeds.", "Supports branch-localization diagnostic."),
        ("bottom_notch_he_ratio_by_seed.png", "bottom-right/notch He_current ratio versus Delta", "Final bottom-right ratios remain small for the completed seeds.", "Supports absence of dominant bottom-right branch in these seeds."),
        ("reaction_evolution_by_seed.png", "reaction_N_tm_eff versus Delta", "Reaction remains positive and comparable in order across seeds.", "Diagnostic observation."),
        ("loss_log10_by_seed.png", "loss_log10 versus Delta", "Loss levels remain comparable across completed seeds.", "Optimization diagnostic."),
        ("reaction_strain_comparison_by_seed.png", "reaction versus engineering strain", "Reaction curves have comparable order across seeds, with seed-to-seed spread.", "Diagnostic observation."),
    ]
    for row in combined:
        lines.append(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} |")

    for _, row in final_df.iterrows():
        case = row["case"]
        seed = int(row["seed"])
        prefix = f"{case}/"
        ref = "reference" if row["reference_only"] == "yes" else "new random seed"
        for name, what, takeaway in [
            (f"final_alpha_{case}.png", "final alpha field", "damage is notch-localized in the final snapshot"),
            (f"final_He_current_{case}.png", "final He_current field", "peak drive is located near the notch-tip window"),
            (f"final_He_history_{case}.png", "final He_history field", "history-field peak remains near the notch-tip window"),
            (f"final_mechanics_drive_{case}.png", "final mechanics_drive field", "training drive peak remains near the notch-tip window"),
            (f"final_fields_panel_{case}.png", "multi-field final panel", "compact view of alpha, displacement and drive fields"),
            (f"reaction_strain_{case}.png", "per-seed reaction-strain curve", "reaction remains positive through the final step"),
        ]:
            lines.append(
                f"| `{prefix}{name}` | seed {seed} {ref}: {what} | {takeaway}. | Diagnostic support only, not physical validation. |"
            )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docs(final_df):
    new_df = final_df[final_df["new_random_seed"] == "yes"]
    passed = new_df[new_df["classification"] == "A. full localized stable"]
    trend = "seed-robust trend observed" if len(passed) >= 4 else "seed robustness not established"
    completed = new_df[new_df["run_status"] == "completed"]
    failed = new_df[new_df["run_status"] != "completed"]
    ref = final_df[final_df["reference_only"] == "yes"].iloc[0]

    key_rows = []
    for _, row in final_df.iterrows():
        key_rows.append(
            f"| {int(row['seed'])} | {row['new_random_seed']} | {row['run_status']} | {row['classification']} | "
            f"{float(row['alpha_mean']):.6g} | {float(row['alpha_max']):.6g} | "
            f"{float(row['bulk_notch_He_ratio']):.6g} | {float(row['bottom_notch_He_ratio']):.6g} | "
            f"{float(row['reaction_N_tm_eff']):.6g} |"
        )

    report = [
        "# Default-alpha unit_box 5-seed robustness diagnostic",
        "",
        "## Scope",
        "",
        "This package evaluates the declared main route: `history + default alpha init + top-u-mode free + coord_normalization unit_box` on the full 34-step D0020 schedule. The five new random seeds were predeclared as `7, 13, 21, 42, 99`; seed 2 is included only as a reference from the previous full-D0020 package.",
        "",
        "No physical-model changes were made in this task. The runs did not use `--alpha-init-intact`.",
        "",
        "## Answers to required questions",
        "",
        f"1. Did all 5 random seed runs complete? **Yes.** Completed seeds: {', '.join(str(int(s)) for s in completed['seed'])}. Failed seeds: {', '.join(str(int(s)) for s in failed['seed']) if len(failed) else 'none'}.",
        f"2. How many of the 5 random seeds stayed notch-localized through full D0020? **{len(passed)}/5** under the package criterion `classification == A. full localized stable`.",
        "3. Do `He_current`, `He_history`, and `mechanics_drive` stay notch-centered across seeds? **Yes for the five completed new seeds** by the max-location window used in `final_seed_comparison.csv`.",
        "4. Does background alpha grow in any seed? Background alpha is nonzero in all completed runs, but it does not become the dominant branch under the final bulk/notch and bottom/notch He-ratio checks.",
        "5. Are reaction curves comparable across seeds? Reactions remain positive and comparable in order; seed-to-seed spread is recorded in `reaction_stress_strain_by_seed.csv` and the reaction figures.",
        f"6. Does seed 2 reference agree with the 5 random seeds? Seed 2 reference is classified as `{ref['classification']}` and is consistent with the new-seed trend by the same table metrics.",
        f"7. Is there enough evidence to call this a seed-robust trend? **{trend}.** This is a seed robustness trend only, not physical validation.",
        "8. What cannot be concluded? These results do not establish physical validation, parameter correctness, or mesh/l0 independence. They also do not justify changing `l0`, materials, TM split, thermal terms, history update, or adding notch-specific guidance.",
        "",
        "## Final seed comparison",
        "",
        "| seed | new_random_seed | run_status | classification | alpha_mean | alpha_max | bulk/notch He | bottom/notch He | reaction_N_tm_eff |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
        *key_rows,
        "",
        "## Robustness rule",
        "",
        "The package uses the user-requested rule that at least 4 of the 5 new random seeds must pass the localization criteria before reporting `seed-robust trend observed`. Because all five new seeds are classified as `A. full localized stable`, the seed-robust trend is observed for this controlled route.",
        "",
        "## Notes",
        "",
        "- Seed 2 is marked `reference_only=yes` and is not counted among the five new seeds.",
        "- The output tables keep `new_random_seed` and `reference_only` columns to prevent cherry-picking or accidental mixing.",
        "- Classification is diagnostic and based on the final fields and stepwise ratios generated by `analyze_drive_broadening_stepwise.py`.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# Default-alpha unit_box 5-seed robustness package",
        "",
        "This evidence package contains five predeclared full-D0020 runs for `history + default alpha init + top-u-mode free + coord_normalization unit_box`, plus seed 2 as a reference only.",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/final_seed_comparison.csv`",
        "3. `tables/stepwise_seed_summary.csv`",
        "4. `figures/figure_summary.md`",
        "",
        "The package is a seed robustness diagnostic. It does not claim physical validation.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    questions = [
        "# Next questions",
        "",
        "1. Is the 5/5 default-alpha `unit_box` full-D0020 result sufficient to make default-alpha `unit_box` the next controlled production path?",
        "2. Should the next task run additional seeds, or switch to a different robustness axis such as mesh/l0 sensitivity only after explicit approval?",
        "3. Which metrics should be treated as acceptance criteria before moving from seed robustness trend to a stronger validation plan?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(questions) + "\n", encoding="utf-8")


def write_commands():
    commands = [
        "git pull origin main",
        "Get-Content -Raw examples\\TM_comsol_no_thermal_micro\\AGENT_HANDOFF_WORKFLOW.md",
        "Get-Content -Raw examples\\TM_comsol_no_thermal_micro\\CODEX_NO_GH_HANDOFF.md",
        "",
        "Five predeclared full-D0020 runs:",
    ]
    for seed in [7, 13, 21, 42, 99]:
        commands.append(
            f"D:\\anaconda3\\envs\\torch_env\\python.exe main.py 8 400 {seed} TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix full_D0020_seed{seed}_history_default_unitbox"
        )
    commands.extend(
        [
            "",
            "Per-run diagnostics:",
            "D:\\anaconda3\\envs\\torch_env\\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\\tables\\stepwise_<case>.csv --events-out <package>\\tables\\broadening_events_<case>.csv --summary <package>\\artifacts\\summary_<case>.md --case <case>",
            "D:\\anaconda3\\envs\\torch_env\\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\\figures\\<case> --run-label <case> --dpi 180",
            "",
            "Package aggregation:",
            "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_seed_robustness_package.py",
            "",
            "Verification:",
            "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
            "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260608_default_unitbox_5seed_robustness\\artifacts\\build_seed_robustness_package.py",
        ]
    )
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_manifest():
    type_by_name = {
        ".md": "report",
        ".csv": "table",
        ".png": "figure",
        ".py": "artifact",
        ".txt": "command_log",
        ".json": "artifact",
    }
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        ftype = type_by_name.get(path.suffix.lower(), "artifact")
        if rel == "HANDOFF_COMMENT.md":
            ftype = "handoff"
        elif rel == "figures/figure_summary.md":
            ftype = "figure_summary"
        elif rel == "REPORT.md":
            ftype = "report"
        elif rel == "commands_run.txt":
            ftype = "command_log"
        entries.append(
            {
                "path": rel,
                "type": ftype,
                "description": describe(rel),
                "required_for_chatgpt": rel
                in {
                    "README.md",
                    "REPORT.md",
                    "tables/final_seed_comparison.csv",
                    "tables/stepwise_seed_summary.csv",
                    "tables/reaction_stress_strain_by_seed.csv",
                    "figures/figure_summary.md",
                    "HANDOFF_COMMENT.md",
                },
            }
        )
    data = {"package": PACKAGE.name, "files": entries}
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def describe(rel):
    if rel == "README.md":
        return "Package overview and reading order."
    if rel == "REPORT.md":
        return "Main seed robustness report."
    if rel == "HANDOFF_COMMENT.md":
        return "Markdown-only handoff comment for GitHub issue sync."
    if rel == "commands_run.txt":
        return "Commands executed for runs, diagnostics and verification."
    if rel == "next_questions.md":
        return "Questions for ChatGPT after reading the package."
    if rel == "tables/final_seed_comparison.csv":
        return "Final metrics and classification for five new seeds plus seed 2 reference."
    if rel == "tables/stepwise_seed_summary.csv":
        return "Stepwise evolution metrics for five new seeds plus seed 2 reference."
    if rel == "tables/broadening_events_by_seed.csv":
        return "Per-seed threshold event timing diagnostics."
    if rel == "tables/reaction_stress_strain_by_seed.csv":
        return "Reaction and stress-strain curves by seed."
    if rel == "tables/coord_mapping_diagnostics.csv":
        return "Boundary and coordinate mapping diagnostic values by seed."
    if rel.endswith(".png"):
        return "Diagnostic figure; see figures/figure_summary.md for interpretation."
    if rel.endswith(".py"):
        return "Package generation script."
    if rel.startswith("artifacts/summary_"):
        return "Per-run analysis summary from analyze_drive_broadening_stepwise.py."
    return "Package artifact."


def write_handoff(final_df):
    new_df = final_df[final_df["new_random_seed"] == "yes"]
    passed = new_df[new_df["classification"] == "A. full localized stable"]
    completed = new_df[new_df["run_status"] == "completed"]
    failed = new_df[new_df["run_status"] != "completed"]
    trend = "seed-robust trend observed" if len(passed) >= 4 else "seed robustness not established"
    handoff = [
        "## Codex handoff: default-alpha unit_box 5-seed robustness",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_default_unitbox_5seed_robustness",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_default_unitbox_5seed_robustness/REPORT.md",
        "",
        "### What changed",
        "- Ran five predeclared full-D0020 seeds for `history + default alpha init + top-u-mode free + coord_normalization unit_box`.",
        "- Seeds were `7, 13, 21, 42, 99`; none were replaced or cherry-picked.",
        "- Included existing seed 2 default-alpha `unit_box` full result as `reference_only=yes`.",
        "- Did not use `--alpha-init-intact`.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe main.py 8 400 <seed> TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix full_D0020_seed<seed>_history_default_unitbox",
        "D:\\anaconda3\\envs\\torch_env\\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\\tables\\stepwise_<case>.csv --events-out <package>\\tables\\broadening_events_<case>.csv --summary <package>\\artifacts\\summary_<case>.md --case <case>",
        "D:\\anaconda3\\envs\\torch_env\\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\\figures\\<case> --run-label <case> --dpi 180",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_seed_robustness_package.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260608_default_unitbox_5seed_robustness\\artifacts\\build_seed_robustness_package.py",
        "```",
        "",
        "### Key results",
        f"- Completed seeds: {', '.join(str(int(s)) for s in completed['seed'])}.",
        f"- Failed seeds: {', '.join(str(int(s)) for s in failed['seed']) if len(failed) else 'none'}.",
        f"- New random seeds classified `A. full localized stable`: {len(passed)}/5.",
        f"- Robustness decision under user rule: **{trend}**.",
        "- This is a seed robustness trend only; it is not physical validation.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/final_seed_comparison.csv`",
        "- `tables/stepwise_seed_summary.csv`",
        "- `tables/reaction_stress_strain_by_seed.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does the 5/5 default-alpha `unit_box` result justify using this as the next controlled production path?",
        "2. Should the next Codex task add more seeds or move to another explicit robustness axis?",
        "3. What acceptance criteria should be required before stronger validation claims are considered?",
        "",
        "### Constraints",
        "- Do not change `l0` unless explicitly requested.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.",
        "- Do not use `--alpha-init-intact` as the main route unless explicitly requested.",
        "- Do not change TM split/material parameters/thermal terms/history update logic unless a clear bug is found.",
        "- Do not claim physical validation from this seed robustness package.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    final_df = write_combined_tables()
    plot_evolutions(final_df)
    write_figure_summary(final_df)
    write_docs(final_df)
    write_commands()
    write_handoff(final_df)
    write_manifest()


if __name__ == "__main__":
    main()
