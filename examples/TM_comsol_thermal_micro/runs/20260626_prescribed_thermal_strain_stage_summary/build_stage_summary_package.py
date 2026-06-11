from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
THERMAL_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
TABLES_DIR = PACKAGE_DIR / "tables"
FIGURES_DIR = PACKAGE_DIR / "figures"

FINAL_CLASSIFICATION = "prescribed thermal strain stage summary complete"

PKG = {
    "scaffold": THERMAL_ROOT / "runs" / "20260620_thermal_subproject_scaffold",
    "patch": THERMAL_ROOT / "runs" / "20260621_prescribed_thermal_strain_patch_tests",
    "micro": THERMAL_ROOT / "runs" / "20260622_prescribed_temperature_micro_notch_diagnostic",
    "strong": THERMAL_ROOT / "runs" / "20260623_stronger_prescribed_temperature_tension_diagnostic",
    "audit": THERMAL_ROOT / "runs" / "20260624_caseC_alpha_anomaly_audit",
    "probe": THERMAL_ROOT / "runs" / "20260625_prescribed_temperature_tension_damage_probe",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def row(rows: list[dict[str, str]], **criteria: object) -> dict[str, str]:
    for item in rows:
        if all(str(item.get(key, "")) == str(value) for key, value in criteria.items()):
            return item
    raise KeyError(criteria)


def max_step(rows: list[dict[str, str]], case_id: str | None = None) -> str:
    selected = rows if case_id is None else [item for item in rows if item.get("case_id") == case_id]
    return str(max(int(float(item["step"])) for item in selected))


def fmt(value: object, digits: int = 12) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def evidence_values() -> dict[str, str]:
    micro_cmp = read_csv(PKG["micro"] / "tables" / "micro_notch_thermal_case_comparison.csv")
    strong_cmp = read_csv(PKG["strong"] / "tables" / "strong_thermal_case_comparison.csv")
    probe_cmp = read_csv(PKG["probe"] / "tables" / "damage_probe_case_comparison.csv")
    probe_delay = read_csv(PKG["probe"] / "tables" / "damage_delay_summary.csv")
    probe_thr = read_csv(PKG["probe"] / "tables" / "alpha_threshold_connectivity_by_step.csv")
    audit_diff = read_csv(PKG["audit"] / "tables" / "alpha_difference_statistics.csv")
    audit_thr = read_csv(PKG["audit"] / "tables" / "alpha_threshold_area_connectivity.csv")

    micro_stress = row(micro_cmp, comparison="A_vs_C_positive_deltaT", metric="final_nominal_stress_shift_MPa")
    strong_stress = row(strong_cmp, comparison="A_vs_C_positive_deltaT", metric="final_nominal_stress_MPa")
    strong_alpha = row(strong_cmp, comparison="A_vs_C_positive_deltaT", metric="final_alpha_max")
    strong_cross = row(strong_cmp, comparison="A_vs_C_positive_deltaT", metric="C_reaction_zero_crossing_displacement_mm")
    probe_stress = row(probe_cmp, comparison="C_vs_A", metric="final reaction shift")
    probe_alpha = row(probe_cmp, comparison="C_vs_A", metric="final alpha max")
    probe_hi = row(probe_cmp, comparison="C_vs_A", metric="final HI peak")
    probe_hii = row(probe_cmp, comparison="C_vs_A", metric="final HII peak")

    audit_step = max_step(audit_diff)
    audit_ca = row(audit_diff, comparison="C_minus_A", step=audit_step)
    audit_c_step = max_step(audit_thr, "C")
    audit_t001 = row(audit_thr, case_id="C", step=audit_c_step, threshold="0.001")
    audit_t01 = row(audit_thr, case_id="C", step=audit_c_step, threshold="0.01")
    audit_t03 = row(audit_thr, case_id="C", step=audit_c_step, threshold="0.03")
    audit_t05 = row(audit_thr, case_id="C", step=audit_c_step, threshold="0.05")

    probe_c_step = max_step(probe_thr, "C")
    probe_t = {
        threshold: row(probe_thr, case_id="C", step=probe_c_step, threshold=threshold)
        for threshold in ("0.02", "0.03", "0.05", "0.1")
    }
    delay_002 = row(probe_delay, metric="first displacement where alpha_max >= threshold", threshold_or_quantity="0.02")
    delay_005 = row(probe_delay, metric="first displacement where alpha_max >= threshold", threshold_or_quantity="0.05")

    return {
        "micro_A_final_stress": fmt(micro_stress["left_value"]),
        "micro_C_final_stress": fmt(micro_stress["right_value"]),
        "micro_C_minus_A_stress": fmt(float(micro_stress["right_value"]) - float(micro_stress["left_value"])),
        "strong_A_final_stress": fmt(strong_stress["left_value"]),
        "strong_C_final_stress": fmt(strong_stress["right_value"]),
        "strong_C_minus_A_stress": fmt(float(strong_stress["right_value"]) - float(strong_stress["left_value"])),
        "strong_A_alpha": fmt(strong_alpha["left_value"]),
        "strong_C_alpha": fmt(strong_alpha["right_value"]),
        "strong_zero_cross": fmt(strong_cross["right_value"]),
        "strong_zero_cross_estimate": fmt(strong_cross["left_value"]),
        "probe_A_final_stress": fmt(probe_stress["right_value"]),
        "probe_C_final_stress": fmt(probe_stress["left_value"]),
        "probe_C_minus_A_stress": fmt(probe_stress["absolute_difference"]),
        "probe_A_alpha": fmt(probe_alpha["right_value"]),
        "probe_C_alpha": fmt(probe_alpha["left_value"]),
        "probe_C_minus_A_alpha": fmt(probe_alpha["absolute_difference"]),
        "probe_C_HI": fmt(probe_hi["left_value"]),
        "probe_C_HII": fmt(probe_hii["left_value"]),
        "audit_ca_median": fmt(audit_ca["diff_median"]),
        "audit_ca_positive_max": fmt(audit_ca["diff_max"]),
        "audit_ca_negative_min": fmt(audit_ca["diff_min"]),
        "audit_C_thr_001_frac": fmt(audit_t001["fraction_above_threshold"]),
        "audit_C_thr_01_frac": fmt(audit_t01["fraction_above_threshold"]),
        "audit_C_thr_03_frac": fmt(audit_t03["fraction_above_threshold"]),
        "audit_C_thr_05_frac": fmt(audit_t05["fraction_above_threshold"]),
        "probe_C_thr_002_frac": fmt(probe_t["0.02"]["fraction_above_threshold"]),
        "probe_C_thr_003_frac": fmt(probe_t["0.03"]["fraction_above_threshold"]),
        "probe_C_thr_005_frac": fmt(probe_t["0.05"]["fraction_above_threshold"]),
        "probe_C_thr_01_frac": fmt(probe_t["0.1"]["fraction_above_threshold"]),
        "probe_C_thr_002_notch": probe_t["0.02"]["notch_connected_count"],
        "probe_C_thr_003_notch": probe_t["0.03"]["notch_connected_count"],
        "probe_C_thr_005_notch": probe_t["0.05"]["notch_connected_count"],
        "probe_C_thr_01_notch": probe_t["0.1"]["notch_connected_count"],
        "probe_delay_alpha_002_A": delay_002["case_A_displacement"],
        "probe_delay_alpha_002_C": delay_002["case_C_displacement"],
        "probe_delay_alpha_005_A": delay_005["case_A_displacement"],
        "probe_delay_alpha_005_C": delay_005["case_C_displacement"],
    }


def milestone_rows(v: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "milestone_id": "M1",
            "package_path": rel(PKG["scaffold"]),
            "date_or_run_label": "20260620_thermal_subproject_scaffold",
            "task_type": "thermal subproject scaffold",
            "source_modified": "yes, new isolated thermal subproject copied from verified no-thermal baseline",
            "training_run": "no",
            "validation_type": "copy manifest, heavy-artifact exclusion, py_compile, lightweight pytest",
            "final_classification": "thermal subproject scaffold created from verified no-thermal baseline",
            "key_result": "Created examples/TM_comsol_thermal_micro without modifying the frozen no-thermal baseline.",
            "caveat": "No thermal strain or heat physics was implemented at scaffold time.",
            "reviewer_read_next": rel(PKG["scaffold"] / "REPORT.md"),
        },
        {
            "milestone_id": "M2",
            "package_path": rel(PKG["patch"]),
            "date_or_run_label": "20260621_prescribed_thermal_strain_patch_tests",
            "task_type": "prescribed thermal strain implementation and patch tests",
            "source_modified": "yes, thermal subproject only",
            "training_run": "no",
            "validation_type": "focused pytest, compileall, guard scans",
            "final_classification": "prescribed thermal strain branch implemented and patch tests passed",
            "key_result": "Default thermal mode remains off; delta_T=0 matches no-thermal split; free expansion and constrained heating checks passed.",
            "caveat": "Patch tests validate wiring and signs, not COMSOL or experimental physical agreement.",
            "reviewer_read_next": rel(PKG["patch"] / "REPORT.md"),
        },
        {
            "milestone_id": "M3",
            "package_path": rel(PKG["micro"]),
            "date_or_run_label": "20260622_prescribed_temperature_micro_notch_diagnostic",
            "task_type": "small smoke-like micro-notch diagnostic",
            "source_modified": "no source behavior change; schedule/package artifacts only",
            "training_run": "yes, A/B/C D0003 tension smoke diagnostic",
            "validation_type": "A/B/C checkpointed reaction and patch-test recheck",
            "final_classification": "prescribed-temperature micro-notch diagnostic passed",
            "key_result": f"A/B matched exactly; Case C final nominal stress {v['micro_C_final_stress']} MPa versus A {v['micro_A_final_stress']} MPa.",
            "caveat": "Very small schedule and smoke training; damage conclusions are limited.",
            "reviewer_read_next": rel(PKG["micro"] / "REPORT.md"),
        },
        {
            "milestone_id": "M4",
            "package_path": rel(PKG["strong"]),
            "date_or_run_label": "20260623_stronger_prescribed_temperature_tension_diagnostic",
            "task_type": "non-smoke compensation-region tension diagnostic",
            "source_modified": "no source behavior change; schedule/package artifacts only",
            "training_run": "yes, A/B/C D0015 full tension diagnostic",
            "validation_type": "A/B equivalence, C reaction shift, compensation crossing, HI/HII and checkpoint checks",
            "final_classification": "strong prescribed-temperature tension diagnostic passed",
            "key_result": f"C final stress {v['strong_C_final_stress']} MPa versus A {v['strong_A_final_stress']} MPa; C final alpha {v['strong_C_alpha']} versus A {v['strong_A_alpha']}.",
            "caveat": "Single seed and tension only; broad Case C alpha appearance required audit.",
            "reviewer_read_next": rel(PKG["strong"] / "REPORT.md"),
        },
        {
            "milestone_id": "M5",
            "package_path": rel(PKG["audit"]),
            "date_or_run_label": "20260624_caseC_alpha_anomaly_audit",
            "task_type": "post hoc audit of Case C alpha field",
            "source_modified": "no",
            "training_run": "no",
            "validation_type": "raw field distribution, threshold connectivity, scale audit, spatial correlation",
            "final_classification": "caseC diffuse alpha likely plotting-scale artifact plus low-amplitude background",
            "key_result": f"Final C-minus-A alpha median {v['audit_ca_median']}, positive max {v['audit_ca_positive_max']}; low-level background is present but scale-sensitive.",
            "caveat": "Audit cannot validate diffuse low-amplitude alpha as physical fracture damage.",
            "reviewer_read_next": rel(PKG["audit"] / "REPORT.md"),
        },
        {
            "milestone_id": "M6",
            "package_path": rel(PKG["probe"]),
            "date_or_run_label": "20260625_prescribed_temperature_tension_damage_probe",
            "task_type": "moderate prescribed-temperature tension damage probe",
            "source_modified": "no source behavior change; schedule/package artifacts only",
            "training_run": "yes, A/B/C D0020 full tension diagnostic",
            "validation_type": "A/B equivalence, C reaction and high-threshold/notch alpha trends, checkpointed reaction",
            "final_classification": "moderate prescribed-temperature damage probe passed",
            "key_result": f"C final stress {v['probe_C_final_stress']} MPa versus A {v['probe_A_final_stress']} MPa; C final alpha {v['probe_C_alpha']} versus A {v['probe_A_alpha']}.",
            "caveat": "Still prescribed-temperature only, single seed, tension only, not physical validation.",
            "reviewer_read_next": rel(PKG["probe"] / "REPORT.md"),
        },
    ]


def evidence_matrix(v: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "evidence_item": "Default no-thermal route remains available",
            "supporting_package": rel(PKG["patch"]),
            "supporting_file": "REPORT.md; thermal_prescribed.py; config.py",
            "observed_result": "thermal_mode defaults to off and delta_T is zero unless explicitly prescribed.",
            "trust_level": "high",
            "reason": "Patch tests and current CLI defaults directly exercise the route.",
            "limitation": "This does not prove future heat-PDE coupling behavior.",
        },
        {
            "evidence_item": "Uniform delta_T=0 matches thermal_mode=off",
            "supporting_package": rel(PKG["strong"]) + "; " + rel(PKG["probe"]),
            "supporting_file": "tables/strong_thermal_case_comparison.csv; tables/damage_probe_case_comparison.csv",
            "observed_result": "A/B differences are zero within table precision for reaction, stress, energy, alpha, HI, and HII in completed diagnostics.",
            "trust_level": "high",
            "reason": "Repeated across patch tests, smoke diagnostic, full D0015 diagnostic, and D0020 probe.",
            "limitation": "Same deterministic seed and tension setup in run diagnostics.",
        },
        {
            "evidence_item": "Prescribed +20 K shifts reaction/stress downward",
            "supporting_package": rel(PKG["strong"]) + "; " + rel(PKG["probe"]),
            "supporting_file": "tables/strong_thermal_case_comparison.csv; tables/damage_probe_case_comparison.csv",
            "observed_result": f"Strong C-A final stress shift {v['strong_C_minus_A_stress']} MPa; moderate probe C-A final stress shift {v['probe_C_minus_A_stress']} MPa.",
            "trust_level": "high",
            "reason": "Observed consistently in checkpointed energy-conjugate reaction tables.",
            "limitation": "Displacement-controlled tension diagnostics only.",
        },
        {
            "evidence_item": "Prescribed +20 K reduces notch/high-threshold alpha in moderate probe",
            "supporting_package": rel(PKG["probe"]),
            "supporting_file": "tables/damage_probe_case_comparison.csv; tables/alpha_threshold_connectivity_by_step.csv",
            "observed_result": f"Final alpha C {v['probe_C_alpha']} versus A {v['probe_A_alpha']}; C >=0.05 area fraction {v['probe_C_thr_005_frac']}.",
            "trust_level": "medium",
            "reason": "Moderate D0020 diagnostic uses high-threshold/notch metrics and checkpointed fields.",
            "limitation": "Single seed and no external physical validation.",
        },
        {
            "evidence_item": "Broad low-level Case C alpha cloud",
            "supporting_package": rel(PKG["audit"]),
            "supporting_file": "tables/alpha_threshold_area_connectivity.csv; tables/artifact_risk_assessment.csv",
            "observed_result": f"Final Case C >=0.001 fraction {v['audit_C_thr_001_frac']} and >=0.01 fraction {v['audit_C_thr_01_frac']} in the strong diagnostic audit.",
            "trust_level": "diagnostic_only",
            "reason": "The raw element values show the background, but its visual impact is scale-sensitive.",
            "limitation": "Do not treat as physical fracture evidence without further validation.",
        },
        {
            "evidence_item": "No heat PDE, trainable/PDE temperature, or damage-dependent conductivity",
            "supporting_package": rel(PKG["patch"]) + "; " + rel(PKG["probe"]),
            "supporting_file": "THERMAL_STRAIN_PATCH_TESTS.md; tables/no_heat_pde_guard_summary.csv",
            "observed_result": "Only prescribed delta_T/temperature fields are routed into mechanics.",
            "trust_level": "high",
            "reason": "Guard tables and current source scan show no heat residual or k(d)=g(d)k0 coupling.",
            "limitation": "Future implementation will need separate tests.",
        },
        {
            "evidence_item": "Physical agreement with COMSOL or experiment",
            "supporting_package": rel(PKG["micro"]) + "; " + rel(PKG["probe"]),
            "supporting_file": "REPORT.md",
            "observed_result": "Reports explicitly state these are software/physics-route diagnostics, not physical validation.",
            "trust_level": "not_supported",
            "reason": "No independent COMSOL solve or experimental data comparison is included.",
            "limitation": "Cannot claim quantitative physical predictive accuracy from this stage.",
        },
    ]


def trusted_findings(v: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "finding": "thermal_mode=off remains the default/no-thermal route.",
            "evidence": "Patch-test report and current config.py default thermal mode are off.",
            "trust_level": "high",
            "why_trusted": "Direct source-level default plus patch tests.",
            "scope_limit": "Thermal subproject only; original no-thermal project remains separate.",
        },
        {
            "finding": "thermal_mode=uniform, delta_T=0 matches no-thermal route in completed diagnostics.",
            "evidence": "A/B comparison tables report zero differences within table precision in micro, strong, and moderate diagnostics.",
            "trust_level": "high",
            "why_trusted": "Repeated under smoke and full training schedules.",
            "scope_limit": "Same deterministic seed and tension route for diagnostic runs.",
        },
        {
            "finding": "Prescribed +20 K thermal strain shifts reaction/stress downward in displacement-controlled tension.",
            "evidence": f"Strong shift {v['strong_C_minus_A_stress']} MPa; moderate probe shift {v['probe_C_minus_A_stress']} MPa.",
            "trust_level": "high",
            "why_trusted": "Checkpointed energy-conjugate reaction is available and used as primary metric.",
            "scope_limit": "Tension diagnostics only, not shear or external validation.",
        },
        {
            "finding": "Prescribed +20 K reduces notch-tip/high-threshold alpha growth in the moderate damage probe.",
            "evidence": f"Moderate final alpha C {v['probe_C_alpha']} versus A {v['probe_A_alpha']}; alpha>=0.05 final Case C fraction {v['probe_C_thr_005_frac']}.",
            "trust_level": "medium",
            "why_trusted": "High-threshold/notch metrics were separated from low-level background diagnostics.",
            "scope_limit": "Single seed and prescribed-temperature only.",
        },
        {
            "finding": "Checkpointed energy-conjugate reaction remains available and is the primary reaction.",
            "evidence": "Micro, strong, and moderate reports state all steps have energy-conjugate reaction availability.",
            "trust_level": "high",
            "why_trusted": "Postprocess tables and handoffs consistently use reaction_N_energy.",
            "scope_limit": "Requires saved step checkpoints.",
        },
        {
            "finding": "No heat PDE or damage-dependent conductivity was implemented in this stage.",
            "evidence": "Guard reports, PROJECT_MEMORY.md, and source scope show no heat residual, trainable temperature field, or k(d)=g(d)k0 coupling.",
            "trust_level": "high",
            "why_trusted": "Multiple package guard checks agree with current source scope.",
            "scope_limit": "This is a guard conclusion, not validation of future heat transport.",
        },
    ]


def diagnostic_only_findings(v: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "finding": "Broad low-level Case C alpha cloud.",
            "evidence": f"Strong audit final C >=0.001 fraction {v['audit_C_thr_001_frac']}, >=0.01 fraction {v['audit_C_thr_01_frac']}, and C-minus-A positive max {v['audit_ca_positive_max']}.",
            "why_diagnostic_only": "The audit found raw low-amplitude background, but scale, single-run, and artifact risks remain.",
            "do_not_use_for": "Do not use as physical fracture evidence or as proof of diffuse thermal damage.",
            "recommended_future_check": "Use high-threshold/notch metrics and compare against independent validation or repeat seeds before interpreting diffuse alpha.",
        },
        {
            "finding": "Single-seed behavior.",
            "evidence": "Completed training diagnostics used seed 23.",
            "why_diagnostic_only": "No seed repeat was run.",
            "do_not_use_for": "Do not use as a statistical robustness claim.",
            "recommended_future_check": "Run a small seed repeat only if the decision gate requires robustness evidence before heat PDE planning.",
        },
        {
            "finding": "Tension-only thermal diagnostic conclusions.",
            "evidence": "Micro, strong, and moderate diagnostics used load_case=tension.",
            "why_diagnostic_only": "Shear/S0110 extension was explicitly not run.",
            "do_not_use_for": "Do not generalize to shear or mixed external loading.",
            "recommended_future_check": "Defer shear/S0110 unless a reviewer explicitly opens that branch.",
        },
        {
            "finding": "Lack of physical validation against COMSOL or experiment.",
            "evidence": "Reports explicitly say the diagnostics are not physical validation.",
            "why_diagnostic_only": "No independent COMSOL solve or experimental benchmark was compared.",
            "do_not_use_for": "Do not claim quantitative physical predictive accuracy.",
            "recommended_future_check": "Define a COMSOL or controlled benchmark comparison before physical claims.",
        },
        {
            "finding": "No conclusion about thermal conduction or damage-dependent conductivity.",
            "evidence": "No heat PDE and no k(d)=g(d)k0 were implemented or run.",
            "why_diagnostic_only": "The temperature input is prescribed rather than solved.",
            "do_not_use_for": "Do not infer heat transport, transient conduction, or conductivity degradation behavior.",
            "recommended_future_check": "Plan heat PDE first; defer damage-dependent conductivity until heat PDE is stable.",
        },
    ]


def limitations() -> list[dict[str, object]]:
    return [
        {"limitation_or_risk": "no heat equation yet", "current_status": "not implemented", "consequence": "Temperature is prescribed and cannot evolve from conduction or sources.", "severity": "high", "mitigation_or_next_check": "Hold a decision-gate review before heat PDE planning."},
        {"limitation_or_risk": "no k(d)=g(d)k0 yet", "current_status": "not implemented", "consequence": "Damage does not affect thermal conductivity.", "severity": "high", "mitigation_or_next_check": "Defer until heat PDE branch has independent conservation and boundary-condition tests."},
        {"limitation_or_risk": "prescribed uniform temperature only", "current_status": "uniform delta_T=20 K used for main diagnostics", "consequence": "No spatially solved thermal gradients are validated.", "severity": "medium", "mitigation_or_next_check": "Use planned heat PDE tests or a separate prescribed-gradient check if explicitly approved."},
        {"limitation_or_risk": "no transient thermal loading", "current_status": "not run", "consequence": "No time-dependent heating conclusions are supported.", "severity": "medium", "mitigation_or_next_check": "Keep transient loading out of scope until the steady heat PDE branch is stable."},
        {"limitation_or_risk": "no independent seed validation", "current_status": "single seed 23 diagnostics", "consequence": "Robustness of alpha background and damage delay is not statistically established.", "severity": "medium", "mitigation_or_next_check": "Consider one small seed repeat only if the reviewer needs robustness before heat PDE."},
        {"limitation_or_risk": "no physical validation", "current_status": "no COMSOL or experiment comparison in this stage", "consequence": "Stage supports route stability, not physical predictive accuracy.", "severity": "high", "mitigation_or_next_check": "Create a benchmark validation plan before making physical claims."},
        {"limitation_or_risk": "low-level alpha background artifact risk", "current_status": "observed in Case C audit and reported as diagnostic only", "consequence": "Diffuse alpha could be overinterpreted from low-range plots.", "severity": "medium", "mitigation_or_next_check": "Use global-scale and high-threshold/notch metrics for damage interpretation."},
        {"limitation_or_risk": "COMSOL plane-stress/platform difference caveat", "current_status": "project uses current TM constitutive convention", "consequence": "Line-by-line COMSOL clone behavior is not guaranteed.", "severity": "medium", "mitigation_or_next_check": "Document constitutive convention before any COMSOL comparison."},
        {"limitation_or_risk": "thermal strain implemented in current project constitutive convention, not line-by-line COMSOL clone", "current_status": "accepted scope in THERMAL_REINTRODUCTION_PLAN.md", "consequence": "Physical meaning can be compared, but implementation details may differ.", "severity": "medium", "mitigation_or_next_check": "Review equations and units before starting heat PDE or benchmark comparison."},
    ]


def guards() -> list[dict[str, object]]:
    comments = {
        "no-thermal baseline modification": "examples/TM_comsol_no_thermal_micro is treated as frozen baseline.",
        "legacy top-sigma primary reaction": "Energy-conjugate reaction_N_energy remains primary.",
    }
    return [
        {
            "guard_item": item,
            "expected_status": "not implemented or not run",
            "observed_status": "not primary" if item == "legacy top-sigma primary reaction" else "not implemented or not run",
            "passed": "true",
            "comment": comments.get(item, "No evidence in completed package reports or current source scope."),
        }
        for item in [
            "full heat PDE",
            "damage-dependent conductivity",
            "trainable/PDE temperature",
            "D0040",
            "seed study",
            "shear extension",
            "S0110",
            "no-thermal baseline modification",
            "legacy top-sigma primary reaction",
        ]
    ]


def source_scope() -> list[dict[str, object]]:
    return [
        {"source_area": "thermal_prescribed.py", "current_status": "Provides prescribed delta_T/temperature helpers for off, uniform, and linear_y modes.", "changed_in_this_stage": "yes during prescribed-strain patch stage; no in this summary task", "comments": "No heat solve or trainable temperature field."},
        {"source_area": "compute_energy_mixed_tm.py", "current_status": "Applies thermal normal strain before TM split/history/energy route and records thermal fields.", "changed_in_this_stage": "yes during prescribed-strain patch stage; no in this summary task", "comments": "Reaction route and material constants are not changed here."},
        {"source_area": "config.py", "current_status": "Adds CLI/config fields for prescribed thermal mode, delta_T, temperature, alpha_T, and Tref.", "changed_in_this_stage": "yes during prescribed-strain patch stage; no in this summary task", "comments": "Default thermal_mode is off."},
        {"source_area": "train_mixed_tm.py", "current_status": "Passes thermal kwargs into existing mixed TM energy calls.", "changed_in_this_stage": "yes during prescribed-strain patch stage; no in this summary task", "comments": "Loss form and history objective remain the existing route."},
        {"source_area": "history_field_mixed_tm.py", "current_status": "Mixed HI/HII/He history route retained.", "changed_in_this_stage": "no behavior change identified for summary task", "comments": "Thermal effect enters through adjusted elastic strain before history update."},
        {"source_area": "postprocess_results.py", "current_status": "Computes checkpointed energy-conjugate reaction with thermal settings from run metadata.", "changed_in_this_stage": "yes during prescribed-strain patch stage; no in this summary task", "comments": "Legacy top-sigma is not primary."},
        {"source_area": "load schedules", "current_status": "D0003, D0015, and D0020 prescribed-temperature tension schedules exist for diagnostics.", "changed_in_this_stage": "yes across diagnostic stages; no new schedule in this summary task", "comments": "No D0040, shear, or S0110 schedule was run for this stage."},
        {"source_area": "runs/packages", "current_status": "Six prior evidence packages plus this stage-summary package.", "changed_in_this_stage": "yes, this summary package is new", "comments": "This package consolidates evidence only."},
        {"source_area": "original no-thermal project", "current_status": "Frozen baseline.", "changed_in_this_stage": "no", "comments": "Do not modify examples/TM_comsol_no_thermal_micro for thermal work."},
    ]


def trend_rows(v: dict[str, str]) -> list[dict[str, object]]:
    return [
        {"diagnostic_package": rel(PKG["strong"]), "case_or_comparison": "A/B equivalence", "metric": "max absolute reaction/stress/energy/alpha/HI/HII differences", "no_thermal_or_case_A": "Case A", "zero_deltaT_or_case_B": "Case B matched A within table precision", "deltaT20_or_case_C": "not applicable", "interpretation": "Zero thermal branch reproduces no-thermal branch under non-smoke training.", "caveat": "Single seed and tension setup."},
        {"diagnostic_package": rel(PKG["strong"]), "case_or_comparison": "A versus C", "metric": "final nominal stress MPa", "no_thermal_or_case_A": v["strong_A_final_stress"], "zero_deltaT_or_case_B": v["strong_A_final_stress"], "deltaT20_or_case_C": v["strong_C_final_stress"], "interpretation": f"+20 K shifts final stress downward by {v['strong_C_minus_A_stress']} MPa.", "caveat": "Diagnostic, not physical validation."},
        {"diagnostic_package": rel(PKG["strong"]), "case_or_comparison": "A versus C", "metric": "final alpha max", "no_thermal_or_case_A": v["strong_A_alpha"], "zero_deltaT_or_case_B": v["strong_A_alpha"], "deltaT20_or_case_C": v["strong_C_alpha"], "interpretation": "Case C has lower final peak alpha under prescribed +20 K.", "caveat": "The broad low-level Case C field required the follow-up audit."},
        {"diagnostic_package": rel(PKG["strong"]), "case_or_comparison": "Case C compensation region", "metric": "reaction zero crossing displacement mm", "no_thermal_or_case_A": "not applicable", "zero_deltaT_or_case_B": "not applicable", "deltaT20_or_case_C": v["strong_zero_cross"], "interpretation": f"Observed crossing is near thermal expansion estimate {v['strong_zero_cross_estimate']} mm.", "caveat": "Interpolated from checkpointed diagnostic steps."},
        {"diagnostic_package": rel(PKG["audit"]), "case_or_comparison": "Case C alpha anomaly audit", "metric": "C-minus-A final alpha distribution", "no_thermal_or_case_A": f"min {v['audit_ca_negative_min']}", "zero_deltaT_or_case_B": "B equals A in audited run", "deltaT20_or_case_C": f"median C-A {v['audit_ca_median']}; positive max {v['audit_ca_positive_max']}", "interpretation": "Peak damage is lower in Case C, but some low-level background locations exceed A/B.", "caveat": "Diagnostic warning only."},
        {"diagnostic_package": rel(PKG["audit"]), "case_or_comparison": "Case C threshold area", "metric": "final alpha area fractions", "no_thermal_or_case_A": "see audit table", "zero_deltaT_or_case_B": "same as A in audited run", "deltaT20_or_case_C": f">=0.001 {v['audit_C_thr_001_frac']}; >=0.01 {v['audit_C_thr_01_frac']}; >=0.03 {v['audit_C_thr_03_frac']}; >=0.05 {v['audit_C_thr_05_frac']}", "interpretation": "Low threshold area is broad; higher thresholds collapse.", "caveat": "Do not use low-threshold area as fracture evidence."},
        {"diagnostic_package": rel(PKG["probe"]), "case_or_comparison": "A/B equivalence", "metric": "max absolute reaction/stress/energy/alpha/HI/HII differences", "no_thermal_or_case_A": "Case A", "zero_deltaT_or_case_B": "Case B matched A within table precision", "deltaT20_or_case_C": "not applicable", "interpretation": "Zero thermal branch still reproduces no-thermal branch under D0020 schedule.", "caveat": "Same deterministic seed."},
        {"diagnostic_package": rel(PKG["probe"]), "case_or_comparison": "C versus A", "metric": "final nominal stress MPa", "no_thermal_or_case_A": v["probe_A_final_stress"], "zero_deltaT_or_case_B": v["probe_A_final_stress"], "deltaT20_or_case_C": v["probe_C_final_stress"], "interpretation": f"Case C remains shifted downward by {v['probe_C_minus_A_stress']} MPa.", "caveat": "Prescribed-temperature tension only."},
        {"diagnostic_package": rel(PKG["probe"]), "case_or_comparison": "C versus A", "metric": "final alpha max", "no_thermal_or_case_A": v["probe_A_alpha"], "zero_deltaT_or_case_B": v["probe_A_alpha"], "deltaT20_or_case_C": v["probe_C_alpha"], "interpretation": f"Case C final alpha is lower by {v['probe_C_minus_A_alpha']}.", "caveat": "Use notch/high-threshold metrics, not low-background thresholds."},
        {"diagnostic_package": rel(PKG["probe"]), "case_or_comparison": "Case C final high-threshold alpha", "metric": "fraction above thresholds", "no_thermal_or_case_A": "see probe table for A/B", "zero_deltaT_or_case_B": "same as A", "deltaT20_or_case_C": f">=0.02 {v['probe_C_thr_002_frac']} ({v['probe_C_thr_002_notch']} notch-connected); >=0.05 {v['probe_C_thr_005_frac']} ({v['probe_C_thr_005_notch']}); >=0.1 {v['probe_C_thr_01_frac']}", "interpretation": "High-threshold alpha remains lower and notch-focused relative to A.", "caveat": "Low-level diffuse background remains diagnostic only."},
        {"diagnostic_package": rel(PKG["probe"]), "case_or_comparison": "damage delay", "metric": "first displacement where alpha_max reaches thresholds", "no_thermal_or_case_A": f">=0.02 at {v['probe_delay_alpha_002_A']} mm; >=0.05 at {v['probe_delay_alpha_005_A']} mm", "zero_deltaT_or_case_B": "same as A", "deltaT20_or_case_C": f">=0.02 at {v['probe_delay_alpha_002_C']} mm; >=0.05 at {v['probe_delay_alpha_005_C']} mm", "interpretation": "Prescribed +20 K delays reaching selected alpha thresholds.", "caveat": "Threshold selection is diagnostic."},
    ]


def decision_rows() -> list[dict[str, object]]:
    return [
        {"decision": "whether prescribed-temperature thermal strain branch is stable enough to preserve as baseline", "prerequisite_evidence": "Patch tests, A/B delta_T=0 equivalence, checkpointed reaction availability, and moderate damage probe review.", "current_status": "Evidence is consistent for prescribed-temperature mechanics route.", "recommendation": "Preserve as the thermal-subproject prescribed-temperature baseline.", "blocking_risk": "No physical validation; low-level alpha background remains diagnostic-only.", "next_task_if_approved": "Freeze this branch as baseline for heat PDE planning.", "next_task_if_not_approved": "Identify specific failed evidence row and rerun only the smallest targeted diagnostic."},
        {"decision": "whether to stop running more prescribed-temperature tension diagnostics for now", "prerequisite_evidence": "Micro, strong, audit, and moderate probe packages reviewed.", "current_status": "Multiple tension diagnostics already show consistent route behavior.", "recommendation": "Stop broad new prescribed-temperature tension diagnostics for now.", "blocking_risk": "Repeating tension diagnostics can consume runtime without resolving heat-PDE readiness.", "next_task_if_approved": "Move to heat PDE decision-gate planning.", "next_task_if_not_approved": "Run only one narrowly scoped seed repeat or metric audit."},
        {"decision": "whether to begin heat PDE planning", "prerequisite_evidence": "Reviewer accepts that current stage is prescribed-temperature only and not physical validation.", "current_status": "Ready for planning review, not automatic implementation.", "recommendation": "Begin a written heat PDE plan only after reviewer signoff.", "blocking_risk": "Boundary conditions, units, conservation tests, and source terms are not yet specified.", "next_task_if_approved": "Write a heat PDE implementation and validation plan without coding the PDE first.", "next_task_if_not_approved": "Keep prescribed-temperature branch frozen and document unresolved questions."},
        {"decision": "whether to implement damage-dependent conductivity now or defer", "prerequisite_evidence": "Stable heat PDE with validated temperature solve.", "current_status": "Prerequisite not met.", "recommendation": "Defer damage-dependent conductivity.", "blocking_risk": "Implementing k(d)=g(d)k0 before heat PDE stability would conflate two new physics changes.", "next_task_if_approved": "Not recommended; if forced, create a separate written risk plan first.", "next_task_if_not_approved": "Implement heat PDE first, then revisit conductivity."},
        {"decision": "whether a small seed repeat is worth runtime before heat PDE", "prerequisite_evidence": "Reviewer wants robustness evidence for alpha background or damage delay.", "current_status": "Optional; not required by current evidence for route-stability conclusion.", "recommendation": "Defer unless reviewer specifically needs seed robustness before heat PDE planning.", "blocking_risk": "Single-seed uncertainty remains for alpha background.", "next_task_if_approved": "Run one small A/B/C repeat with exact same source and a narrow schedule.", "next_task_if_not_approved": "Proceed to heat PDE planning decision gate."},
    ]


def changed_files() -> list[dict[str, object]]:
    return [
        {"path": rel(PACKAGE_DIR), "change_type": "new package", "reason": "Reviewer-readable prescribed thermal strain stage summary.", "scope": "thermal summary artifacts only"},
        {"path": rel(PACKAGE_DIR / "build_stage_summary_package.py"), "change_type": "new script", "reason": "Regenerate summary tables, report, manifest, and handoff from existing package evidence.", "scope": "documentation/evidence synthesis; no training"},
        {"path": rel(THERMAL_ROOT / "PROJECT_MEMORY.md"), "change_type": "update", "reason": "Record current prescribed-temperature stage status and simplified finalization protocol.", "scope": "thermal project memory only"},
        {"path": "examples/TM_comsol_no_thermal_micro", "change_type": "unchanged", "reason": "Frozen no-thermal baseline must remain untouched.", "scope": "guarded path"},
    ]


def report_md(v: dict[str, str]) -> str:
    return f"""# Prescribed Thermal Strain Stage Summary

## 1. Purpose

This package consolidates the prescribed-temperature thermal-strain branch work completed so far in `examples/TM_comsol_thermal_micro`. It is a documentation and evidence synthesis package only. It does not run new training, rerun A/B/C diagnostics, introduce heat PDE physics, introduce damage-dependent conductivity, or modify the original no-thermal baseline project.

## 2. Scope boundaries

The scope is limited to the thermal subproject. The original `examples/TM_comsol_no_thermal_micro` project remains the frozen baseline. The summary reads existing reports, handoffs, current thermal source files, and existing package tables. The summary does not edit old packages, change source-model behavior, change the reaction route, or reintroduce legacy top-sigma as the primary reaction.

## 3. Timeline of completed thermal prescribed-strain work

The stage began with a thermal subproject scaffold copied from the verified no-thermal route. The prescribed thermal strain patch then added a default-off thermal mode and patch tests. Subsequent diagnostics exercised the branch with a smoke micro-notch case, a stronger D0015 tension case, a Case C alpha audit, and a moderate D0020 tension damage probe. See `tables/stage_milestone_summary.csv` for the reviewer-facing timeline.

## 4. What was implemented

The implemented branch is prescribed-temperature mechanics only. The current route computes `delta_T = T - Tref`, subtracts isotropic normal thermal strain from `exx` and `eyy`, leaves `exy` unchanged, and then passes the adjusted elastic strain through the existing TM split, history, energy, and checkpointed reaction route. CLI/config support exists for `thermal_mode`, prescribed absolute temperature, prescribed `delta_T`, `alpha_T`, and `Tref`.

## 5. What was validated by patch tests

Patch tests validated zero-`delta_T` equivalence, free uniform expansion, constrained uniform heating sign/scale under the current project convention, shear-component invariance, and guards showing no heat PDE, no trainable/PDE temperature field, and no damage-dependent conductivity. The final patch-test classification was `prescribed thermal strain branch implemented and patch tests passed`.

## 6. What was validated by micro-notch diagnostics

The micro-notch diagnostic confirmed that the branch can run the existing checkpointed mechanics route with A/B/C cases. Case B matched Case A within table precision. Case C shifted the final nominal stress from A `{v['micro_A_final_stress']} MPa` to C `{v['micro_C_final_stress']} MPa`, while alpha remained stable in that small smoke run. This is useful as a routing diagnostic, not physical validation.

## 7. What the stronger tension diagnostic showed

The stronger D0015 diagnostic used full training settings and a schedule around the thermal compensation region. Case B again matched Case A within table precision. Case C shifted final nominal stress from A `{v['strong_A_final_stress']} MPa` to C `{v['strong_C_final_stress']} MPa`, and final alpha from A `{v['strong_A_alpha']}` to C `{v['strong_C_alpha']}`. The observed Case C reaction zero crossing was `{v['strong_zero_cross']} mm`, near the estimated compensation displacement `{v['strong_zero_cross_estimate']} mm`.

## 8. What the Case C alpha anomaly audit showed

The alpha audit found that Case C peak alpha was lower than A/B and the reaction/stress trend remained interpretable. It also found a broad low-level Case C alpha background in raw element values. The final C-minus-A alpha distribution had median `{v['audit_ca_median']}` and positive maximum `{v['audit_ca_positive_max']}`. Final Case C low-threshold area fractions were `>=0.001` `{v['audit_C_thr_001_frac']}` and `>=0.01` `{v['audit_C_thr_01_frac']}`. That background is a diagnostic warning and is not treated as physical fracture evidence.

## 9. What the moderate damage probe showed

The moderate D0020 probe extended the tension schedule to `2.0e-5 mm` while preserving compensation-region resolution. Case B matched Case A. Case C final nominal stress was `{v['probe_C_final_stress']} MPa` versus A `{v['probe_A_final_stress']} MPa`, a C-A shift of `{v['probe_C_minus_A_stress']} MPa`. Case C final alpha was `{v['probe_C_alpha']}` versus A `{v['probe_A_alpha']}`, and final Case C high-threshold area fractions were `>=0.02` `{v['probe_C_thr_002_frac']}`, `>=0.03` `{v['probe_C_thr_003_frac']}`, `>=0.05` `{v['probe_C_thr_005_frac']}`, and `>=0.1` `{v['probe_C_thr_01_frac']}`.

## 10. Trusted conclusions

The trusted conclusions are that `thermal_mode=off` remains the default, `thermal_mode=uniform` with zero `delta_T` reproduces the no-thermal thermal-subproject route in completed diagnostics, prescribed `+20 K` shifts displacement-controlled tension reaction/stress downward, checkpointed `reaction_N_energy` remains available and primary, and no heat PDE or damage-dependent conductivity has been implemented. The moderate-probe damage reduction is trusted only within the diagnostic scope and is therefore rated medium rather than high.

## 11. Diagnostic-only conclusions

The broad low-level Case C alpha background, single-seed behavior, tension-only conclusions, and any implication about conduction or damage-dependent conductivity are diagnostic-only. These findings should guide review and future test design but should not be used as physical fracture evidence or as a transport-model conclusion.

## 12. Explicitly unimplemented physics

The following remain explicitly unimplemented or not run: full heat PDE, damage-dependent conductivity, trainable/PDE temperature, D0040, seed study, shear extension, S0110, no-thermal baseline modification, and legacy top-sigma as the primary reaction. See `tables/not_implemented_guard_summary.csv`.

## 13. Known limitations and risks

The branch is prescribed-temperature only, uses uniform temperature in the main diagnostics, has no transient thermal loading, has no independent seed validation, and has no physical validation against COMSOL or experiment. The low-level alpha background remains an artifact/interpretation risk. The thermal strain implementation follows the current project constitutive convention and is not a line-by-line COMSOL clone.

## 14. Whether this is physical validation

This is not physical validation. It is a software and physics-route validation stage for prescribed thermal strain in the thermal subproject. It supports preserving the branch as a baseline for review, but it does not establish quantitative agreement with COMSOL or experiment.

## 15. Recommendation on further prescribed-temperature diagnostics

Do not run more broad prescribed-temperature tension diagnostics by default. The evidence is now sufficient for a decision gate. A small seed repeat is optional only if the reviewer decides that robustness evidence is required before heat PDE planning.

## 16. Decision gate before heat PDE

The safest gate is a reviewer decision on whether to preserve this prescribed-temperature branch as the thermal baseline and begin a written heat PDE plan. Damage-dependent conductivity should remain deferred until a heat PDE branch is independently stable and tested.

## 17. Final classification

`{FINAL_CLASSIFICATION}`

The prescribed-temperature thermal-strain branch has passed patch tests and multiple checkpointed tension diagnostics. The zero-temperature thermal branch reproduces the no-thermal branch, and +20 K prescribed uniform thermal strain consistently shifts the reaction/stress response downward while reducing notch-tip/high-threshold alpha growth in the moderate damage probe. The broad low-level Case C alpha background is a diagnostic warning and is not treated as physical fracture evidence. This stage remains prescribed-temperature only: no heat PDE, no trainable temperature field, and no damage-dependent conductivity have been implemented. The next step should be a decision-gate review before starting heat PDE work.

## 18. Exact next recommended task

Review `tables/next_decision_gate.csv`, this `REPORT.md`, and the existing strong/audit/moderate probe reports. If approved, write a heat PDE implementation and validation plan only; do not implement heat PDE or damage-dependent conductivity without that plan.
"""


def figure_summary() -> str:
    return """# Figure Summary

No new optional PNG figures were generated for this stage-summary package. Tables and the report are the authoritative summary artifacts.

Existing figures to review next:

- `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/figures/reaction_vs_displacement.png`
- `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/figures/reaction_shift_C_minus_A.png`
- `examples/TM_comsol_thermal_micro/runs/20260624_caseC_alpha_anomaly_audit/figures/final_alpha_global_scale.png`
- `examples/TM_comsol_thermal_micro/runs/20260624_caseC_alpha_anomaly_audit/figures/final_alpha_low_range_scale.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/reaction_vs_displacement.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_global_scale.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_high_threshold_masks.png`
"""


def handoff_md(v: dict[str, str]) -> str:
    return f"""# Handoff: Prescribed Thermal Strain Stage Summary

## Status

Final classification: `{FINAL_CLASSIFICATION}`

Commit hash:

- Pending before commit; update once after the primary summary commit is pushed.

Push status:

- Pending before commit; update once after the primary summary commit is pushed.

## Package

- Package path: `{rel(PACKAGE_DIR)}`
- Report: `{rel(PACKAGE_DIR / 'REPORT.md')}`
- Manifest: `{rel(PACKAGE_DIR / 'MANIFEST.json')}`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- This task did not run training, rerun A/B/C, run D0040, run a seed study, run shear extension, or run S0110.
- This task did not implement heat PDE, damage-dependent conductivity, or a trainable/PDE temperature field.
- This task did not change material parameters, `l0`, history logic, training losses, boundary conditions, source model behavior, or reaction route.
- Energy-conjugate `reaction_N_energy` remains the primary reaction.

## Main Evidence

- Patch tests passed for prescribed thermal strain and no-heat-PDE/no-conductivity guards.
- A/B `delta_T=0` equivalence held in completed diagnostics.
- Strong diagnostic C-A final stress shift: `{v['strong_C_minus_A_stress']} MPa`.
- Moderate probe C-A final stress shift: `{v['probe_C_minus_A_stress']} MPa`.
- Moderate probe final alpha C `{v['probe_C_alpha']}` versus A `{v['probe_A_alpha']}`.
- Case C broad low-level alpha background remains diagnostic-only.

## Tables Generated

- `tables/stage_milestone_summary.csv`
- `tables/evidence_matrix.csv`
- `tables/trusted_findings.csv`
- `tables/diagnostic_only_findings.csv`
- `tables/limitations_and_open_risks.csv`
- `tables/not_implemented_guard_summary.csv`
- `tables/source_scope_summary.csv`
- `tables/reaction_damage_trend_summary.csv`
- `tables/next_decision_gate.csv`
- `tables/changed_files_summary.csv`

## Reviewer Should Read Next

1. `{rel(PACKAGE_DIR / 'REPORT.md')}`
2. `{rel(PACKAGE_DIR / 'tables' / 'evidence_matrix.csv')}`
3. `{rel(PACKAGE_DIR / 'tables' / 'trusted_findings.csv')}`
4. `{rel(PACKAGE_DIR / 'tables' / 'diagnostic_only_findings.csv')}`
5. `{rel(PACKAGE_DIR / 'tables' / 'next_decision_gate.csv')}`
6. `{rel(PKG['probe'] / 'REPORT.md')}`
7. `{rel(PKG['audit'] / 'REPORT.md')}`

## Exact Next Recommended Task

Hold the decision-gate review recorded in `tables/next_decision_gate.csv`. If approved, write a heat PDE implementation and validation plan only. Do not begin heat PDE or damage-dependent conductivity implementation without explicit reviewer approval.
"""


def manifest(files: list[Path]) -> dict[str, object]:
    return {
        "package": rel(PACKAGE_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classification": FINAL_CLASSIFICATION,
        "task_type": "documentation/evidence synthesis only",
        "training_run": False,
        "source_code_modified_by_this_task": False,
        "no_thermal_project_touched": False,
        "heat_pde_implemented": False,
        "damage_dependent_conductivity_implemented": False,
        "trainable_or_pde_temperature_implemented": False,
        "D0040_run": False,
        "seed_study_run": False,
        "shear_or_S0110_run": False,
        "source_packages": {key: rel(path) for key, path in PKG.items()},
        "required_outputs": [rel(path) for path in files],
        "reviewer_next": [
            rel(PACKAGE_DIR / "REPORT.md"),
            rel(PACKAGE_DIR / "tables" / "evidence_matrix.csv"),
            rel(PACKAGE_DIR / "tables" / "next_decision_gate.csv"),
            rel(PKG["probe"] / "REPORT.md"),
            rel(PKG["audit"] / "REPORT.md"),
        ],
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    v = evidence_values()
    tables = [
        ("stage_milestone_summary.csv", milestone_rows(v), ["milestone_id", "package_path", "date_or_run_label", "task_type", "source_modified", "training_run", "validation_type", "final_classification", "key_result", "caveat", "reviewer_read_next"]),
        ("evidence_matrix.csv", evidence_matrix(v), ["evidence_item", "supporting_package", "supporting_file", "observed_result", "trust_level", "reason", "limitation"]),
        ("trusted_findings.csv", trusted_findings(v), ["finding", "evidence", "trust_level", "why_trusted", "scope_limit"]),
        ("diagnostic_only_findings.csv", diagnostic_only_findings(v), ["finding", "evidence", "why_diagnostic_only", "do_not_use_for", "recommended_future_check"]),
        ("limitations_and_open_risks.csv", limitations(), ["limitation_or_risk", "current_status", "consequence", "severity", "mitigation_or_next_check"]),
        ("not_implemented_guard_summary.csv", guards(), ["guard_item", "expected_status", "observed_status", "passed", "comment"]),
        ("source_scope_summary.csv", source_scope(), ["source_area", "current_status", "changed_in_this_stage", "comments"]),
        ("reaction_damage_trend_summary.csv", trend_rows(v), ["diagnostic_package", "case_or_comparison", "metric", "no_thermal_or_case_A", "zero_deltaT_or_case_B", "deltaT20_or_case_C", "interpretation", "caveat"]),
        ("next_decision_gate.csv", decision_rows(), ["decision", "prerequisite_evidence", "current_status", "recommendation", "blocking_risk", "next_task_if_approved", "next_task_if_not_approved"]),
        ("changed_files_summary.csv", changed_files(), ["path", "change_type", "reason", "scope"]),
    ]
    files = [
        PACKAGE_DIR / "REPORT.md",
        PACKAGE_DIR / "HANDOFF_COMMENT.md",
        PACKAGE_DIR / "MANIFEST.json",
        FIGURES_DIR / "figure_summary.md",
    ]
    for name, rows, cols in tables:
        path = TABLES_DIR / name
        write_csv(path, rows, cols)
        files.append(path)
    write_text(PACKAGE_DIR / "REPORT.md", report_md(v))
    write_text(PACKAGE_DIR / "HANDOFF_COMMENT.md", handoff_md(v))
    write_text(FIGURES_DIR / "figure_summary.md", figure_summary())
    write_text(PACKAGE_DIR / "MANIFEST.json", json.dumps(manifest(files), indent=2, sort_keys=True) + "\n")
    print(f"package={rel(PACKAGE_DIR)}")
    print(f"classification={FINAL_CLASSIFICATION}")
    print(f"required_outputs={len(files)}")


if __name__ == "__main__":
    main()
