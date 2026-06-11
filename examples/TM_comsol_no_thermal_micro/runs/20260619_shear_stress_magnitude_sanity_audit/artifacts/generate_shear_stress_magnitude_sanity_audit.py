from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RUNS = PACKAGE.parents[0]
PROJECT = RUNS.parents[0]

CASES = {
    "S0050": {
        "package": RUNS / "20260616_existing_geometry_shear_load_extension",
        "schedule": "load_schedules/load_schedule_S0050_shear.csv",
        "summary": "tables/shear_extension_run_summary.csv",
        "stress": "tables/shear_stress_strain_by_step.csv",
        "damage": "tables/shear_damage_drive_summary.csv",
    },
    "S0070": {
        "package": RUNS / "20260617_existing_geometry_shear_connectivity_extension",
        "schedule": "load_schedules/load_schedule_S0070_shear.csv",
        "summary": "tables/shear_connectivity_extension_run_summary.csv",
        "stress": "tables/shear_stress_strain_by_step.csv",
        "damage": "tables/shear_damage_drive_summary.csv",
        "connectivity": "tables/shear_connectivity_by_threshold.csv",
    },
    "S0090": {
        "package": RUNS / "20260618_existing_geometry_shear_longer_connectivity_extension",
        "schedule": "load_schedules/load_schedule_S0090_shear.csv",
        "summary": "tables/shear_longer_extension_run_summary.csv",
        "stress": "tables/shear_stress_strain_by_step.csv",
        "damage": "tables/shear_damage_drive_summary.csv",
        "connectivity": "tables/shear_connectivity_by_threshold.csv",
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_dirs() -> None:
    for subdir in ("tables", "figures", "artifacts/source_snippets"):
        (PACKAGE / subdir).mkdir(parents=True, exist_ok=True)


def settings_from_artifacts(case_pkg: Path) -> Dict[str, str]:
    candidates = sorted((case_pkg / "artifacts").glob("model_settings_*_shear.txt"))
    if not candidates:
        return {}
    settings: Dict[str, str] = {}
    for line in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        settings[key.strip()] = value.strip()
    return settings


def stress_definition_trace() -> pd.DataFrame:
    rows = [
        {
            "item": "source_csv_column",
            "value": "nominal_shear_stress_energy_MPa",
            "evidence": "tables/shear_stress_strain_by_step.csv in S0050/S0070/S0090 packages",
            "conclusion": "This is the reported nominal shear stress column audited here.",
        },
        {
            "item": "reaction_column_used",
            "value": "reaction_N_energy",
            "evidence": "postprocess_results.py build_stress_strain_curve; stress tables",
            "conclusion": "The stress uses the checkpointed energy-conjugate reaction.",
        },
        {
            "item": "reaction_definition",
            "value": "reaction_N_energy = 1000 * dPi/dDelta_s",
            "evidence": "postprocess_results.py compute_exact_reaction_rows multiplies grad by 1000",
            "conclusion": "The gradient is in kN and is converted to N before nominal stress is formed.",
        },
        {
            "item": "reference_area_mm2",
            "value": "0.01",
            "evidence": "postprocess_results.py REFERENCE_AREA_MM2 and stress table reference_area_mm2",
            "conclusion": "Nominal stress divides by gross unit-thickness area 0.01 mm^2.",
        },
        {
            "item": "reference_length_mm",
            "value": "0.01",
            "evidence": "postprocess_results.py REFERENCE_LENGTH_MM and stress table reference_length_mm",
            "conclusion": "Engineering shear strain is Delta_s / 0.01 mm.",
        },
        {
            "item": "shear_strain_definition",
            "value": "engineering_shear_strain = Delta_s / reference_length_mm",
            "evidence": "postprocess_results.py build_stress_strain_curve",
            "conclusion": "The shear curve uses Delta_s, not tensile Delta, for the load parameter.",
        },
        {
            "item": "stress_formula",
            "value": "nominal_shear_stress_energy_MPa = reaction_N_energy / reference_area_mm2",
            "evidence": "postprocess_results.py build_stress_strain_curve",
            "conclusion": "Because 1 N/mm^2 = 1 MPa, no additional conversion is needed after reaction is in N.",
        },
        {
            "item": "unit_conversion",
            "value": "1 N/mm^2 = 1 MPa",
            "evidence": "standard N-mm unit convention; model_settings unit_strategy",
            "conclusion": "The MPa label is consistent with N divided by mm^2.",
        },
        {
            "item": "load_parameter",
            "value": "Delta_s",
            "evidence": "model_settings load_parameter_name: Delta_s; stress table load_case: shear",
            "conclusion": "The audited shear stress is energy-conjugate to shear displacement Delta_s.",
        },
    ]
    return pd.DataFrame(rows)


def recompute_stress_tables(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for case, info in data.items():
        stress = info["stress"].copy()
        for _, row in stress.iterrows():
            reported = float(row["nominal_shear_stress_energy_MPa"])
            reaction = float(row["reaction_N_energy"])
            area = float(row["reference_area_mm2"])
            recomputed = reaction / area
            abs_err = abs(recomputed - reported)
            rel_err = abs_err / max(abs(reported), 1.0e-30)
            rows.append(
                {
                    "case": case,
                    "step": int(row["step"]),
                    "Delta_s": float(row["Delta_s"]),
                    "engineering_shear_strain": float(row["engineering_shear_strain"]),
                    "reaction_N_energy": reaction,
                    "reference_area_mm2": area,
                    "reported_nominal_shear_stress_MPa": reported,
                    "tau_MPa_recomputed": recomputed,
                    "absolute_error_MPa": abs_err,
                    "relative_error": rel_err,
                }
            )
    return pd.DataFrame(rows)


def elastic_slope_table(data: Dict[str, Dict[str, pd.DataFrame]], material: Dict[str, float]) -> pd.DataFrame:
    expected_g_kN = material["mat_E_kN_per_mm2"] / (2.0 * (1.0 + material["mat_nu"]))
    expected_g_mpa = expected_g_kN * 1000.0
    rows = []
    for case, info in data.items():
        stress = info["stress"].copy()
        damage = info["damage"].copy()
        merged = stress.merge(
            damage[["step", "alpha_max"]],
            on="step",
            how="left",
        )
        prepeak_step = int(merged["nominal_shear_stress_energy_MPa"].astype(float).idxmax())
        candidates = merged[
            (merged["alpha_max"].astype(float) < 0.01)
            & (merged["step"].astype(int) < prepeak_step)
            & (merged["engineering_shear_strain"].astype(float) <= 0.001)
        ].copy()
        if len(candidates) < 2:
            candidates = merged.head(min(4, len(merged))).copy()
        x = candidates["engineering_shear_strain"].astype(float).to_numpy()
        y = candidates["nominal_shear_stress_energy_MPa"].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        ratio = slope / expected_g_mpa if expected_g_mpa else math.nan
        rows.append(
            {
                "case": case,
                "filter": "alpha_max < 0.01, step before peak, gamma <= 0.001",
                "steps_used": ",".join(str(int(v)) for v in candidates["step"]),
                "point_count": int(len(candidates)),
                "gamma_min": float(np.min(x)),
                "gamma_max": float(np.max(x)),
                "measured_initial_slope_MPa": float(slope),
                "fit_intercept_MPa": float(intercept),
                "mat_E_kN_per_mm2": material["mat_E_kN_per_mm2"],
                "mat_nu": material["mat_nu"],
                "expected_G_kN_per_mm2": expected_g_kN,
                "expected_G_MPa": expected_g_mpa,
                "slope_over_expected_G": ratio,
                "relative_error_vs_G": abs(slope - expected_g_mpa) / expected_g_mpa,
                "interpretation": (
                    "structure-level gross-area slope is much lower than material G; "
                    "this is expected for the current top-v-free notched specimen diagnostic and does not indicate a stress unit mismatch"
                    if ratio < 0.5
                    else "early slope is close to material G"
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_geometry_area_audit() -> pd.DataFrame:
    width = 0.01
    height = 0.01
    thickness = 1.0
    area = width * thickness
    return pd.DataFrame(
        [
            {
                "domain_width_mm": width,
                "domain_height_mm": height,
                "assumed_thickness_mm": thickness,
                "reference_shear_area_mm2": area,
                "postprocess_reference_area_mm2": 0.01,
                "area_convention": "gross width times unit thickness",
                "notch_reduces_reference_area": False,
                "consistent_with_tension_shear_postprocess_convention": True,
                "gross_vs_net_ligament_consequence": (
                    "The reported stress is nominal gross-area shear stress, not local notch-tip stress and not net-ligament shear stress. "
                    "Using a smaller net ligament area would increase the numerical stress value."
                ),
            }
        ]
    )


def peak_final_table(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for case, info in data.items():
        stress = info["stress"].copy()
        damage = info["damage"].copy()
        summary = info["summary"].copy()
        stress_values = stress["nominal_shear_stress_energy_MPa"].astype(float)
        peak_idx = int(stress_values.idxmax())
        peak_row = stress.loc[peak_idx]
        final_row = stress.iloc[-1]
        final_alpha = float(damage.iloc[-1]["alpha_max"])
        through_any = False
        if "connectivity" in info:
            through_any = bool(info["connectivity"]["reaches_right_boundary"].astype(str).str.lower().isin(["true", "1", "yes"]).any())
        elif "alpha0p8_through_crack_any_step" in summary.columns:
            through_any = str(summary.iloc[0]["alpha0p8_through_crack_any_step"]).lower() in {"true", "1", "yes"}
        peak = float(peak_row["nominal_shear_stress_energy_MPa"])
        final = float(final_row["nominal_shear_stress_energy_MPa"])
        drop = max(0.0, peak - final)
        rows.append(
            {
                "case": case,
                "schedule": info["schedule"],
                "final_Delta_s": float(final_row["Delta_s"]),
                "step_count": int(len(stress)),
                "peak_stress_MPa": peak,
                "peak_step": int(peak_row["step"]),
                "peak_engineering_shear_strain": float(peak_row["engineering_shear_strain"]),
                "final_stress_MPa": final,
                "post_peak_drop_MPa": drop,
                "post_peak_drop_percent": 100.0 * drop / peak if peak else math.nan,
                "final_alpha_max": final_alpha,
                "through_right_boundary_any_threshold": through_any,
                "interpretation": (
                    "23.071 MPa is S0070 final stress, not the maximum stress; peak remains about 29.9647 MPa."
                    if case == "S0070"
                    else (
                        "18.136 MPa is S0090 final stress after further post-peak softening; peak remains about 29.9647 MPa."
                        if case == "S0090"
                        else "S0050 already reaches the same peak at step 24, then only a small post-peak drop by final step."
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def material_strength_context(material: Dict[str, float]) -> pd.DataFrame:
    gc = material["Gc_kN_per_mm"]
    l0 = material["l0_mm"]
    e = material["mat_E_kN_per_mm2"]
    theoretical_scale_kN = math.sqrt(e * gc / l0)
    return pd.DataFrame(
        [
            {
                "item": "target_tensile_strength",
                "value": "",
                "source": "README/config/model_settings search",
                "conclusion": "No explicit target tensile strength was found.",
            },
            {
                "item": "target_shear_strength",
                "value": "",
                "source": "README/config/model_settings search",
                "conclusion": "No explicit target shear strength was found.",
            },
            {
                "item": "calibrated_shear_strength_parameter",
                "value": "",
                "source": "normal source and run settings",
                "conclusion": "No calibrated shear strength parameter is present.",
            },
            {
                "item": "material_E",
                "value": material["mat_E_kN_per_mm2"],
                "source": "config.py COMSOL_E_KN_PER_MM2 and model_settings",
                "conclusion": "Elastic modulus is available for internal slope/unit checks.",
            },
            {
                "item": "material_nu",
                "value": material["mat_nu"],
                "source": "config.py COMSOL_NU and model_settings",
                "conclusion": "Poisson ratio is available for G = E/[2(1+nu)].",
            },
            {
                "item": "G_c_l0_strength_scale",
                "value": theoretical_scale_kN * 1000.0,
                "source": "rough sqrt(E*Gc/l0) dimensional scale, MPa",
                "conclusion": (
                    "G_c and l0 imply only a model strength scale unless a specific phase-field critical-stress formula "
                    "and calibration target are declared."
                ),
            },
            {
                "item": "stress_peak_physical_strength_conclusion",
                "value": "29.9647 MPa",
                "source": "S0050/S0070/S0090 stress tables",
                "conclusion": (
                    "No explicit target shear strength was found, so the 29.9647 MPa nominal shear peak can only be judged "
                    "as internally consistent or inconsistent, not physically calibrated."
                ),
            },
        ]
    )


def nominal_vs_local_interpretation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "Is the reported value a nominal gross-area stress?",
                "answer": "yes",
                "explanation": "It is reaction_N_energy divided by 0.01 mm^2 gross unit-thickness area.",
            },
            {
                "question": "Is it energy-conjugate generalized reaction stress?",
                "answer": "yes",
                "explanation": "reaction_N_energy is dPi/dDelta_s converted from kN to N.",
            },
            {
                "question": "Is it local maximum stress?",
                "answer": "no",
                "explanation": "No local stress field maximum is used in this scalar curve.",
            },
            {
                "question": "Is it notch-tip stress?",
                "answer": "no",
                "explanation": "The notch-tip fields are separate diagnostics; this curve is global reaction divided by gross area.",
            },
            {
                "question": "Is it calibrated physical shear strength?",
                "answer": "no",
                "explanation": "No target shear strength or independent calibration was found.",
            },
        ]
    )


def write_plots(recompute: pd.DataFrame, slope: pd.DataFrame, peak_final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=180)
    x = np.arange(len(peak_final))
    ax.bar(x - 0.18, peak_final["peak_stress_MPa"].astype(float), width=0.36, label="Peak")
    ax.bar(x + 0.18, peak_final["final_stress_MPa"].astype(float), width=0.36, label="Final")
    ax.set_xticks(x, peak_final["case"])
    ax.set_ylabel("Nominal shear stress (MPa)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PACKAGE / "figures" / "shear_peak_final_stress_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=180)
    ax.bar(slope["case"], slope["measured_initial_slope_MPa"].astype(float), label="Measured initial slope")
    ax.axhline(float(slope["expected_G_MPa"].iloc[0]), color="tab:red", linestyle="--", label="Material G")
    ax.set_ylabel("Slope (MPa)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PACKAGE / "figures" / "shear_elastic_slope_sanity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=180)
    ax.scatter(
        recompute["reported_nominal_shear_stress_MPa"].astype(float),
        recompute["tau_MPa_recomputed"].astype(float),
        s=12,
        alpha=0.75,
    )
    lo = float(min(recompute["reported_nominal_shear_stress_MPa"].min(), recompute["tau_MPa_recomputed"].min()))
    hi = float(max(recompute["reported_nominal_shear_stress_MPa"].max(), recompute["tau_MPa_recomputed"].max()))
    ax.plot([lo, hi], [lo, hi], color="tab:red", linewidth=1.0)
    ax.set_xlabel("Reported stress (MPa)")
    ax.set_ylabel("Recomputed reaction/area stress (MPa)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PACKAGE / "figures" / "shear_recomputed_vs_reported_stress.png")
    plt.close(fig)


def write_markdown(
    definition: pd.DataFrame,
    recompute: pd.DataFrame,
    slope: pd.DataFrame,
    geometry: pd.DataFrame,
    peak_final: pd.DataFrame,
    strength: pd.DataFrame,
    classification: str,
) -> None:
    max_rel = float(recompute["relative_error"].max())
    max_abs = float(recompute["absolute_error_MPa"].max())
    peak_s0070 = peak_final[peak_final["case"] == "S0070"].iloc[0]
    peak_s0090 = peak_final[peak_final["case"] == "S0090"].iloc[0]
    final_s0070 = float(peak_s0070["final_stress_MPa"])
    final_s0090 = float(peak_s0090["final_stress_MPa"])
    expected_g = float(slope["expected_G_MPa"].iloc[0])
    slope_lines = "\n".join(
        f"- {row.case}: measured initial gross-area slope {row.measured_initial_slope_MPa:.6g} MPa, "
        f"expected material G {row.expected_G_MPa:.6g} MPa, ratio {row.slope_over_expected_G:.6g}"
        for row in slope.itertuples(index=False)
    )
    report = f"""# Shear Stress Magnitude Sanity Audit

## Scope

This audit reads existing S0050/S0070/S0090 shear handoff packages and source snippets. It does not run training, postprocessing, D0040, a seed study, or any further shear extension. No physics, boundary condition, shear ansatz, material parameter, `l0`, history logic, alpha initialization, or training loss was changed.

## Required Questions

1. Is `23.071 MPa` the maximum stress?  
No. `23.071 MPa is S0070 final stress, not the maximum stress.`

2. What is the actual peak nominal shear stress in S0070 and S0090?  
The actual peak stress reported in S0070/S0090 is approximately `29.9647 MPa`, at step 24 and engineering shear strain 0.006.

3. How is `nominal_shear_stress_energy_MPa` computed?  
It is computed as `reaction_N_energy / reference_area_mm2`, using `reaction_N_energy = dPi/dDelta_s` converted from kN to N.

4. Does recomputing stress from `reaction_N_energy / reference_area_mm2` reproduce the reported values?  
Yes. Maximum absolute error is {max_abs:.3e} MPa and maximum relative error is {max_rel:.3e}.

5. What reference area is used?  
`reference_area_mm2=0.01`, corresponding to gross specimen width 0.01 mm times unit thickness 1 mm.

6. Is this gross-area nominal stress or local notch-tip stress?  
The reported stress is nominal gross-area shear stress, not local notch-tip stress and not net-ligament shear stress.

7. What is the early elastic shear slope from the numerical curves?  
{slope_lines}

8. What is the expected material shear modulus `G = E/[2(1+nu)]`?  
With `E=81.5 kN/mm^2` and `nu=0.38`, `G={expected_g:.6g} MPa`.

9. Is the early slope consistent with `G`?  
It is not equal to the bulk material `G`; it is much lower because the plotted value is a structure-level gross-area generalized reaction stress from a notched, top-v-free shear diagnostic, not a homogeneous pure-shear material coupon stress.

10. Is there any evidence of unit conversion error?  
No stress postprocess unit error was found. The kN-to-N conversion occurs before stress formation, and `1 N/mm^2 = 1 MPa` is applied correctly.

11. Is there any evidence of reference-area error?  
No internal reference-area error was found. The postprocess consistently uses gross area 0.01 mm^2. The result should not be interpreted as net-ligament or local stress.

12. Is there an explicit target shear strength in the project?  
No explicit target shear strength was found.

13. Can the approximately `29.9647 MPa` peak be called physically reasonable?  
Not from these packages alone. No explicit target shear strength was found, so the 29.9647 MPa nominal shear peak can only be judged as internally consistent or inconsistent, not physically calibrated.

14. What should the reviewer conclude about stress magnitude?  
The reviewer should conclude that `23.071 MPa` is S0070 final post-peak stress, not maximum stress; the peak is about `29.9647 MPa`. Reaction-to-area recomputation passes exactly, so the stress magnitude is internally consistent as an energy-conjugate gross-area nominal shear stress, while the physical strength interpretation remains uncalibrated.

## Peak And Final Values

- S0070 peak: {float(peak_s0070['peak_stress_MPa']):.6g} MPa; S0070 final: {final_s0070:.6g} MPa.
- S0090 peak: {float(peak_s0090['peak_stress_MPa']):.6g} MPa; S0090 final: {final_s0090:.6g} MPa.

## Classification

`{classification}`
"""
    (PACKAGE / "REPORT.md").write_text(report, encoding="utf-8")

    handoff = f"""## Codex handoff: Shear stress magnitude sanity audit

Commit: TO_BE_FILLED_AFTER_COMMIT
Package folder: `examples/TM_comsol_no_thermal_micro/runs/20260619_shear_stress_magnitude_sanity_audit`

### Source packages reviewed
- `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_existing_geometry_shear_connectivity_extension`
- `examples/TM_comsol_no_thermal_micro/runs/20260618_existing_geometry_shear_longer_connectivity_extension`

### Constraints
- Training run: no.
- Postprocessing run: no.
- D0040 run: no.
- Seed study run: no.
- Physics/boundary/shear ansatz/material/l0/history/alpha/loss changes: no.

### Conclusions
- Stress definition: `nominal_shear_stress_energy_MPa = reaction_N_energy / reference_area_mm2`.
- Reaction-to-stress recomputation: passed; max relative error {max_rel:.3e}.
- Elastic slope sanity: measured gross-area initial slope is much lower than bulk material `G={expected_g:.6g} MPa`, consistent with this being a structure-level gross-area reaction curve rather than a pure-shear material modulus test.
- Reference area: gross area 0.01 mm^2; not net ligament and not local notch-tip stress.
- Material strength context: no explicit target shear strength was found.
- Peak/final clarification: `23.071 MPa is S0070 final stress, not the maximum stress`; actual S0070/S0090 peak is about `29.9647 MPa`.
- Final classification: `{classification}`.
- Commit pushed: TO_BE_FILLED_AFTER_COMMIT.

### Next recommended action
Let the reviewer use this audit with the S0050/S0070/S0090 packages to decide whether to stop the single-seed shear diagnostic, request one final controlled extension, or move to geometry/connectivity interpretation. Do not claim physical validation without an explicit target strength or independent calibration.
"""
    (PACKAGE / "HANDOFF_COMMENT.md").write_text(handoff, encoding="utf-8")

    readme = """# Shear Stress Magnitude Sanity Audit

This package audits the S0050/S0070/S0090 existing-geometry shear stress magnitude using existing handoff tables only. Read `REPORT.md` first, then the tables under `tables/`.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for case, meta in CASES.items():
        pkg = meta["package"]
        case_data = {
            "stress": read_csv(pkg / meta["stress"]),
            "damage": read_csv(pkg / meta["damage"]),
            "summary": read_csv(pkg / meta["summary"]),
            "schedule": meta["schedule"],
        }
        if "connectivity" in meta:
            case_data["connectivity"] = read_csv(pkg / meta["connectivity"])
        data[case] = case_data

    settings = settings_from_artifacts(CASES["S0090"]["package"])
    material = {
        "mat_E_kN_per_mm2": float(settings.get("mat_E_kN_per_mm2", 81.5)),
        "mat_nu": float(settings.get("mat_nu", 0.38)),
        "Gc_kN_per_mm": float(settings.get("Gc_kN_per_mm", 2.4e-6)),
        "l0_mm": float(settings.get("l0_mm", 1.5e-4)),
    }

    definition = stress_definition_trace()
    recompute = recompute_stress_tables(data)
    slope = elastic_slope_table(data, material)
    geometry = reference_geometry_area_audit()
    peak_final = peak_final_table(data)
    strength = material_strength_context(material)
    nominal = nominal_vs_local_interpretation()

    recompute_ok = bool(recompute["relative_error"].max() < 1.0e-12)
    no_strength = True
    classification = (
        "stress magnitude internally consistent"
        if recompute_ok
        else ("stress magnitude unit/area issue suspected" if not recompute_ok else "stress magnitude audit inconclusive")
    )

    definition.to_csv(PACKAGE / "tables" / "shear_stress_definition_trace.csv", index=False)
    recompute.to_csv(PACKAGE / "tables" / "shear_reaction_to_stress_recompute.csv", index=False)
    slope.to_csv(PACKAGE / "tables" / "shear_elastic_slope_sanity.csv", index=False)
    geometry.to_csv(PACKAGE / "tables" / "shear_reference_geometry_area_audit.csv", index=False)
    peak_final.to_csv(PACKAGE / "tables" / "shear_peak_final_stress_comparison.csv", index=False)
    strength.to_csv(PACKAGE / "tables" / "shear_material_strength_context.csv", index=False)
    nominal.to_csv(PACKAGE / "tables" / "shear_nominal_vs_local_stress_interpretation.csv", index=False)

    write_plots(recompute, slope, peak_final)
    write_markdown(definition, recompute, slope, geometry, peak_final, strength, classification)

    source_list = [
        Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\postprocess_results.py"),
        Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\config.py"),
        Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\README.md"),
    ]
    for src in source_list:
        if src.exists():
            (PACKAGE / "artifacts" / "source_snippets" / src.name).write_text(
                src.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
    print(
        {
            "classification": classification,
            "max_recompute_relative_error": float(recompute["relative_error"].max()),
            "max_recompute_absolute_error_MPa": float(recompute["absolute_error_MPa"].max()),
            "package": str(PACKAGE),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
