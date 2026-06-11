from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
REPO_ROOT = PACKAGE.parents[3]
RUNS_ROOT = REPO_ROOT / "examples" / "TM_comsol_no_thermal_micro" / "runs"
EXECUTION_TREE = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")

CLASSIFICATION = (
    "no-thermal physical core acceptable as thermal baseline with documented platform differences"
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def exists(path: Path) -> str:
    return "present" if path.exists() else "missing"


def evidence_status() -> dict[str, str]:
    paths = {
        "repo_memory": RUNS_ROOT / "CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md",
        "project_memory": REPO_ROOT / "examples" / "TM_comsol_no_thermal_micro" / "PROJECT_MEMORY.md",
        "current_config": EXECUTION_TREE / "config.py",
        "current_energy": EXECUTION_TREE / "compute_energy_mixed_tm.py",
        "current_split": EXECUTION_TREE / "mixed_mode_tm.py",
        "current_history": EXECUTION_TREE / "history_field_mixed_tm.py",
        "current_fields": EXECUTION_TREE / "field_computation.py",
        "current_postprocess": EXECUTION_TREE / "postprocess_results.py",
        "current_material": EXECUTION_TREE / "source" / "material_properties.py",
        "current_pff": EXECUTION_TREE / "source" / "pff_model.py",
        "current_readme": EXECUTION_TREE / "README.md",
        "postprocess_workflow": EXECUTION_TREE / "POSTPROCESS_WORKFLOW.md",
        "project_structure": EXECUTION_TREE / "PROJECT_STRUCTURE.md",
        "s0090_handoff": RUNS_ROOT
        / "20260618_existing_geometry_shear_longer_connectivity_extension"
        / "HANDOFF_COMMENT.md",
        "stress_audit_handoff": RUNS_ROOT
        / "20260619_shear_stress_magnitude_sanity_audit"
        / "HANDOFF_COMMENT.md",
    }
    return {name: exists(path) for name, path in paths.items()}


def physical_core_rows() -> list[dict[str, object]]:
    return [
        {
            "topic": "material parameters",
            "COMSOL_comp3_reference": "E0=81.5 GPa, nu=0.38, Gf0=0.0024 MPa*mm, kappa=1e-5",
            "current_PINN_route": "config.py uses E=81.5 kN/mm^2, nu=0.38, Gc=2.4e-6 kN/mm; eta_residual default 1e-5",
            "same_physical_core": "true",
            "implementation_difference": "Unit expression differs but kN/mm units are equivalent to GPa/MPa*mm baseline.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "config.py: COMSOL_E_KN_PER_MM2, COMSOL_NU, Gc, tm_eps_r",
            "audit_comment": "Material and residual stiffness constants should remain unchanged before thermal work.",
        },
        {
            "topic": "geometry scale and thickness/reference area",
            "COMSOL_comp3_reference": "Single-notch micro specimen, 2D plane-stress branch with unit out-of-plane thickness convention",
            "current_PINN_route": "0..0.01 mm by 0..0.01 mm domain; postprocess reference length 0.01 mm and gross reference area 0.01 mm^2",
            "same_physical_core": "true",
            "implementation_difference": "PINN stress is a gross-area nominal reaction measure, not local notch-tip stress.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "config.py SPECIMEN_SIZE_MM; postprocess_results.py REFERENCE_AREA_MM2",
            "audit_comment": "Stress-audit package confirmed reaction/area recomputation consistency.",
        },
        {
            "topic": "2D assumption",
            "COMSOL_comp3_reference": "COMSOL comp3 uses 2D Plane Stress",
            "current_PINN_route": "2D displacement fields with eps_zz reconstructed in TM split; material_properties.py uses 3D Lame form",
            "same_physical_core": "partial",
            "implementation_difference": "Current code is not a line-by-line plane-stress constitutive copy; eps_zz is set in mixed_mode_tm.",
            "acceptable_difference": "conditionally_true",
            "risk_level": "medium",
            "evidence_file_or_function": "mixed_mode_tm._tm_source_split; source/material_properties.py",
            "audit_comment": "Do not change now; document and run thermal-strain patch tests before relying on plane-stress equivalence.",
        },
        {
            "topic": "elastic strain definition",
            "COMSOL_comp3_reference": "Small-strain solid mechanics with elastic strain corrected for thermal strain in thermal branch",
            "current_PINN_route": "Small strains from autograd gradients of u,v; no thermal correction in no-thermal baseline",
            "same_physical_core": "true_for_no_thermal",
            "implementation_difference": "PINN evaluates strain from neural ansatz and mesh-element gradients rather than FEM shape functions.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "compute_energy_mixed_tm.compute_mixed_tm_fields; compute_energy.gradients",
            "audit_comment": "Thermal correction should enter before TM split when a prescribed-temperature branch is added.",
        },
        {
            "topic": "thermal strain status",
            "COMSOL_comp3_reference": "exx_e=exx-alpha_T*(T-Tref), eyy_e=eyy-alpha_T*(T-Tref), exy_e=exy",
            "current_PINN_route": "No temperature field and no thermal strain in current route",
            "same_physical_core": "not_applicable_no_thermal",
            "implementation_difference": "Thermal physics intentionally omitted from this baseline.",
            "acceptable_difference": "true",
            "risk_level": "medium",
            "evidence_file_or_function": "README.md no temperature field statement",
            "audit_comment": "Future implementation must use T-Tref, not raw T.",
        },
        {
            "topic": "tensile/opening drive",
            "COMSOL_comp3_reference": "Opening-like positive split contribution psi_I",
            "current_PINN_route": "tm_source split computes psiI from positive principal-like strain trace part",
            "same_physical_core": "true",
            "implementation_difference": "Symbolic decomposition is TM source split, not required to match COMSOL expressions exactly.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "mixed_mode_tm._tm_source_split psiI",
            "audit_comment": "Physical meaning is opening/tensile drive separation.",
        },
        {
            "topic": "deviatoric/shear drive",
            "COMSOL_comp3_reference": "Shear/deviatoric contribution psi_II with Gf0/GcII weighting",
            "current_PINN_route": "tm_source split computes psiII=mu*ep2 and He_current=psiI+ratio*psiII",
            "same_physical_core": "true",
            "implementation_difference": "PINN uses configured Gc/GcII ratio from current material route.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "mixed_mode_tm._tm_source_split; compute_energy_mixed_tm.compute_mixed_tm_fields",
            "audit_comment": "Recent shear diagnostics show HII active and notch-localized.",
        },
        {
            "topic": "mixed history",
            "COMSOL_comp3_reference": "HI=max(HI,psi_I), HII=max(HII,psi_II), He=HI+(Gf0/GcII)*HII",
            "current_PINN_route": "HI_trial=max(HI_old,psiI), HII_trial=max(HII_old,psiII), He_trial=HI_trial+ratio*HII_trial",
            "same_physical_core": "true",
            "implementation_difference": "PINN commits history across load steps outside a COMSOL state feature.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "compute_energy_mixed_tm; history_field_mixed_tm.commit_mixed_history_from_fields",
            "audit_comment": "Irreversibility concept is preserved; do not alter history logic.",
        },
        {
            "topic": "phase-field model class",
            "COMSOL_comp3_reference": "AT2-like PDE with c=Gf0*l0, a=Gf0/l0+2*He, f=2*He",
            "current_PINN_route": "PFF_model=AT2; fracture density uses w1/c_w*(alpha^2+l0^2|grad alpha|^2), w1=Gc/l0, c_w=2",
            "same_physical_core": "true",
            "implementation_difference": "PINN minimizes energy/loss instead of assembling COMSOL PDE coefficients directly.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "config.py PFF_model_dict; pff_model.damageFun; compute_energy_mixed_tm",
            "audit_comment": "Same AT2 model class; exact solver form need not match.",
        },
        {
            "topic": "degradation function",
            "COMSOL_comp3_reference": "g(d)=(1-d)^2+kappa degrades positive/crack-driving stress part",
            "current_PINN_route": "g_alpha=(1-alpha)^2+eta_residual multiplies history/current crack-driving energy; stress postprocess has tm effective stress",
            "same_physical_core": "true",
            "implementation_difference": "COMSOL may use external stress trick; PINN primarily degrades energy.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "compute_energy_mixed_tm.g_alpha; mixed_mode_tm.tm_source_effective_stress_fields",
            "audit_comment": "Energy degradation is acceptable if reaction remains energy-conjugate.",
        },
        {
            "topic": "residual stiffness/kappa",
            "COMSOL_comp3_reference": "kappa=1e-5 residual stiffness",
            "current_PINN_route": "eta_residual/tm_eps_r baseline 1e-5; g_alpha includes eta_residual",
            "same_physical_core": "true",
            "implementation_difference": "Name differs: eta_residual in energy, tm_eps_r in TM split regularization.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "config.py training_dict; compute_energy_mixed_tm.g_alpha",
            "audit_comment": "Keep the current values unless a separate bug-fix task proves otherwise.",
        },
        {
            "topic": "crack length scale/l0",
            "COMSOL_comp3_reference": "l0=0.15 um = 1.5e-4 mm",
            "current_PINN_route": "Default --l0 is 1.5e-4 mm; model settings record l0_mm",
            "same_physical_core": "true",
            "implementation_difference": "None relevant; unit conversion must stay explicit.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "config.py parser --l0; mat_prop_dict",
            "audit_comment": "Changing l0 would invalidate this baseline audit.",
        },
        {
            "topic": "irreversibility/history update",
            "COMSOL_comp3_reference": "State variables preserve max history",
            "current_PINN_route": "history_field_mixed_tm commits max history after load step and saves He_history diagnostics",
            "same_physical_core": "true",
            "implementation_difference": "PINN history is explicit Python step state rather than COMSOL state3.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "history_field_mixed_tm.commit_mixed_tm_history_from_model",
            "audit_comment": "Do not replace with current-energy-only drive before thermal reintroduction.",
        },
        {
            "topic": "reaction definition",
            "COMSOL_comp3_reference": "Reaction can be extracted by solver/postprocess boundary quantities",
            "current_PINN_route": "Primary reaction_N_energy=1000*dPi/dDelta or dPi/dDelta_s from saved checkpoints",
            "same_physical_core": "true",
            "implementation_difference": "Energy derivative route differs from FEM boundary reaction extraction.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "postprocess_results.compute_exact_reaction_rows",
            "audit_comment": "Do not reintroduce legacy top-sigma as primary reaction.",
        },
        {
            "topic": "stress normalization",
            "COMSOL_comp3_reference": "Stress interpretation depends on selected cross-section and units",
            "current_PINN_route": "Nominal stress equals reaction_N_energy/reference_area_mm2; shear uses nominal_shear_stress_energy_MPa",
            "same_physical_core": "true",
            "implementation_difference": "This is gross-area nominal stress, not local material shear strength.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "postprocess_results.build_stress_strain_curve; 20260619 stress audit",
            "audit_comment": "Stress audit classified magnitude as internally consistent.",
        },
        {
            "topic": "heat equation status",
            "COMSOL_comp3_reference": "ht3 active heat-transfer physics in comp3",
            "current_PINN_route": "No heat equation in no-thermal PINN route",
            "same_physical_core": "not_applicable_no_thermal",
            "implementation_difference": "Thermal transport omitted intentionally.",
            "acceptable_difference": "true_for_baseline",
            "risk_level": "medium",
            "evidence_file_or_function": "README.md no heat equation statement",
            "audit_comment": "Do not jump directly to full heat PDE without thermal-strain patch tests.",
        },
        {
            "topic": "damage-dependent conductivity status",
            "COMSOL_comp3_reference": "k_d=g(d)*k0 in heat transfer",
            "current_PINN_route": "No conductivity model in current no-thermal route",
            "same_physical_core": "not_applicable_no_thermal",
            "implementation_difference": "Damage-dependent conductivity is deferred.",
            "acceptable_difference": "true_for_baseline",
            "risk_level": "medium",
            "evidence_file_or_function": "README.md no thermal transport coupling statement",
            "audit_comment": "Future heat branch needs separate conservation and units checks.",
        },
        {
            "topic": "boundary condition treatment",
            "COMSOL_comp3_reference": "FEM boundary constraints/selections",
            "current_PINN_route": "Dirichlet constraints encoded in displacement ansatz; bottom fixed, top prescribed, left/right free",
            "same_physical_core": "true",
            "implementation_difference": "Ansatz replaces explicit FEM constraint nodes.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "field_computation.FieldComputation.fieldCalculation",
            "audit_comment": "Do not modify tension/shear boundary treatment in this audit.",
        },
        {
            "topic": "displacement ansatz",
            "COMSOL_comp3_reference": "Direct FEM displacement degrees of freedom",
            "current_PINN_route": "Top-u/free tension default; shear ansatz u=Delta_s*(eta+bubble*raw_u), v=Delta_s*(eta+bubble)*raw_v",
            "same_physical_core": "true",
            "implementation_difference": "PINN ansatz enforces selected essential boundary conditions analytically.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "field_computation.py lines for shear and tension branches",
            "audit_comment": "Prompt explicitly forbids shear ansatz changes.",
        },
        {
            "topic": "postprocess route",
            "COMSOL_comp3_reference": "COMSOL results/postprocess tables can be solver-derived",
            "current_PINN_route": "postprocess_results.py reconstructs checkpoint energy and writes reaction/stress tables",
            "same_physical_core": "true",
            "implementation_difference": "Checkpoint reconstruction differs from COMSOL result evaluation.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "POSTPROCESS_WORKFLOW.md; postprocess_results.py",
            "audit_comment": "This is the trusted current route.",
        },
        {
            "topic": "generated-artifact hygiene",
            "COMSOL_comp3_reference": "Not a physics invariant",
            "current_PINN_route": "Project docs require generated outputs under outputs/ and audit packages under runs/",
            "same_physical_core": "not_applicable",
            "implementation_difference": "Repository hygiene rule, not model physics.",
            "acceptable_difference": "true",
            "risk_level": "low",
            "evidence_file_or_function": "PROJECT_STRUCTURE.md",
            "audit_comment": "This audit package is confined to runs/ and does not alter source.",
        },
    ]


def comsol_reference_rows() -> list[dict[str, object]]:
    return [
        {
            "item": "component",
            "value": "comp3 / solid3 / ht3 / phase-field PDE c / state3 / std1",
            "role_in_physics": "Single-notch thermo-mechanical phase-field reference branch",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "false",
            "comment": "Tags identify the reference branch; PINN implementation need not reuse COMSOL tags.",
        },
        {
            "item": "ignored branch",
            "value": "comp4 and TFinal ignored",
            "role_in_physics": "Prevents mixing multi-pore or unrelated thermal settings into single-notch route",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Do not use comp4 facts for this baseline.",
        },
        {
            "item": "plane stress",
            "value": "2D Plane Stress",
            "role_in_physics": "Constitutive assumption for comp3 solid mechanics",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "false",
            "comment": "The physical assumption must be documented; code symbols need not match COMSOL line by line.",
        },
        {
            "item": "thermal strain exx",
            "value": "exx_e = exx - alpha_T*(T - Tref)",
            "role_in_physics": "Thermoelastic strain correction",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Reference-temperature convention is a physical invariant.",
        },
        {
            "item": "thermal strain eyy",
            "value": "eyy_e = eyy - alpha_T*(T - Tref)",
            "role_in_physics": "Thermoelastic strain correction",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Using raw T instead of T-Tref would be high risk.",
        },
        {
            "item": "thermal strain exy",
            "value": "exy_e = exy",
            "role_in_physics": "No isotropic thermal shear strain",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Shear strain should not receive alpha_T*(T-Tref).",
        },
        {
            "item": "history",
            "value": "HI=max(HI,psi_I), HII=max(HII,psi_II), He=HI+(Gf0/GcII)*HII",
            "role_in_physics": "Irreversible mixed-mode crack drive",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Implementation details may differ, but max-history physics must be preserved.",
        },
        {
            "item": "AT2 PDE",
            "value": "c=Gf0*l0, a=Gf0/l0+2*He, f=2*He",
            "role_in_physics": "Phase-field damage model class",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Same AT2 class is required; energy vs PDE implementation can differ.",
        },
        {
            "item": "degradation",
            "value": "g(d)=(1-d)^2+kappa",
            "role_in_physics": "Residual-stiffness damage degradation",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Parameter and physical form should remain consistent.",
        },
        {
            "item": "conductivity",
            "value": "k_d=g(d)*k0",
            "role_in_physics": "Damage-dependent heat transfer in COMSOL",
            "required_for_future_thermal_branch": "later",
            "must_match_exactly_in_PINN": "false",
            "comment": "Not required for first prescribed-temperature thermal-strain branch.",
        },
        {
            "item": "thermal material constants",
            "value": "alpha_T=18.9 ppm/K, rho=1040 kg/m^3, k0=418 W/m/K, c=170 J/kg/K, Tref=273.15 K, T0=0 degC",
            "role_in_physics": "Thermal expansion and heat equation parameters",
            "required_for_future_thermal_branch": "true",
            "must_match_exactly_in_PINN": "true",
            "comment": "Keep units explicit when thermal physics is introduced.",
        },
    ]


def pinn_route_rows() -> list[dict[str, object]]:
    return [
        {
            "item": "route",
            "current_status": "mixedH_TM + tm_source + history",
            "evidence": "config.py history_mode, mixed_split_mode, mixed_mechanics_mode",
            "acceptable_as_baseline": "true",
            "comment": "This is the cleaned current no-thermal route.",
        },
        {
            "item": "phase field",
            "current_status": "AT2",
            "evidence": "config.py PFF_model_dict; source/pff_model.py damageFun",
            "acceptable_as_baseline": "true",
            "comment": "Matches required model class.",
        },
        {
            "item": "alpha initialization",
            "current_status": "default alpha initialization; no pre-existing phase-field crack",
            "evidence": "config.py initial_phase_field_crack_enabled=False; README.md",
            "acceptable_as_baseline": "true",
            "comment": "Do not reintroduce alpha-init-intact route.",
        },
        {
            "item": "coordinate normalization",
            "current_status": "unit_box network input normalization by default",
            "evidence": "config.py coord_normalization default; field_computation.network_input",
            "acceptable_as_baseline": "true",
            "comment": "Gradients still use physical x,y.",
        },
        {
            "item": "tension boundary",
            "current_status": "top-u/free default, bottom fixed, left/right free",
            "evidence": "field_computation.py tension branch",
            "acceptable_as_baseline": "true",
            "comment": "Do not alter boundary conditions in this audit.",
        },
        {
            "item": "shear boundary",
            "current_status": "top u=Delta_s, top v free through ansatz, bottom fixed",
            "evidence": "field_computation.py shear branch; memory shear policy",
            "acceptable_as_baseline": "true",
            "comment": "Do not modify shear ansatz.",
        },
        {
            "item": "history logic",
            "current_status": "HI/HII max history committed after load steps",
            "evidence": "history_field_mixed_tm.commit_mixed_history_from_fields",
            "acceptable_as_baseline": "true",
            "comment": "Conceptually consistent with COMSOL comp3 history state.",
        },
        {
            "item": "reaction",
            "current_status": "energy-conjugate checkpoint derivative primary reaction",
            "evidence": "postprocess_results.compute_exact_reaction_rows",
            "acceptable_as_baseline": "true",
            "comment": "Legacy top-sigma route is not the normal primary output.",
        },
        {
            "item": "stress normalization",
            "current_status": "gross-area nominal stress from reaction/reference_area",
            "evidence": "postprocess_results.build_stress_strain_curve; 20260619 stress audit",
            "acceptable_as_baseline": "true",
            "comment": "Internally consistent but not local material strength.",
        },
        {
            "item": "thermal physics",
            "current_status": "not implemented in current no-thermal baseline",
            "evidence": "README.md",
            "acceptable_as_baseline": "true",
            "comment": "Expected for this audit; add later via prescribed-temperature patch tests.",
        },
    ]


def platform_difference_rows() -> list[dict[str, object]]:
    return [
        {
            "difference": "Positive stress degradation route",
            "COMSOL_implementation": "ExternalStress-like degradation of positive stress contribution",
            "PINN_implementation": "Energy degradation with g_alpha*He plus effective-stress diagnostics",
            "acceptable": "true",
            "reason": "Both express residual-stiffness degradation of crack-driving positive energy/stress.",
            "risk_if_ignored": "May incorrectly demand line-by-line COMSOL source matching.",
            "recommended_action_before_thermal_reintroduction": "Keep energy-conjugate reaction as primary metric and document stress diagnostics as secondary.",
        },
        {
            "difference": "FEM solver versus PINN optimizer",
            "COMSOL_implementation": "FEM PDE solve, likely segregated/nonlinear study",
            "PINN_implementation": "Neural ansatz optimized by energy/residual losses",
            "acceptable": "true",
            "reason": "Platform implementation differs while physical energy/history model is preserved.",
            "risk_if_ignored": "False assumption that identical residual equations are required.",
            "recommended_action_before_thermal_reintroduction": "Use patch tests and energy identities instead of line-by-line comparisons.",
        },
        {
            "difference": "Heat equation active in reference",
            "COMSOL_implementation": "ht3 heat-transfer physics present with thermal material properties",
            "PINN_implementation": "No heat equation in no-thermal baseline",
            "acceptable": "true_for_baseline",
            "reason": "Audit target is no-thermal baseline before thermal reintroduction.",
            "risk_if_ignored": "Prematurely coupling heat transport before mechanics baseline is frozen.",
            "recommended_action_before_thermal_reintroduction": "Start with prescribed-temperature thermal strain, then add heat PDE separately.",
        },
        {
            "difference": "Boundary treatment",
            "COMSOL_implementation": "FEM boundary constraints/selections",
            "PINN_implementation": "Boundary conditions encoded in displacement ansatz",
            "acceptable": "true",
            "reason": "Essential boundary meanings are preserved.",
            "risk_if_ignored": "Unnecessary ansatz edits could invalidate verified tension/shear routes.",
            "recommended_action_before_thermal_reintroduction": "Do not change boundary conditions while adding thermal strain.",
        },
        {
            "difference": "Postprocess reaction extraction",
            "COMSOL_implementation": "Boundary or solver-derived reaction quantities",
            "PINN_implementation": "Checkpoint mechanics-energy derivative dPi/dDelta",
            "acceptable": "true",
            "reason": "Energy derivative is conjugate to prescribed displacement in the current route.",
            "risk_if_ignored": "Reintroducing legacy top-sigma metric as primary could misclassify curves.",
            "recommended_action_before_thermal_reintroduction": "Keep reaction_N_energy primary and mark unavailable when checkpoints are missing.",
        },
        {
            "difference": "Plane stress documentation",
            "COMSOL_implementation": "2D Plane Stress",
            "PINN_implementation": "2D fields with eps_zz reconstruction and 3D Lamé constants in material helper",
            "acceptable": "conditionally_true",
            "reason": "No source change requested; current mechanics route can be baseline if documented and patch-tested.",
            "risk_if_ignored": "Thermal expansion may be mapped with the wrong constitutive assumption.",
            "recommended_action_before_thermal_reintroduction": "Run zero-damage thermoelastic patch tests and document plane-stress/plane-strain convention before full coupling.",
        },
        {
            "difference": "Exact symbolic TM split",
            "COMSOL_implementation": "COMSOL expressions for psi_I and psi_II",
            "PINN_implementation": "tm_source split with positive strain parts and psiI/psiII fields",
            "acceptable": "true",
            "reason": "Prompt allows non-identical symbolic decomposition when physical meaning is preserved.",
            "risk_if_ignored": "Unnecessary rewrite of verified split formulas.",
            "recommended_action_before_thermal_reintroduction": "Preserve current split; only add thermal strain input upstream in a controlled branch.",
        },
    ]


def readiness_rows() -> list[dict[str, object]]:
    return [
        {
            "readiness_item": "no-thermal baseline stability",
            "current_status": "Cleaned verified route documented; S0090 and stress audit completed",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Freeze current no-thermal route for thermal branch baseline.",
        },
        {
            "readiness_item": "energy-conjugate reaction availability",
            "current_status": "postprocess_results.py supports checkpointed dPi/dDelta and dPi/dDelta_s",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Keep checkpoint saving enabled for thermal patch tests.",
        },
        {
            "readiness_item": "material parameter consistency",
            "current_status": "E, nu, Gf0, l0, residual stiffness match current baseline",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Do not retune material constants during thermal reintroduction.",
        },
        {
            "readiness_item": "AT2 consistency",
            "current_status": "AT2 fracture energy class present",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Preserve AT2 form and w1=Gc/l0.",
        },
        {
            "readiness_item": "history consistency",
            "current_status": "HI/HII max-history behavior present",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Thermal branch should feed thermoelastic strains into the same history update.",
        },
        {
            "readiness_item": "stress normalization consistency",
            "current_status": "Stress audit classified nominal shear stress internally consistent",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Continue reporting gross-area nominal stress with clear labels.",
        },
        {
            "readiness_item": "COMSOL comp3 thermal strain mapping",
            "current_status": "Reference formula identified; not yet implemented",
            "ready": "partial",
            "blocking_issue": "Need patch tests for T-Tref convention and plane-stress assumption.",
            "suggested_next_step": "Implement prescribed-temperature thermal-strain branch in a separate task.",
        },
        {
            "readiness_item": "prescribed-temperature branch feasibility",
            "current_status": "Mechanics energy route has a clear insertion point before TM split",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Add fixed temperature field input and test zero-damage thermoelastic response.",
        },
        {
            "readiness_item": "heat PDE branch feasibility",
            "current_status": "Not implemented and not needed for first thermal-strain branch",
            "ready": "partial",
            "blocking_issue": "Requires heat residual, units, boundary conditions, and thermal data validation.",
            "suggested_next_step": "Defer until prescribed-temperature mechanics branch passes.",
        },
        {
            "readiness_item": "damage-dependent conductivity feasibility",
            "current_status": "Not implemented",
            "ready": "partial",
            "blocking_issue": "Requires heat PDE branch and k_d=g(d)*k0 validation.",
            "suggested_next_step": "Defer behind heat PDE conservation tests.",
        },
        {
            "readiness_item": "required patch tests",
            "current_status": "Not part of this read-only audit",
            "ready": "false",
            "blocking_issue": "Patch tests must be created with the future thermal branch.",
            "suggested_next_step": "Add tests for uniform DeltaT free expansion, constrained thermal stress, and zero-DeltaT equivalence.",
        },
        {
            "readiness_item": "rollback safety",
            "current_status": "This package changes only runs/ audit artifacts",
            "ready": "true",
            "blocking_issue": "",
            "suggested_next_step": "Keep thermal work in a separate branch/package and preserve current no-thermal outputs.",
        },
    ]


def constraints_rows() -> list[dict[str, object]]:
    rows = [
        ("source code modified", "false", "No production/source model files were changed."),
        ("physics changed", "false", "No equations, losses, split, or model parameters were edited."),
        ("boundary condition changed", "false", "No tension or shear boundary/ansatz code was edited."),
        ("shear ansatz changed", "false", "Shear ansatz was only reviewed."),
        ("material parameter changed", "false", "E, nu, Gf0, GcII, kappa/eta_residual, eps_r were not changed."),
        ("l0 changed", "false", "l0 remains 1.5e-4 mm."),
        ("history logic changed", "false", "HI/HII max-history code was not edited."),
        ("training loss changed", "false", "No loss code was edited."),
        ("new training run", "false", "No training command was run."),
        ("D0040 run", "false", "D0040 was not run."),
        ("seed study run", "false", "No seed study was run."),
        ("heat PDE implemented", "false", "No heat PDE or conductivity model was implemented."),
        ("COMSOL exact-matching imposed", "false", "Audit accepts documented platform implementation differences."),
        ("comp4 or TFinal used", "false", "Reference restricted to comp3 prompt facts."),
    ]
    return [
        {
            "constraint": name,
            "observed": observed,
            "evidence_or_comment": comment,
        }
        for name, observed, comment in rows
    ]


def manifest(status: dict[str, str]) -> dict[str, object]:
    files = [
        "REPORT.md",
        "HANDOFF_COMMENT.md",
        "MANIFEST.json",
        "tables/physical_core_consistency_matrix.csv",
        "tables/comsol_comp3_reference_summary.csv",
        "tables/pinn_current_route_summary.csv",
        "tables/platform_difference_acceptance.csv",
        "tables/thermal_reintroduction_readiness.csv",
        "tables/do_not_change_constraints.csv",
        "artifacts/generate_physical_core_audit.py",
    ]
    return {
        "package": str(PACKAGE.relative_to(REPO_ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": CLASSIFICATION,
        "scope": "read-only physical-core audit before thermal coupling reintroduction",
        "source_code_modified": False,
        "training_run": False,
        "postprocessing_run": False,
        "D0040_run": False,
        "seed_study_run": False,
        "comsol_runtime_used": False,
        "reference_branch": "COMSOL comp3 only; comp4 and TFinal ignored",
        "source_evidence_tree": str(EXECUTION_TREE),
        "repo_package_tree": str(REPO_ROOT),
        "evidence_status": status,
        "files": files,
        "commit_hash": "PENDING_HANDOFF_UPDATE",
        "pushed": "PENDING_HANDOFF_UPDATE",
    }


def report_md(status: dict[str, str]) -> str:
    evidence_lines = "\n".join(f"- {key}: {value}" for key, value in status.items())
    return f"""# Existing Project Physical-Core Audit

## 1. Purpose

This audit checks whether the existing no-thermal PINN phase-field fracture route has a clean, physically interpretable core that can serve as the baseline for later thermal coupling. It does not attempt to make the PINN implementation identical to COMSOL line by line.

## 2. Existing no-thermal PINN route reviewed

The reviewed route is `mixedH_TM + tm_source + history` with AT2 phase field, default alpha initialization, top-u/free tension ansatz, the existing top-v-free shear ansatz, unit-box coordinate normalization, and checkpointed energy-conjugate reaction as the primary reaction route.

## 3. Relevant COMSOL branch

The only COMSOL reference branch used here is `comp3 / solid3 / ht3 / c / state3 / std1`. The COMSOL facts are treated as theoretical reference facts supplied by the task prompt, not as a requirement that every implementation detail must match.

## 4. Why comp4 is ignored

`comp4`, multi-pore settings, and `TFinal` are explicitly outside this single-notch audit. Mixing them into this baseline would create an invalid reference for the current no-thermal route.

## 5. Material parameters to keep unchanged

- `E0 = 81.5 GPa` / `81.5 kN/mm^2`
- `nu = 0.38`
- `Gf0 = 0.0024 MPa*mm` / `2.4e-6 kN/mm`
- `kappa = 1e-5`
- `l0 = 0.15 um` / `1.5e-4 mm`
- `eps_r = 1e-5`
- future thermal constants: `alpha_T=18.9 ppm/K`, `rho=1040 kg/m^3`, `k0=418 W/m/K`, `c=170 J/kg/K`, `Tref=273.15 K`, `T0=0 degC`

## 6. Same physical core assessment

The current no-thermal PINN route shares the required physical core for a baseline: material units, AT2 model class, `l0`, residual stiffness convention, mixed `psiI/psiII` drive, max-history irreversibility, degradation of crack-driving energy, and energy-conjugate reaction are internally consistent. The main caveat is the constitutive convention: COMSOL comp3 is documented as plane stress, while the current PINN uses 2D fields with `eps_zz` reconstruction and a Lame helper that should be documented and patch-tested before thermal strain is trusted.

## 7. Acceptable platform implementation differences

Acceptable differences include COMSOL external-stress degradation versus PINN energy degradation, FEM PDE solve versus neural energy/residual optimization, COMSOL boundary constraints versus PINN ansatz constraints, COMSOL heat transfer being active while the current baseline is intentionally no-thermal, and COMSOL postprocess reactions versus PINN checkpoint energy derivatives.

## 8. Unacceptable differences before thermal coupling

High-risk differences to avoid later are wrong material units, wrong `l0` units, using thermal strain as `alpha_T*T` instead of `alpha_T*(T-Tref)`, silently degrading compressive stress without justification, losing max-history irreversibility, reintroducing legacy top-sigma as the primary reaction metric, changing the shear ansatz, and mixing comp4/TFinal into the single-notch route.

## 9. Exact COMSOL matching

Exact line-by-line COMSOL matching is not required. Physical-model invariants must match, while solver mechanics, state storage, boundary enforcement, and postprocess extraction may differ when the physical meaning is preserved and documented.

## 10. Baseline classification

Final classification: `{CLASSIFICATION}`.

The existing no-thermal PINN route is acceptable as the baseline for thermal reintroduction if the AT2 phase-field class, mixed history concept, degradation logic, material units, `l0`, and energy-conjugate reaction route remain internally consistent. Exact COMSOL implementation matching is not required. Differences caused by FEM versus PINN formulations are acceptable when the physical meaning is preserved and documented.

## 11. What must not change before thermal reintroduction

Do not change source physics, boundary conditions, shear ansatz, material parameters, `l0`, history logic, alpha initialization, training losses, reaction policy, or load schedules as part of this audit baseline. Do not run D0040 or a seed study as a substitute for thermal patch tests.

## 12. Safest next step

The safest next task is a prescribed-temperature thermal-strain branch with patch tests, not a full heat-equation/damage-conductivity coupling. Start with zero-damage thermoelastic tests for uniform free expansion, constrained thermal stress, zero-DeltaT equivalence to the current no-thermal route, and the `T-Tref` convention.

## Constraints Observed

- no source code was modified;
- no physics was changed;
- no boundary condition was changed;
- no shear ansatz was changed;
- no material parameter was changed;
- no `l0` was changed;
- no history logic was changed;
- no training loss was changed;
- no new training was run;
- no D0040 was run;
- no seed study was run;
- no heat PDE was implemented;
- no COMSOL exact-matching requirement was imposed.

## Evidence Status

{evidence_lines}
"""


def handoff_md(status: dict[str, str]) -> str:
    evidence_lines = "\n".join(f"- {key}: {value}" for key, value in status.items())
    return f"""# Handoff Comment: Existing Project Physical-Core Audit

Package folder: `examples/TM_comsol_no_thermal_micro/runs/20260620_existing_project_physical_core_audit`
Commit hash: `PENDING_HANDOFF_UPDATE`
Commit pushed: `PENDING_HANDOFF_UPDATE`

## Source Packages and Files Reviewed

- Project memory: `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`
- Project memory pointer: `examples/TM_comsol_no_thermal_micro/PROJECT_MEMORY.md`
- Current execution tree: `{EXECUTION_TREE}`
- Current source files reviewed: `config.py`, `compute_energy_mixed_tm.py`, `mixed_mode_tm.py`, `history_field_mixed_tm.py`, `field_computation.py`, `postprocess_results.py`, `source/material_properties.py`, `source/pff_model.py`
- Current docs reviewed: `README.md`, `POSTPROCESS_WORKFLOW.md`, `PROJECT_STRUCTURE.md`
- Tests reviewed by source inspection: `test_single_verified_pipeline.py`, `test_history_mode_controls.py`, `test_postprocess_results.py`, `test_shear_load_case.py`, `test_shear_connectivity.py`, `test_shear_package_builder.py`, `test_coord_normalization.py`, `test_project_cleanup_interface.py`, `test_project_directory_hygiene.py`
- Prior packages reviewed: `20260618_existing_geometry_shear_longer_connectivity_extension`, `20260619_shear_stress_magnitude_sanity_audit`

## Execution Status

- Source code modified: no
- Training run: no
- Postprocessing run: no
- D0040 run: no
- Seed study run: no
- COMSOL runtime used: no
- Physics/boundary/shear ansatz/material/l0/history/alpha/loss changes: no

## Final Physical-Core Classification

`{CLASSIFICATION}`

## Biggest Acceptable Differences

- COMSOL can degrade positive stress through an external-stress construction; PINN can degrade crack-driving energy directly.
- COMSOL solves FEM PDEs; PINN optimizes a neural ansatz with energy/residual losses.
- COMSOL heat transfer is active in comp3; the current baseline intentionally omits heat transfer.
- COMSOL boundary constraints and PINN ansatz constraints differ in implementation but preserve boundary meaning.
- COMSOL reaction extraction and PINN checkpoint energy derivative differ, with `reaction_N_energy` remaining the trusted current route.

## Biggest Unacceptable Risks To Avoid Later

- Wrong material or `l0` units.
- Silent plane-stress versus plane-strain mismatch in thermal strain tests.
- Thermal strain using `T` instead of `T-Tref`.
- Degrading compressive stress without justification.
- Reintroducing legacy top-sigma reaction as the primary metric.
- Losing HI/HII max-history irreversibility.
- Mixing comp4 or `TFinal` into the single-notch branch.

## Recommended Next Step

Create a prescribed-temperature thermal-strain branch with patch tests first. Do not start with full heat-equation or damage-dependent-conductivity coupling.

## Evidence Status

{evidence_lines}
"""


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    status = evidence_status()

    write_csv(
        TABLES / "physical_core_consistency_matrix.csv",
        [
            "topic",
            "COMSOL_comp3_reference",
            "current_PINN_route",
            "same_physical_core",
            "implementation_difference",
            "acceptable_difference",
            "risk_level",
            "evidence_file_or_function",
            "audit_comment",
        ],
        physical_core_rows(),
    )
    write_csv(
        TABLES / "comsol_comp3_reference_summary.csv",
        [
            "item",
            "value",
            "role_in_physics",
            "required_for_future_thermal_branch",
            "must_match_exactly_in_PINN",
            "comment",
        ],
        comsol_reference_rows(),
    )
    write_csv(
        TABLES / "pinn_current_route_summary.csv",
        ["item", "current_status", "evidence", "acceptable_as_baseline", "comment"],
        pinn_route_rows(),
    )
    write_csv(
        TABLES / "platform_difference_acceptance.csv",
        [
            "difference",
            "COMSOL_implementation",
            "PINN_implementation",
            "acceptable",
            "reason",
            "risk_if_ignored",
            "recommended_action_before_thermal_reintroduction",
        ],
        platform_difference_rows(),
    )
    write_csv(
        TABLES / "thermal_reintroduction_readiness.csv",
        ["readiness_item", "current_status", "ready", "blocking_issue", "suggested_next_step"],
        readiness_rows(),
    )
    write_csv(
        TABLES / "do_not_change_constraints.csv",
        ["constraint", "observed", "evidence_or_comment"],
        constraints_rows(),
    )
    (PACKAGE / "REPORT.md").write_text(report_md(status), encoding="utf-8")
    (PACKAGE / "HANDOFF_COMMENT.md").write_text(handoff_md(status), encoding="utf-8")
    (PACKAGE / "MANIFEST.json").write_text(
        json.dumps(manifest(status), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
