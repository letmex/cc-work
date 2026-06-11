import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
TABLE_DIR = PACKAGE_DIR / "tables"
FIGURE_DIR = PACKAGE_DIR / "figures"

PACKAGE_REL = "examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan"
MEMORY_REL = "examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md"

REPORT_SECTIONS = [
    "Purpose",
    "Scope boundaries",
    "Current prescribed-temperature baseline status",
    "Heat PDE target equation",
    "Unit system and conversion blockers",
    "Temperature variable and representation options",
    "Boundary and initial condition plan",
    "First implementation phase recommendation",
    "Validation ladder",
    "Patch test plan",
    "Coupling strategy",
    "Deferred features",
    "COMSOL comp3 alignment and non-alignment",
    "Source touch plan",
    "Risks and blockers",
    "Decision gate",
    "Final classification",
    "Exact next recommended task",
]

FINAL_CLASSIFICATION = "heat PDE implementation plan complete"
TARGET_EQUATION = "rho * c * dT/dt - div(k0 * grad T) = Q"


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def heat_pde_scope_summary_rows():
    return [
        {
            "item": "temperature field",
            "proposed_status": "future implementation",
            "first_phase_behavior": "Introduce a solved or approximated T only after unit convention approval; preserve prescribed-temperature baseline.",
            "deferred_behavior": "No trainable/PDE temperature field in this planning task.",
            "reason": "Current branch uses prescribed delta_T only; solved T adds a new state variable and optimization surface.",
        },
        {
            "item": "heat PDE residual",
            "proposed_status": "future implementation",
            "first_phase_behavior": "Plan residual for constant-conductivity heat transfer only.",
            "deferred_behavior": "No heat residual loss is added by this package.",
            "reason": "Residual scaling and units must be decided before code changes.",
        },
        {
            "item": "constant conductivity",
            "proposed_status": "recommended first heat PDE physics",
            "first_phase_behavior": "Use constant k0 with Q=0 for patch tests.",
            "deferred_behavior": "Do not combine with k(d) during initial validation.",
            "reason": "Constant k0 isolates thermal transport from damage coupling.",
        },
        {
            "item": "damage-dependent conductivity",
            "proposed_status": "deferred",
            "first_phase_behavior": "Explicit guard that k_d = g(d) * k0 is not active.",
            "deferred_behavior": "Consider only after constant-k0 heat solve and solved-T mechanics coupling pass independently.",
            "reason": "Adding k(d) before the heat PDE is stable would conflate two new physics changes.",
        },
        {
            "item": "heat source Q",
            "proposed_status": "deferred except zero source",
            "first_phase_behavior": "Start with Q=0.",
            "deferred_behavior": "Nonzero source and thermomechanical generation require a separate decision.",
            "reason": "Zero source enables closed-form patch tests and conservation checks.",
        },
        {
            "item": "thermal strain coupling to mechanics",
            "proposed_status": "preserve existing relation",
            "first_phase_behavior": "Keep delta_T = T - Tref; exx_e = exx - alpha_T * delta_T; eyy_e = eyy - alpha_T * delta_T; exy_e = exy.",
            "deferred_behavior": "No change to split/history/loss route.",
            "reason": "Patch tests already validate this prescribed-temperature mechanics path.",
        },
        {
            "item": "thermal boundary conditions",
            "proposed_status": "future implementation",
            "first_phase_behavior": "Support minimal Dirichlet T and insulated/Neumann flux only after unit gate.",
            "deferred_behavior": "No boundary-condition code in this task.",
            "reason": "Boundary semantics must be validated before coupled diagnostics.",
        },
        {
            "item": "thermal initial condition",
            "proposed_status": "future implementation",
            "first_phase_behavior": "Uniform initial T first; linear initial T only for controlled patch tests.",
            "deferred_behavior": "No transient IC code in this package.",
            "reason": "Transient behavior needs a clear time and thermal diffusivity convention.",
        },
        {
            "item": "transient time stepping",
            "proposed_status": "later phase",
            "first_phase_behavior": "Plan tests but do not implement before steady or residual patch tests.",
            "deferred_behavior": "Full transient schedules and time-dependent heating.",
            "reason": "Time units and rho*c scaling are unresolved.",
        },
        {
            "item": "steady-state option",
            "proposed_status": "recommended before transient",
            "first_phase_behavior": "Use div(k0 * grad T) + Q = 0 form for simple conduction patch tests.",
            "deferred_behavior": "No physical notch run until patch tests pass.",
            "reason": "Steady tests reduce moving parts while validating gradients and BCs.",
        },
        {
            "item": "prescribed-temperature fallback",
            "proposed_status": "preserve",
            "first_phase_behavior": "Keep thermal_mode=off default and existing prescribed-temperature modes available.",
            "deferred_behavior": "Do not replace the baseline branch during heat PDE work.",
            "reason": "Existing diagnostics rely on this reviewed fallback.",
        },
        {
            "item": "reaction route",
            "proposed_status": "unchanged",
            "first_phase_behavior": "Keep checkpointed energy-conjugate reaction_N_energy as primary.",
            "deferred_behavior": "Do not reintroduce legacy top-sigma as primary reaction.",
            "reason": "Reaction route is a protected invariant in project memory.",
        },
        {
            "item": "phase-field/history coupling",
            "proposed_status": "unchanged in first heat PDE phase",
            "first_phase_behavior": "Heat affects mechanics only through solved T to thermal strain after separate approval.",
            "deferred_behavior": "No direct heat-to-history or damage-conductivity feedback initially.",
            "reason": "History irreversibility and AT2 route should not change while thermal transport is introduced.",
        },
    ]


def thermal_variables_units_rows():
    return [
        {
            "variable": "T",
            "meaning": "absolute temperature field",
            "proposed_unit": "K",
            "current_project_unit_or_note": "Currently only prescribed scalar/field inputs are supported.",
            "conversion_needed": "no for temperature difference; yes when combined with transport constants",
            "source_reference": "thermal_prescribed.py; THERMAL_REINTRODUCTION_PLAN.md",
            "risk_note": "Solved representation and checkpoint format are not yet defined.",
        },
        {
            "variable": "Tref",
            "meaning": "reference temperature for thermal strain",
            "proposed_unit": "K",
            "current_project_unit_or_note": "273.15 K default",
            "conversion_needed": "no",
            "source_reference": "thermal_prescribed.DEFAULT_TREF_K",
            "risk_note": "Must remain consistent between prescribed and solved-T routes.",
        },
        {
            "variable": "delta_T",
            "meaning": "temperature change T - Tref",
            "proposed_unit": "K",
            "current_project_unit_or_note": "Existing prescribed branch uses K.",
            "conversion_needed": "no",
            "source_reference": "delta_T_from_temperature",
            "risk_note": "Signs and reference temperature must be preserved.",
        },
        {
            "variable": "alpha_T",
            "meaning": "isotropic thermal expansion coefficient",
            "proposed_unit": "1/K",
            "current_project_unit_or_note": "18.9e-6 1/K",
            "conversion_needed": "no",
            "source_reference": "DEFAULT_ALPHA_T; COMSOL alpha_T = 18.9 ppm/K",
            "risk_note": "Do not change material constants in heat PDE phase 1.",
        },
        {
            "variable": "rho",
            "meaning": "mass density",
            "proposed_unit": "kg/m^3 in COMSOL reference",
            "current_project_unit_or_note": "Future transport constant only; mechanics uses kN/mm style quantities.",
            "conversion_needed": "yes",
            "source_reference": "THERMAL_REINTRODUCTION_PLAN.md",
            "risk_note": "Exact SI-to-project conversion is an implementation blocker.",
        },
        {
            "variable": "c",
            "meaning": "specific heat capacity",
            "proposed_unit": "J/kg/K in COMSOL reference",
            "current_project_unit_or_note": "Future transport constant only.",
            "conversion_needed": "yes",
            "source_reference": "THERMAL_REINTRODUCTION_PLAN.md",
            "risk_note": "Combines with rho and time units in transient residual.",
        },
        {
            "variable": "k0",
            "meaning": "constant thermal conductivity",
            "proposed_unit": "W/m/K in COMSOL reference",
            "current_project_unit_or_note": "Future transport constant only.",
            "conversion_needed": "yes",
            "source_reference": "THERMAL_REINTRODUCTION_PLAN.md",
            "risk_note": "Must be converted consistently with mm geometry and time convention.",
        },
        {
            "variable": "Q",
            "meaning": "volumetric heat source",
            "proposed_unit": "W/m^3 in SI if used",
            "current_project_unit_or_note": "Not implemented; start Q=0.",
            "conversion_needed": "yes for nonzero Q",
            "source_reference": "planned target equation",
            "risk_note": "Nonzero source is deferred to avoid residual scale ambiguity.",
        },
        {
            "variable": "time",
            "meaning": "thermal time coordinate",
            "proposed_unit": "s unless reviewer approves another convention",
            "current_project_unit_or_note": "Current mechanics uses displacement steps, not physical time.",
            "conversion_needed": "yes for transient coupling",
            "source_reference": "future transient heat PDE",
            "risk_note": "Mapping load steps to physical time is unresolved.",
        },
        {
            "variable": "x/y coordinates",
            "meaning": "physical coordinates",
            "proposed_unit": "mm internally",
            "current_project_unit_or_note": "Mesh converted from m to mm; NN input normalized to unit box by default.",
            "conversion_needed": "yes for thermal gradients using SI constants",
            "source_reference": "config.py model settings; PROJECT_MEMORY.md",
            "risk_note": "Need clear distinction between physical mm and normalized NN coordinates.",
        },
        {
            "variable": "grad T",
            "meaning": "temperature gradient",
            "proposed_unit": "K/mm internally or K/m before conversion",
            "current_project_unit_or_note": "Not implemented.",
            "conversion_needed": "yes",
            "source_reference": "planned heat residual",
            "risk_note": "Incorrect coordinate basis would scale conduction by powers of 1000.",
        },
        {
            "variable": "heat flux",
            "meaning": "-k0 * grad T",
            "proposed_unit": "converted project heat flux unit",
            "current_project_unit_or_note": "Not implemented.",
            "conversion_needed": "yes",
            "source_reference": "Fourier conduction relation",
            "risk_note": "Boundary flux tests must prove sign and magnitude.",
        },
        {
            "variable": "heat PDE residual",
            "meaning": "rho*c*dT/dt - div(k0*grad T) - Q",
            "proposed_unit": "converted heat source density unit",
            "current_project_unit_or_note": "Not implemented.",
            "conversion_needed": "yes",
            "source_reference": "planned target equation",
            "risk_note": "Residual normalization must be documented before training.",
        },
        {
            "variable": "thermal strain",
            "meaning": "alpha_T * delta_T normal strain",
            "proposed_unit": "dimensionless",
            "current_project_unit_or_note": "Implemented in prescribed branch.",
            "conversion_needed": "no",
            "source_reference": "apply_thermal_strain",
            "risk_note": "Must use solved T only after heat solve has passed patch tests.",
        },
        {
            "variable": "energy units if relevant",
            "meaning": "mechanical and thermal residual normalization scales",
            "proposed_unit": "project kN-mm mechanics plus converted thermal scales",
            "current_project_unit_or_note": "Mechanical route uses kN/mm material values and mm geometry.",
            "conversion_needed": "yes",
            "source_reference": "config.py mat_E_kN_per_mm2; Gc_kN_per_mm",
            "risk_note": "Do not mix SI heat energy with kN-mm losses without explicit scaling.",
        },
    ]


def implementation_phases_rows():
    return [
        {
            "phase_id": "P0",
            "phase_name": "Planning only",
            "source_files_expected_to_touch": "none for behavior; package builder and PROJECT_MEMORY.md only",
            "physics_added": "none",
            "physics_not_added": "heat PDE, solved temperature field, heat residual, k(d), new BC code",
            "required_tests_before_next_phase": "package schema, py_compile builder, git diff checks, no-thermal guard",
            "allowed_outputs": "REPORT.md, tables, manifest, handoff",
            "stop_conditions": "repo dirty outside allowed thermal paths or reviewer rejects scope",
        },
        {
            "phase_id": "P1",
            "phase_name": "Analytical prescribed temperature compatibility",
            "source_files_expected_to_touch": "thermal_prescribed.py; tests; docs",
            "physics_added": "none beyond prescribed analytical T/delta_T compatibility",
            "physics_not_added": "heat PDE residual and trainable T",
            "required_tests_before_next_phase": "default-off no-thermal regression; prescribed fallback regression",
            "allowed_outputs": "small focused tests and field summaries",
            "stop_conditions": "delta_T=0 no longer matches thermal_mode=off",
        },
        {
            "phase_id": "P2",
            "phase_name": "Constant-conductivity steady-state heat patch tests",
            "source_files_expected_to_touch": "new heat PDE module; config.py; focused tests",
            "physics_added": "steady constant-k0 heat residual with Q=0",
            "physics_not_added": "transient rho*c*dT/dt and k(d)",
            "required_tests_before_next_phase": "constant T residual; linear 1D Dirichlet conduction; insulated boundary check; unit sanity",
            "allowed_outputs": "heat patch-test tables only",
            "stop_conditions": "unit conversion convention unresolved or residual scale unstable",
        },
        {
            "phase_id": "P3",
            "phase_name": "Constant-conductivity transient heat patch tests",
            "source_files_expected_to_touch": "heat PDE module; config.py; tests",
            "physics_added": "rho*c*dT/dt term with constant k0",
            "physics_not_added": "damage-dependent conductivity and thermomechanical heat generation",
            "required_tests_before_next_phase": "uniform transient T remains constant; manufactured transient solution if feasible",
            "allowed_outputs": "transient patch-test summaries",
            "stop_conditions": "physical time convention or rho*c scaling not approved",
        },
        {
            "phase_id": "P4",
            "phase_name": "One-way solved T to mechanics thermal strain",
            "source_files_expected_to_touch": "compute_energy_mixed_tm.py; train_mixed_tm.py; history_field_mixed_tm.py; tests",
            "physics_added": "solved T routed through existing thermal strain relation",
            "physics_not_added": "mechanics-to-heat feedback and k(d)",
            "required_tests_before_next_phase": "free expansion; constrained heating stress sign/scale; no history change when drive is zero",
            "allowed_outputs": "mechanics patch tests under solved uniform T",
            "stop_conditions": "prescribed fallback regresses or history changes unexpectedly",
        },
        {
            "phase_id": "P5",
            "phase_name": "Checkpointed reaction diagnostics under solved T",
            "source_files_expected_to_touch": "postprocess_results.py; checkpoint payload support; docs",
            "physics_added": "solved temperature outputs and checkpoint metadata",
            "physics_not_added": "physical notch validation claims",
            "required_tests_before_next_phase": "reaction_N_energy availability; solved T stored/read consistently",
            "allowed_outputs": "small checkpoint diagnostics only",
            "stop_conditions": "reaction route unavailable or top-sigma becomes primary",
        },
        {
            "phase_id": "P6",
            "phase_name": "Damage-dependent conductivity planning review",
            "source_files_expected_to_touch": "planning docs only unless separately approved",
            "physics_added": "none at review time",
            "physics_not_added": "k(d) implementation",
            "required_tests_before_next_phase": "all constant-k0 heat and solved-T mechanics tests pass",
            "allowed_outputs": "risk plan and decision table",
            "stop_conditions": "constant-k0 branch not stable",
        },
        {
            "phase_id": "P7",
            "phase_name": "Damage-dependent conductivity implementation only after approval",
            "source_files_expected_to_touch": "heat PDE module; tests; docs",
            "physics_added": "k_d = g(d) * k0 if approved",
            "physics_not_added": "COMSOL line-by-line clone or comp4 branch",
            "required_tests_before_next_phase": "guarded k(d) tests and no-thermal regression",
            "allowed_outputs": "separate approved implementation package",
            "stop_conditions": "reviewer approval absent",
        },
    ]


def validation_matrix_rows():
    return [
        {
            "validation_id": "V01",
            "validation_name": "default-off no-thermal regression",
            "phase": "P1 and every later phase",
            "purpose": "Protect baseline behavior.",
            "expected_result": "thermal_mode=off follows current mechanics route.",
            "required_outputs": "pytest table or focused regression report",
            "pass_criteria": "No numerical drift beyond existing tolerance.",
            "failure_action": "Stop and restore default-off behavior before continuing.",
        },
        {
            "validation_id": "V02",
            "validation_name": "prescribed-temperature fallback regression",
            "phase": "P1 and every later phase",
            "purpose": "Protect reviewed prescribed-temperature branch.",
            "expected_result": "delta_T=0 matches off; +20 K keeps sign/scale expectations.",
            "required_outputs": "thermal strain patch-test results",
            "pass_criteria": "A/B equivalence and existing patch-test stress sign are preserved.",
            "failure_action": "Do not enable heat PDE.",
        },
        {
            "validation_id": "V03",
            "validation_name": "zero-gradient constant T heat residual",
            "phase": "P2",
            "purpose": "Validate residual baseline.",
            "expected_result": "Residual is zero or near numerical zero for constant T and Q=0.",
            "required_outputs": "element residual summary",
            "pass_criteria": "max residual below approved tolerance after unit scaling.",
            "failure_action": "Debug gradient basis and residual scaling.",
        },
        {
            "validation_id": "V04",
            "validation_name": "linear steady-state 1D conduction with Dirichlet boundaries",
            "phase": "P2",
            "purpose": "Validate spatial gradients and Dirichlet BC handling.",
            "expected_result": "T is linear between prescribed boundary temperatures.",
            "required_outputs": "temperature profile and residual table",
            "pass_criteria": "profile slope and residual match analytical solution within tolerance.",
            "failure_action": "Fix coordinate scaling or BC enforcement.",
        },
        {
            "validation_id": "V05",
            "validation_name": "transient uniform T with zero source remains constant",
            "phase": "P3",
            "purpose": "Validate time derivative and zero-flux behavior.",
            "expected_result": "Uniform T does not drift.",
            "required_outputs": "time-step temperature summary",
            "pass_criteria": "max |T(t)-T0| below tolerance.",
            "failure_action": "Stop transient work and inspect time scaling.",
        },
        {
            "validation_id": "V06",
            "validation_name": "transient manufactured solution if feasible",
            "phase": "P3",
            "purpose": "Validate rho*c and k0 scaling with a known source or decay.",
            "expected_result": "Residual and solution follow manufactured reference.",
            "required_outputs": "MMS residual and error table",
            "pass_criteria": "error below reviewer-approved tolerance.",
            "failure_action": "Do not couple solved T to mechanics.",
        },
        {
            "validation_id": "V07",
            "validation_name": "insulated boundary zero-flux check",
            "phase": "P2 or P3",
            "purpose": "Validate Neumann sign and normal direction.",
            "expected_result": "normal heat flux is zero on insulated edges.",
            "required_outputs": "boundary flux table",
            "pass_criteria": "net and local flux below tolerance.",
            "failure_action": "Fix boundary normal handling.",
        },
        {
            "validation_id": "V08",
            "validation_name": "unit-conversion sanity check",
            "phase": "P2 gate",
            "purpose": "Prevent SI/project unit mixing.",
            "expected_result": "rho, c, k0, Q, length, and time conversions are explicit.",
            "required_outputs": "unit conversion table and checked residual dimensions",
            "pass_criteria": "reviewer approves conversion convention before code uses constants.",
            "failure_action": "Block implementation.",
        },
        {
            "validation_id": "V09",
            "validation_name": "free thermal expansion under solved uniform T",
            "phase": "P4",
            "purpose": "Validate solved-T mechanics coupling.",
            "expected_result": "Near-zero elastic strain/stress for compatible free expansion.",
            "required_outputs": "mechanics field summary",
            "pass_criteria": "stress and energy near zero within existing patch tolerance.",
            "failure_action": "Fix T to delta_T routing.",
        },
        {
            "validation_id": "V10",
            "validation_name": "constrained heating stress sign/scale under solved uniform T",
            "phase": "P4",
            "purpose": "Protect mechanical constitutive convention.",
            "expected_result": "Compressive normal stress with current project sign and scale.",
            "required_outputs": "stress patch table",
            "pass_criteria": "Matches prescribed-branch expectation after replacing prescribed T with solved uniform T.",
            "failure_action": "Stop before diagnostics.",
        },
        {
            "validation_id": "V11",
            "validation_name": "checkpointed energy reaction availability with solved T",
            "phase": "P5",
            "purpose": "Preserve primary reaction route.",
            "expected_result": "reaction_N_energy remains available.",
            "required_outputs": "reaction_metric_availability.csv and stress_strain_by_step.csv",
            "pass_criteria": "energy_conjugate status true for every checkpointed step.",
            "failure_action": "Do not use legacy top-sigma as primary.",
        },
        {
            "validation_id": "V12",
            "validation_name": "no damage-conductivity guard",
            "phase": "P2 through P5",
            "purpose": "Ensure k(d) remains deferred.",
            "expected_result": "No active k_d = g(d) * k0 code path or output.",
            "required_outputs": "guard scan table",
            "pass_criteria": "Guard passes before every constant-k0 report.",
            "failure_action": "Remove or disable k(d) until separately approved.",
        },
    ]


def patch_test_plan_rows():
    return [
        {
            "patch_test_id": "PT01",
            "name": "constant T residual",
            "geometry_or_field": "full current square mesh or simple rectangular patch",
            "boundary_condition": "no-flux or compatible Dirichlet constant T",
            "initial_condition": "uniform T0",
            "source_term": "Q=0",
            "expected_temperature_solution": "T remains constant and grad T = 0",
            "expected_mechanical_response": "none if heat residual only; if coupled, delta_T follows uniform value",
            "tolerance": "reviewer-approved residual tolerance after unit scaling",
            "required_artifacts": "residual summary table",
            "notes": "First thermal residual smoke test.",
        },
        {
            "patch_test_id": "PT02",
            "name": "1D linear conduction",
            "geometry_or_field": "rectangular strip in x or y",
            "boundary_condition": "Dirichlet T_left and T_right; insulated remaining edges",
            "initial_condition": "optional analytical linear T",
            "source_term": "Q=0",
            "expected_temperature_solution": "linear T profile",
            "expected_mechanical_response": "not evaluated until coupling phase",
            "tolerance": "profile error and residual below approved thresholds",
            "required_artifacts": "temperature profile CSV and flux balance table",
            "notes": "Validates gradient basis, BCs, and k0 sign.",
        },
        {
            "patch_test_id": "PT03",
            "name": "insulated boundary zero flux",
            "geometry_or_field": "constant or compatible field on current mesh",
            "boundary_condition": "Neumann q_n=0",
            "initial_condition": "uniform or linear field consistent with insulated edges",
            "source_term": "Q=0",
            "expected_temperature_solution": "no normal flux through insulated boundary",
            "expected_mechanical_response": "none",
            "tolerance": "net boundary flux near zero",
            "required_artifacts": "boundary normal flux table",
            "notes": "Catches normal orientation and unit mistakes.",
        },
        {
            "patch_test_id": "PT04",
            "name": "uniform transient no source",
            "geometry_or_field": "current mesh",
            "boundary_condition": "insulated",
            "initial_condition": "uniform T0",
            "source_term": "Q=0",
            "expected_temperature_solution": "T(t)=T0 for all thermal steps",
            "expected_mechanical_response": "no drift if coupled with delta_T=0",
            "tolerance": "max temperature drift below tolerance",
            "required_artifacts": "transient temperature table",
            "notes": "Requires approved time convention.",
        },
        {
            "patch_test_id": "PT05",
            "name": "free expansion with solved uniform T",
            "geometry_or_field": "mechanics patch with compatible displacement",
            "boundary_condition": "mechanically free expansion; solved uniform T",
            "initial_condition": "uniform T = Tref + DeltaT",
            "source_term": "Q=0 or prescribed uniform solved T surrogate",
            "expected_temperature_solution": "uniform solved T",
            "expected_mechanical_response": "near-zero elastic stress and energy",
            "tolerance": "same scale as prescribed thermal strain patch test",
            "required_artifacts": "stress/energy patch table",
            "notes": "Only after heat residual tests pass.",
        },
        {
            "patch_test_id": "PT06",
            "name": "constrained uniform heating with solved T",
            "geometry_or_field": "mechanics patch with zero displacement normal strain",
            "boundary_condition": "mechanically constrained; solved uniform T",
            "initial_condition": "uniform T = Tref + DeltaT",
            "source_term": "Q=0 or prescribed uniform solved T surrogate",
            "expected_temperature_solution": "uniform solved T",
            "expected_mechanical_response": "compressive normal stress under current project convention",
            "tolerance": "matches prescribed branch sign/scale tolerance",
            "required_artifacts": "stress field table",
            "notes": "Protects current constitutive convention.",
        },
    ]


def boundary_initial_condition_rows():
    return [
        {
            "bc_or_ic": "prescribed Dirichlet T",
            "first_phase_support": "yes for steady patch tests after unit gate",
            "later_phase_support": "time-dependent prescribed temperature",
            "mathematical_form": "T = T_D on selected boundary",
            "implementation_note": "Start with constant boundary values and explicit boundary tags.",
            "validation_test": "linear steady-state 1D conduction",
            "risk": "Wrong boundary group mapping can mimic a physics error.",
        },
        {
            "bc_or_ic": "Neumann heat flux",
            "first_phase_support": "minimal zero or constant flux only after Dirichlet tests",
            "later_phase_support": "spatial/time-dependent flux",
            "mathematical_form": "-k0 * grad(T) dot n = q_n",
            "implementation_note": "Define sign convention in table before implementation.",
            "validation_test": "insulated boundary zero-flux check",
            "risk": "Normal direction and unit scaling are high risk.",
        },
        {
            "bc_or_ic": "insulated boundary",
            "first_phase_support": "yes as q_n=0",
            "later_phase_support": "mixed boundary sets",
            "mathematical_form": "grad(T) dot n = 0",
            "implementation_note": "Use as default natural condition only if documented.",
            "validation_test": "zero-flux check and uniform transient no-source test",
            "risk": "Implicit natural BC can hide missing boundary tags.",
        },
        {
            "bc_or_ic": "uniform initial T",
            "first_phase_support": "yes for transient after time gate",
            "later_phase_support": "case-specific T0 fields",
            "mathematical_form": "T(x,y,0)=T0",
            "implementation_note": "Use T0 = 273.15 K or documented COMSOL T0 mapping as approved.",
            "validation_test": "transient uniform T remains constant",
            "risk": "T0 degC versus absolute K confusion.",
        },
        {
            "bc_or_ic": "linear initial T",
            "first_phase_support": "patch-test only",
            "later_phase_support": "manufactured solution initialization",
            "mathematical_form": "T(x,y,0)=a+b*x or a+b*y",
            "implementation_note": "Keep analytical and not used for physical notch runs initially.",
            "validation_test": "linear conduction or transient MMS",
            "risk": "Coordinate normalization could scale the gradient incorrectly.",
        },
        {
            "bc_or_ic": "time-dependent boundary temperature",
            "first_phase_support": "deferred",
            "later_phase_support": "yes after transient heat tests",
            "mathematical_form": "T_D(t)",
            "implementation_note": "Requires physical time/load-step mapping.",
            "validation_test": "transient manufactured or ramp response test",
            "risk": "Ambiguous time scale would make diagnostics uninterpretable.",
        },
        {
            "bc_or_ic": "heat source Q",
            "first_phase_support": "Q=0 only",
            "later_phase_support": "constant or manufactured source after conversion approval",
            "mathematical_form": "Q(x,y,t)",
            "implementation_note": "Do not include thermomechanical heat generation in first phase.",
            "validation_test": "manufactured solution with known source",
            "risk": "Wrong units can dominate residual optimization.",
        },
    ]


def coupling_dependency_rows():
    return [
        {
            "coupling": "T to thermal strain",
            "current_status": "prescribed T/delta_T to thermal strain implemented",
            "planned_first_status": "reuse relation with solved T only after heat patch tests",
            "dependency": "validated solved T representation and unit convention",
            "risk": "Can regress prescribed branch if routing is shared carelessly.",
            "validation_before_enablement": "free expansion and constrained heating under solved uniform T",
        },
        {
            "coupling": "mechanics to heat",
            "current_status": "not implemented",
            "planned_first_status": "deferred",
            "dependency": "constant-k0 heat and one-way solved-T mechanics stability",
            "risk": "Thermomechanical source terms introduce new units and feedback.",
            "validation_before_enablement": "separate heat generation plan and conservation tests",
        },
        {
            "coupling": "damage to conductivity",
            "current_status": "not implemented",
            "planned_first_status": "explicitly disabled",
            "dependency": "stable constant-k0 heat PDE and reviewer approval",
            "risk": "Conflates damage evolution with thermal transport implementation defects.",
            "validation_before_enablement": "no damage-conductivity guard and later k(d) decision gate",
        },
        {
            "coupling": "heat to phase-field/history",
            "current_status": "only indirect through thermal strain mechanics when prescribed",
            "planned_first_status": "no direct coupling",
            "dependency": "unchanged history logic",
            "risk": "Direct heat-history coupling could violate irreversibility assumptions.",
            "validation_before_enablement": "history patch tests showing no false drive",
        },
        {
            "coupling": "solved T to reaction diagnostics",
            "current_status": "not implemented",
            "planned_first_status": "record solved T metadata while preserving reaction_N_energy",
            "dependency": "checkpoint schema update",
            "risk": "Reaction route could become unavailable if checkpoint payload is incomplete.",
            "validation_before_enablement": "reaction_metric_availability energy_conjugate for all steps",
        },
        {
            "coupling": "checkpointing solved T",
            "current_status": "not implemented",
            "planned_first_status": "future checkpoint metadata only after P2/P3 tests",
            "dependency": "T representation strategy",
            "risk": "Large or incompatible checkpoints could break postprocess.",
            "validation_before_enablement": "load/save smoke test for T field and thermal metadata",
        },
        {
            "coupling": "postprocessing solved T",
            "current_status": "not implemented",
            "planned_first_status": "read and report T, grad T, residual, and flux summaries",
            "dependency": "checkpoint/export format",
            "risk": "Plots can be mistaken for physical validation before patch tests pass.",
            "validation_before_enablement": "figure/table labels must state diagnostic scope",
        },
    ]


def deferred_features_rows():
    return [
        {
            "feature": "damage-dependent conductivity k(d)=g(d)k0",
            "reason_deferred": "Constant-conductivity heat PDE is not yet implemented or validated.",
            "prerequisite": "P2 through P5 pass and reviewer approves k(d) plan.",
            "risk_if_implemented_now": "Conflates heat PDE errors with damage coupling.",
            "future_decision_gate": "P6 damage-conductivity planning review.",
        },
        {
            "feature": "thermomechanical heat generation",
            "reason_deferred": "Requires mechanics-to-heat source law and units.",
            "prerequisite": "Stable heat PDE and agreed energy conversion.",
            "risk_if_implemented_now": "Unbounded or mis-scaled source dominates heat residual.",
            "future_decision_gate": "Separate source-term validation plan.",
        },
        {
            "feature": "crack-surface heat transfer",
            "reason_deferred": "Requires crack geometry/interface treatment not present in first heat phase.",
            "prerequisite": "Validated boundary flux and damage field interpretation.",
            "risk_if_implemented_now": "Boundary/source ambiguity at crack surfaces.",
            "future_decision_gate": "Crack heat-transfer scope review.",
        },
        {
            "feature": "temperature-dependent material properties",
            "reason_deferred": "Would change mechanics material constants.",
            "prerequisite": "Validated solved T and material law review.",
            "risk_if_implemented_now": "Violates first-phase no material change invariant.",
            "future_decision_gate": "Material-property sensitivity plan.",
        },
        {
            "feature": "full COMSOL line-by-line matching",
            "reason_deferred": "Project memory allows platform differences when physical meaning is preserved.",
            "prerequisite": "Conceptual comp3 validation case selected.",
            "risk_if_implemented_now": "Overfits COMSOL implementation details before local unit/BC tests pass.",
            "future_decision_gate": "Benchmark comparison plan.",
        },
        {
            "feature": "multi-pore comp4 branch",
            "reason_deferred": "Out of current single-notch comp3 scope.",
            "prerequisite": "Separate reviewer approval.",
            "risk_if_implemented_now": "Mixes unrelated geometry and physics branches.",
            "future_decision_gate": "Comp4 scope decision.",
        },
        {
            "feature": "D0040",
            "reason_deferred": "Not part of heat PDE planning and expensive diagnostic.",
            "prerequisite": "Heat PDE branch passes patch tests and reviewer requests physical diagnostic.",
            "risk_if_implemented_now": "Runtime without resolving implementation blockers.",
            "future_decision_gate": "Post-implementation diagnostic review.",
        },
        {
            "feature": "seed study",
            "reason_deferred": "Current task is planning only.",
            "prerequisite": "Stable heat PDE implementation or specific robustness question.",
            "risk_if_implemented_now": "Consumes runtime before heat PDE exists.",
            "future_decision_gate": "Robustness evidence request.",
        },
        {
            "feature": "shear thermal study",
            "reason_deferred": "Tension prescribed branch is the current evidence base; heat PDE not implemented.",
            "prerequisite": "Thermal transport and solved-T mechanics validated in core route.",
            "risk_if_implemented_now": "Expands mechanics scope while thermal basics are unresolved.",
            "future_decision_gate": "Shear thermal extension plan.",
        },
    ]


def risk_register_rows():
    return [
        {
            "risk_id": "R01",
            "risk": "SI-to-project unit conversion risk",
            "severity": "high",
            "likelihood": "high",
            "evidence_or_context": "Geometry/mechanics use mm and kN/mm style quantities; rho, c, and k0 are COMSOL SI constants.",
            "mitigation": "Create and approve a unit-conversion table before any heat residual uses constants.",
            "decision_required": "Approve exact length/time/energy conversion convention.",
        },
        {
            "risk_id": "R02",
            "risk": "heat PDE residual scaling risk",
            "severity": "high",
            "likelihood": "medium",
            "evidence_or_context": "PINN loss scale will add thermal residual to existing energy route.",
            "mitigation": "Patch-test residual magnitudes before training diagnostics.",
            "decision_required": "Choose residual normalization and weighting.",
        },
        {
            "risk_id": "R03",
            "risk": "PINN optimization instability from adding T field",
            "severity": "high",
            "likelihood": "medium",
            "evidence_or_context": "Current network solves u, v, alpha only.",
            "mitigation": "Stage analytical/prescribed compatibility before solved T and use small patch tests.",
            "decision_required": "Choose T representation strategy.",
        },
        {
            "risk_id": "R04",
            "risk": "conflating heat PDE with damage conductivity",
            "severity": "high",
            "likelihood": "medium",
            "evidence_or_context": "COMSOL eventually uses k_d, but local heat PDE is absent.",
            "mitigation": "Keep k(d)=g(d)k0 deferred and guard constant-k0 branch.",
            "decision_required": "Confirm damage conductivity remains deferred.",
        },
        {
            "risk_id": "R05",
            "risk": "preserving no-thermal and prescribed-temperature regressions",
            "severity": "high",
            "likelihood": "medium",
            "evidence_or_context": "Existing evidence depends on default-off and delta_T=0 equivalence.",
            "mitigation": "Run focused regressions at every phase gate.",
            "decision_required": "Approve these tests as mandatory gates.",
        },
        {
            "risk_id": "R06",
            "risk": "boundary condition ambiguity",
            "severity": "medium",
            "likelihood": "medium",
            "evidence_or_context": "Thermal BCs are not implemented and COMSOL boundary tags are not mapped in code.",
            "mitigation": "Start with simple Dirichlet/insulated patch tests and explicit tags.",
            "decision_required": "Choose first supported BC set.",
        },
        {
            "risk_id": "R07",
            "risk": "transient time-scale ambiguity",
            "severity": "high",
            "likelihood": "medium",
            "evidence_or_context": "Current load schedule is displacement steps, not physical time.",
            "mitigation": "Do steady-state first or define time convention before transient.",
            "decision_required": "Choose steady-state first versus transient first.",
        },
        {
            "risk_id": "R08",
            "risk": "output/checkpoint compatibility risk",
            "severity": "medium",
            "likelihood": "medium",
            "evidence_or_context": "Postprocess currently reads mechanics checkpoints and recomputes prescribed thermal strain.",
            "mitigation": "Version checkpoint payloads and add read/write smoke tests for T.",
            "decision_required": "Approve checkpoint schema extension.",
        },
        {
            "risk_id": "R09",
            "risk": "low-level alpha background misinterpretation risk",
            "severity": "medium",
            "likelihood": "medium",
            "evidence_or_context": "Case C audit found broad low-level alpha background that is diagnostic-only.",
            "mitigation": "Use high-threshold/notch metrics and label diagnostics clearly.",
            "decision_required": "Confirm no physical fracture claim from heat PDE patch tests.",
        },
        {
            "risk_id": "R10",
            "risk": "COMSOL platform-convention mismatch",
            "severity": "medium",
            "likelihood": "medium",
            "evidence_or_context": "Project memory permits conceptual comp3 alignment, not line-by-line matching.",
            "mitigation": "Document non-alignment topics before benchmark comparison.",
            "decision_required": "Approve conceptual comparison criteria.",
        },
    ]


def comsol_alignment_rows():
    return [
        {
            "topic": "alpha_T",
            "COMSOL_comp3_reference": "18.9 ppm/K",
            "proposed_PINN_plan": "Keep alpha_T = 18.9e-6 1/K.",
            "exact_match_required": "yes for constant value",
            "caveat": "Do not change material constants in heat PDE phase 1.",
        },
        {
            "topic": "rho/c/k0",
            "COMSOL_comp3_reference": "rho=1040 kg/m^3; c=170 J/kg/K; k0=418 W/m/K",
            "proposed_PINN_plan": "Use only after explicit SI-to-project unit conversion approval.",
            "exact_match_required": "conceptual constants yes; internal unit values need conversion",
            "caveat": "This is the main implementation blocker.",
        },
        {
            "topic": "Tref/T0",
            "COMSOL_comp3_reference": "Tref=273.15 K; T0=0 degC",
            "proposed_PINN_plan": "Preserve Tref=273.15 K and document T0 mapping before transient ICs.",
            "exact_match_required": "yes for Tref; T0 mapping must be explicit",
            "caveat": "Avoid degC versus K confusion.",
        },
        {
            "topic": "heat PDE",
            "COMSOL_comp3_reference": "ht3 heat transfer branch",
            "proposed_PINN_plan": "Start with rho*c*dT/dt - div(k0*grad T)=Q, Q=0, constant k0.",
            "exact_match_required": "no line-by-line match",
            "caveat": "Patch tests precede any notch diagnostic.",
        },
        {
            "topic": "k_d",
            "COMSOL_comp3_reference": "k_d = g(d) * k0 eventually",
            "proposed_PINN_plan": "Explicitly defer until constant-k0 branch is stable.",
            "exact_match_required": "not in first phase",
            "caveat": "Guard against accidental k(d) activation.",
        },
        {
            "topic": "thermal strain",
            "COMSOL_comp3_reference": "Thermal expansion coupled to solid mechanics",
            "proposed_PINN_plan": "Use existing relation delta_T = T-Tref; subtract alpha_T*delta_T from normal strains.",
            "exact_match_required": "physical relation yes",
            "caveat": "Current project constitutive convention is not forced into a COMSOL line clone.",
        },
        {
            "topic": "plane stress/platform difference",
            "COMSOL_comp3_reference": "COMSOL platform conventions may differ.",
            "proposed_PINN_plan": "Keep current TM source split/stress convention unless separately reviewed.",
            "exact_match_required": "no",
            "caveat": "Document before any quantitative COMSOL comparison.",
        },
        {
            "topic": "phase-field/history coupling",
            "COMSOL_comp3_reference": "phase-field PDE c and history state3",
            "proposed_PINN_plan": "Keep AT2 TM history route unchanged in first heat PDE phases.",
            "exact_match_required": "conceptual only",
            "caveat": "Do not alter history logic while adding heat residual.",
        },
        {
            "topic": "comp4 ignored",
            "COMSOL_comp3_reference": "comp4/multi-pore branch excluded",
            "proposed_PINN_plan": "Ignore comp4, solid2, ht2, c2, state4, TFinal, and multi-pore settings.",
            "exact_match_required": "yes for exclusion",
            "caveat": "Only single-notch comp3 scope is relevant.",
        },
    ]


def source_touch_plan_rows():
    return [
        {
            "source_file": "thermal_prescribed.py",
            "future_touch_expected": "possible",
            "reason": "May remain as fallback and shared thermal-strain helper.",
            "first_phase_change_type": "none in planning; later compatibility helper only if needed",
            "tests_required": "prescribed-temperature fallback regression",
            "risk": "Breaking existing delta_T semantics.",
        },
        {
            "source_file": "config.py",
            "future_touch_expected": "yes",
            "reason": "Future heat mode flags, constants, BC/IC options, and unit convention.",
            "first_phase_change_type": "add options only after unit gate",
            "tests_required": "CLI defaults and no-thermal regression",
            "risk": "Changing defaults could activate thermal behavior accidentally.",
        },
        {
            "source_file": "train_mixed_tm.py",
            "future_touch_expected": "yes",
            "reason": "Future heat residual and T representation integration.",
            "first_phase_change_type": "constant-k0 residual plumbing after patch tests",
            "tests_required": "heat residual patch tests and mechanics loss regression",
            "risk": "Loss scale or optimization instability.",
        },
        {
            "source_file": "compute_energy_mixed_tm.py",
            "future_touch_expected": "yes",
            "reason": "Solved T to thermal strain and possibly energy summaries.",
            "first_phase_change_type": "read solved T only after heat tests",
            "tests_required": "free expansion and constrained heating with solved T",
            "risk": "History/energy route regression.",
        },
        {
            "source_file": "history_field_mixed_tm.py",
            "future_touch_expected": "yes",
            "reason": "Checkpointed field summaries may need solved T and residual outputs.",
            "first_phase_change_type": "metadata/export extension only after schema plan",
            "tests_required": "field save/load smoke and history invariance patch test",
            "risk": "Checkpoint compatibility.",
        },
        {
            "source_file": "postprocess_results.py",
            "future_touch_expected": "yes",
            "reason": "Report solved T while preserving reaction_N_energy.",
            "first_phase_change_type": "read T metadata after checkpoint schema update",
            "tests_required": "reaction availability and T field readback",
            "risk": "Reaction route could become unavailable.",
        },
        {
            "source_file": "new heat PDE module if recommended",
            "future_touch_expected": "yes",
            "reason": "Keep heat residual, BC, IC, and unit utilities isolated.",
            "first_phase_change_type": "new module for constant k0 residual",
            "tests_required": "constant T, linear conduction, zero flux, unit sanity",
            "risk": "New abstraction may duplicate existing gradient logic unless scoped carefully.",
        },
        {
            "source_file": "tests",
            "future_touch_expected": "yes",
            "reason": "Patch tests are the main gate.",
            "first_phase_change_type": "focused heat PDE and regression tests",
            "tests_required": "all validation matrix rows relevant to phase",
            "risk": "Skipping tests would make coupled diagnostics unreviewable.",
        },
        {
            "source_file": "docs/project memory",
            "future_touch_expected": "yes",
            "reason": "Record gates, approvals, and finalization protocol.",
            "first_phase_change_type": "planning memory update only in this task",
            "tests_required": "review for scope boundaries",
            "risk": "Future tasks may forget deferred features.",
        },
        {
            "source_file": "load schedules",
            "future_touch_expected": "later maybe",
            "reason": "Thermal transient schedules may need time columns.",
            "first_phase_change_type": "none",
            "tests_required": "time convention review before schedule changes",
            "risk": "Conflating displacement increments with physical time.",
        },
    ]


def next_decision_gate_rows():
    return [
        {
            "decision": "approve heat PDE implementation phase 1 or not",
            "prerequisite_evidence": "Reviewer reads this package and accepts constant-conductivity scope.",
            "current_status": "ready for review; no implementation performed",
            "recommendation": "Approve only Phase 1/P2 constant-k0 patch-test implementation after unit gate.",
            "blocking_risk": "Unit conversion unresolved.",
            "next_task_if_approved": "Implement only constant-conductivity heat PDE utilities and patch tests; do not add k(d).",
            "next_task_if_not_approved": "Keep prescribed-temperature branch frozen and refine open questions.",
        },
        {
            "decision": "choose steady-state first versus transient first",
            "prerequisite_evidence": "Boundary and unit plan reviewed.",
            "current_status": "not decided",
            "recommendation": "Start steady-state first.",
            "blocking_risk": "Transient time-scale ambiguity.",
            "next_task_if_approved": "Create steady constant-k0 residual and Dirichlet/insulated patch tests.",
            "next_task_if_not_approved": "Define physical time convention before any heat residual work.",
        },
        {
            "decision": "choose solved temperature representation strategy",
            "prerequisite_evidence": "FieldComputation/network design reviewed.",
            "current_status": "not decided",
            "recommendation": "Prefer isolated design review before adding T as a trainable output.",
            "blocking_risk": "Optimization instability and checkpoint schema changes.",
            "next_task_if_approved": "Prototype representation in patch-test branch only.",
            "next_task_if_not_approved": "Use analytical T fields while heat residual plumbing is reviewed.",
        },
        {
            "decision": "approve unit conversion convention",
            "prerequisite_evidence": "Complete dimensional table for rho, c, k0, Q, length, time, and residual scale.",
            "current_status": "blocked until specified",
            "recommendation": "Make this the first implementation gate.",
            "blocking_risk": "SI constants cannot be mixed directly with mm/kN mechanics quantities.",
            "next_task_if_approved": "Encode constants in approved internal units and test residual dimensions.",
            "next_task_if_not_approved": "Do not implement heat PDE.",
        },
        {
            "decision": "keep damage-dependent conductivity deferred",
            "prerequisite_evidence": "Constant-k0 heat PDE not yet implemented.",
            "current_status": "deferred",
            "recommendation": "Keep deferred.",
            "blocking_risk": "k(d) would obscure heat PDE defects.",
            "next_task_if_approved": "Add guard tests proving k(d) inactive in constant-k0 phases.",
            "next_task_if_not_approved": "Write a separate risk plan before any k(d) code.",
        },
        {
            "decision": "keep prescribed-temperature fallback frozen",
            "prerequisite_evidence": "Existing patch tests and diagnostics support it as baseline.",
            "current_status": "recommended frozen",
            "recommendation": "Preserve thermal_mode=off default and prescribed branch.",
            "blocking_risk": "Fallback regression would remove comparison baseline.",
            "next_task_if_approved": "Run fallback regressions at every heat PDE phase.",
            "next_task_if_not_approved": "Re-review prescribed-temperature evidence before heat work.",
        },
        {
            "decision": "decide whether any physical validation case is needed before implementation",
            "prerequisite_evidence": "Reviewer decides whether conceptual planning is enough.",
            "current_status": "not required by this package",
            "recommendation": "Do not run physical validation before patch-test implementation.",
            "blocking_risk": "No heat PDE exists, so physical heat comparison cannot be meaningful yet.",
            "next_task_if_approved": "Select a small benchmark only after patch tests pass.",
            "next_task_if_not_approved": "Proceed to implementation gate and patch tests.",
        },
    ]


def changed_files_summary_rows():
    return [
        {
            "path": f"{PACKAGE_REL}/build_heat_pde_plan_package.py",
            "change_type": "added",
            "purpose": "Rebuild planning-only package artifacts.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": f"{PACKAGE_REL}/REPORT.md",
            "change_type": "added",
            "purpose": "Reviewer-readable heat PDE implementation and validation plan.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": f"{PACKAGE_REL}/HANDOFF_COMMENT.md",
            "change_type": "added",
            "purpose": "Continuation handoff with scope, validation, and finalization status.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": f"{PACKAGE_REL}/MANIFEST.json",
            "change_type": "added",
            "purpose": "Package manifest and schema inventory.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": f"{PACKAGE_REL}/tables/*.csv",
            "change_type": "added",
            "purpose": "Required planning tables for scope, units, phases, validation, risks, COMSOL alignment, source touch plan, and decision gates.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": f"{PACKAGE_REL}/figures/figure_summary.md",
            "change_type": "added",
            "purpose": "Documents that no PNG figures were required for this planning package.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
        {
            "path": MEMORY_REL,
            "change_type": "modified",
            "purpose": "Record heat PDE planning package, constant-k0 recommendation, unit gate, deferred k(d), and finalization protocol.",
            "behavior_source_modified": "false",
            "no_thermal_project_touched": "false",
        },
    ]


TABLES = {
    "heat_pde_scope_summary.csv": (
        ["item", "proposed_status", "first_phase_behavior", "deferred_behavior", "reason"],
        heat_pde_scope_summary_rows,
    ),
    "thermal_variables_units.csv": (
        ["variable", "meaning", "proposed_unit", "current_project_unit_or_note", "conversion_needed", "source_reference", "risk_note"],
        thermal_variables_units_rows,
    ),
    "implementation_phases.csv": (
        ["phase_id", "phase_name", "source_files_expected_to_touch", "physics_added", "physics_not_added", "required_tests_before_next_phase", "allowed_outputs", "stop_conditions"],
        implementation_phases_rows,
    ),
    "validation_matrix.csv": (
        ["validation_id", "validation_name", "phase", "purpose", "expected_result", "required_outputs", "pass_criteria", "failure_action"],
        validation_matrix_rows,
    ),
    "patch_test_plan.csv": (
        ["patch_test_id", "name", "geometry_or_field", "boundary_condition", "initial_condition", "source_term", "expected_temperature_solution", "expected_mechanical_response", "tolerance", "required_artifacts", "notes"],
        patch_test_plan_rows,
    ),
    "boundary_initial_condition_plan.csv": (
        ["bc_or_ic", "first_phase_support", "later_phase_support", "mathematical_form", "implementation_note", "validation_test", "risk"],
        boundary_initial_condition_rows,
    ),
    "coupling_dependency_plan.csv": (
        ["coupling", "current_status", "planned_first_status", "dependency", "risk", "validation_before_enablement"],
        coupling_dependency_rows,
    ),
    "deferred_features.csv": (
        ["feature", "reason_deferred", "prerequisite", "risk_if_implemented_now", "future_decision_gate"],
        deferred_features_rows,
    ),
    "risk_register.csv": (
        ["risk_id", "risk", "severity", "likelihood", "evidence_or_context", "mitigation", "decision_required"],
        risk_register_rows,
    ),
    "comsol_alignment_notes.csv": (
        ["topic", "COMSOL_comp3_reference", "proposed_PINN_plan", "exact_match_required", "caveat"],
        comsol_alignment_rows,
    ),
    "source_touch_plan.csv": (
        ["source_file", "future_touch_expected", "reason", "first_phase_change_type", "tests_required", "risk"],
        source_touch_plan_rows,
    ),
    "next_decision_gate.csv": (
        ["decision", "prerequisite_evidence", "current_status", "recommendation", "blocking_risk", "next_task_if_approved", "next_task_if_not_approved"],
        next_decision_gate_rows,
    ),
    "changed_files_summary.csv": (
        ["path", "change_type", "purpose", "behavior_source_modified", "no_thermal_project_touched"],
        changed_files_summary_rows,
    ),
}


def report_text():
    return f"""# Heat PDE Implementation and Validation Plan

## 1. Purpose

This planning package defines the next safe implementation path for adding a heat PDE branch to `examples/TM_comsol_thermal_micro` after the prescribed-temperature thermal-strain stage. It is documentation and validation planning only. It does not implement a heat equation, a trainable or PDE temperature field, damage-dependent conductivity, new boundary-condition code, new losses, or any training run.

## 2. Scope boundaries

Work is scoped to the thermal subproject. The original `examples/TM_comsol_no_thermal_micro` baseline remains frozen and untouched. Existing mechanics behavior, material constants, `l0`, TM split/history logic, phase-field route, reaction route, boundary conditions, and training losses are not changed by this package.

Allowed future work is staged heat PDE implementation after reviewer approval. Disallowed in the first implementation phase are `k(d)=g(d)k0`, thermomechanical heat generation, crack-surface heat transfer, COMSOL line-by-line matching, D0040, seed studies, shear thermal studies, S0110, and multi-pore or `comp4` work.

## 3. Current prescribed-temperature baseline status

The current reviewed thermal branch is prescribed-temperature mechanics only. It keeps `thermal_mode=off` as the default, supports prescribed absolute temperature or prescribed `delta_T`, and applies thermal strain before the existing TM split/history/energy route.

The trusted mechanics relation to preserve is:

```text
delta_T = T - Tref
exx_e = exx - alpha_T * delta_T
eyy_e = eyy - alpha_T * delta_T
exy_e = exy
```

Patch tests and diagnostics show that `thermal_mode=uniform` with `delta_T=0` reproduces the no-thermal thermal-subproject route in completed checks. Prescribed `+20 K` shifts displacement-controlled tension reaction/stress downward and reduces high-threshold/notch alpha in the moderate diagnostic. The broad low-level Case C alpha background remains diagnostic-only and is not physical fracture evidence.

## 4. Heat PDE target equation

The planned heat-transfer equation is:

```text
{TARGET_EQUATION}
```

The recommended first heat PDE implementation uses constant conductivity `k0` and starts with `Q=0`. It should not use damage-dependent conductivity. A steady-state patch-test route may use the corresponding constant-k0 steady residual before any transient term is enabled.

## 5. Unit system and conversion blockers

The main implementation blocker is a precise SI-to-project unit conversion convention. The current project geometry and mechanics use mm and kN/mm style quantities, and neural-network coordinates may be normalized to a unit box. The COMSOL transport constants are SI values: `rho = 1040 kg/m^3`, `c = 170 J/kg/K`, and `k0 = 418 W/m/K`.

This plan does not hand-wave that conversion. Before heat PDE implementation, a reviewer-approved table must define internal units for length, time, thermal energy or power, `rho*c`, `k0`, `Q`, heat flux, and the residual scale. Without that decision, the heat PDE implementation should remain blocked.

See `tables/thermal_variables_units.csv` for the unit and conversion inventory.

## 6. Temperature variable and representation options

The future temperature variable can be represented as a separate trainable output, a separate network, a deterministic analytical field for patch tests, or a staged hybrid. This package does not choose a trainable representation. It recommends using analytical or fixed fields for compatibility checks first, then adding a solved temperature representation only after unit conversion and heat residual patch tests are approved.

Any solved T representation must preserve the existing prescribed-temperature fallback and must be checkpointed and postprocessed without breaking `reaction_N_energy`.

## 7. Boundary and initial condition plan

First supported thermal boundary conditions should be minimal: prescribed Dirichlet temperature and insulated or constant Neumann heat flux, with explicit sign and unit conventions. First initial conditions should be uniform T, with linear initial T reserved for controlled patch tests. Time-dependent boundary temperature and nonzero `Q` are later-phase items.

See `tables/boundary_initial_condition_plan.csv` for the BC/IC staging plan.

## 8. First implementation phase recommendation

The next approved coding task should implement only constant-conductivity heat-transfer patch-test infrastructure, not coupled fracture diagnostics. The safest sequence is:

1. approve unit conversion convention;
2. create isolated heat PDE utilities for constant `k0`, `Q=0`;
3. validate constant T, linear 1D conduction, and insulated boundary checks;
4. only then route solved uniform T into the existing thermal-strain mechanics relation;
5. preserve `thermal_mode=off` default and prescribed-temperature fallback throughout.

## 9. Validation ladder

Validation must start with regressions: default-off no-thermal behavior and prescribed-temperature fallback. Heat residual tests should then proceed from constant T to linear steady conduction, insulated flux, transient uniform no-source, and manufactured transient checks if feasible. Mechanics coupling follows only after thermal residuals pass.

The full ladder is encoded in `tables/validation_matrix.csv`.

## 10. Patch test plan

The patch-test plan prioritizes closed-form checks with simple expected outputs: constant T residual zero, 1D linear conduction, insulated zero flux, transient uniform no-source, free expansion under solved uniform T, and constrained heating under solved uniform T.

See `tables/patch_test_plan.csv`.

## 11. Coupling strategy

First coupling should be one-way: solved T to the already reviewed thermal strain relation. Mechanics-to-heat feedback, heat-to-history direct coupling, and damage-to-conductivity coupling remain deferred. Checkpointing and postprocessing of solved T are separate dependencies and must not break the checkpointed energy-conjugate reaction route.

See `tables/coupling_dependency_plan.csv`.

## 12. Deferred features

Damage-dependent conductivity `k(d)=g(d)k0` is explicitly deferred. The same is true for thermomechanical heat generation, crack-surface heat transfer, temperature-dependent material properties, full COMSOL line-by-line matching, multi-pore `comp4`, D0040, seed studies, and shear thermal studies.

See `tables/deferred_features.csv`.

## 13. COMSOL comp3 alignment and non-alignment

The relevant COMSOL branch remains `comp3 / solid3 / ht3 / c / state3 / std1`. `comp4`, `solid2`, `ht2`, `c2`, `state4`, `TFinal`, and multi-pore settings are ignored. The PINN plan should align conceptually with comp3 constants and physics, but it does not require line-by-line COMSOL platform matching.

The eventual COMSOL relation `k_d = g(d) * k0` is acknowledged but not part of the first heat PDE phase. See `tables/comsol_alignment_notes.csv`.

## 14. Source touch plan

This package modifies no behavior source code. Future implementation is expected to touch an isolated new heat PDE module, `config.py`, `train_mixed_tm.py`, `compute_energy_mixed_tm.py`, `history_field_mixed_tm.py`, `postprocess_results.py`, focused tests, and documentation only after the relevant gates pass.

See `tables/source_touch_plan.csv`.

## 15. Risks and blockers

The highest risk is unit conversion from SI thermal constants into the current mm/kN mechanics code. Other material risks are heat residual scaling, PINN instability from adding a T field, conflating heat PDE with damage-dependent conductivity, boundary-condition ambiguity, transient time-scale ambiguity, checkpoint compatibility, low-level alpha-background misinterpretation, and COMSOL convention mismatch.

See `tables/risk_register.csv`.

## 16. Decision gate

Before implementation, the reviewer should decide whether to approve Phase 1 heat PDE implementation, whether steady-state should precede transient, which solved-temperature representation to use, the exact unit-conversion convention, whether to keep damage-dependent conductivity deferred, and whether the prescribed-temperature fallback remains frozen.

See `tables/next_decision_gate.csv`.

## 17. Final classification

`{FINAL_CLASSIFICATION}`

This plan recommends adding heat PDE support in a staged way, starting with constant-conductivity heat transfer and patch tests, while preserving the prescribed-temperature branch and the no-thermal default route. Damage-dependent conductivity `k(d)=g(d)k0` should remain deferred until the constant-conductivity heat solve and solved-temperature-to-thermal-strain coupling are independently validated. The main implementation blocker is a precise unit-conversion convention for SI thermal constants in the current mm/kN mechanics code.

## 18. Exact next recommended task

Hold a reviewer decision-gate review of this package. If approved, implement only Phase 1 constant-conductivity heat PDE/unit-conversion infrastructure and patch tests in `examples/TM_comsol_thermal_micro`; do not implement damage-dependent conductivity, run D0040, run seed studies, run shear thermal studies, or modify `examples/TM_comsol_no_thermal_micro`.
"""


def handoff_text():
    return f"""# Handoff: Heat PDE Implementation and Validation Plan

## Status

Final classification: `{FINAL_CLASSIFICATION}`

Commit hash:

- Primary plan commit: `PENDING_PRIMARY_COMMIT`.
- Handoff sync commit: `PENDING_HANDOFF_SYNC_COMMIT`; do not chase this file's own sync hash indefinitely.

Push status:

- Primary plan commit push: `PENDING_PUSH_STATUS`.
- Final status at generation time: `PENDING_FINAL_GIT_STATUS`.
- Final HEAD known at handoff-sync edit time: `PENDING_FINAL_HEAD`.

## Package

- Package path: `{PACKAGE_REL}`
- Report: `{PACKAGE_REL}/REPORT.md`
- Manifest: `{PACKAGE_REL}/MANIFEST.json`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- This task did not run training, rerun A/B/C, run D0040, run a seed study, run shear extension, or run S0110.
- This task did not implement heat PDE, damage-dependent conductivity, or a trainable/PDE temperature field.
- This task did not change material parameters, `l0`, history logic, training losses, boundary conditions, source model behavior, or reaction route.
- Energy-conjugate `reaction_N_energy` remains the primary reaction.

## Key Planning Decisions

- First heat PDE implementation should be constant-conductivity heat transfer: `{TARGET_EQUATION}` with `Q=0` initially.
- Use constant `k0`; do not implement `k(d)=g(d)k0` until constant-k0 heat PDE and solved-T-to-mechanics coupling pass independently.
- Preserve `thermal_mode=off` default and the prescribed-temperature fallback branch.
- Preserve thermal strain mechanics: `delta_T = T - Tref`; `exx_e = exx - alpha_T * delta_T`; `eyy_e = eyy - alpha_T * delta_T`; `exy_e = exy`.
- Treat SI-to-project unit conversion as an implementation blocker, not a solved detail.

## Tables Generated

- `tables/heat_pde_scope_summary.csv`
- `tables/thermal_variables_units.csv`
- `tables/implementation_phases.csv`
- `tables/validation_matrix.csv`
- `tables/patch_test_plan.csv`
- `tables/boundary_initial_condition_plan.csv`
- `tables/coupling_dependency_plan.csv`
- `tables/deferred_features.csv`
- `tables/risk_register.csv`
- `tables/comsol_alignment_notes.csv`
- `tables/source_touch_plan.csv`
- `tables/next_decision_gate.csv`
- `tables/changed_files_summary.csv`

## Validation To Report

- `git status`
- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile {PACKAGE_REL}/build_heat_pde_plan_package.py`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples/TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `{PACKAGE_REL}/REPORT.md`
2. `{PACKAGE_REL}/tables/thermal_variables_units.csv`
3. `{PACKAGE_REL}/tables/implementation_phases.csv`
4. `{PACKAGE_REL}/tables/validation_matrix.csv`
5. `{PACKAGE_REL}/tables/risk_register.csv`
6. `{PACKAGE_REL}/tables/next_decision_gate.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Hold the reviewer decision-gate review. If approved, implement only Phase 1 constant-conductivity heat PDE/unit-conversion infrastructure and patch tests. Do not implement damage-dependent conductivity, do not run training, do not run D0040 or seed/shear studies, and do not touch `examples/TM_comsol_no_thermal_micro`.
"""


def figure_summary_text():
    return """# Figure Summary

No PNG figures were generated for this planning-only package. The required scope, dependency, validation, and risk information is captured in the CSV tables and `REPORT.md`.

Optional future figures, if a reviewer requests them, could show:

1. heat PDE stage flow from planning to constant-k0 patch tests;
2. coupling dependency diagram from solved T to thermal strain;
3. validation ladder from default-off regression to solved-T reaction diagnostics.
"""


def manifest():
    return {
        "package": PACKAGE_REL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": FINAL_CLASSIFICATION,
        "planning_only": True,
        "behavior_source_modified": False,
        "training_run": False,
        "heat_pde_implemented": False,
        "damage_dependent_conductivity_implemented": False,
        "no_thermal_project_touched": False,
        "comsol_reference_scope": "comp3 / solid3 / ht3 / c / state3 / std1",
        "ignored_comsol_scope": ["comp4", "solid2", "ht2", "c2", "state4", "TFinal", "multi-pore settings"],
        "target_equation": TARGET_EQUATION,
        "first_phase_recommendation": "constant conductivity k0, Q=0, patch tests first",
        "main_blocker": "exact SI-to-project unit conversion for rho, c, k0, Q, time, heat flux, and residual scale",
        "required_report_sections": REPORT_SECTIONS,
        "tables": sorted(TABLES.keys()),
        "required_outputs": [
            "REPORT.md",
            "HANDOFF_COMMENT.md",
            "MANIFEST.json",
            "figures/figure_summary.md",
            *[f"tables/{name}" for name in sorted(TABLES.keys())],
        ],
        "reviewer_read_next": [
            f"{PACKAGE_REL}/REPORT.md",
            f"{PACKAGE_REL}/tables/thermal_variables_units.csv",
            f"{PACKAGE_REL}/tables/implementation_phases.csv",
            f"{PACKAGE_REL}/tables/validation_matrix.csv",
            f"{PACKAGE_REL}/tables/risk_register.csv",
            f"{PACKAGE_REL}/tables/next_decision_gate.csv",
            MEMORY_REL,
        ],
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for filename, (columns, rows_fn) in TABLES.items():
        write_csv(TABLE_DIR / filename, rows_fn(), columns)

    write_text(PACKAGE_DIR / "REPORT.md", report_text())
    write_text(PACKAGE_DIR / "HANDOFF_COMMENT.md", handoff_text())
    write_text(FIGURE_DIR / "figure_summary.md", figure_summary_text())

    with open(PACKAGE_DIR / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest(), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
