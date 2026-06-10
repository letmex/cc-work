import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_exposes_mixed_mechanics_mode_cli():
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "--mixed-mechanics-mode" in text
    assert "_mech_{mixed_mechanics_mode}" in text
    assert "history_phase_current_mechanics" in text
    assert 'TM_COMSOL_MICRO_MIXED_MECHANICS_MODE", "history"' in text


def test_result_plotting_separates_history_current_and_training_drive():
    text = (ROOT / "plot_results.py").read_text(encoding="utf-8")
    assert '"He_history"' in text
    assert '"He_current"' in text
    assert '"mechanics_drive"' in text


def test_readme_documents_no_thermal_scope_and_known_risks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "temperature field T" in text
    assert "history / dual-history is the closest paper-COMSOL source-model route" in text
    assert "current_split is a diagnostic/ablation route" in text
    assert "current_split is the intended physical route" not in text
    assert "bottom-boundary He_current/mechanics_drive artifact" in text
    assert "single continuous global neural network displacement ansatz" in text
    assert "NonsmoothSigmoid" in text


def test_diagnostics_include_alpha_bounds_drive_locations_and_local_regions():
    text = (ROOT / "history_field_mixed_tm.py").read_text(encoding="utf-8")
    for token in [
        "alpha_min",
        "n_alpha_lt_0",
        "n_alpha_gt_1",
        "max_He_current_x",
        "max_He_history_x",
        "max_mechanics_drive_x",
        "notch_tip_region",
        "bottom_right_region",
        "elastic_energy_density",
        "fracture_energy_density",
    ]:
        assert token in text


def test_project_structure_documents_managed_debug_outputs():
    text = (ROOT / "PROJECT_STRUCTURE.md").read_text(encoding="utf-8")
    assert "outputs/debug/" in text
    assert "Debug scripts should not be left in the example root" in text


def test_history_phase_current_mechanics_mode_documents_split_gradient_diagnostics():
    energy_text = (ROOT / "compute_energy_mixed_tm.py").read_text(encoding="utf-8")
    summary_text = (ROOT / "history_field_mixed_tm.py").read_text(encoding="utf-8")
    for token in [
        "history_phase_current_mechanics",
        "mechanics_current_energy_density",
        "phase_history_energy_density",
        "phase_history_total_density",
        "detach()",
    ]:
        assert token in energy_text
    assert "mechanics_current_energy" in summary_text
    assert "phase_history_energy" in summary_text


def test_root_debug_scripts_are_not_formal_workflow_entries():
    assert not list(ROOT.glob("debug_*.py"))
    assert not (ROOT / "analyze_drive_broadening_stepwise.py").exists()
    assert not (ROOT / "mesh_l0_diagnostics.py").exists()


def test_platform_equivalence_diagnostic_scripts_and_alpha_intact_flag_exist():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "--alpha-init-intact" in config_text
    assert "apply_alpha_init_intact" in main_text

    text = (ROOT / "PROJECT_STRUCTURE.md").read_text(encoding="utf-8")
    assert "one-off diagnostics" in text


def test_save_step_checkpoints_defaults_to_true():
    config_path = ROOT / "config.py"
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    default_value = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != "--save-step-checkpoints":
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                default_value = ast.literal_eval(keyword.value)
                break
        break
    assert default_value is True


def test_uniform_background_damage_diagnostics_and_experimental_controls_exist():
    for filename, tokens in {
        "field_computation.py": [
            "top_u_mode",
            "top_u_raw",
        ],
        "config.py": [
            "--top-u-mode",
            "--phase-proximal",
            "--eta-eff",
            "--dt",
        ],
        "compute_energy_mixed_tm.py": [
            "phase_proximal_mode",
            "phase_proximal_energy_density",
            "alpha_step_change_mean",
        ],
        "history_field_mixed_tm.py": [
            "phase_proximal_energy",
            "alpha_step_change_max",
        ],
    }.items():
        path = ROOT / filename
        assert path.exists(), filename
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            assert token in text


def test_top_u_free_boundary_diagnostics_exist():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    field_text = (ROOT / "field_computation.py").read_text(encoding="utf-8")
    summary_text = (ROOT / "history_field_mixed_tm.py").read_text(encoding="utf-8")
    train_text = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    for token in ["--top-u-mode", "fixed", "free"]:
        assert token in config_text
    for token in ["top_u_mode", "top_u_raw", "u_shape"]:
        assert token in field_text
    for token in [
        "_boundary_displacement_stats",
        "top_u_abs_max",
        "top_v_error_max",
        "bottom_u_abs_max",
        "bottom_v_abs_max",
        "top_u_mode",
    ]:
        assert token in summary_text
    assert "top_u_mode=training_dict.get" in train_text


def test_root_generated_debug_outputs_are_absent():
    assert not list(ROOT.glob("debug_*.csv"))
    assert not list(ROOT.glob("debug_*.npz"))
    assert not list(ROOT.glob("*_fields.npz"))


def test_true_staggered_cli_and_training_diagnostics_exist():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    train_text = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    for token in [
        "--solve-scheme",
        "coupled",
        "staggered",
        "--stagger-iters",
        'TM_COMSOL_MICRO_SOLVE_SCHEME", "coupled"',
    ]:
        assert token in config_text
    for token in [
        "_run_mixed_tm_step_staggered",
        "mechanics_substep_loss",
        "phase_substep_loss",
        "stagger_iter",
        "diagnostics_staggered_substeps.csv",
        "solve_scheme",
        "optim_rel_tol",
    ]:
        assert token in train_text


def test_fedof_debug_scripts_are_not_root_workflow_entries():
    assert not list(ROOT.glob("debug_fedof*.py"))
