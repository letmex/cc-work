import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_keeps_single_verified_route_defaults():
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert 'mixed_mechanics_mode = "history"' in text
    assert 'mixed_split_mode = "tm_source"' in text
    assert '"PFF_model": "AT2"' in text
    for removed in [
        "--mixed-mechanics-mode",
        "--alpha-init-intact",
        "--solve-scheme",
        "--stagger-iters",
        "phase_proximal",
    ]:
        assert removed not in text


def test_result_plotting_separates_history_current_and_training_drive():
    text = (ROOT / "plot_results.py").read_text(encoding="utf-8")
    assert '"He_history"' in text
    assert '"He_current"' in text
    assert '"mechanics_drive"' in text


def test_readme_documents_no_thermal_scope_and_energy_reaction():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "No temperature field" in text
    assert "history_mode = mixedH_TM" in text
    assert "mixed_split_mode = tm_source" in text
    assert "reaction_N_energy" in text
    assert "does not claim physical validation" in text


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
    assert "The root folder should not contain generated run directories" in text


def test_energy_module_uses_history_drive_without_alternate_branches():
    energy_text = (ROOT / "compute_energy_mixed_tm.py").read_text(encoding="utf-8")
    summary_text = (ROOT / "history_field_mixed_tm.py").read_text(encoding="utf-8")
    for token in [
        "mechanics_current_energy_density",
        "phase_history_energy_density",
        "phase_history_total_density",
        "He_trial",
    ]:
        assert token in energy_text
    assert "mechanics_current_energy" in summary_text
    assert "phase_history_energy" in summary_text
    for removed in ["phase_proximal", "split_mode=", "mechanics_mode="]:
        assert removed not in energy_text
        assert removed not in summary_text


def test_root_debug_scripts_are_not_formal_workflow_entries():
    assert not list(ROOT.glob("debug_*.py"))
    assert not (ROOT / "analyze_drive_broadening_stepwise.py").exists()
    assert not (ROOT / "mesh_l0_diagnostics.py").exists()


def test_alpha_intact_route_is_absent():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "alpha-init-intact" not in config_text
    assert "apply_alpha_init_intact" not in main_text
    assert not (ROOT / "alpha_initialization.py").exists()


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
