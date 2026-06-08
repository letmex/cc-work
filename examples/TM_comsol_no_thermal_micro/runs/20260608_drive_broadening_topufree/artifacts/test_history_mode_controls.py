from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_exposes_mixed_mechanics_mode_cli():
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "--mixed-mechanics-mode" in text
    assert "_mech_{mixed_mechanics_mode}" in text
    assert "history_phase_current_mechanics" in text
    assert 'TM_COMSOL_MICRO_MIXED_MECHANICS_MODE", "history"' in text


def test_clean_plotting_separates_history_current_and_training_drive():
    text = (ROOT / "plot_clean_tm_results.py").read_text(encoding="utf-8")
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


def test_debug_recompute_script_exposes_expected_cli_and_recomputes_drive():
    path = ROOT / "debug_recompute_he_current.py"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for token in [
        "--npz",
        "--out",
        "--alpha-mode",
        "mixed_mode_energy_split",
        "He_current",
        "max_abs_diff",
        "bottom_right",
        "notch_tip",
    ]:
        assert token in text


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


def test_mesh_l0_diagnostics_script_documents_regions_candidates_and_outputs():
    path = ROOT / "mesh_l0_diagnostics.py"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for token in [
        "geo_coarse_with_groups_mm.msh",
        "h_eq",
        "bottom_right",
        "bottom_boundary_band",
        "top_boundary_band",
        "1.5e-4",
        "4.0e-4",
        "mesh_l0_diagnostics.csv",
        "mesh_l0_diagnostics_summary.txt",
    ]:
        assert token in text


def test_platform_equivalence_diagnostic_scripts_and_alpha_intact_flag_exist():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "--alpha-init-intact" in config_text
    assert "apply_alpha_init_intact" in main_text

    for filename, tokens in {
        "debug_alpha_initial_state.py": [
            "--alpha-init-intact",
            "raw_alpha_min",
            "raw_alpha_mean",
            "raw_alpha_max",
            "alpha_min",
            "alpha_mean",
            "alpha_max",
            "n_alpha_lt_0",
            "n_alpha_gt_1",
        ],
        "debug_elastic_only_pinn.py": [
            "alpha fixed to zero",
            "He_current",
            "notch_tip_He_current_max",
            "bottom_right_He_current_max",
        ],
        "debug_boundary_notch_lips.py": [
            "top",
            "bottom",
            "notch_upper",
            "notch_lower",
            "displacement_jump",
        ],
    }.items():
        text = (ROOT / filename).read_text(encoding="utf-8")
        for token in tokens:
            assert token in text


def test_uniform_background_damage_diagnostics_and_experimental_controls_exist():
    for filename, tokens in {
        "debug_alpha_equilibrium_from_drive.py": [
            "--npz",
            "--drive-field",
            "alpha_eq_min",
            "notch_tip_alpha_eq_max",
            "bottom_right_alpha_eq_max",
            "correlation",
        ],
        "debug_fixed_uv_alpha_only.py": [
            "--npz",
            "--drive-field",
            "--phase-proximal",
            "alpha_step_change_mean",
            "notch_tip_alpha_max",
            "bottom_right_alpha_max",
        ],
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


def test_drive_broadening_stepwise_script_exists_and_outputs_required_fields():
    path = ROOT / "analyze_drive_broadening_stepwise.py"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for token in [
        "--run-dir",
        "--out",
        "--summary",
        "alpha_gt_0p5_area_fraction",
        "bulk_He_current_p95_over_notch_tip_He_current_max",
        "bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max",
        "first_alpha_mean_gt_0p05",
        "first_bulk_He_ratio_gt_0p25",
    ]:
        assert token in text


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


def test_fedof_staggered_baseline_script_exists_and_uses_dual_history():
    path = ROOT / "debug_fedof_staggered_baseline.py"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for token in [
        "--load-schedule-file",
        "--max-steps",
        "mechanics_epochs",
        "phase_epochs",
        "HI_trial",
        "HII_trial",
        "mechanics_substep_loss",
        "phase_substep_loss",
        "reaction_N_tm_eff",
        "debug_fedof_staggered_baseline_fields.npz",
    ]:
        assert token in text
