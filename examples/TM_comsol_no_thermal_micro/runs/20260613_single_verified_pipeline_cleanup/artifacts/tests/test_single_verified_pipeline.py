from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NORMAL_SOURCE_FILES = [
    "config.py",
    "main.py",
    "mixed_mode_tm.py",
    "compute_energy_mixed_tm.py",
    "train_mixed_tm.py",
    "postprocess_results.py",
    "plot_results.py",
    "README.md",
    "POSTPROCESS_WORKFLOW.md",
    "PROJECT_STRUCTURE.md",
]


def _text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_normal_cli_exposes_only_verified_route():
    config = _text("config.py")
    for removed in [
        "--mixed-mechanics-mode",
        "current_split",
        "history_phase_current_mechanics",
        "--alpha-init-intact",
        "alpha_init_intact",
        "--solve-scheme",
        "staggered",
        "--stagger-iters",
    ]:
        assert removed not in config
    assert "mixed_mechanics_mode = \"history\"" in config
    assert "mixed_split_mode = \"tm_source\"" in config


def test_alpha_init_intact_route_is_removed():
    assert not (ROOT / "alpha_initialization.py").exists()
    assert "apply_alpha_init_intact" not in _text("main.py")


def test_mixed_split_module_keeps_only_tm_source_split():
    text = _text("mixed_mode_tm.py")
    for removed in [
        "MIXED_SPLIT_MODES",
        "voldev_tension_only",
        "voldev",
        "_current_split",
        "split_mode=",
        "Unsupported mixed split mode",
    ]:
        assert removed not in text
    assert "tm_source" in text


def test_training_code_has_no_user_selectable_solver_or_mechanics_alternative():
    text = _text("train_mixed_tm.py")
    for removed in [
        "current_split",
        "history_phase_current_mechanics",
        "solve_scheme",
        "staggered",
        "diagnostics_staggered_substeps",
    ]:
        assert removed not in text


def test_normal_postprocess_outputs_do_not_include_legacy_or_compatibility_metrics():
    for filename in ["postprocess_results.py", "plot_results.py"]:
        text = _text(filename)
        for removed in [
            "legacy_top_sigma_integral_N",
            "reaction_N_legacy_top_sigma",
            "legacy_top_sigma",
            "legacy_curve_status",
            "corrected",
            "clean",
            "run_corrected",
            "make_clean",
        ]:
            assert removed not in text


def test_docs_describe_single_verified_workflow_only():
    combined = "\n".join(_text(path) for path in ["README.md", "POSTPROCESS_WORKFLOW.md", "PROJECT_STRUCTURE.md"])
    for removed in [
        "current_split",
        "alpha-init-intact",
        "apply_alpha_init_intact",
        "staggered",
        "legacy top",
        "reaction_N_legacy_top_sigma",
        "corrected_reaction_postprocess",
        "plot_clean_tm_results",
    ]:
        assert removed not in combined
    assert "postprocess_results.py" in combined


def test_obsolete_root_reports_are_removed():
    for filename in [
        "ALPHA_INIT_INTACT_COMPARISON_REPORT.md",
        "TRUE_STAGGERED_DIAGNOSTIC_REPORT.md",
        "UNIFORM_BACKGROUND_DAMAGE_ROOT_CAUSE_REPORT.md",
        "PLATFORM_EQUIVALENCE_DIAGNOSTIC_REPORT.md",
    ]:
        assert not (ROOT / filename).exists()
