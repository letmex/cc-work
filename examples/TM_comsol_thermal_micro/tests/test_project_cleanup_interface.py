from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_minimal_field(result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        result_dir / "fields_mixed_tm_step_0000.npz",
        x=np.array([0.0, 0.01, 0.0], dtype=float),
        y=np.array([0.0, 0.0, 0.01], dtype=float),
        triangles=np.array([[0, 1, 2]], dtype=int),
        u=np.zeros(3, dtype=float),
        v=np.array([0.0, 0.0, 1.0e-6], dtype=float),
        alpha=np.zeros(3, dtype=float),
        displacement_mm=np.array(1.0e-6),
    )


def _write_minimal_stress_strain(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed,step,Delta,nominal_strain,stress_strain_primary_metric,reaction_metric_status,nominal_stress_energy_MPa,reaction_N_energy,is_energy_conjugate",
                "42,0,1e-6,1e-4,nominal_stress_energy_MPa,energy_conjugate,8.0,0.08,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_normal_route_defaults_are_configured_without_route_flags():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert 'mixed_mechanics_mode = "history"' in config_text
    assert 'mixed_split_mode = "tm_source"' in config_text
    assert 'TM_COMSOL_MICRO_TOP_U_MODE", "free"' in config_text
    assert 'TM_COMSOL_MICRO_COORD_NORMALIZATION", "unit_box"' in config_text
    for removed in ["--mixed-mechanics-mode", "--alpha-init-intact", "--solve-scheme"]:
        assert removed not in config_text


def test_functional_postprocess_and_plot_modules_exist_and_main_uses_entry():
    assert (ROOT / "postprocess_results.py").exists()
    assert (ROOT / "plot_results.py").exists()
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from postprocess_results import run_results_postprocess" in main_text
    assert "run_results_postprocess(" in main_text


def test_postprocess_output_names_are_functional():
    import postprocess_results as pp

    names = [
        pp.STRESS_STRAIN_TABLE_NAME,
        pp.REACTION_TABLE_NAME,
        pp.REACTION_AVAILABILITY_NAME,
    ]
    assert names == [
        "stress_strain_by_step.csv",
        "reaction_by_step.csv",
        "reaction_metric_availability.csv",
    ]
    for name in names:
        assert "reaction" in name or "stress_strain" in name


def test_postprocess_generates_csv_and_figures_with_functional_names(tmp_path):
    from postprocess_results import STRESS_STRAIN_TABLE_NAME, generate_figures_for_stress_strain_curve

    result_dir = tmp_path / "results" / "full_hl_8_Neurons_400_Seed_42_D0020_case"
    _write_minimal_field(result_dir)
    curve_path = result_dir / "curves" / STRESS_STRAIN_TABLE_NAME
    _write_minimal_stress_strain(curve_path)

    generated = generate_figures_for_stress_strain_curve(result_dir, curve_path, dpi=80)
    names = {path.name for path in generated}

    assert "stress_strain_seed42_D0020.png" in names
    assert "reaction_strain_seed42_D0020.png" in names
    assert "final_fields_panel_seed42_D0020.png" in names


def test_figure_failure_is_nonfatal_by_default(tmp_path):
    from postprocess_results import STRESS_STRAIN_TABLE_NAME, run_results_postprocess

    model_dir = tmp_path / "model"
    result_dir = tmp_path / "results" / "run_without_fields"
    model_dir.mkdir()
    result_dir.mkdir(parents=True)
    (model_dir / "model_settings.txt").write_text("seed: 42\n", encoding="utf-8")
    (result_dir / "displacement_list.csv").write_text("step,displacement_mm\n0,1e-6\n", encoding="utf-8")

    result = run_results_postprocess(model_dir, result_dir)

    assert result["status"] == "no_step_checkpoints"
    assert result["figure_status"].startswith("failed_")
    assert (result_dir / "curves" / STRESS_STRAIN_TABLE_NAME).exists()


def test_fail_on_figure_error_makes_figure_failure_fatal(tmp_path):
    from postprocess_results import run_results_postprocess

    model_dir = tmp_path / "model"
    result_dir = tmp_path / "results" / "run_without_fields"
    model_dir.mkdir()
    result_dir.mkdir(parents=True)
    (model_dir / "model_settings.txt").write_text("seed: 42\n", encoding="utf-8")
    (result_dir / "displacement_list.csv").write_text("step,displacement_mm\n0,1e-6\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_results_postprocess(model_dir, result_dir, fail_on_figure_error=True)


def test_metric_policy_keeps_energy_reaction_as_only_normal_metric():
    from postprocess_results import build_stress_strain_curve

    curve = build_stress_strain_curve(
        pd.DataFrame(
            [
                {
                    "seed": 42,
                    "step": 0,
                    "Delta": 1.0e-6,
                    "reaction_N_energy": 0.08,
                }
            ]
        )
    )

    row = curve.iloc[0]
    assert row["stress_strain_primary_metric"] == "nominal_stress_energy_MPa"
    assert row["reaction_metric_status"] == "energy_conjugate"
    assert "nominal_stress_energy_MPa" in curve.columns
