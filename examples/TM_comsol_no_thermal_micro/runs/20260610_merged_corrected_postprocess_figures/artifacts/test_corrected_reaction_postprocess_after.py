from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_corrected_curve_builder_promotes_energy_exact_metric():
    from corrected_reaction_postprocess import build_corrected_stress_strain_curve

    reactions = pd.DataFrame(
        [
            {
                "seed": 42,
                "step": 0,
                "Delta": 1.0e-6,
                "reaction_N_energy_exact": 0.08,
                "reaction_N_energy_virtual_work": 0.08000001,
                "reaction_N_legacy_top_sigma": 0.03,
                "reaction_N_bottom_sigma_legacy": -0.04,
                "reaction_N_internal_cut_above": 0.02,
                "reaction_N_internal_cut_below": 0.025,
                "alpha0p8_through_crack": False,
            }
        ]
    )

    curve = build_corrected_stress_strain_curve(reactions, reference_length_mm=0.01, reference_area_mm2=0.01)

    row = curve.iloc[0]
    assert row["stress_strain_primary_metric"] == "nominal_stress_energy_exact_MPa"
    assert row["stress_strain_metric_status"] == "energy_conjugate_primary"
    assert row["legacy_curve_status"] == "legacy_diagnostic_only"
    assert abs(row["nominal_strain"] - 1.0e-4) <= 1.0e-12
    assert abs(row["nominal_stress_energy_exact_MPa"] - 8.0) <= 1.0e-12
    assert abs(row["nominal_stress_legacy_top_sigma_MPa"] - 3.0) <= 1.0e-12


def test_unavailable_curve_is_written_when_checkpoints_are_absent(tmp_path):
    from corrected_reaction_postprocess import write_unavailable_corrected_curve

    result_dir = tmp_path / "results" / "run_without_checkpoints"
    result_dir.mkdir(parents=True)
    (result_dir / "displacement_list.csv").write_text(
        "step,displacement_mm\n0,0.0\n1,1e-6\n",
        encoding="utf-8",
    )

    out_path = write_unavailable_corrected_curve(
        result_dir,
        result_dir / "curves" / "corrected_stress_strain_by_step.csv",
        reason="no_step_checkpoints",
    )

    curve = pd.read_csv(out_path)
    assert list(curve["stress_strain_primary_metric"]) == ["reaction_metric_unavailable", "reaction_metric_unavailable"]
    assert list(curve["stress_strain_metric_status"]) == ["no_step_checkpoints", "no_step_checkpoints"]
    assert "nominal_stress_energy_exact_MPa" in curve.columns
    assert curve["nominal_stress_energy_exact_MPa"].isna().all()


def test_main_invokes_corrected_reaction_postprocess_after_training():
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from corrected_reaction_postprocess import run_corrected_reaction_postprocess" in main_text
    assert "run_corrected_reaction_postprocess(" in main_text


def test_clean_plotting_auto_discovers_corrected_curve_under_result_curves(tmp_path):
    from plot_clean_tm_results import resolve_corrected_stress_strain

    result_dir = tmp_path / "full_Seed_42_case"
    curves = result_dir / "curves"
    out_dir = tmp_path / "figures"
    curves.mkdir(parents=True)
    out_dir.mkdir()
    (curves / "corrected_stress_strain_by_step.csv").write_text(
        "\n".join(
            [
                "seed,step,Delta,nominal_strain,stress_strain_primary_metric,stress_strain_metric_status,nominal_stress_energy_exact_MPa,reaction_N_energy_exact",
                "42,0,1e-6,1e-4,nominal_stress_energy_exact_MPa,energy_conjugate_primary,8.0,0.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, path, seed = resolve_corrected_stress_strain(result_dir, out_dir)

    assert path == curves / "corrected_stress_strain_by_step.csv"
    assert seed == 42
    assert rows[0]["stress_strain_primary_metric"] == "nominal_stress_energy_exact_MPa"


def test_corrected_postprocess_generates_clean_figures_from_corrected_curve(tmp_path):
    from corrected_reaction_postprocess import generate_clean_figures_for_corrected_curve

    result_dir = tmp_path / "results" / "smoke_Seed_42_case"
    curves = result_dir / "curves"
    figures = result_dir / "figures"
    curves.mkdir(parents=True)
    x = np.array([0.0, 0.01, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.01], dtype=float)
    np.savez(
        result_dir / "fields_mixed_tm_step_0000.npz",
        x=x,
        y=y,
        triangles=np.array([[0, 1, 2]], dtype=int),
        u=np.zeros(3, dtype=float),
        v=np.array([0.0, 0.0, 1.0e-6], dtype=float),
        alpha=np.zeros(3, dtype=float),
        displacement_mm=np.array(1.0e-6),
    )
    corrected_curve = curves / "corrected_stress_strain_by_step.csv"
    corrected_curve.write_text(
        "\n".join(
            [
                "seed,step,Delta,nominal_strain,stress_strain_primary_metric,stress_strain_metric_status,nominal_stress_energy_exact_MPa,reaction_N_energy_exact,legacy_curve_status",
                "42,0,1e-6,1e-4,nominal_stress_energy_exact_MPa,energy_conjugate_primary,8.0,0.08,legacy_diagnostic_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generated = generate_clean_figures_for_corrected_curve(
        result_dir,
        corrected_curve,
        figure_dir=figures,
        run_label="merged",
        dpi=80,
    )

    assert figures / "stress_strain_merged.png" in generated
    assert figures / "reaction_strain_merged.png" in generated
    assert (figures / "stress_strain_merged.png").exists()
    assert (figures / "reaction_strain_merged.png").exists()
