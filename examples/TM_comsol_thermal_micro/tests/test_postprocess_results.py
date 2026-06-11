from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_stress_strain_builder_uses_energy_reaction_metric():
    from postprocess_results import build_stress_strain_curve

    reactions = pd.DataFrame(
        [
            {
                "seed": 42,
                "step": 0,
                "Delta": 1.0e-6,
                "reaction_N_energy": 0.08,
                "reaction_N_virtual_work": 0.08000001,
                "alpha0p8_through_crack": False,
            }
        ]
    )

    curve = build_stress_strain_curve(reactions, reference_length_mm=0.01, reference_area_mm2=0.01)

    row = curve.iloc[0]
    assert row["stress_strain_primary_metric"] == "nominal_stress_energy_MPa"
    assert row["reaction_metric_status"] == "energy_conjugate"
    assert row["is_energy_conjugate"] is True or row["is_energy_conjugate"] == True
    assert abs(row["nominal_strain"] - 1.0e-4) <= 1.0e-12
    assert abs(row["nominal_stress_energy_MPa"] - 8.0) <= 1.0e-12
    assert "reaction_N_energy" in curve.columns


def test_unavailable_stress_strain_table_is_written_when_checkpoints_are_absent(tmp_path):
    from postprocess_results import STRESS_STRAIN_TABLE_NAME, write_unavailable_stress_strain_curve

    result_dir = tmp_path / "results" / "run_without_checkpoints"
    result_dir.mkdir(parents=True)
    (result_dir / "displacement_list.csv").write_text(
        "step,displacement_mm\n0,0.0\n1,1e-6\n",
        encoding="utf-8",
    )

    out_path = write_unavailable_stress_strain_curve(
        result_dir,
        result_dir / "curves" / STRESS_STRAIN_TABLE_NAME,
        reason="no_step_checkpoints",
    )

    curve = pd.read_csv(out_path)
    assert list(curve["stress_strain_primary_metric"]) == ["reaction_metric_unavailable", "reaction_metric_unavailable"]
    assert list(curve["reaction_metric_status"]) == ["no_step_checkpoints", "no_step_checkpoints"]
    assert "nominal_stress_energy_MPa" in curve.columns
    assert curve["nominal_stress_energy_MPa"].isna().all()


def test_main_invokes_results_postprocess_after_training():
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from postprocess_results import run_results_postprocess" in main_text
    assert "run_results_postprocess(" in main_text


def test_plotting_auto_discovers_stress_strain_table_under_result_curves(tmp_path):
    from plot_results import resolve_stress_strain_table

    result_dir = tmp_path / "full_Seed_42_case"
    curves = result_dir / "curves"
    out_dir = tmp_path / "figures"
    curves.mkdir(parents=True)
    out_dir.mkdir()
    (curves / "stress_strain_by_step.csv").write_text(
        "\n".join(
            [
                "seed,step,Delta,nominal_strain,stress_strain_primary_metric,reaction_metric_status,nominal_stress_energy_MPa,reaction_N_energy",
                "42,0,1e-6,1e-4,nominal_stress_energy_MPa,energy_conjugate,8.0,0.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, path, seed = resolve_stress_strain_table(result_dir, out_dir)

    assert path == curves / "stress_strain_by_step.csv"
    assert seed == 42
    assert rows[0]["stress_strain_primary_metric"] == "nominal_stress_energy_MPa"


def test_postprocess_generates_figures_from_stress_strain_table(tmp_path):
    from postprocess_results import generate_figures_for_stress_strain_curve

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
    table = curves / "stress_strain_by_step.csv"
    table.write_text(
        "\n".join(
            [
                "seed,step,Delta,nominal_strain,stress_strain_primary_metric,reaction_metric_status,nominal_stress_energy_MPa,reaction_N_energy",
                "42,0,1e-6,1e-4,nominal_stress_energy_MPa,energy_conjugate,8.0,0.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generated = generate_figures_for_stress_strain_curve(
        result_dir,
        table,
        figure_dir=figures,
        run_label="merged",
        dpi=80,
    )

    assert figures / "stress_strain_merged.png" in generated
    assert figures / "reaction_strain_merged.png" in generated
    assert (figures / "stress_strain_merged.png").exists()
    assert (figures / "reaction_strain_merged.png").exists()
