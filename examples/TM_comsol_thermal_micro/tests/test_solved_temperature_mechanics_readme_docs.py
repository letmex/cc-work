from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_solved_temperature_smoke_cli_as_opt_in():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_snippets = [
        "Solved-Temperature Mechanics Smoke CLI",
        "run_solved_temperature_mechanics_smoke.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe",
        "--output-dir",
        "mechanics_smoke_results.csv",
        "solved_frozen_lift",
        "element_centroid",
        "DeltaT",
        "6.666666666666686",
        "13.333333333333371",
        "mechanics_max_abs_diff = 0.0",
        "opt-in",
        "train_mixed_tm.py remains unmodified",
    ]
    for snippet in required_snippets:
        assert snippet in readme


def test_readme_keeps_solved_temperature_smoke_boundaries_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    boundary_snippets = [
        "does not implement coupled heat-mechanics training",
        "does not add heat-fracture diagnostics",
        "does not add damage-dependent conductivity",
        "does not run D0040, seed, shear, S0110, transient, or bottom-cooling production studies",
        "examples/TM_comsol_no_thermal_micro",
    ]
    for snippet in boundary_snippets:
        assert snippet in readme

    train_source = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    assert "run_solved_temperature_mechanics_smoke" not in train_source
    assert "mechanics_smoke_results.csv" not in train_source
