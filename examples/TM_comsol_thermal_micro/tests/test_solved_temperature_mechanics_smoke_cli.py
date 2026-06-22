from pathlib import Path
import csv
import importlib
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def _smoke_module():
    return importlib.import_module("run_solved_temperature_mechanics_smoke")


def _metrics(table_path):
    with Path(table_path).open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def test_smoke_runner_writes_compact_mechanics_result_table(tmp_path):
    smoke = _smoke_module()

    result = smoke.run_smoke(tmp_path)
    table_path = tmp_path / "mechanics_smoke_results.csv"
    metrics = _metrics(table_path)

    assert Path(result["table_path"]) == table_path
    assert result["classification"] == "solved-temperature mechanics smoke cli passed"
    assert metrics["classification"] == "solved-temperature mechanics smoke cli passed"
    assert metrics["source_mode"] == "solved_frozen_lift"
    assert metrics["case_id"] == "H1"
    assert metrics["evaluation_location"] == "element_centroid"
    assert metrics["network_training_run"] == "false"
    assert int(metrics["n_nodes"]) == 4
    assert int(metrics["n_elements"]) == 2
    assert float(metrics["delta_T_min_K"]) == pytest.approx(20.0 / 3.0)
    assert float(metrics["delta_T_max_K"]) == pytest.approx(40.0 / 3.0)
    assert float(metrics["mechanics_max_abs_diff"]) == pytest.approx(0.0)


def test_smoke_cli_runs_as_explicit_script_and_writes_requested_output_dir(tmp_path):
    script = ROOT / "run_solved_temperature_mechanics_smoke.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    table_path = tmp_path / "mechanics_smoke_results.csv"
    metrics = _metrics(table_path)
    assert str(table_path) in completed.stdout
    assert metrics["classification"] == "solved-temperature mechanics smoke cli passed"
    assert metrics["train_mixed_tm_modified"] == "false"


def test_smoke_cli_is_opt_in_and_has_no_forbidden_tokens():
    _smoke_module()

    train_source = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    assert "run_solved_temperature_mechanics_smoke" not in train_source
    assert "mechanics_smoke_results.csv" not in train_source

    script_source = (ROOT / "run_solved_temperature_mechanics_smoke.py").read_text(encoding="utf-8")
    forbidden_tokens = [
        "k(d)",
        "g(d)",
        "k_d",
        "damage_dependent_conductivity",
        "alpha_conductivity",
        "heat_fracture",
        "D0040",
        "seed study",
    ]
    for token in forbidden_tokens:
        assert token not in script_source
