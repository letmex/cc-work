import ast
import csv
import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def _reload_config(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    sys.modules.pop("config", None)
    return importlib.import_module("config")


def _two_element_fixture():
    inp = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]],
        dtype=torch.float64,
    )
    t_conn = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    return inp, t_conn


def test_config_flag_defaults_off_and_records_fixed_h1_source_settings(monkeypatch, tmp_path):
    config = _reload_config(
        monkeypatch,
        ["config.py", "--output-root", str(tmp_path), "--run-suffix", "train_guard_default"],
    )

    assert config.args.solved_temperature_mechanics_smoke is False
    assert config.training_dict["solved_temperature_mechanics_smoke"] is False
    assert config.training_dict["solved_temperature_source_mode"] == "solved_frozen_lift"
    assert config.training_dict["solved_temperature_case_id"] == "H1"
    assert config.training_dict["solved_temperature_evaluation_location"] == "element_centroid"
    assert config.training_dict["solved_temperature_bounds"] == [[0.0, 0.01], [0.0, 0.01]]
    assert config.training_dict["solved_temperature_T_bottom"] == 300.0
    assert config.training_dict["solved_temperature_T_top"] == 320.0
    assert config.training_dict["solved_temperature_T_ref"] == 300.0


def test_config_flag_opt_in_sets_training_dict_boolean(monkeypatch, tmp_path):
    config = _reload_config(
        monkeypatch,
        [
            "config.py",
            "--output-root",
            str(tmp_path),
            "--run-suffix",
            "train_guard_enabled",
            "--solved-temperature-mechanics-smoke",
        ],
    )

    assert config.args.solved_temperature_mechanics_smoke is True
    assert config.training_dict["solved_temperature_mechanics_smoke"] is True


@pytest.mark.parametrize(
    "thermal_args",
    [
        ["--thermal-temperature-K", "310"],
        ["--thermal-delta-T", "10"],
        ["--thermal-mode", "uniform"],
    ],
)
def test_config_rejects_solved_temperature_smoke_with_prescribed_thermal_options(
    monkeypatch,
    tmp_path,
    thermal_args,
):
    with pytest.raises(ValueError, match="solved_temperature_mechanics_smoke.*thermal"):
        _reload_config(
            monkeypatch,
            [
                "config.py",
                "--output-root",
                str(tmp_path),
                "--run-suffix",
                "train_guard_conflict",
                "--solved-temperature-mechanics-smoke",
                *thermal_args,
            ],
        )


def test_no_flag_helper_returns_unchanged_kwargs_and_does_not_import_adapter():
    sys.modules.pop("thermal_mechanics_adapter", None)
    import train_mixed_tm

    inp, t_conn = _two_element_fixture()
    thermal_kwargs = {"thermal_mode": "off", "thermal_delta_T": None}
    training_dict = {"solved_temperature_mechanics_smoke": False}

    result = train_mixed_tm._apply_solved_temperature_mechanics_smoke(
        training_dict,
        thermal_kwargs,
        inp,
        t_conn,
    )

    assert result is thermal_kwargs
    assert "thermal_mechanics_adapter" not in sys.modules
    assert "solved_temperature_mechanics_smoke_diagnostics" not in training_dict


def test_opt_in_helper_returns_element_delta_t_and_scalar_diagnostics():
    import train_mixed_tm

    inp, t_conn = _two_element_fixture()
    training_dict = {
        "solved_temperature_mechanics_smoke": True,
        "solved_temperature_source_mode": "solved_frozen_lift",
        "solved_temperature_case_id": "H1",
        "solved_temperature_evaluation_location": "element_centroid",
        "solved_temperature_bounds": [[0.0, 0.01], [0.0, 0.01]],
        "solved_temperature_T_bottom": 300.0,
        "solved_temperature_T_top": 320.0,
        "solved_temperature_T_ref": 300.0,
    }

    result = train_mixed_tm._apply_solved_temperature_mechanics_smoke(
        training_dict,
        {"thermal_mode": "off", "thermal_delta_T": None},
        inp,
        t_conn,
    )

    assert set(result) == {"thermal_mode", "thermal_delta_T"}
    assert result["thermal_mode"] == "off"
    assert torch.allclose(
        result["thermal_delta_T"],
        torch.tensor([20.0 / 3.0, 40.0 / 3.0], dtype=torch.float64),
    )
    diagnostics = training_dict["solved_temperature_mechanics_smoke_diagnostics"]
    assert diagnostics["active"] is True
    assert diagnostics["network_training_run"] is False
    assert diagnostics["evaluation_location"] == training_dict["solved_temperature_evaluation_location"]
    assert diagnostics["delta_T_min_K"] == pytest.approx(20.0 / 3.0)
    assert diagnostics["delta_T_max_K"] == pytest.approx(40.0 / 3.0)


def test_opt_in_helper_persists_scalar_diagnostics_csv(tmp_path):
    import train_mixed_tm

    inp, t_conn = _two_element_fixture()
    training_dict = {
        "results_path": tmp_path,
        "solved_temperature_mechanics_smoke": True,
        "solved_temperature_source_mode": "solved_frozen_lift",
        "solved_temperature_case_id": "H1",
        "solved_temperature_evaluation_location": "element_centroid",
        "solved_temperature_bounds": [[0.0, 0.01], [0.0, 0.01]],
        "solved_temperature_T_bottom": 300.0,
        "solved_temperature_T_top": 320.0,
        "solved_temperature_T_ref": 300.0,
    }

    train_mixed_tm._apply_solved_temperature_mechanics_smoke(
        training_dict,
        {"thermal_mode": "off", "thermal_delta_T": None},
        inp,
        t_conn,
    )

    path = tmp_path / "solved_temperature_mechanics_smoke_diagnostics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}

    assert rows["active"] == "true"
    assert rows["source_mode"] == "solved_frozen_lift"
    assert rows["case_id"] == "H1"
    assert rows["evaluation_location"] == "element_centroid"
    assert float(rows["delta_T_min_K"]) == pytest.approx(20.0 / 3.0)
    assert float(rows["delta_T_max_K"]) == pytest.approx(40.0 / 3.0)
    assert rows["network_training_run"] == "false"


@pytest.mark.parametrize(
    "thermal_kwargs",
    [
        {"thermal_mode": "off", "thermal_temperature": 310.0, "thermal_delta_T": None},
        {"thermal_mode": "off", "thermal_temperature": None, "thermal_delta_T": 10.0},
        {"thermal_mode": "uniform", "thermal_temperature": None, "thermal_delta_T": None},
    ],
)
def test_opt_in_helper_rejects_conflicting_prescribed_thermal_inputs_before_adapter_import(
    thermal_kwargs,
):
    sys.modules.pop("thermal_mechanics_adapter", None)
    import train_mixed_tm

    inp, t_conn = _two_element_fixture()
    training_dict = {
        "solved_temperature_mechanics_smoke": True,
        "solved_temperature_source_mode": "solved_frozen_lift",
        "solved_temperature_case_id": "H1",
        "solved_temperature_bounds": [[0.0, 0.01], [0.0, 0.01]],
        "solved_temperature_T_bottom": 300.0,
        "solved_temperature_T_top": 320.0,
        "solved_temperature_T_ref": 300.0,
    }

    with pytest.raises(ValueError, match="solved_temperature_mechanics_smoke.*thermal"):
        train_mixed_tm._apply_solved_temperature_mechanics_smoke(
            training_dict,
            thermal_kwargs,
            inp,
            t_conn,
        )

    assert "thermal_mechanics_adapter" not in sys.modules


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"solved_temperature_source_mode": "prescribed_uniform"}, "source_mode"),
        ({"solved_temperature_case_id": "H2"}, "case_id"),
        ({"solved_temperature_evaluation_location": "node"}, "evaluation_location"),
        ({"solved_temperature_bounds": [[0.0, 0.02], [0.0, 0.01]]}, "bounds"),
        ({"solved_temperature_bounds": [0.0, 0.01]}, "bounds"),
        ({"solved_temperature_bounds": [[0.0, 0.01]]}, "bounds"),
        ({"solved_temperature_T_bottom": 299.0}, "T_bottom"),
        ({"solved_temperature_T_top": 321.0}, "T_top"),
        ({"solved_temperature_T_ref": 301.0}, "T_ref"),
    ],
)
def test_opt_in_helper_rejects_non_fixed_h1_smoke_settings_before_adapter_import(
    override,
    match,
):
    sys.modules.pop("thermal_mechanics_adapter", None)
    import train_mixed_tm

    inp, t_conn = _two_element_fixture()
    training_dict = {
        "solved_temperature_mechanics_smoke": True,
        "solved_temperature_source_mode": "solved_frozen_lift",
        "solved_temperature_case_id": "H1",
        "solved_temperature_bounds": [[0.0, 0.01], [0.0, 0.01]],
        "solved_temperature_T_bottom": 300.0,
        "solved_temperature_T_top": 320.0,
        "solved_temperature_T_ref": 300.0,
    }
    training_dict.update(override)

    with pytest.raises(ValueError, match=match):
        train_mixed_tm._apply_solved_temperature_mechanics_smoke(
            training_dict,
            {"thermal_mode": "off", "thermal_temperature": None, "thermal_delta_T": None},
            inp,
            t_conn,
        )

    assert "thermal_mechanics_adapter" not in sys.modules


def test_opt_in_helper_is_guarded_when_connectivity_is_missing():
    sys.modules.pop("thermal_mechanics_adapter", None)
    import train_mixed_tm

    inp, _ = _two_element_fixture()
    thermal_kwargs = {"thermal_mode": "off"}
    training_dict = {"solved_temperature_mechanics_smoke": True}

    result = train_mixed_tm._apply_solved_temperature_mechanics_smoke(
        training_dict,
        thermal_kwargs,
        inp,
        None,
    )

    assert result is thermal_kwargs
    assert "thermal_mechanics_adapter" not in sys.modules


def test_train_mixed_tm_uses_only_local_adapter_import():
    source = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(alias.name == "thermal_mechanics_adapter" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "thermal_mechanics_adapter"
            )
        )
    ]
    assert module_imports == []

    helper_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_solved_temperature_mechanics_smoke"
    ]
    assert len(helper_defs) == 1
    helper_imports = [
        node
        for node in ast.walk(helper_defs[0])
        if isinstance(node, ast.ImportFrom)
        and node.module == "thermal_mechanics_adapter"
        and any(alias.name == "build_mechanics_thermal_kwargs_from_bridge" for alias in node.names)
    ]
    assert len(helper_imports) == 1


def test_thermal_energy_kwargs_defaults_remain_off_and_none():
    import train_mixed_tm

    kwargs = train_mixed_tm._thermal_energy_kwargs({})

    assert kwargs["thermal_mode"] == "off"
    assert kwargs["thermal_temperature"] is None
    assert kwargs["thermal_delta_T"] is None
