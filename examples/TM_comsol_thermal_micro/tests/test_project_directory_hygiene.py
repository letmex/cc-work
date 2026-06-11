from pathlib import Path
import importlib
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORBIDDEN_ROOT_PATTERNS = (
    "debug_*.csv",
    "debug_*.npz",
    "debug_*_fields.npz",
    "debug_*.py",
    "full_hl_*",
    "medium_hl_*",
    "smoke_hl_*",
    "*_logs",
    "*logs",
)


def test_root_contains_no_generated_artifacts_or_one_off_debug_scripts():
    offenders = []
    for pattern in FORBIDDEN_ROOT_PATTERNS:
        offenders.extend(path.name for path in ROOT.glob(pattern))
    for cache_name in (".pytest_cache", "__pycache__"):
        if (ROOT / cache_name).exists():
            offenders.append(cache_name)

    assert offenders == []


def test_managed_output_directories_are_declared_and_used_by_default():
    old_argv = sys.argv[:]
    sys.argv = ["config.py"]
    try:
        sys.modules.pop("config", None)
        import config
        config = importlib.reload(config)
    finally:
        sys.argv = old_argv

    assert config.DEFAULT_OUTPUT_ROOT == "outputs"
    assert config.DEFAULT_RESULTS_ROOT == "outputs/results"
    assert config.DEFAULT_CHECKPOINT_ROOT == "outputs/checkpoints"
    assert config.DEFAULT_LOG_ROOT == "outputs/logs"

    assert config.model_path.is_relative_to(config.PATH_ROOT / "outputs" / "checkpoints")
    assert config.results_path.is_relative_to(config.PATH_ROOT / "outputs" / "results")
    assert config.writer.log_dir.startswith(str(config.PATH_ROOT / "outputs" / "logs"))


def test_postprocess_and_plot_defaults_do_not_write_to_example_root(tmp_path):
    from postprocess_results import default_curve_dir, default_figure_dir
    from plot_results import default_plot_out_dir

    result_dir = tmp_path / "outputs" / "results" / "seed42_D0020"
    assert default_curve_dir(result_dir) == result_dir / "curves"
    assert default_figure_dir(result_dir) == result_dir / "figures"
    assert default_plot_out_dir("seed42_D0020").parts[-3:] == ("outputs", "figures", "seed42_D0020")
