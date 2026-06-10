# Auto Checkpointed Corrected Curve Mainflow

## Question

Can training produce the corrected stress-strain curve automatically, instead of depending on pre-existing checkpoints?

## Root Cause

The training code already had the mechanism to write per-step checkpoints:

- `train_mixed_tm.py::save_mixed_tm_step_checkpoint(...)`
- `train_mixed_tm.py::should_save_mixed_tm_step_checkpoint(...)`

The gate was controlled by `training_dict["save_step_checkpoints"]`, which came from `config.py`. Its command-line default was `False`, so ordinary training runs did not save the checkpoint states required by the exact energy-conjugate reaction postprocessor.

## Change

`config.py` now defaults `--save-step-checkpoints` to `True`:

```python
parser.add_argument(
    "--save-step-checkpoints",
    nargs="?",
    const=True,
    default=True,
    type=_str_to_bool,
    help="Save composite per-loading-step checkpoints. Default true; pass false to disable.",
)
```

This is a mainflow behavior change, not a physics/model change. It does not change `l0`, material parameters, TM split, phase-field notch handling, alpha initialization, history update, or the loss formulation.

## Verification

A minimal smoke run was executed without passing `--save-step-checkpoints`:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --max-steps 2 --delta-max 1e-6 --n-rprop 1 --n-lbfgs 0 --top-u-mode free --coord-normalization unit_box --run-suffix auto_ckpt_default_smoke
```

The generated `model_settings.txt` recorded:

```text
save_step_checkpoints: True
checkpoint_every_step: True
```

The run generated:

- `best_models/step_checkpoints/checkpoint_mixedH_TM_step_0000.pt`
- `results/<run>/curves/corrected_reaction_by_step.csv`
- `results/<run>/curves/corrected_stress_strain_by_step.csv`
- `results/<run>/curves/corrected_reaction_availability.csv`

The availability table reports `energy_exact_primary` and `exact_reaction_computable=True`.

## Tests

Commands run:

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest tests\test_history_mode_controls.py::test_save_step_checkpoints_defaults_to_true -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile config.py main.py train_mixed_tm.py corrected_reaction_postprocess.py plot_clean_tm_results.py tests\test_history_mode_controls.py tests\test_corrected_reaction_postprocess.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

Results:

- New focused test failed before the code change: `assert False is True`.
- Same focused test passed after the code change.
- Full local example test suite passed: `23 passed, 8 warnings`.
- `py_compile` passed.

## Interpretation

The corrected stress-strain curve is now part of the normal training completion path for this example. A new run should no longer end with `reaction_metric_unavailable` merely because checkpoints were not saved.

The legacy top-boundary sigma reaction remains a diagnostic metric. The primary stress-strain source is still:

`nominal_stress_energy_exact_MPa`

## Limitations

This smoke run verifies the software pathway, not physical validation. It confirms that default training writes checkpoints and that the corrected reaction postprocessor can compute an energy-exact curve from them. It does not make a new claim about physical softening across full D0020 or multi-seed runs.
