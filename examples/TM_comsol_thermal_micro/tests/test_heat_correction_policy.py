from pathlib import Path
import inspect
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_heat_only  # noqa: E402


DTYPE = torch.float64


def test_h1_default_uses_frozen_lift_with_zero_correction_and_residual():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H1",
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["correction_policy"] == "frozen_lift"
    assert result["is_correction_trainable"] is False
    assert result["primary_loss"] == train_heat_only.PRIMARY_LOSS
    assert result["used_strong_residual_as_primary_loss"] is False
    assert result["max_abs_temperature_error_K"] < 1.0e-10
    assert result["l2_temperature_error_K"] < 1.0e-10
    assert result["bottom_boundary_max_abs_error_K"] < 1.0e-12
    assert result["top_boundary_max_abs_error_K"] < 1.0e-12
    assert result["max_abs_correction_K"] < 1.0e-12
    assert result["l2_correction_K"] < 1.0e-12
    assert result["max_abs_strong_residual_W_per_m3"] < 1.0e-4
    assert result["normalized_residual_max"] < 1.0e-12


def test_h1_trainable_correction_override_preserves_old_diagnostic_behavior():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H1",
        correction_policy="trainable_correction",
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["correction_policy"] == "trainable_correction"
    assert result["is_correction_trainable"] is True
    assert result["used_strong_residual_as_primary_loss"] is False
    assert result["max_abs_correction_K"] > 0.0
    assert result["l2_correction_K"] > 0.0


def test_regularized_correction_is_explicit_opt_in_not_h1_default():
    default_result = train_heat_only.run_steady_heat_patch_case(
        case_id="H1",
        nx=2,
        ny=2,
        num_epochs=0,
        dtype=DTYPE,
    )
    regularized = train_heat_only.run_steady_heat_patch_case(
        case_id="H1",
        correction_policy="regularized_correction",
        regularization_weight=1.0e12,
        nx=2,
        ny=2,
        num_epochs=2,
        dtype=DTYPE,
    )

    assert default_result["correction_policy"] == "frozen_lift"
    assert regularized["correction_policy"] == "regularized_correction"
    assert regularized["regularization"] == "correction_l2"
    assert regularized["is_correction_trainable"] is True


def test_h2_default_uses_frozen_zero_correction_with_zero_residual_and_flux():
    result = train_heat_only.run_steady_heat_patch_case(
        case_id="H2",
        nx=4,
        ny=4,
        num_epochs=8,
        dtype=DTYPE,
    )

    assert result["correction_policy"] == "frozen_lift"
    assert result["is_correction_trainable"] is False
    assert result["max_abs_temperature_error_K"] < 1.0e-12
    assert result["l2_temperature_error_K"] < 1.0e-12
    assert result["bottom_boundary_max_abs_error_K"] < 1.0e-12
    assert result["max_abs_correction_K"] < 1.0e-12
    assert result["l2_correction_K"] < 1.0e-12
    assert result["max_abs_strong_residual_W_per_m3"] < 1.0e-4
    assert result["normalized_residual_max"] < 1.0e-12
    assert result["max_abs_top_side_flux_W_per_m2"] < 1.0e-12


def test_policy_comparison_reports_required_heat_only_cases():
    results = train_heat_only.run_correction_policy_comparison(nx=4, ny=4, dtype=DTYPE)

    assert set(results) >= {
        "H1_frozen_lift",
        "H1_trainable_correction",
        "H1_regularized_correction",
        "H2_frozen_lift",
    }
    assert results["H1_frozen_lift"]["correction_policy"] == "frozen_lift"
    assert results["H1_trainable_correction"]["correction_policy"] == "trainable_correction"
    assert results["H1_regularized_correction"]["correction_policy"] == "regularized_correction"
    assert results["H2_frozen_lift"]["correction_policy"] == "frozen_lift"
    assert results["H1_frozen_lift"]["max_abs_correction_K"] < 1.0e-12
    assert results["H1_trainable_correction"]["max_abs_correction_K"] > 0.0


def test_heat_policy_api_has_no_damage_dependent_conductivity_inputs_or_tokens():
    public_functions = [
        train_heat_only.run_steady_heat_patch_case,
        train_heat_only.run_h1_quadrature_diagnostic_case,
        train_heat_only.run_correction_policy_comparison,
    ]
    forbidden_parameter_names = {"alpha", "damage", "d", "g_d", "k_d"}

    for fn in public_functions:
        assert not (set(inspect.signature(fn).parameters) & forbidden_parameter_names)

    source = Path(train_heat_only.__file__).read_text(encoding="utf-8")
    forbidden_tokens = [
        "strong_residual_loss",
        "train_mixed_tm",
        "compute_energy_mixed_tm",
        "k(d)",
        "g(d)",
        "damage_dependent_conductivity",
    ]
    for token in forbidden_tokens:
        assert token not in source


def test_invalid_correction_policy_is_rejected():
    with pytest.raises(ValueError, match="correction_policy"):
        train_heat_only.run_steady_heat_patch_case(
            case_id="H1",
            correction_policy="not_a_policy",
            nx=2,
            ny=2,
            num_epochs=0,
            dtype=DTYPE,
        )
