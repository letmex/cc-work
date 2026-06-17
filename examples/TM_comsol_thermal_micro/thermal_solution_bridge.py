"""Bridge solved-temperature fields into the prescribed thermal-strain route."""

from __future__ import annotations

import torch

import thermal_field
from thermal_prescribed import DEFAULT_TREF_K, delta_T_from_temperature, thermal_strain_2d


DEFAULT_TEMPERATURE_SOURCE_MODE = "prescribed_uniform"
FROZEN_LIFT_POLICY = "frozen_lift"


def _as_coords(coords_mm):
    coords_mm = torch.as_tensor(coords_mm)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_points, 2)")
    return coords_mm


def _as_like(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _as_bounds(bounds_mm, coords_mm):
    if bounds_mm is None:
        raise ValueError("bounds_mm is required for solved_frozen_lift")
    bounds_mm = torch.as_tensor(bounds_mm, device=coords_mm.device, dtype=coords_mm.dtype)
    if bounds_mm.shape != (2, 2):
        raise ValueError("bounds_mm must have shape (2, 2)")
    return bounds_mm


def _uniform_temperature(coords_mm, uniform_temperature_K, uniform_delta_T_K, T_ref_K):
    if uniform_temperature_K is not None and uniform_delta_T_K is not None:
        raise ValueError("Specify either uniform_temperature_K or uniform_delta_T_K, not both")
    if uniform_temperature_K is None:
        delta_T = 0.0 if uniform_delta_T_K is None else uniform_delta_T_K
        uniform_temperature_K = _as_like(T_ref_K, coords_mm) + _as_like(delta_T, coords_mm)
    return torch.zeros(coords_mm.shape[0], device=coords_mm.device, dtype=coords_mm.dtype) + _as_like(
        uniform_temperature_K,
        coords_mm,
    )


def evaluate_temperature_source_at_coords_mm(
    coords_mm,
    source_mode=DEFAULT_TEMPERATURE_SOURCE_MODE,
    *,
    uniform_temperature_K=None,
    uniform_delta_T_K=None,
    case_id=None,
    bounds_mm=None,
    T_bottom_K=300.0,
    T_top_K=320.0,
    T_ref_K=DEFAULT_TREF_K,
):
    """Evaluate a supported temperature source at mechanics/material coordinates."""
    coords_mm = _as_coords(coords_mm)
    if source_mode == "prescribed_uniform":
        return _uniform_temperature(coords_mm, uniform_temperature_K, uniform_delta_T_K, T_ref_K)

    if source_mode == "solved_frozen_lift":
        bounds_mm = _as_bounds(bounds_mm, coords_mm)
        if case_id == "H1":
            return thermal_field.top_bottom_dirichlet_temperature_lift(
                coords_mm,
                T_bottom_K=T_bottom_K,
                T_top_K=T_top_K,
                y_min_mm=bounds_mm[1, 0],
                y_max_mm=bounds_mm[1, 1],
            )
        if case_id == "H2":
            return thermal_field.bottom_dirichlet_temperature_lift(coords_mm, T_bottom_K=T_bottom_K)
        raise ValueError("case_id must be 'H1' or 'H2' for solved_frozen_lift")

    raise ValueError("source_mode must be 'prescribed_uniform' or 'solved_frozen_lift'")


def temperature_increment_from_reference(temperature_K, T_ref_K=DEFAULT_TREF_K):
    """Return DeltaT = T - T_ref using the existing prescribed-temperature helper."""
    temperature_K = torch.as_tensor(temperature_K)
    return delta_T_from_temperature(temperature_K, Tref=_as_like(T_ref_K, temperature_K))


def build_thermal_strain_input_from_temperature(
    coords_mm,
    source_mode=DEFAULT_TEMPERATURE_SOURCE_MODE,
    *,
    uniform_temperature_K=None,
    uniform_delta_T_K=None,
    case_id=None,
    bounds_mm=None,
    T_bottom_K=300.0,
    T_top_K=320.0,
    T_ref_K=DEFAULT_TREF_K,
    alpha_T=None,
):
    """Return the temperature-derived input consumed by the mechanics route."""
    coords_mm = _as_coords(coords_mm)
    temperature_K = evaluate_temperature_source_at_coords_mm(
        coords_mm,
        source_mode=source_mode,
        uniform_temperature_K=uniform_temperature_K,
        uniform_delta_T_K=uniform_delta_T_K,
        case_id=case_id,
        bounds_mm=bounds_mm,
        T_bottom_K=T_bottom_K,
        T_top_K=T_top_K,
        T_ref_K=T_ref_K,
    )
    thermal_delta_T = temperature_increment_from_reference(temperature_K, T_ref_K=T_ref_K)
    strain = thermal_strain_2d(thermal_delta_T) if alpha_T is None else thermal_strain_2d(thermal_delta_T, alpha_T)
    correction_K = torch.zeros_like(temperature_K)
    result = {
        "source_mode": source_mode,
        "case_id": case_id,
        "temperature_K": temperature_K,
        "thermal_temperature": temperature_K,
        "thermal_delta_T": thermal_delta_T,
        "delta_T": thermal_delta_T,
        "thermal_eps_xx": strain["eps_xx"],
        "thermal_eps_yy": strain["eps_yy"],
        "thermal_eps_xy": strain["eps_xy"],
        "T_ref_K": _as_like(T_ref_K, temperature_K),
        "network_training_run": False,
        "correction_K": correction_K,
    }
    if source_mode == "solved_frozen_lift":
        result["correction_policy"] = FROZEN_LIFT_POLICY
    else:
        result["correction_policy"] = "not_applicable"
    return result
