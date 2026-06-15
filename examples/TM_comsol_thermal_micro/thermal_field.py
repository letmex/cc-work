"""Solved temperature field ansatz helpers for heat-only patch cases."""

from __future__ import annotations

import torch


def _as_tensor_like(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _as_coords(coords_mm):
    coords_mm = torch.as_tensor(coords_mm)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_points, 2)")
    return coords_mm


def _as_bounds(bounds_mm, coords_mm):
    bounds_mm = torch.as_tensor(bounds_mm, device=coords_mm.device, dtype=coords_mm.dtype)
    if bounds_mm.shape != (2, 2):
        raise ValueError("bounds_mm must have shape (2, 2): [[x_min, x_max], [y_min, y_max]]")
    spans = bounds_mm[:, 1] - bounds_mm[:, 0]
    if torch.any(spans <= 0.0):
        raise ValueError("bounds_mm spans must be positive")
    return bounds_mm


def _net_values(net, normalized_coords):
    values = net(normalized_coords)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim == 1:
        return values
    raise ValueError("temperature net must return shape (n_points,) or (n_points, 1)")


def normalize_coords_for_heat_net(coords_mm, bounds_mm):
    """Map mm coordinates in rectangular bounds to the network interval [-1, 1]."""
    coords_mm = _as_coords(coords_mm)
    bounds_mm = _as_bounds(bounds_mm, coords_mm)
    lower = bounds_mm[:, 0]
    upper = bounds_mm[:, 1]
    return 2.0 * (coords_mm - lower) / (upper - lower) - 1.0


def thermal_eta_y(coords_mm, y_min_mm, y_max_mm):
    """Return eta = (y - y_min) / (y_max - y_min)."""
    coords_mm = _as_coords(coords_mm)
    y_min = _as_tensor_like(y_min_mm, coords_mm)
    y_max = _as_tensor_like(y_max_mm, coords_mm)
    span = y_max - y_min
    if float(span.detach().cpu()) <= 0.0:
        raise ValueError("y_max_mm must be greater than y_min_mm")
    return (coords_mm[:, 1] - y_min) / span


def thermal_bubble_y(coords_mm, y_min_mm, y_max_mm):
    """Return the y-direction bubble eta * (1 - eta)."""
    eta = thermal_eta_y(coords_mm, y_min_mm, y_max_mm)
    return eta * (1.0 - eta)


def bottom_dirichlet_temperature_lift(coords_mm, T_bottom_K):
    """Return the constant lift that satisfies the bottom temperature."""
    coords_mm = _as_coords(coords_mm)
    return torch.zeros(coords_mm.shape[0], device=coords_mm.device, dtype=coords_mm.dtype) + _as_tensor_like(
        T_bottom_K, coords_mm
    )


def top_bottom_dirichlet_temperature_lift(coords_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm):
    """Return the linear top-bottom temperature lift."""
    coords_mm = _as_coords(coords_mm)
    eta = thermal_eta_y(coords_mm, y_min_mm, y_max_mm)
    bottom = _as_tensor_like(T_bottom_K, coords_mm)
    top = _as_tensor_like(T_top_K, coords_mm)
    return (1.0 - eta) * bottom + eta * top


def evaluate_bottom_dirichlet_temperature(coords_mm, net, bounds_mm, T_bottom_K):
    """Evaluate T = T_bottom + (y - y_min) * N_T(x_hat, y_hat)."""
    coords_mm = _as_coords(coords_mm)
    bounds_mm = _as_bounds(bounds_mm, coords_mm)
    normalized = normalize_coords_for_heat_net(coords_mm, bounds_mm)
    net_values = _net_values(net, normalized)
    y_min = bounds_mm[1, 0]
    lift = bottom_dirichlet_temperature_lift(coords_mm, T_bottom_K)
    return lift + (coords_mm[:, 1] - y_min) * net_values


def evaluate_top_bottom_dirichlet_temperature(
    coords_mm,
    net,
    bounds_mm,
    T_bottom_K,
    T_top_K,
    y_min_mm,
    y_max_mm,
):
    """Evaluate T = T_lift + eta*(1-eta)*N_T(x_hat, y_hat)."""
    coords_mm = _as_coords(coords_mm)
    bounds_mm = _as_bounds(bounds_mm, coords_mm)
    normalized = normalize_coords_for_heat_net(coords_mm, bounds_mm)
    lift = top_bottom_dirichlet_temperature_lift(coords_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm)
    bubble = thermal_bubble_y(coords_mm, y_min_mm, y_max_mm)
    return lift + bubble * _net_values(net, normalized)
