"""Constant-conductivity heat PDE utilities for Phase 1 patch tests.

All heat-transfer quantities in this module use SI units. The mechanics mesh in
this project is stored in mm, so coordinate inputs must be converted explicitly
before heat gradients are interpreted as derivatives per meter.
"""

from __future__ import annotations

import torch


MM_TO_M = 1.0e-3
DEFAULT_THERMAL_RHO_KG_PER_M3 = 1040.0
DEFAULT_THERMAL_C_J_PER_KGK = 170.0
DEFAULT_THERMAL_K0_W_PER_MK = 418.0
DEFAULT_HEAT_SOURCE_Q_W_PER_M3 = 0.0


def _as_like(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def coords_mm_to_m(coords_mm):
    """Convert physical coordinates from mm to m."""
    return torch.as_tensor(coords_mm) * MM_TO_M


def _coordinate_scale_to_m(coordinate_unit):
    if coordinate_unit == "m":
        return 1.0
    if coordinate_unit == "mm":
        return MM_TO_M
    raise ValueError("coordinate_unit must be 'm' or 'mm'")


def _grad(outputs, inputs, create_graph=True):
    if not torch.is_tensor(outputs):
        outputs = torch.as_tensor(outputs, device=inputs.device, dtype=inputs.dtype)
    if outputs.ndim == 0:
        outputs = torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype) + outputs
    if not outputs.requires_grad:
        return torch.zeros_like(inputs)
    grad_outputs = torch.ones_like(outputs)
    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(inputs)
    return grad


def temperature_gradient_m(temperature_K, coords, coordinate_unit="mm"):
    """Return grad(T) in K/m for coordinates supplied in mm or m."""
    coords = torch.as_tensor(coords)
    temperature_K = torch.as_tensor(temperature_K, device=coords.device, dtype=coords.dtype)
    scale_to_m = _coordinate_scale_to_m(coordinate_unit)
    grad_native = _grad(temperature_K, coords, create_graph=True)
    return grad_native / _as_like(scale_to_m, coords)


def heat_flux_W_per_m2(
    temperature_K,
    coords,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    coordinate_unit="mm",
):
    """Return Fourier heat flux q = -k0 * grad(T) in W/m^2."""
    grad_T = temperature_gradient_m(temperature_K, coords, coordinate_unit=coordinate_unit)
    return -_as_like(thermal_k0_W_per_mK, grad_T) * grad_T


def divergence_m(vector_field, coords, coordinate_unit="mm"):
    """Return divergence of a vector field using derivatives per meter."""
    coords = torch.as_tensor(coords)
    vector_field = torch.as_tensor(vector_field, device=coords.device, dtype=coords.dtype)
    scale_to_m = _coordinate_scale_to_m(coordinate_unit)
    divergence = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    for dim in range(vector_field.shape[1]):
        component_grad_native = _grad(vector_field[:, dim], coords, create_graph=True)
        divergence = divergence + component_grad_native[:, dim] / _as_like(scale_to_m, coords[:, dim])
    return divergence


def steady_heat_residual_W_per_m3(
    temperature_K,
    coords,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    heat_source_Q_W_per_m3=DEFAULT_HEAT_SOURCE_Q_W_PER_M3,
    coordinate_unit="mm",
):
    """Return -div(k0 grad T) - Q for constant k0 in W/m^3."""
    flux = heat_flux_W_per_m2(
        temperature_K,
        coords,
        thermal_k0_W_per_mK=thermal_k0_W_per_mK,
        coordinate_unit=coordinate_unit,
    )
    residual_without_source = divergence_m(flux, coords, coordinate_unit=coordinate_unit)
    return residual_without_source - _as_like(heat_source_Q_W_per_m3, residual_without_source)


def transient_heat_residual_W_per_m3(
    temperature_K,
    coords,
    dTdt_K_per_s,
    thermal_rho_kg_per_m3=DEFAULT_THERMAL_RHO_KG_PER_M3,
    thermal_c_J_per_kgK=DEFAULT_THERMAL_C_J_PER_KGK,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    heat_source_Q_W_per_m3=DEFAULT_HEAT_SOURCE_Q_W_PER_M3,
    coordinate_unit="mm",
):
    """Return rho*c*dTdt - div(k0 grad T) - Q in W/m^3."""
    steady_residual = steady_heat_residual_W_per_m3(
        temperature_K,
        coords,
        thermal_k0_W_per_mK=thermal_k0_W_per_mK,
        heat_source_Q_W_per_m3=heat_source_Q_W_per_m3,
        coordinate_unit=coordinate_unit,
    )
    dTdt_K_per_s = torch.as_tensor(dTdt_K_per_s, device=steady_residual.device, dtype=steady_residual.dtype)
    volumetric_heat_capacity = _as_like(thermal_rho_kg_per_m3, steady_residual) * _as_like(
        thermal_c_J_per_kgK, steady_residual
    )
    return volumetric_heat_capacity * dTdt_K_per_s + steady_residual


def normal_heat_flux_W_per_m2(
    temperature_K,
    coords,
    normals,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    coordinate_unit="mm",
):
    """Return q dot n with q = -k0 * grad(T), in W/m^2."""
    flux = heat_flux_W_per_m2(
        temperature_K,
        coords,
        thermal_k0_W_per_mK=thermal_k0_W_per_mK,
        coordinate_unit=coordinate_unit,
    )
    normals = torch.as_tensor(normals, device=flux.device, dtype=flux.dtype)
    return torch.sum(flux * normals, dim=1)
