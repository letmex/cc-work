"""Constant-conductivity heat PDE utilities for Phase 1 patch tests.

All heat-transfer quantities in this module use SI units. The mechanics mesh in
this project is stored in mm, so coordinate inputs must be converted explicitly
before heat gradients are interpreted as derivatives per meter.

The strong-form residual utilities are diagnostics and patch-test sanity checks.
The future main heat loss route should use the thermal functional / weak-form
layer.
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


def _is_constant_tensor(values):
    if values.ndim == 0 or values.numel() <= 1:
        return True
    return bool(torch.all(values == values.reshape(-1)[0]).item())


def _grad(outputs, inputs, create_graph=True, allow_constant=False):
    if not torch.is_tensor(outputs):
        outputs = torch.as_tensor(outputs, device=inputs.device, dtype=inputs.dtype)
    if outputs.ndim == 0:
        outputs = torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype) + outputs
    if not inputs.requires_grad:
        if allow_constant and _is_constant_tensor(outputs):
            return torch.zeros_like(inputs)
        raise ValueError("inputs must have requires_grad=True for nonconstant derivative calls")
    if not outputs.requires_grad:
        if allow_constant and _is_constant_tensor(outputs):
            return torch.zeros_like(inputs)
        raise ValueError(
            "outputs must have requires_grad=True unless allow_constant=True is used for a constant field; "
            "check for detached nonconstant temperature values"
        )
    grad_outputs = torch.ones_like(outputs)
    try:
        grad = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]
    except RuntimeError as exc:
        if allow_constant and "does not require grad" in str(exc):
            return torch.zeros_like(inputs)
        raise ValueError("autograd could not connect outputs to inputs for this derivative call") from exc
    return grad


def temperature_gradient_m(temperature_K, coords, coordinate_unit="mm"):
    """Return grad(T) in K/m for coordinates supplied in mm or m."""
    coords = torch.as_tensor(coords)
    temperature_K = torch.as_tensor(temperature_K, device=coords.device, dtype=coords.dtype)
    scale_to_m = _coordinate_scale_to_m(coordinate_unit)
    grad_native = _grad(temperature_K, coords, create_graph=True, allow_constant=True)
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
        component_grad_native = _grad(vector_field[:, dim], coords, create_graph=True, allow_constant=True)
        divergence = divergence + component_grad_native[:, dim] / _as_like(scale_to_m, coords[:, dim])
    return divergence


def steady_heat_residual_W_per_m3(
    temperature_K,
    coords,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    heat_source_Q_W_per_m3=DEFAULT_HEAT_SOURCE_Q_W_PER_M3,
    coordinate_unit="mm",
):
    """Return diagnostic strong-form residual -div(k0 grad T) - Q in W/m^3.

    This residual is retained for patch-test diagnostics and sign/unit sanity
    checks. Future heat loss implementation should use the thermal functional /
    weak-form route instead of treating this strong-form residual as primary.
    """
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
    """Return diagnostic transient residual rho*c*dTdt - div(k0 grad T) - Q.

    Units are W/m^3. This strong-form utility is for analytical diagnostics and
    patch tests; it is not the intended primary future training loss route.
    """
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


def steady_thermal_energy_density_J_per_m3(
    temperature_K,
    coords,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    heat_source_Q_W_per_m3=DEFAULT_HEAT_SOURCE_Q_W_PER_M3,
    coordinate_unit="mm",
):
    """Return constant-k0 steady thermal functional density at sample points.

    The computed expression is `0.5 * k0 * |grad_m T|^2 - Q * T`. With
    `k0` in W/m/K, `grad_m T` in K/m, `Q` in W/m^3, and `T` in K, this is the
    steady variational thermal-power density under the temperature-test-function
    convention. The historical function name keeps `J_per_m3` for loss-scaling
    compatibility, but the steady expression should be interpreted as W/m^3
    density, or a J/m^3-equivalent only after a chosen time/load scaling.
    """
    coords = torch.as_tensor(coords)
    temperature_K = torch.as_tensor(temperature_K, device=coords.device, dtype=coords.dtype)
    grad_T = temperature_gradient_m(temperature_K, coords, coordinate_unit=coordinate_unit)
    k0 = _as_like(thermal_k0_W_per_mK, grad_T)
    source = _as_like(heat_source_Q_W_per_m3, temperature_K)
    grad_energy = 0.5 * k0 * torch.sum(grad_T * grad_T, dim=1)
    return grad_energy - source * temperature_K


def transient_thermal_incremental_energy_density_J_per_m3(
    temperature_K,
    previous_temperature_K,
    coords,
    dt_s,
    thermal_rho_kg_per_m3=DEFAULT_THERMAL_RHO_KG_PER_M3,
    thermal_c_J_per_kgK=DEFAULT_THERMAL_C_J_PER_KGK,
    thermal_k0_W_per_mK=DEFAULT_THERMAL_K0_W_PER_MK,
    heat_source_Q_W_per_m3=DEFAULT_HEAT_SOURCE_Q_W_PER_M3,
    coordinate_unit="mm",
):
    """Return backward-Euler-style incremental thermal functional density.

    The density is
    `rho*c/(2*dt) * (T - T_prev)^2 + 0.5*k0*|grad_m T|^2 - Q*T`.
    The storage and conduction/source terms are expressed as thermal-power
    density for the incremental variational statement, using SI units:
    `rho` kg/m^3, `c` J/kg/K, `dt` s, `k0` W/m/K, and `Q` W/m^3.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0 for transient thermal incremental energy density")
    coords = torch.as_tensor(coords)
    temperature_K = torch.as_tensor(temperature_K, device=coords.device, dtype=coords.dtype)
    previous_temperature_K = torch.as_tensor(previous_temperature_K, device=coords.device, dtype=coords.dtype)
    steady_density = steady_thermal_energy_density_J_per_m3(
        temperature_K,
        coords,
        thermal_k0_W_per_mK=thermal_k0_W_per_mK,
        heat_source_Q_W_per_m3=heat_source_Q_W_per_m3,
        coordinate_unit=coordinate_unit,
    )
    rho_c = _as_like(thermal_rho_kg_per_m3, steady_density) * _as_like(thermal_c_J_per_kgK, steady_density)
    dt = _as_like(dt_s, steady_density)
    storage_density = rho_c / (2.0 * dt) * (temperature_K - previous_temperature_K) ** 2
    return storage_density + steady_density


def mean_thermal_functional_density(functional_density):
    """Return the pointwise mean of a thermal functional density for patch tests.

    This is not mesh quadrature or a domain integral. It only averages supplied
    sample-point densities for focused analytical tests.
    """
    return torch.as_tensor(functional_density).mean()
