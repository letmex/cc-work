"""Independent heat-only weak-form patch trainer for constant-k0 cases."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import heat_pde
import thermal_field
import thermal_quadrature


PRIMARY_LOSS = "thermal_functional_area_weighted_mean"
USED_STRONG_RESIDUAL_AS_PRIMARY_LOSS = False


class HeatOnlyMLP(nn.Module):
    """Small deterministic MLP for heat-only patch cases."""

    def __init__(self, width=16, dtype=torch.float64, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        self.to(device=device, dtype=dtype)
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, coords):
        return self.net(coords)


def build_rectangular_tri_mesh(
    x_min_mm=0.0,
    x_max_mm=0.01,
    y_min_mm=0.0,
    y_max_mm=0.01,
    nx=4,
    ny=4,
    dtype=torch.float64,
    device=None,
):
    """Build a structured rectangular triangle mesh with coordinates in mm."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    xs = torch.linspace(x_min_mm, x_max_mm, nx + 1, dtype=dtype, device=device)
    ys = torch.linspace(y_min_mm, y_max_mm, ny + 1, dtype=dtype, device=device)
    nodes = []
    for y in ys:
        for x in xs:
            nodes.append(torch.stack((x, y)))
    nodes_mm = torch.stack(nodes, dim=0)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elems.append((n0, n1, n3))
            elems.append((n0, n3, n2))
    elements = torch.tensor(elems, dtype=torch.long, device=device)
    bounds_mm = torch.tensor([[x_min_mm, x_max_mm], [y_min_mm, y_max_mm]], dtype=dtype, device=device)
    return {"nodes_mm": nodes_mm, "elements": elements, "bounds_mm": bounds_mm}


def _exact_top_bottom(coords_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm):
    eta = thermal_field.thermal_eta_y(coords_mm, y_min_mm, y_max_mm)
    return (1.0 - eta) * T_bottom_K + eta * T_top_K


def _evaluate_temperature(case_id, coords_mm, net, bounds_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm):
    if case_id == "H1":
        return thermal_field.evaluate_top_bottom_dirichlet_temperature(
            coords_mm,
            net,
            bounds_mm,
            T_bottom_K=T_bottom_K,
            T_top_K=T_top_K,
            y_min_mm=y_min_mm,
            y_max_mm=y_max_mm,
        )
    if case_id == "H2":
        return thermal_field.evaluate_bottom_dirichlet_temperature(
            coords_mm,
            net,
            bounds_mm,
            T_bottom_K=T_bottom_K,
        )
    raise ValueError("case_id must be 'H1' or 'H2'")


def _exact_temperature(case_id, coords_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm):
    if case_id == "H1":
        return _exact_top_bottom(coords_mm, T_bottom_K, T_top_K, y_min_mm, y_max_mm)
    if case_id == "H2":
        return torch.zeros(coords_mm.shape[0], device=coords_mm.device, dtype=coords_mm.dtype) + T_bottom_K
    raise ValueError("case_id must be 'H1' or 'H2'")


def _loss_on_centroids(case_id, net, centroid_coords_mm, area_m2, bounds_mm, T_bottom_K, T_top_K):
    y_min_mm = float(bounds_mm[1, 0].detach().cpu())
    y_max_mm = float(bounds_mm[1, 1].detach().cpu())
    temperature_K = _evaluate_temperature(
        case_id,
        centroid_coords_mm,
        net,
        bounds_mm,
        T_bottom_K,
        T_top_K,
        y_min_mm,
        y_max_mm,
    )
    density = heat_pde.steady_thermal_energy_density_J_per_m3(
        temperature_K,
        centroid_coords_mm,
        thermal_k0_W_per_mK=heat_pde.DEFAULT_THERMAL_K0_W_PER_MK,
        heat_source_Q_W_per_m3=0.0,
        coordinate_unit="mm",
    )
    return thermal_quadrature.area_weighted_mean_density(density, area_m2), density, temperature_K


def _boundary_coords(bounds_mm, count=9):
    dtype = bounds_mm.dtype
    device = bounds_mm.device
    xs = torch.linspace(bounds_mm[0, 0], bounds_mm[0, 1], count, dtype=dtype, device=device)
    ys = torch.linspace(bounds_mm[1, 0], bounds_mm[1, 1], count, dtype=dtype, device=device)
    bottom = torch.stack((xs, torch.zeros_like(xs) + bounds_mm[1, 0]), dim=1)
    top = torch.stack((xs, torch.zeros_like(xs) + bounds_mm[1, 1]), dim=1)
    left = torch.stack((torch.zeros_like(ys) + bounds_mm[0, 0], ys), dim=1)
    right = torch.stack((torch.zeros_like(ys) + bounds_mm[0, 1], ys), dim=1)
    return bottom, top, left, right


def _max_abs_top_side_flux(case_id, net, bounds_mm, T_bottom_K, T_top_K):
    if case_id != "H2":
        return math.nan
    bottom, top, left, right = _boundary_coords(bounds_mm)
    coords = torch.cat((top, left, right), dim=0).detach().clone().requires_grad_(True)
    normals = torch.cat(
        (
            torch.tensor([[0.0, 1.0]], dtype=bounds_mm.dtype, device=bounds_mm.device).repeat(top.shape[0], 1),
            torch.tensor([[-1.0, 0.0]], dtype=bounds_mm.dtype, device=bounds_mm.device).repeat(left.shape[0], 1),
            torch.tensor([[1.0, 0.0]], dtype=bounds_mm.dtype, device=bounds_mm.device).repeat(right.shape[0], 1),
        ),
        dim=0,
    )
    temperature = _evaluate_temperature(
        case_id,
        coords,
        net,
        bounds_mm,
        T_bottom_K,
        T_top_K,
        float(bounds_mm[1, 0].detach().cpu()),
        float(bounds_mm[1, 1].detach().cpu()),
    )
    flux = heat_pde.normal_heat_flux_W_per_m2(temperature, coords, normals, coordinate_unit="mm")
    return float(torch.max(torch.abs(flux)).detach().cpu())


def _diagnostics(case_id, net, mesh, loss_trace, T_bottom_K, T_top_K):
    nodes_mm = mesh["nodes_mm"]
    elements = mesh["elements"]
    bounds_mm = mesh["bounds_mm"]
    area_m2 = thermal_quadrature.triangle_areas_m2(nodes_mm, elements)
    centroids = thermal_quadrature.triangle_centroids_mm(nodes_mm, elements).detach().clone().requires_grad_(True)

    final_loss, _density, temperature = _loss_on_centroids(
        case_id,
        net,
        centroids,
        area_m2,
        bounds_mm,
        T_bottom_K,
        T_top_K,
    )
    exact = _exact_temperature(
        case_id,
        centroids.detach(),
        T_bottom_K,
        T_top_K,
        float(bounds_mm[1, 0].detach().cpu()),
        float(bounds_mm[1, 1].detach().cpu()),
    )
    err = temperature.detach() - exact

    residual = heat_pde.steady_heat_residual_W_per_m3(temperature, centroids, coordinate_unit="mm")
    grad_T = heat_pde.temperature_gradient_m(temperature, centroids, coordinate_unit="mm")
    grad_norm = torch.linalg.norm(grad_T, dim=1)

    bottom, top, _left, _right = _boundary_coords(bounds_mm)
    with torch.no_grad():
        bottom_T = _evaluate_temperature(
            case_id,
            bottom,
            net,
            bounds_mm,
            T_bottom_K,
            T_top_K,
            float(bounds_mm[1, 0].detach().cpu()),
            float(bounds_mm[1, 1].detach().cpu()),
        )
        top_T = _evaluate_temperature(
            case_id,
            top,
            net,
            bounds_mm,
            T_bottom_K,
            T_top_K,
            float(bounds_mm[1, 0].detach().cpu()),
            float(bounds_mm[1, 1].detach().cpu()),
        )

    out = {
        "case_id": case_id,
        "primary_loss": PRIMARY_LOSS,
        "used_strong_residual_as_primary_loss": USED_STRONG_RESIDUAL_AS_PRIMARY_LOSS,
        "final_area_weighted_functional_loss": float(final_loss.detach().cpu()),
        "loss_trace": [float(v) for v in loss_trace],
        "max_abs_temperature_error_K": float(torch.max(torch.abs(err)).detach().cpu()),
        "l2_temperature_error_K": float(torch.sqrt(torch.mean(err * err)).detach().cpu()),
        "bottom_boundary_max_abs_error_K": float(torch.max(torch.abs(bottom_T - T_bottom_K)).detach().cpu()),
        "max_abs_strong_residual_W_per_m3": float(torch.max(torch.abs(residual)).detach().cpu()),
        "max_temperature_gradient_K_per_m": float(torch.max(grad_norm).detach().cpu()),
    }
    if case_id == "H1":
        out["top_boundary_max_abs_error_K"] = float(torch.max(torch.abs(top_T - T_top_K)).detach().cpu())
    if case_id == "H2":
        out["max_abs_top_side_flux_W_per_m2"] = _max_abs_top_side_flux(case_id, net, bounds_mm, T_bottom_K, T_top_K)
    return out


def run_steady_heat_patch_case(
    case_id,
    nx=4,
    ny=4,
    num_epochs=25,
    learning_rate=1.0e-4,
    dtype=torch.float64,
    device=None,
):
    """Run a small independent heat-only weak-form patch case."""
    torch.manual_seed(20260630)
    mesh = build_rectangular_tri_mesh(nx=nx, ny=ny, dtype=dtype, device=device)
    bounds_mm = mesh["bounds_mm"]
    nodes_mm = mesh["nodes_mm"]
    elements = mesh["elements"]
    area_m2 = thermal_quadrature.triangle_areas_m2(nodes_mm, elements)
    centroid_base = thermal_quadrature.triangle_centroids_mm(nodes_mm, elements)
    T_bottom_K = 300.0
    T_top_K = 320.0 if case_id == "H1" else 300.0
    net = HeatOnlyMLP(dtype=dtype, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_trace = []

    for _epoch in range(num_epochs):
        centroid_coords = centroid_base.detach().clone().requires_grad_(True)
        loss, _density, _temperature = _loss_on_centroids(
            case_id,
            net,
            centroid_coords,
            area_m2,
            bounds_mm,
            T_bottom_K,
            T_top_K,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_trace.append(float(loss.detach().cpu()))

    if num_epochs == 0:
        centroid_coords = centroid_base.detach().clone().requires_grad_(True)
        loss, _density, _temperature = _loss_on_centroids(
            case_id,
            net,
            centroid_coords,
            area_m2,
            bounds_mm,
            T_bottom_K,
            T_top_K,
        )
        loss_trace.append(float(loss.detach().cpu()))

    return _diagnostics(case_id, net, mesh, loss_trace, T_bottom_K, T_top_K)


def run_quadrature_sanity_case(
    x_min_mm=0.0,
    x_max_mm=0.02,
    y_min_mm=0.0,
    y_max_mm=0.01,
    constant_density=1.0,
    dtype=torch.float64,
    device=None,
):
    """Check triangle-area integration for a constant density rectangle."""
    mesh = build_rectangular_tri_mesh(
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        y_min_mm=y_min_mm,
        y_max_mm=y_max_mm,
        nx=2,
        ny=1,
        dtype=dtype,
        device=device,
    )
    area_m2 = thermal_quadrature.triangle_areas_m2(mesh["nodes_mm"], mesh["elements"])
    density = torch.zeros_like(area_m2) + constant_density
    integral = thermal_quadrature.integrate_density_2d(density, area_m2)
    mean = thermal_quadrature.area_weighted_mean_density(density, area_m2)
    rectangle_area_m2 = (x_max_mm - x_min_mm) * heat_pde.MM_TO_M * (y_max_mm - y_min_mm) * heat_pde.MM_TO_M
    return {
        "integral": float(integral.detach().cpu()),
        "area_weighted_mean_density": float(mean.detach().cpu()),
        "rectangle_area_m2": float(rectangle_area_m2),
        "total_triangle_area_m2": float(torch.sum(area_m2).detach().cpu()),
    }


def run_transient_storage_sanity_case(delta_T_K=0.0, dt_s=2.0, dtype=torch.float64, device=None):
    """Evaluate transient incremental density for a uniform temperature increment."""
    coords_mm = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    previous_temperature = coords_mm[:, 0] * 0.0 + 300.0
    temperature = previous_temperature + delta_T_K
    density = heat_pde.transient_thermal_incremental_energy_density_J_per_m3(
        temperature,
        previous_temperature,
        coords_mm,
        dt_s=dt_s,
        coordinate_unit="mm",
    )
    return {
        "delta_T_K": float(delta_T_K),
        "dt_s": float(dt_s),
        "mean_incremental_density": float(torch.mean(density).detach().cpu()),
    }


def save_heat_only_diagnostics(result, output_path):
    """Write one heat-only diagnostics dictionary as a simple CSV file."""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("metric,value\n")
        for key, value in result.items():
            if key == "loss_trace":
                file.write(f"{key},{';'.join(str(v) for v in value)}\n")
            else:
                file.write(f"{key},{value}\n")
