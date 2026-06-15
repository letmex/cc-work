"""Triangle-area quadrature helpers for heat-only weak-form patch tests."""

from __future__ import annotations

import torch

import heat_pde


def _as_tensor(value):
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def triangle_centroids_mm(nodes_mm, elements):
    """Return triangle centroid coordinates in mm."""
    nodes_mm = _as_tensor(nodes_mm)
    elements = _as_tensor(elements).to(device=nodes_mm.device, dtype=torch.long)
    return nodes_mm[elements].mean(dim=1)


def triangle_areas_m2(nodes_mm, elements):
    """Return positive triangle areas in m^2 for nodal coordinates supplied in mm."""
    nodes_mm = _as_tensor(nodes_mm)
    elements = _as_tensor(elements).to(device=nodes_mm.device, dtype=torch.long)
    tri_m = heat_pde.coords_mm_to_m(nodes_mm)[elements]
    edge_a = tri_m[:, 1, :] - tri_m[:, 0, :]
    edge_b = tri_m[:, 2, :] - tri_m[:, 0, :]
    cross_z = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    return 0.5 * torch.abs(cross_z)


def integrate_density_2d(density, area_m2):
    """Return sum(area * density) without any thickness factor."""
    density = _as_tensor(density)
    area_m2 = _as_tensor(area_m2).to(device=density.device, dtype=density.dtype)
    return torch.sum(area_m2 * density)


def area_weighted_mean_density(density, area_m2):
    """Return sum(area * density) / sum(area)."""
    density = _as_tensor(density)
    area_m2 = _as_tensor(area_m2).to(device=density.device, dtype=density.dtype)
    total_area = torch.sum(area_m2)
    if float(total_area.detach().cpu()) <= 0.0:
        raise ValueError("total quadrature area must be positive")
    return integrate_density_2d(density, area_m2) / total_area


def triangle_quadrature_points_mm(nodes_mm, elements, rule="centroid"):
    """Return triangle quadrature points in mm and corresponding m^2 weights.

    Supported rules are:

    - `centroid`: one point per triangle with full triangle area as weight.
    - `triangle_3point`: standard degree-2 barycentric rule with each sample
      carrying one third of the triangle area.
    """
    nodes_mm = _as_tensor(nodes_mm)
    elements = _as_tensor(elements).to(device=nodes_mm.device, dtype=torch.long)
    tri = nodes_mm[elements]
    area_m2 = triangle_areas_m2(nodes_mm, elements)

    if rule == "centroid":
        return tri.mean(dim=1), area_m2

    if rule == "triangle_3point":
        bary = torch.tensor(
            [
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            ],
            dtype=nodes_mm.dtype,
            device=nodes_mm.device,
        )
        points = torch.einsum("qa,eac->eqc", bary, tri).reshape(-1, 2)
        weights = (area_m2[:, None].expand(-1, 3) / 3.0).reshape(-1)
        return points, weights

    raise ValueError("rule must be 'centroid' or 'triangle_3point'")


def integrate_density_with_weights(density, weights_m2):
    """Return sum(weight * density) for quadrature weights in m^2."""
    density = _as_tensor(density)
    weights_m2 = _as_tensor(weights_m2).to(device=density.device, dtype=density.dtype)
    return torch.sum(weights_m2 * density)


def weighted_mean_density(density, weights_m2):
    """Return sum(weight * density) / sum(weight)."""
    density = _as_tensor(density)
    weights_m2 = _as_tensor(weights_m2).to(device=density.device, dtype=density.dtype)
    total_weight = torch.sum(weights_m2)
    if float(total_weight.detach().cpu()) <= 0.0:
        raise ValueError("total quadrature weight must be positive")
    return integrate_density_with_weights(density, weights_m2) / total_weight
