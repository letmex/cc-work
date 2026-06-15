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
