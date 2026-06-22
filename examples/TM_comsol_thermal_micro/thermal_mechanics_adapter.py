"""Element-centroid adapter from solved-temperature sources to mechanics inputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

import thermal_solution_bridge


def _as_nodes_mm(nodes_mm):
    nodes_mm = torch.as_tensor(nodes_mm)
    if nodes_mm.ndim != 2 or nodes_mm.shape[1] != 2:
        raise ValueError("nodes_mm must have shape (n_nodes, 2)")
    return nodes_mm


def _as_triangle_elements(elements, nodes_mm):
    if elements is None:
        raise ValueError("elements is required to evaluate element-centroid temperatures")
    elements = torch.as_tensor(elements, device=nodes_mm.device, dtype=torch.long)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements must have shape (n_elements, 3)")
    if elements.numel() > 0:
        min_index = int(torch.min(elements).detach().cpu())
        max_index = int(torch.max(elements).detach().cpu())
        if min_index < 0 or max_index >= nodes_mm.shape[0]:
            raise ValueError("elements contain node indices outside nodes_mm")
    return elements


def _source_options(source_config, source_kwargs):
    if source_config is None:
        options = {}
    elif isinstance(source_config, Mapping):
        options = dict(source_config)
    else:
        raise TypeError("source_config must be a mapping or None")
    options.update(source_kwargs)
    return options


def element_centroid_coords_mm(nodes_mm, elements):
    """Return triangle element centroid coordinates in the same units as nodes_mm."""
    nodes_mm = _as_nodes_mm(nodes_mm)
    elements = _as_triangle_elements(elements, nodes_mm)
    return nodes_mm[elements].mean(dim=1)


def build_element_thermal_delta_T_from_bridge(nodes_mm, elements, source_config=None, **source_kwargs):
    """Evaluate bridge temperature sources at mechanics element centroids."""
    centroids_mm = element_centroid_coords_mm(nodes_mm, elements)
    result = thermal_solution_bridge.build_thermal_strain_input_from_temperature(
        centroids_mm,
        **_source_options(source_config, source_kwargs),
    )
    result = dict(result)
    result["evaluation_location"] = "element_centroid"
    result["element_centroid_coords_mm"] = centroids_mm
    return result


def build_mechanics_thermal_kwargs_from_bridge(nodes_mm, elements, source_config=None, **source_kwargs):
    """Return thermal keyword arguments for compute_mixed_tm_fields plus diagnostics."""
    diagnostics = build_element_thermal_delta_T_from_bridge(
        nodes_mm,
        elements,
        source_config=source_config,
        **source_kwargs,
    )
    return {
        "thermal_kwargs": {"thermal_delta_T": diagnostics["thermal_delta_T"]},
        "diagnostics": diagnostics,
    }
