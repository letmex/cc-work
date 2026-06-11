"""Connectivity diagnostics for shear damage fields."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_TIP_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
RIGHT_BOUNDARY_BAND_MM = 2.5e-4
BOUNDARY_BAND_MM = 2.5e-4
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.8, 0.95)


def _as_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def field_paths(result_dir: Path) -> list[Path]:
    return sorted(Path(result_dir).glob("fields_mixed_tm_step_*.npz"), key=_as_step)


def edge_map(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for elem_idx, nodes in enumerate(np.asarray(triangles, dtype=int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges[tuple(sorted((int(a), int(b))))].append(int(elem_idx))
    return edges


def element_adjacency(triangles: np.ndarray) -> list[list[int]]:
    adjacency = [[] for _ in range(len(triangles))]
    for elems in edge_map(triangles).values():
        if len(elems) == 2:
            a, b = elems
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def connected_components(mask: np.ndarray, adjacency: list[list[int]]) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape[0], dtype=bool)
    components = []
    for seed in np.flatnonzero(mask):
        if visited[seed]:
            continue
        comp = np.zeros(mask.shape[0], dtype=bool)
        queue: deque[int] = deque([int(seed)])
        visited[seed] = True
        comp[seed] = True
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if mask[nxt] and not visited[nxt]:
                    visited[nxt] = True
                    comp[nxt] = True
                    queue.append(int(nxt))
        components.append(comp)
    return components


def notch_tip_seed_mask(
    x: np.ndarray,
    y: np.ndarray,
    notch_tip_x: float = NOTCH_TIP_X_MM,
    notch_tip_y: float = NOTCH_TIP_Y_MM,
    half_window: float = TIP_HALF_WINDOW_MM,
) -> np.ndarray:
    return (
        (x >= notch_tip_x - half_window)
        & (x <= notch_tip_x + half_window)
        & (np.abs(y - notch_tip_y) <= half_window)
    )


def _principal_direction_angle_deg(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    coords = np.column_stack([x - np.mean(x), y - np.mean(y)])
    if np.allclose(coords, 0.0):
        return math.nan
    cov = coords.T @ coords / max(coords.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    vector = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if angle > 90.0:
        angle -= 180.0
    if angle <= -90.0:
        angle += 180.0
    return angle


def triangle_areas(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if "x" not in data or "y" not in data or "triangles" not in data:
        return None
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    triangles = np.asarray(data["triangles"], dtype=int)
    if triangles.size == 0:
        return None
    x1 = x[triangles[:, 0]]
    y1 = y[triangles[:, 0]]
    x2 = x[triangles[:, 1]]
    y2 = y[triangles[:, 1]]
    x3 = x[triangles[:, 2]]
    y3 = y[triangles[:, 2]]
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _component_metrics(
    comp: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    total_count: int,
    right_boundary_x: float,
    right_boundary_band: float,
    boundary_band: float,
    areas: np.ndarray | None = None,
) -> dict[str, object]:
    if areas is not None and areas.shape[0] != total_count:
        areas = None
    if not np.any(comp):
        return {
            "notch_connected_component_count": 0,
            "notch_connected_x_span": 0.0,
            "notch_connected_y_span": 0.0,
            "notch_connected_area_fraction": 0.0,
            "component_min_x": math.nan,
            "component_max_x": math.nan,
            "component_min_y": math.nan,
            "component_max_y": math.nan,
            "component_centroid_x": math.nan,
            "component_centroid_y": math.nan,
            "component_mean_y": math.nan,
            "reaches_right_boundary": False,
            "reaches_top_boundary": False,
            "reaches_bottom_boundary": False,
            "crack_angle_deg": math.nan,
            "principal_direction_angle_deg": math.nan,
        }

    comp_x = x[comp]
    comp_y = y[comp]
    centroid_x = float(np.mean(comp_x))
    centroid_y = float(np.mean(comp_y))
    count = int(np.sum(comp))
    if areas is not None and float(np.sum(areas)) > 0.0:
        area_fraction = float(np.sum(areas[comp]) / np.sum(areas))
    else:
        area_fraction = float(count / total_count) if total_count else 0.0
    return {
        "notch_connected_component_count": count,
        "notch_connected_x_span": float(np.max(comp_x) - np.min(comp_x)),
        "notch_connected_y_span": float(np.max(comp_y) - np.min(comp_y)),
        "notch_connected_area_fraction": area_fraction,
        "component_min_x": float(np.min(comp_x)),
        "component_max_x": float(np.max(comp_x)),
        "component_min_y": float(np.min(comp_y)),
        "component_max_y": float(np.max(comp_y)),
        "component_centroid_x": centroid_x,
        "component_centroid_y": centroid_y,
        "component_mean_y": centroid_y,
        "reaches_right_boundary": bool(np.any(comp_x >= right_boundary_x - right_boundary_band)),
        "reaches_top_boundary": bool(np.any(comp_y >= SPECIMEN_SIZE_MM - boundary_band)),
        "reaches_bottom_boundary": bool(np.any(comp_y <= boundary_band)),
        "crack_angle_deg": float(math.degrees(math.atan2(centroid_y - NOTCH_TIP_Y_MM, centroid_x - NOTCH_TIP_X_MM))),
        "principal_direction_angle_deg": _principal_direction_angle_deg(comp_x, comp_y),
    }


def connectivity_metrics_for_threshold(
    data: dict[str, np.ndarray],
    threshold: float,
    step: int,
    right_boundary_x: float = SPECIMEN_SIZE_MM,
    right_boundary_band: float = RIGHT_BOUNDARY_BAND_MM,
    boundary_band: float = BOUNDARY_BAND_MM,
) -> dict[str, object]:
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    triangles = np.asarray(data["triangles"], dtype=int)
    mask = alpha >= float(threshold)
    components = connected_components(mask, element_adjacency(triangles))
    largest = max((int(np.sum(comp)) for comp in components), default=0)
    seed_mask = notch_tip_seed_mask(x, y)
    notch_components = [comp for comp in components if np.any(comp & seed_mask)]
    notch_comp = max(notch_components, key=lambda comp: int(np.sum(comp)), default=np.zeros(mask.shape[0], dtype=bool))
    row = {
        "step": int(step),
        "threshold": float(threshold),
        "threshold_label": f"alpha_ge_{str(threshold).replace('.', 'p')}",
        "connected_component_count": int(len(components)),
        "largest_connected_component_count": int(largest),
    }
    row.update(
        _component_metrics(
            notch_comp,
            x,
            y,
            total_count=int(alpha.shape[0]),
            right_boundary_x=right_boundary_x,
            right_boundary_band=right_boundary_band,
            boundary_band=boundary_band,
            areas=triangle_areas(data),
        )
    )
    return row


def compute_connectivity_by_threshold(
    result_dir: Path,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
):
    import pandas as pd

    rows = []
    for path in field_paths(result_dir):
        step = _as_step(path)
        with np.load(path) as data:
            field = {key: data[key] for key in ("element_x", "element_y", "alpha_elem", "triangles")}
            for optional_key in ("x", "y"):
                if optional_key in data:
                    field[optional_key] = data[optional_key]
        for threshold in thresholds:
            rows.append(connectivity_metrics_for_threshold(field, threshold=threshold, step=step))
    return pd.DataFrame(rows)


def first_event_steps(connectivity):
    import pandas as pd

    rows = []
    for threshold, group in connectivity.groupby("threshold", sort=True):
        notch = group[group["notch_connected_component_count"] > 0]
        through = group[group["reaches_right_boundary"].astype(bool)]
        rows.append(
            {
                "threshold": float(threshold),
                "first_notch_connected_step": int(notch["step"].min()) if not notch.empty else math.nan,
                "first_right_boundary_through_step": int(through["step"].min()) if not through.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)
