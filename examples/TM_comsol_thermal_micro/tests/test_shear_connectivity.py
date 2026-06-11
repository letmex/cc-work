from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strip_step_fields(row):
    return {key: value for key, value in row.items() if key not in {"step", "threshold"}}


def test_notch_connected_component_reports_span_bounds_and_boundary_reach():
    from shear_connectivity import connectivity_metrics_for_threshold

    data = {
        "element_x": np.array([0.00495, 0.00525, 0.00990, 0.00990], dtype=float),
        "element_y": np.array([0.00500, 0.00505, 0.00510, 0.00980], dtype=float),
        "alpha_elem": np.array([0.9, 0.85, 0.82, 0.95], dtype=float),
        "triangles": np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [5, 6, 7]], dtype=int),
    }

    row = connectivity_metrics_for_threshold(data, threshold=0.8, step=7)

    assert row["step"] == 7
    assert row["threshold"] == 0.8
    assert row["largest_connected_component_count"] == 3
    assert row["notch_connected_component_count"] == 3
    assert row["notch_connected_x_span"] == np.float64(0.00990 - 0.00495)
    assert row["notch_connected_y_span"] == np.float64(0.00510 - 0.00500)
    assert row["component_min_x"] == 0.00495
    assert row["component_max_x"] == 0.00990
    assert row["reaches_right_boundary"] is True
    assert row["reaches_top_boundary"] is False
    assert row["reaches_bottom_boundary"] is False
    assert row["notch_connected_area_fraction"] == 3 / 4
    assert row["crack_angle_deg"] < 5.0


def test_notch_connected_component_is_empty_when_thresholded_elements_miss_tip_window():
    from shear_connectivity import connectivity_metrics_for_threshold

    data = {
        "element_x": np.array([0.0060, 0.0062], dtype=float),
        "element_y": np.array([0.0050, 0.0051], dtype=float),
        "alpha_elem": np.array([0.9, 0.9], dtype=float),
        "triangles": np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    }

    row = connectivity_metrics_for_threshold(data, threshold=0.8, step=2)

    assert row["largest_connected_component_count"] == 2
    assert row["notch_connected_component_count"] == 0
    assert row["notch_connected_x_span"] == 0.0
    assert row["notch_connected_y_span"] == 0.0
    assert row["reaches_right_boundary"] is False
    assert np.isnan(row["component_min_x"])
    assert np.isnan(row["principal_direction_angle_deg"])


def test_principal_direction_angle_follows_elongated_notch_component():
    from shear_connectivity import connectivity_metrics_for_threshold

    data = {
        "element_x": np.array([0.00495, 0.00520, 0.00545, 0.00570], dtype=float),
        "element_y": np.array([0.00500, 0.00525, 0.00550, 0.00575], dtype=float),
        "alpha_elem": np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
        "triangles": np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5]], dtype=int),
    }

    row = connectivity_metrics_for_threshold(data, threshold=0.8, step=4)

    assert 40.0 <= row["principal_direction_angle_deg"] <= 50.0
    assert 40.0 <= row["crack_angle_deg"] <= 50.0


def test_area_fraction_uses_triangle_areas_when_nodal_coordinates_are_available():
    from shear_connectivity import connectivity_metrics_for_threshold

    data = {
        "x": np.array([0.0049, 0.0051, 0.0049, 0.0090, 0.0091, 0.0090], dtype=float),
        "y": np.array([0.0049, 0.0049, 0.0051, 0.0090, 0.0090, 0.0091], dtype=float),
        "element_x": np.array([0.00497, 0.00903], dtype=float),
        "element_y": np.array([0.00497, 0.00903], dtype=float),
        "alpha_elem": np.array([0.9, 0.9], dtype=float),
        "triangles": np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    }

    row = connectivity_metrics_for_threshold(data, threshold=0.8, step=0)

    assert row["notch_connected_component_count"] == 1
    assert 0.78 <= row["notch_connected_area_fraction"] <= 0.82
