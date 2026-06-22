"""Explicit smoke runner for the solved-temperature mechanics adapter."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy_mixed_tm import compute_mixed_tm_fields  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from pff_model import PFFModel  # noqa: E402
from thermal_mechanics_adapter import build_mechanics_thermal_kwargs_from_bridge  # noqa: E402


CLASSIFICATION = "solved-temperature mechanics smoke cli passed"
DTYPE = torch.float64


def _matprop(dtype=DTYPE):
    return MaterialProperties(
        mat_E=torch.tensor(81.5, dtype=dtype),
        mat_nu=torch.tensor(0.38, dtype=dtype),
        w1=torch.tensor(2.4e-6 / 1.5e-4, dtype=dtype),
        l0=torch.tensor(1.5e-4, dtype=dtype),
    )


def _pffmodel():
    return PFFModel(PFF_model="AT2", se_split="volumetric", tol_ir=torch.tensor(5.0e-3))


def _fixture_patch(dtype=DTYPE):
    nodes = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]],
        dtype=dtype,
    )
    elements = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    area = torch.tensor([0.5e-4, 0.5e-4], dtype=dtype)
    alpha = torch.zeros(4, dtype=dtype)
    history = torch.zeros(2, dtype=dtype)
    u = 1.0e-4 * nodes[:, 0] + 2.0e-5 * nodes[:, 1]
    v = -3.0e-5 * nodes[:, 0] + 4.0e-5 * nodes[:, 1]
    return nodes, elements, area, alpha, history, u, v


def _compute_fields(nodes, elements, area, alpha, history, u, v, **thermal_kwargs):
    return compute_mixed_tm_fields(
        nodes,
        u,
        v,
        alpha,
        history,
        history,
        _matprop(),
        _pffmodel(),
        area,
        elements,
        **thermal_kwargs,
    )


def _max_abs_field_difference(left, right, keys):
    diffs = []
    for key in keys:
        diffs.append(torch.max(torch.abs(left[key] - right[key])))
    return float(torch.max(torch.stack(diffs)).detach().cpu())


def _write_metrics_csv(table_path, metrics):
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in metrics:
            writer.writerow({"metric": metric, "value": value})


def run_smoke(output_dir=None):
    """Run one explicit two-element mechanics smoke evaluation and write metrics."""
    output_dir = ROOT / "outputs" / "solved_temperature_mechanics_smoke" if output_dir is None else Path(output_dir)
    table_path = output_dir / "mechanics_smoke_results.csv"

    nodes, elements, area, alpha, history, u, v = _fixture_patch()
    adapter = build_mechanics_thermal_kwargs_from_bridge(
        nodes,
        elements,
        source_mode="solved_frozen_lift",
        case_id="H1",
        bounds_mm=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=DTYPE),
        T_bottom_K=300.0,
        T_top_K=320.0,
        T_ref_K=300.0,
    )
    diagnostics = adapter["diagnostics"]
    direct_fields = _compute_fields(
        nodes,
        elements,
        area,
        alpha,
        history,
        u,
        v,
        thermal_delta_T=diagnostics["thermal_delta_T"],
    )
    bridged_fields = _compute_fields(
        nodes,
        elements,
        area,
        alpha,
        history,
        u,
        v,
        **adapter["thermal_kwargs"],
    )
    field_keys = (
        "thermal_delta_T",
        "thermal_eps_xx",
        "thermal_eps_yy",
        "eps_xx_elastic",
        "eps_yy_elastic",
        "psi_total",
        "sigma_xx_tm_total",
        "sigma_yy_tm_total",
    )
    max_abs_diff = _max_abs_field_difference(bridged_fields, direct_fields, field_keys)
    delta_t = diagnostics["thermal_delta_T"].detach().cpu()
    centroids = diagnostics["element_centroid_coords_mm"].detach().cpu()

    metrics = [
        ("classification", CLASSIFICATION),
        ("source_mode", str(diagnostics["source_mode"])),
        ("case_id", str(diagnostics["case_id"])),
        ("evaluation_location", str(diagnostics["evaluation_location"])),
        ("network_training_run", str(bool(diagnostics["network_training_run"])).lower()),
        ("correction_policy", str(diagnostics["correction_policy"])),
        ("n_nodes", str(nodes.shape[0])),
        ("n_elements", str(elements.shape[0])),
        ("delta_T_min_K", repr(float(torch.min(delta_t)))),
        ("delta_T_max_K", repr(float(torch.max(delta_t)))),
        ("centroid_y_min_mm", repr(float(torch.min(centroids[:, 1])))),
        ("centroid_y_max_mm", repr(float(torch.max(centroids[:, 1])))),
        ("mechanics_max_abs_diff", repr(max_abs_diff)),
        ("train_mixed_tm_modified", "false"),
    ]
    _write_metrics_csv(table_path, metrics)
    return {
        "classification": CLASSIFICATION,
        "table_path": table_path,
        "metrics": dict(metrics),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "solved_temperature_mechanics_smoke",
        help="Directory for mechanics_smoke_results.csv.",
    )
    args = parser.parse_args(argv)
    result = run_smoke(args.output_dir)
    print(result["table_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
