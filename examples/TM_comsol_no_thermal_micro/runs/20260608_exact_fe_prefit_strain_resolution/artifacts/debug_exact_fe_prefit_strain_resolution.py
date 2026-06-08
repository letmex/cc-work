"""Diagnose exact-FE PINN prefit strain and drive reconstruction.

This script uses only the accepted direct sparse FE alpha=0 top-u-free target.
It audits whether global PINN displacement prefit failures in strain and
He_current are explained by displacement residuals, derivative amplification,
element quality, coordinate scaling, or a T3 reconstruction mismatch.

Training losses remain geometry-agnostic: no notch/lip labels, no local masks,
no local weights, no displacement jump target, no enrichment, and no coupled
phase-field run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


SIM_ROOT_DEFAULT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
REPO_ROOT_DEFAULT = Path(r"d:\Desktop\新建文件夹\cc-work")


def _insert_paths(sim_root: Path) -> None:
    for item in (sim_root, sim_root / "source"):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-FE prefit strain-resolution diagnostic.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--sim-root", type=Path, default=SIM_ROOT_DEFAULT)
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT_DEFAULT
        / "examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/artifacts/"
        / "exact_fe_topufree_alpha0_Delta1e-6_fields.npz",
    )
    parser.add_argument(
        "--old-fedof-target",
        type=Path,
        default=REPO_ROOT_DEFAULT
        / "examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz/artifacts/"
        / "fedof_free_log10_energy_e300_fields.npz",
    )
    parser.add_argument(
        "--reuse-baseline",
        type=Path,
        default=REPO_ROOT_DEFAULT
        / "examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/artifacts/"
        / "disp_global_prefit_end_fields.npz",
    )
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--E", type=float, default=81.5)
    parser.add_argument("--nu", type=float, default=0.38)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--eta-residual", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--capacity-epochs", type=int, default=300)
    parser.add_argument("--continuation-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _prepare_dirs(out_dir: Path) -> None:
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 2:
        return float("nan")
    aa = aa[mask] - float(np.mean(aa[mask]))
    bb = bb[mask] - float(np.mean(bb[mask]))
    denom = math.sqrt(float(np.sum(aa**2) * np.sum(bb**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _quantiles(values, prefix: str) -> dict:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}_{key}": float("nan") for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_p50": float(np.quantile(vals, 0.50)),
        f"{prefix}_p90": float(np.quantile(vals, 0.90)),
        f"{prefix}_p95": float(np.quantile(vals, 0.95)),
        f"{prefix}_p99": float(np.quantile(vals, 0.99)),
        f"{prefix}_max": float(np.max(vals)),
    }


def _load_exact_target(path: Path) -> dict:
    with np.load(path) as data:
        target = {key: np.asarray(data[key]) for key in data.files}
    target["triangles"] = np.asarray(target["triangles"], dtype=np.int64)
    return target


def _target_exact_dict(target: dict) -> dict:
    return {
        "u": np.asarray(target["u"], dtype=np.float64),
        "v": np.asarray(target["v"], dtype=np.float64),
        "eps_xx": np.asarray(target["eps_xx"], dtype=np.float64),
        "eps_yy": np.asarray(target["eps_yy"], dtype=np.float64),
        "eps_xy": np.asarray(target["eps_xy"], dtype=np.float64),
        "He_current": np.asarray(target["He_current"], dtype=np.float64),
        "standard_internal_energy": float(np.asarray(target["standard_energy"])),
        "current_pinn_mechanics_energy": float(np.asarray(target["pinn_mechanics_energy"])),
        "reaction_top_v_N": float(np.asarray(target["reaction_top_v_N"])),
    }


def _t3_strains_np(x, y, tri, u, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x0, y0 = x[tri[:, 0]], y[tri[:, 0]]
    x1, y1 = x[tri[:, 1]], y[tri[:, 1]]
    x2, y2 = x[tri[:, 2]], y[tri[:, 2]]
    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    b = np.column_stack([y1 - y2, y2 - y0, y0 - y1])
    c = np.column_stack([x2 - x1, x0 - x2, x1 - x0])
    denom = 2.0 * area
    grad_u_x = np.sum(b * u[tri], axis=1) / denom
    grad_u_y = np.sum(c * u[tri], axis=1) / denom
    grad_v_x = np.sum(b * v[tri], axis=1) / denom
    grad_v_y = np.sum(c * v[tri], axis=1) / denom
    return grad_u_x, grad_v_y, 0.5 * (grad_u_y + grad_v_x)


def _element_quality(x, y, tri) -> dict:
    pts_x = np.asarray(x)[tri]
    pts_y = np.asarray(y)[tri]
    e01 = np.sqrt((pts_x[:, 0] - pts_x[:, 1]) ** 2 + (pts_y[:, 0] - pts_y[:, 1]) ** 2)
    e12 = np.sqrt((pts_x[:, 1] - pts_x[:, 2]) ** 2 + (pts_y[:, 1] - pts_y[:, 2]) ** 2)
    e20 = np.sqrt((pts_x[:, 2] - pts_x[:, 0]) ** 2 + (pts_y[:, 2] - pts_y[:, 0]) ** 2)
    longest = np.maximum(e01, np.maximum(e12, e20))
    shortest = np.minimum(e01, np.minimum(e12, e20))
    area = 0.5 * np.abs(
        (pts_x[:, 1] - pts_x[:, 0]) * (pts_y[:, 2] - pts_y[:, 0])
        - (pts_x[:, 2] - pts_x[:, 0]) * (pts_y[:, 1] - pts_y[:, 0])
    )
    shortest_altitude = 2.0 * area / np.maximum(longest, 1.0e-30)
    return {
        "area": area,
        "edge_ratio": longest / np.maximum(shortest, 1.0e-30),
        "altitude_aspect": longest / np.maximum(shortest_altitude, 1.0e-30),
    }


def _region_masks(element_x, element_y) -> dict:
    size = 0.01
    notch_x = 0.005
    notch_y = 0.005
    notch_half = 3.0e-4
    bottom_half = 5.0e-4
    boundary_half = 5.0e-4
    notch = (
        (element_x >= notch_x - notch_half)
        & (element_x <= notch_x + notch_half)
        & (np.abs(element_y - notch_y) <= notch_half)
    )
    bottom_right = (
        (element_x >= size - bottom_half)
        & (element_x <= size)
        & (element_y >= 0.0)
        & (element_y <= bottom_half)
    )
    boundary = (
        (element_x <= boundary_half)
        | (element_x >= size - boundary_half)
        | (element_y <= boundary_half)
        | (element_y >= size - boundary_half)
    )
    bulk = (~notch) & (~bottom_right) & (~boundary)
    return {"notch": notch, "bottom_right": bottom_right, "bulk": bulk, "boundary": boundary}


def _stats(values, mask) -> dict:
    vals = np.asarray(values, dtype=np.float64)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"max": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {"max": float(np.max(vals)), "p95": float(np.quantile(vals, 0.95)), "mean": float(np.mean(vals))}


def _ratio(num, den) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= 0.0:
        return float("nan")
    return float(num / den)


def _classify(row: dict) -> str:
    passes = (
        row["displacement_relative_rmse"] < 0.05
        and row["strain_relative_rmse"] < 0.2
        and row["strain_corr"] > 0.95
        and row["He_current_corr"] > 0.8
        and row["standard_energy_ratio_vs_exact"] < 1.5
        and row["pinn_mechanics_energy_ratio_vs_exact"] < 1.5
    )
    if passes:
        return "exact-target-like"
    max_x = row["max_He_current_x"]
    max_y = row["max_He_current_y"]
    boundary_max = max_x <= 5.0e-4 or max_x >= 9.5e-3 or max_y <= 5.0e-4 or max_y >= 9.5e-3
    if boundary_max:
        return "boundary-dominated"
    if row["displacement_relative_rmse"] < 0.05 and row["strain_relative_rmse"] >= 0.2:
        return "displacement-only-good / strain-bad"
    if np.isfinite(row["bulk_to_notch_He_current"]) and 0.5 <= row["bulk_to_notch_He_current"] <= 2.0:
        return "broad/background"
    return "failed"


@dataclass
class Variant:
    case_id: str
    family: str
    loss_kind: str
    strain_weight: float = 0.0
    hidden_layers: int = 8
    neurons: int = 400
    activation: str = "TrainableReLU"
    input_scaled: bool = False
    epochs: int = 500
    source_artifact: Path | None = None
    reuse_reason: str = ""


class ScaledInputNet(nn.Module):
    """Generic input scaling wrapper; it adds no geometry-specific features."""

    def __init__(self, base: nn.Module, xmin: torch.Tensor, xmax: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("xmin", xmin.detach().clone())
        self.register_buffer("xscale", torch.clamp((xmax - xmin).detach().clone(), min=1.0e-12))

    def forward(self, x):
        xmin = self.xmin.to(device=x.device, dtype=x.dtype)
        xscale = self.xscale.to(device=x.device, dtype=x.dtype)
        z = 2.0 * (x - xmin) / xscale - 1.0
        return self.base(z)


def _field_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        E=float(args.E),
        nu=float(args.nu),
        l0=float(args.l0),
        tm_eps_r=float(args.tm_eps_r),
        eta_residual=float(args.eta_residual),
        delta=float(args.delta),
        top_u_mode="free",
    )


def _make_custom_field(args: argparse.Namespace, variant: Variant, device: str):
    from field_computation import FieldComputation
    from network import NeuralNet, init_xavier

    torch.manual_seed(args.seed)
    base = NeuralNet(2, 3, variant.hidden_layers, variant.neurons, variant.activation, args.init_coeff).to(device)
    init_xavier(base)
    net: nn.Module
    if variant.input_scaled:
        xmin = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
        xmax = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
        net = ScaledInputNet(base, xmin, xmax).to(device)
    else:
        net = base
    return FieldComputation(
        net=net,
        domain_extrema=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=torch.float32, device=device),
        lmbda=torch.tensor([args.delta], dtype=torch.float32, device=device),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32, device=device),
        alpha_constraint="nonsmooth",
        top_u_mode="free",
    )


def _evaluate_artifact(path: Path, target_torch: dict, args, device: str) -> dict:
    with np.load(path) as data:
        u = torch.tensor(data["u_pred"], dtype=torch.float32, device=device)
        v = torch.tensor(data["v_pred"], dtype=torch.float32, device=device)
    from debug_prefit_then_energy_mechanics import _compute_energy, _global_prefit_losses

    disp_loss, strain_loss, strains = _global_prefit_losses(target_torch, u, v)
    E_el, E_d, fields = _compute_energy(
        target_torch["inp"], u, v, target_torch["area"], target_torch["triangles"], args, device
    )
    return {
        "u": u,
        "v": v,
        "disp_loss": disp_loss,
        "strain_loss": strain_loss,
        "strains": strains,
        "E_el": E_el,
        "E_d": E_d,
        "fields": fields,
        "energy_loss": torch.log10(torch.clamp(E_el + E_d, min=torch.finfo(E_el.dtype).tiny)),
    }


def _train_variant(
    variant: Variant,
    target_torch: dict,
    args: argparse.Namespace,
    device: str,
    trace_rows: list[dict],
):
    from debug_prefit_then_energy_mechanics import _evaluate

    field = _make_custom_field(args, variant, device)
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    disp_ref = torch.mean(target_torch["u"] ** 2 + target_torch["v"] ** 2).detach()
    strain_ref = torch.mean(
        target_torch["eps_xx"] ** 2 + target_torch["eps_yy"] ** 2 + target_torch["eps_xy"] ** 2
    ).detach()
    for epoch in range(int(variant.epochs)):

        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target_torch, args, device)
            if variant.loss_kind == "global_disp":
                loss = state["disp_loss"]
            elif variant.loss_kind == "normalized_global_disp":
                loss = state["disp_loss"] / disp_ref
            elif variant.loss_kind == "normalized_global_strain":
                loss = state["disp_loss"] / disp_ref + variant.strain_weight * state["strain_loss"] / strain_ref
            elif variant.loss_kind == "energy_log10":
                loss = state["energy_loss"]
            elif variant.loss_kind == "energy_raw":
                loss = state["E_el"] + state["E_d"]
            elif variant.loss_kind == "energy_normalized":
                loss = (state["E_el"] + state["E_d"]) / torch.clamp(target_torch["exact_energy_scale"], min=1.0e-30)
            else:
                raise ValueError(f"unknown loss kind: {variant.loss_kind}")
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == variant.epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target_torch, args, device)
            trace_rows.append(
                {
                    "case_id": variant.case_id,
                    "epoch": epoch + 1,
                    "loss_kind": variant.loss_kind,
                    "strain_weight": variant.strain_weight,
                    "input_scaled": variant.input_scaled,
                    "hidden_layers": variant.hidden_layers,
                    "neurons": variant.neurons,
                    "activation": variant.activation,
                    "loss": float(loss.detach().cpu()),
                    "displacement_mse": float(state["disp_loss"].detach().cpu()),
                    "strain_mse": float(state["strain_loss"].detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )
    return field, _evaluate(field, target_torch, args, device)


def _continue_energy_from_field(
    start_case_id: str,
    start_field,
    energy_mode: str,
    target_torch: dict,
    args: argparse.Namespace,
    device: str,
    trace_rows: list[dict],
):
    from debug_prefit_then_energy_mechanics import _evaluate

    field = copy.deepcopy(start_field)
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    for epoch in range(int(args.continuation_epochs)):

        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target_torch, args, device)
            raw = state["E_el"] + state["E_d"]
            if energy_mode == "raw":
                loss = raw
            elif energy_mode == "log10":
                loss = state["energy_loss"]
            elif energy_mode == "normalized":
                loss = raw / torch.clamp(target_torch["exact_energy_scale"], min=1.0e-30)
            else:
                raise ValueError(f"unknown energy mode: {energy_mode}")
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == args.continuation_epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target_torch, args, device)
            trace_rows.append(
                {
                    "case_id": f"{start_case_id}_energy_{energy_mode}",
                    "epoch": epoch + 1,
                    "loss_kind": f"energy_{energy_mode}",
                    "strain_weight": 0.0,
                    "input_scaled": "inherited",
                    "hidden_layers": "inherited",
                    "neurons": "inherited",
                    "activation": "inherited",
                    "loss": float(loss.detach().cpu()),
                    "displacement_mse": float(state["disp_loss"].detach().cpu()),
                    "strain_mse": float(state["strain_loss"].detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )
    return field, _evaluate(field, target_torch, args, device)


def _save_state_npz(path: Path, target_torch: dict, state: dict) -> None:
    x = target_torch["x"].detach().cpu().numpy()
    y = target_torch["y"].detach().cpu().numpy()
    tri = target_torch["triangles"].detach().cpu().numpy()
    element_x = np.mean(x[tri], axis=1)
    element_y = np.mean(y[tri], axis=1)
    eps_xx, eps_yy, eps_xy = state["strains"]
    np.savez_compressed(
        path,
        x=x,
        y=y,
        triangles=tri,
        element_x=element_x,
        element_y=element_y,
        u_target=target_torch["u"].detach().cpu().numpy(),
        v_target=target_torch["v"].detach().cpu().numpy(),
        u_pred=state["u"].detach().cpu().numpy(),
        v_pred=state["v"].detach().cpu().numpy(),
        eps_xx_target=target_torch["eps_xx"].detach().cpu().numpy(),
        eps_yy_target=target_torch["eps_yy"].detach().cpu().numpy(),
        eps_xy_target=target_torch["eps_xy"].detach().cpu().numpy(),
        eps_xx_pred=eps_xx.detach().cpu().numpy(),
        eps_yy_pred=eps_yy.detach().cpu().numpy(),
        eps_xy_pred=eps_xy.detach().cpu().numpy(),
        He_current_target=target_torch["He_current"].detach().cpu().numpy(),
        He_current_pred=state["fields"]["He_current"].detach().cpu().numpy(),
        psiI_target=target_torch["psiI"].detach().cpu().numpy(),
        psiII_target=target_torch["psiII"].detach().cpu().numpy(),
        psiI_pred=state["fields"]["psiI"].detach().cpu().numpy(),
        psiII_pred=state["fields"]["psiII"].detach().cpu().numpy(),
    )


def _variant_metrics(
    variant: Variant,
    state: dict,
    target_np: dict,
    target_exact: dict,
    target_torch: dict,
    K,
    args: argparse.Namespace,
    evaluate_candidate_arrays,
) -> tuple[dict, dict, dict, dict]:
    x = np.asarray(target_np["x"], dtype=np.float64)
    y = np.asarray(target_np["y"], dtype=np.float64)
    tri = np.asarray(target_np["triangles"], dtype=np.int64)
    area = np.asarray(target_np["area"], dtype=np.float64)
    u_np = state["u"].detach().cpu().numpy().astype(np.float64)
    v_np = state["v"].detach().cpu().numpy().astype(np.float64)
    eps_xx = state["strains"][0].detach().cpu().numpy().astype(np.float64)
    eps_yy = state["strains"][1].detach().cpu().numpy().astype(np.float64)
    eps_xy = state["strains"][2].detach().cpu().numpy().astype(np.float64)
    he = state["fields"]["He_current"].detach().cpu().numpy().astype(np.float64)
    guard = evaluate_candidate_arrays(variant.case_id, x, y, tri, u_np, v_np, target_exact, K, area, args, "free")

    element_x = np.mean(x[tri], axis=1)
    element_y = np.mean(y[tri], axis=1)
    masks = _region_masks(element_x, element_y)
    notch = _stats(he, masks["notch"])["max"]
    bulk = _stats(he, masks["bulk"])["p95"]
    bottom = _stats(he, masks["bottom_right"])["max"]
    max_idx = int(np.nanargmax(he))

    disp_err = (u_np - target_exact["u"]) ** 2 + (v_np - target_exact["v"]) ** 2
    strain_err = (
        (eps_xx - target_exact["eps_xx"]) ** 2
        + (eps_yy - target_exact["eps_yy"]) ** 2
        + (eps_xy - target_exact["eps_xy"]) ** 2
    )
    he_err = (he - target_exact["He_current"]) ** 2
    strain_vec_pred = np.concatenate([eps_xx, eps_yy, eps_xy])
    strain_vec_target = np.concatenate([target_exact["eps_xx"], target_exact["eps_yy"], target_exact["eps_xy"]])
    row = {
        "case_id": variant.case_id,
        "family": variant.family,
        "loss_kind": variant.loss_kind,
        "strain_weight": variant.strain_weight,
        "epochs": variant.epochs,
        "seed": args.seed,
        "hidden_layers": variant.hidden_layers,
        "neurons": variant.neurons,
        "activation": variant.activation,
        "input_scaled_to_minus1_1": variant.input_scaled,
        "source_artifact": str(variant.source_artifact) if variant.source_artifact else "",
        "reuse_reason": variant.reuse_reason,
        "displacement_mse": guard["displacement_mse"],
        "displacement_relative_rmse": guard["displacement_rel_rmse"],
        "strain_mse": guard["strain_mse"],
        "strain_relative_rmse": guard["strain_rel_rmse"],
        "u_corr": guard["u_corr"],
        "v_corr": guard["v_corr"],
        "strain_corr": _safe_corr(strain_vec_pred, strain_vec_target),
        "He_current_corr": _safe_corr(he, target_exact["He_current"]),
        "standard_energy_ratio_vs_exact": guard["standard_energy_ratio"],
        "pinn_mechanics_energy_ratio_vs_exact": guard["pinn_mechanics_energy_ratio"],
        "reaction_N": guard["reaction_N"],
        "reaction_exact_N": guard["reaction_exact_N"],
        "reaction_ratio": guard["reaction_ratio"],
        "reaction_sign_match": guard["reaction_sign_match"],
        "free_residual_L2": guard["free_residual_L2"],
        "boundary_residual_abs_max": guard["boundary_residual_abs_max"],
        "notch_tip_He_current_max": notch,
        "bulk_He_current_p95": bulk,
        "bottom_right_He_current_max": bottom,
        "bulk_to_notch_He_current": _ratio(bulk, notch),
        "bottom_to_notch_He_current": _ratio(bottom, notch),
        "max_He_current": float(np.nanmax(he)),
        "max_He_current_x": float(element_x[max_idx]),
        "max_He_current_y": float(element_y[max_idx]),
    }
    row["classification"] = _classify(row)
    row["passes_suggested_thresholds"] = row["classification"] == "exact-target-like"

    disp_dist = {"case_id": variant.case_id, **_quantiles(np.sqrt(disp_err), "nodal_disp_error")}
    strain_dist = {"case_id": variant.case_id, **_quantiles(np.sqrt(strain_err), "element_strain_error")}
    he_dist = {"case_id": variant.case_id, **_quantiles(np.sqrt(he_err), "element_He_error")}
    quality = _element_quality(x, y, tri)
    boundary_distance = np.minimum.reduce([element_x, element_y, 0.01 - element_x, 0.01 - element_y])
    log_strain = np.log10(np.sqrt(strain_err) + 1.0e-30)
    log_he = np.log10(np.sqrt(he_err) + 1.0e-30)
    corr_row = {
        "case_id": variant.case_id,
        "strain_error_corr_log_area": _safe_corr(log_strain, np.log10(quality["area"] + 1.0e-30)),
        "strain_error_corr_aspect": _safe_corr(log_strain, quality["altitude_aspect"]),
        "strain_error_corr_boundary_distance": _safe_corr(log_strain, boundary_distance),
        "He_error_corr_log_area": _safe_corr(log_he, np.log10(quality["area"] + 1.0e-30)),
        "He_error_corr_aspect": _safe_corr(log_he, quality["altitude_aspect"]),
        "He_error_corr_boundary_distance": _safe_corr(log_he, boundary_distance),
        "mean_strain_error_top_area_decile": float(np.mean(np.sqrt(strain_err)[quality["area"] <= np.quantile(quality["area"], 0.1)])),
        "mean_strain_error_other_area": float(np.mean(np.sqrt(strain_err)[quality["area"] > np.quantile(quality["area"], 0.1)])),
        "mean_strain_error_high_aspect_decile": float(
            np.mean(np.sqrt(strain_err)[quality["altitude_aspect"] >= np.quantile(quality["altitude_aspect"], 0.9)])
        ),
        "mean_strain_error_other_aspect": float(
            np.mean(np.sqrt(strain_err)[quality["altitude_aspect"] < np.quantile(quality["altitude_aspect"], 0.9)])
        ),
    }
    return row, {**disp_dist, **strain_dist}, he_dist, corr_row


def _target_consistency_rows(target_np: dict, baseline_npz: Path | None) -> list[dict]:
    x = np.asarray(target_np["x"], dtype=np.float64)
    y = np.asarray(target_np["y"], dtype=np.float64)
    tri = np.asarray(target_np["triangles"], dtype=np.int64)
    eps = _t3_strains_np(x, y, tri, target_np["u"], target_np["v"])
    rows = [
        {
            "check": "target_t3_recompute_matches_stored_eps",
            "status": "pass" if max(float(np.max(np.abs(eps[i] - target_np[k]))) for i, k in enumerate(("eps_xx", "eps_yy", "eps_xy"))) < 1.0e-12 else "fail",
            "max_abs_eps_xx_diff": float(np.max(np.abs(eps[0] - target_np["eps_xx"]))),
            "max_abs_eps_yy_diff": float(np.max(np.abs(eps[1] - target_np["eps_yy"]))),
            "max_abs_eps_xy_diff": float(np.max(np.abs(eps[2] - target_np["eps_xy"]))),
            "note": "Target strain is recomputed with the same T3 formula used for candidate displacement postprocessing.",
        },
        {
            "check": "current_field_computation_input_scaling",
            "status": "not_scaled",
            "max_x": float(np.max(x)),
            "max_y": float(np.max(y)),
            "note": "Current FieldComputation passes physical millimeter coordinates directly to NeuralNet; coordinate_scaled_network tests a generic [-1,1] wrapper.",
        },
    ]
    if baseline_npz and baseline_npz.exists():
        with np.load(baseline_npz) as data:
            bx = np.asarray(data["x"], dtype=np.float64)
            by = np.asarray(data["y"], dtype=np.float64)
            btri = np.asarray(data["triangles"], dtype=np.int64)
            same = bx.shape == x.shape and by.shape == y.shape and btri.shape == tri.shape
            if same:
                same = bool(np.allclose(bx, x) and np.allclose(by, y) and np.array_equal(btri, tri))
        rows.append(
            {
                "check": "baseline_artifact_connectivity_order_matches_target",
                "status": "pass" if same else "fail",
                "note": str(baseline_npz),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 20)
        args.capacity_epochs = min(args.capacity_epochs, 10)
        args.continuation_epochs = min(args.continuation_epochs, 5)
        args.hidden_layers = min(args.hidden_layers, 2)
        args.neurons = min(args.neurons, 40)
    _insert_paths(args.sim_root)
    _prepare_dirs(args.out_dir)

    from debug_exact_fe_elastic_solve import _assemble_stiffness
    from debug_prefit_then_energy_mechanics import _load_target
    from validate_mechanics_target import evaluate_candidate_arrays

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_np = _load_exact_target(args.target)
    target_exact = _target_exact_dict(target_np)
    target_torch = _load_target(args.target, device)
    target_torch["exact_energy_scale"] = torch.tensor(
        target_exact["current_pinn_mechanics_energy"], dtype=torch.float32, device=device
    )
    K = _assemble_stiffness(target_np["x"], target_np["y"], target_np["triangles"], target_np["area"], args.E, args.nu)

    guard_rows = [
        evaluate_candidate_arrays(
            "accepted_direct_sparse_FE_topufree_alpha0_Delta1e-6",
            target_np["x"],
            target_np["y"],
            target_np["triangles"],
            target_exact["u"],
            target_exact["v"],
            target_exact,
            K,
            target_np["area"],
            args,
            "free",
        )
    ]
    if args.old_fedof_target.exists():
        with np.load(args.old_fedof_target) as old:
            guard_rows.append(
                evaluate_candidate_arrays(
                    "rejected_old_FE_DOF_RPROP_free_log10_e300",
                    target_np["x"],
                    target_np["y"],
                    target_np["triangles"],
                    np.asarray(old["u"], dtype=np.float64),
                    np.asarray(old["v"], dtype=np.float64),
                    target_exact,
                    K,
                    target_np["area"],
                    args,
                    "free",
                )
            )
    _write_csv(args.out_dir / "tables" / "target_guard_check_summary.csv", guard_rows)
    _write_csv(args.out_dir / "tables" / "coordinate_and_connectivity_checks.csv", _target_consistency_rows(target_np, args.reuse_baseline))

    commands = {
        "script": str(Path(__file__).resolve()),
        "device": device,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "constraints": [
            "alpha fixed to zero",
            "accepted exact FE target only",
            "no notch/lip loss, masks, local weights, enrichment, geometry labels, or coupled phase-field run",
        ],
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")

    variants = [
        Variant(
            "baseline_global_disp",
            "A",
            "global_disp",
            epochs=800,
            source_artifact=args.reuse_baseline if args.reuse_baseline.exists() and not args.quick else None,
            reuse_reason="reused previous accepted exact-target 8x400 displacement-only prefit artifact" if args.reuse_baseline.exists() and not args.quick else "",
        ),
        Variant(
            "normalized_global_disp",
            "B",
            "normalized_global_disp",
            epochs=800,
            source_artifact=args.reuse_baseline if args.reuse_baseline.exists() and not args.quick else None,
            reuse_reason="constant positive displacement normalization is RPROP sign-invariant; reused baseline artifact" if args.reuse_baseline.exists() and not args.quick else "",
        ),
    ]
    for weight in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0):
        variants.append(
            Variant(
                f"normalized_global_strain_w{weight:g}".replace(".", "p").replace("-", "m"),
                "C",
                "normalized_global_strain",
                strain_weight=weight,
                epochs=args.epochs,
            )
        )
    variants.extend(
        [
            Variant(
                "coordinate_scaled_network_w1",
                "D",
                "normalized_global_strain",
                strain_weight=1.0,
                input_scaled=True,
                epochs=args.epochs,
            ),
            Variant(
                "capacity_10x500_w1",
                "E",
                "normalized_global_strain",
                strain_weight=1.0,
                hidden_layers=10,
                neurons=500,
                epochs=args.capacity_epochs,
            ),
            Variant(
                "activation_TrainableTanh_w1",
                "F",
                "normalized_global_strain",
                strain_weight=1.0,
                activation="TrainableTanh",
                epochs=args.capacity_epochs,
            ),
        ]
    )
    if args.quick:
        variants = variants[:3]
        for variant in variants:
            variant.source_artifact = None
            variant.reuse_reason = ""
            variant.epochs = min(variant.epochs, args.epochs)
            variant.hidden_layers = min(variant.hidden_layers, args.hidden_layers)
            variant.neurons = min(variant.neurons, args.neurons)

    trace_rows: list[dict] = []
    comparison_rows: list[dict] = []
    strain_rows: list[dict] = []
    he_rows: list[dict] = []
    quality_rows: list[dict] = []
    states_by_case: dict[str, dict] = {}
    fields_by_case: dict[str, object] = {}
    for variant in variants:
        if variant.source_artifact and variant.source_artifact.exists():
            state = _evaluate_artifact(variant.source_artifact, target_torch, args, device)
            field = None
        else:
            field, state = _train_variant(variant, target_torch, args, device, trace_rows)
        states_by_case[variant.case_id] = state
        if field is not None:
            fields_by_case[variant.case_id] = field
        row, strain_dist, he_dist, corr = _variant_metrics(
            variant, state, target_np, target_exact, target_torch, K, args, evaluate_candidate_arrays
        )
        comparison_rows.append(row)
        strain_rows.append(strain_dist)
        he_rows.append(he_dist)
        quality_rows.append(corr)
        _save_state_npz(args.out_dir / "artifacts" / f"{variant.case_id}_fields.npz", target_torch, state)

    _write_csv(args.out_dir / "tables" / "prefit_variant_comparison.csv", comparison_rows)
    _write_csv(args.out_dir / "tables" / "strain_error_distribution.csv", strain_rows)
    _write_csv(args.out_dir / "tables" / "he_error_distribution.csv", he_rows)
    _write_csv(args.out_dir / "tables" / "element_quality_error_correlation.csv", quality_rows)
    _write_csv(args.out_dir / "logs" / "prefit_loss_trace.csv", trace_rows)

    passed = [row for row in comparison_rows if row["passes_suggested_thresholds"]]
    continuation_rows: list[dict] = []
    if passed:
        best = sorted(passed, key=lambda r: (r["strain_relative_rmse"], -r["He_current_corr"]))[0]
        start_field = fields_by_case.get(best["case_id"])
        if start_field is None:
            continuation_rows.append(
                {
                    "status": "not_run",
                    "reason": "best passing case came from a reused artifact without network state",
                    "candidate_case": best["case_id"],
                    "start_elastic_energy": float(states_by_case[best["case_id"]]["E_el"].detach().cpu()),
                }
            )
        else:
            for energy_mode in ("raw", "log10", "normalized"):
                case_id = f"{best['case_id']}_energy_{energy_mode}"
                _field, state = _continue_energy_from_field(
                    best["case_id"], start_field, energy_mode, target_torch, args, device, trace_rows
                )
                cont_variant = Variant(
                    case_id=case_id,
                    family="optional_energy_continuation",
                    loss_kind=f"energy_{energy_mode}",
                    epochs=args.continuation_epochs,
                    hidden_layers=int(best["hidden_layers"]),
                    neurons=int(best["neurons"]),
                    activation=str(best["activation"]),
                    input_scaled=str(best["input_scaled_to_minus1_1"]) == "True",
                )
                row, _strain_dist, _he_dist, _corr = _variant_metrics(
                    cont_variant, state, target_np, target_exact, target_torch, K, args, evaluate_candidate_arrays
                )
                row["start_case_id"] = best["case_id"]
                row["energy_mode"] = energy_mode
                continuation_rows.append(row)
                _save_state_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target_torch, state)
    if continuation_rows:
        _write_csv(args.out_dir / "tables" / "optional_energy_continuation.csv", continuation_rows)

    (args.out_dir / "figures" / "figure_summary.md").write_text(
        "# Figure Summary\n\nNo PNG figures were generated. The diagnostic evidence is in CSV tables and compact NPZ artifacts.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
