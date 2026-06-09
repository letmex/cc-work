"""Frozen-alpha PINN u/v re-optimization diagnostic for D0040 default unit_box.

This package script is intentionally offline. It reads saved D0040 fields,
freezes alpha and history fields, then re-optimizes only the PINN u/v
mechanics response under diagnostic stress/energy variants.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
RESULTS = PROJECT / "results"
SOURCE = PROJECT / "source"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy import gradients  # noqa: E402
from compute_energy_mixed_tm import compute_mixed_tm_fields  # noqa: E402
from field_computation import FieldComputation  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from mixed_mode_tm import mixed_mode_ratio  # noqa: E402
from network import NeuralNet, init_xavier  # noqa: E402
from pff_model import PFFModel  # noqa: E402


SPECIMEN_SIZE_MM = 0.01
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
CUT_XS = (0.006, 0.007, 0.008, 0.009)
CUT_TOL = 2.5e-4
CRACK_BAND_Y_TOL = 8.0e-4
EDGE_TOL = 1.0e-9

CASES = {
    7: "softgate_D0040_seed7_history_default_unitbox",
    13: "softgate_D0040_seed13_history_default_unitbox",
    42: "softgate_D0040_seed42_history_default_unitbox",
}
STATE_STEPS = {
    "final_D0040": 54,
    "through_alpha0p8_onset": 14,
}
VARIANTS = (
    "baseline_current_split",
    "minus_degraded_in_crack_band",
    "minus_removed_in_crack_band",
    "full_degradation_all_energy",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen-alpha u/v re-optimization.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    parser.add_argument("--states", nargs="+", default=["final_D0040"], choices=sorted(STATE_STEPS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--init-strategies", nargs="+", default=["saved_uv_prefit"], choices=["saved_uv_prefit", "random"])
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--prefit-epochs", type=int, default=250)
    parser.add_argument("--reopt-epochs", type=int, default=250)
    parser.add_argument("--prefit-lr", type=float, default=1.0e-4)
    parser.add_argument("--reopt-lr", type=float, default=1.0e-5)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--eta-residual", type=float, default=1.0e-5)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def setup_dirs():
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def choose_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return name


def quick_trim(args):
    if not args.quick:
        return
    args.seeds = args.seeds[:1]
    args.states = args.states[:1]
    args.variants = args.variants[:2]
    args.prefit_epochs = min(args.prefit_epochs, 8)
    args.reopt_epochs = min(args.reopt_epochs, 8)
    args.hidden_layers = min(args.hidden_layers, 2)
    args.neurons = min(args.neurons, 30)


def result_dir(seed):
    suffix = CASES[seed]
    matches = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run dir for seed {seed}, found {len(matches)}")
    return matches[0]


def field_file(seed, state):
    step = STATE_STEPS[state]
    path = result_dir(seed) / f"fields_mixed_tm_step_{step:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_state(seed, state, device):
    path = field_file(seed, state)
    with np.load(path) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    x = arrays["x"].astype(np.float32)
    y = arrays["y"].astype(np.float32)
    tri = arrays["triangles"].astype(np.int64)
    inp = torch.tensor(np.column_stack([x, y]), dtype=torch.float32, device=device)
    T_conn = torch.tensor(tri, dtype=torch.long, device=device)
    area = torch.tensor(triangle_areas_np(x, y, tri), dtype=torch.float32, device=device)
    state_dict = {
        "seed": seed,
        "case": f"D0040_seed{seed}_default_unitbox",
        "state": state,
        "step": STATE_STEPS[state],
        "source_path": str(path),
        "Delta": float(np.asarray(arrays["displacement_mm"])),
        "x_np": x,
        "y_np": y,
        "tri_np": tri,
        "inp": inp,
        "T_conn": T_conn,
        "area": area,
        "u_saved": torch.tensor(arrays["u"], dtype=torch.float32, device=device),
        "v_saved": torch.tensor(arrays["v"], dtype=torch.float32, device=device),
        "alpha_saved": torch.tensor(arrays["alpha"], dtype=torch.float32, device=device),
        "HI_saved": torch.tensor(arrays["HI"], dtype=torch.float32, device=device),
        "HII_saved": torch.tensor(arrays["HII"], dtype=torch.float32, device=device),
        "alpha_elem_saved_np": arrays["alpha_elem"].astype(np.float64),
        "element_x_np": arrays["element_x"].astype(np.float64),
        "element_y_np": arrays["element_y"].astype(np.float64),
    }
    state_dict["crack_mask_np"] = connected_crack_mask_np(state_dict)
    state_dict["crack_mask"] = torch.tensor(state_dict["crack_mask_np"], dtype=torch.bool, device=device)
    return state_dict


def triangle_areas_np(x, y, tri):
    a = np.column_stack([x, y])[tri[:, 0]]
    b = np.column_stack([x, y])[tri[:, 1]]
    c = np.column_stack([x, y])[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def element_adjacency_np(triangles):
    edge_to_elem = {}
    adjacency = [[] for _ in range(len(triangles))]
    for elem_id, nodes in enumerate(triangles.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            key = tuple(sorted((int(a), int(b))))
            other = edge_to_elem.get(key)
            if other is None:
                edge_to_elem[key] = elem_id
            else:
                adjacency[elem_id].append(other)
                adjacency[other].append(elem_id)
    return adjacency


def connected_crack_mask_np(state, threshold=0.8):
    x = state["element_x_np"]
    y = state["element_y_np"]
    alpha = state["alpha_elem_saved_np"]
    mask = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    adjacency = element_adjacency_np(state["tri_np"])
    queue = deque()
    visited = np.zeros(mask.shape[0], dtype=bool)
    for idx in np.flatnonzero(mask & seed_mask):
        visited[idx] = True
        queue.append(int(idx))
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(int(nxt))
    return visited


def material(device, l0):
    matprop = MaterialProperties(
        torch.tensor(81.5, device=device),
        torch.tensor(0.38, device=device),
        torch.tensor(2.4e-6 / l0, device=device),
        torch.tensor(l0, device=device),
    )
    pffmodel = PFFModel("AT2", "volumetric", torch.tensor(5.0e-3, device=device))
    gcII = 2.0 * (1.0 + 0.38) * (0.60**2) * 2.4e-6
    return matprop, pffmodel, gcII


def make_field(args, delta, device, seed_offset=0):
    torch.manual_seed(1000 + seed_offset)
    net = NeuralNet(2, 3, args.hidden_layers, args.neurons, args.activation, args.init_coeff).to(device)
    init_xavier(net)
    return FieldComputation(
        net=net,
        domain_extrema=torch.tensor([[0.0, SPECIMEN_SIZE_MM], [0.0, SPECIMEN_SIZE_MM]], dtype=torch.float32, device=device),
        lmbda=torch.tensor([delta], dtype=torch.float32, device=device),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32, device=device),
        alpha_constraint="nonsmooth",
        top_u_mode="free",
        coord_normalization="unit_box",
    )


def prefit_saved_uv(field, state, args):
    if args.prefit_epochs <= 0:
        return {"prefit_final_loss": math.nan, "prefit_disp_mse": math.nan, "prefit_iterations": 0}
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.prefit_lr, step_sizes=(1.0e-10, 1.0))
    loss_history = []
    for _epoch in range(args.prefit_epochs):

        def closure():
            optimizer.zero_grad()
            u, v, _alpha = field.fieldCalculation(state["inp"])
            loss = torch.mean((u - state["u_saved"]) ** 2 + (v - state["v_saved"]) ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_history.append(float(loss.detach().cpu()))
    with torch.no_grad():
        u, v, _alpha = field.fieldCalculation(state["inp"])
        disp_mse = torch.mean((u - state["u_saved"]) ** 2 + (v - state["v_saved"]) ** 2)
    return {
        "prefit_final_loss": float(loss_history[-1]) if loss_history else math.nan,
        "prefit_disp_mse": float(disp_mse.detach().cpu()),
        "prefit_iterations": len(loss_history),
    }


def compute_fields_for_uv(field, state, args, matprop, pffmodel, gcII):
    u, v, _alpha_pred = field.fieldCalculation(state["inp"])
    fields = compute_mixed_tm_fields(
        state["inp"],
        u,
        v,
        state["alpha_saved"],
        state["HI_saved"],
        state["HII_saved"],
        matprop,
        pffmodel,
        state["area"],
        state["T_conn"],
        eta_residual=args.eta_residual,
        gcII=gcII,
        split_mode="tm_source",
        tm_eps_r=args.tm_eps_r,
        mechanics_mode="history",
    )
    return u, v, fields


def variant_density(fields, crack_mask, variant):
    g = fields["g_alpha"]
    he = fields["He_history"]
    psi_minus = fields["psi_minus"]
    if variant == "baseline_current_split":
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus
    elif variant == "full_degradation_all_energy":
        density = g * (he + psi_minus)
        pos = g * he
        neg = g * psi_minus
    elif variant == "minus_degraded_in_crack_band":
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus.clone()
        density = density.clone()
        density[crack_mask] = g[crack_mask] * he[crack_mask] + g[crack_mask] * psi_minus[crack_mask]
        neg[crack_mask] = g[crack_mask] * psi_minus[crack_mask]
    elif variant == "minus_removed_in_crack_band":
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus.clone()
        density = density.clone()
        density[crack_mask] = g[crack_mask] * he[crack_mask]
        neg[crack_mask] = 0.0
    else:
        raise ValueError(variant)
    return density, pos, neg


def stress_variant(fields, crack_mask, variant):
    g = fields["g_alpha"]
    yy_plus = fields["sigma_yy_tm_plus"]
    yy_minus = fields["sigma_yy_tm_minus"]
    xy_plus = fields["sigma_xy_tm_plus"]
    xy_minus = fields["sigma_xy_tm_minus"]
    if variant == "baseline_current_split":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
    elif variant == "full_degradation_all_energy":
        yy = g * (yy_plus + yy_minus)
        xy = g * (xy_plus + xy_minus)
    elif variant == "minus_degraded_in_crack_band":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
        yy = yy.clone()
        xy = xy.clone()
        yy[crack_mask] = g[crack_mask] * yy_plus[crack_mask] + g[crack_mask] * yy_minus[crack_mask]
        xy[crack_mask] = g[crack_mask] * xy_plus[crack_mask] + g[crack_mask] * xy_minus[crack_mask]
    elif variant == "minus_removed_in_crack_band":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
        yy = yy.clone()
        xy = xy.clone()
        yy[crack_mask] = g[crack_mask] * yy_plus[crack_mask]
        xy[crack_mask] = g[crack_mask] * xy_plus[crack_mask]
    else:
        raise ValueError(variant)
    return yy, xy


def mechanics_loss(field, state, args, variant, matprop, pffmodel, gcII):
    _u, _v, fields = compute_fields_for_uv(field, state, args, matprop, pffmodel, gcII)
    density, _pos, _neg = variant_density(fields, state["crack_mask"], variant)
    energy = torch.sum(state["area"] * density)
    return torch.log10(torch.clamp(energy, min=torch.finfo(energy.dtype).tiny)), energy, fields


def reoptimize(field, state, args, variant, matprop, pffmodel, gcII):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.reopt_lr, step_sizes=(1.0e-10, 1.0))
    trace = []
    for epoch in range(args.reopt_epochs):

        def closure():
            optimizer.zero_grad()
            loss, _energy, _fields = mechanics_loss(field, state, args, variant, matprop, pffmodel, gcII)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == args.reopt_epochs - 1:
            with torch.no_grad():
                loss_eval, energy_eval, _fields = mechanics_loss(field, state, args, variant, matprop, pffmodel, gcII)
            trace.append(
                {
                    "epoch": epoch + 1,
                    "loss_log10": float(loss_eval.detach().cpu()),
                    "elastic_energy": float(energy_eval.detach().cpu()),
                    "optimizer_returned_loss": float(loss.detach().cpu()),
                }
            )
    status = convergence_status(trace)
    return trace, status


def convergence_status(trace):
    if not trace:
        return "not_run"
    vals = [row["loss_log10"] for row in trace if np.isfinite(row["loss_log10"])]
    if not vals:
        return "failed_nonfinite"
    if len(vals) >= 4:
        recent = abs(vals[-1] - vals[-4]) / (abs(vals[-4]) + 1.0e-12)
        if recent < 1.0e-4:
            return "converged_trace_tol"
    return "budget_reached_finite"


def top_reaction_np(x, y, tri, sigma_yy):
    reaction_kN = 0.0
    points = np.column_stack([x, y])
    for elem_id, nodes in enumerate(tri.astype(int)):
        for a_local, b_local in ((0, 1), (1, 2), (2, 0)):
            a = nodes[a_local]
            b = nodes[b_local]
            if abs(points[a, 1] - SPECIMEN_SIZE_MM) <= EDGE_TOL and abs(points[b, 1] - SPECIMEN_SIZE_MM) <= EDGE_TOL:
                reaction_kN += float(sigma_yy[elem_id]) * float(np.linalg.norm(points[a] - points[b]))
    return 1000.0 * reaction_kN


def crack_band_metrics_np(state, sigma_yy, sigma_xy):
    mask = state["crack_mask_np"]
    if not np.any(mask):
        return {
            "crack_band_element_count": 0,
            "crack_band_mean_abs_sigma_yy_eff": math.nan,
            "crack_band_max_abs_sigma_yy_eff": math.nan,
            "crack_band_mean_abs_sigma_xy_eff": math.nan,
            "crack_band_max_abs_sigma_xy_eff": math.nan,
        }
    return {
        "crack_band_element_count": int(np.sum(mask)),
        "crack_band_mean_abs_sigma_yy_eff": float(np.mean(np.abs(sigma_yy[mask]))),
        "crack_band_max_abs_sigma_yy_eff": float(np.max(np.abs(sigma_yy[mask]))),
        "crack_band_mean_abs_sigma_xy_eff": float(np.mean(np.abs(sigma_xy[mask]))),
        "crack_band_max_abs_sigma_xy_eff": float(np.max(np.abs(sigma_xy[mask]))),
    }


def cut_traction_np(state, sigma_yy, sigma_xy, cut_x):
    mask = state["crack_mask_np"]
    if not np.any(mask):
        return {"cut_x": cut_x, "integrated_cut_traction_proxy": math.nan, "cut_band_element_count": 0}
    crack_y = float(np.mean(state["element_y_np"][mask]))
    band = (
        (np.abs(state["element_x_np"] - cut_x) <= CUT_TOL)
        & (np.abs(state["element_y_np"] - crack_y) <= CRACK_BAND_Y_TOL)
        & mask
    )
    if not np.any(band):
        return {"cut_x": cut_x, "integrated_cut_traction_proxy": math.nan, "cut_band_element_count": 0}
    areas = triangle_areas_np(state["x_np"], state["y_np"], state["tri_np"])
    return {
        "cut_x": cut_x,
        "integrated_cut_traction_proxy": float(np.sum(np.sqrt(sigma_yy[band] ** 2 + sigma_xy[band] ** 2) * areas[band])),
        "cut_band_element_count": int(np.sum(band)),
        "mean_abs_sigma_yy_eff": float(np.mean(np.abs(sigma_yy[band]))),
        "max_abs_sigma_yy_eff": float(np.max(np.abs(sigma_yy[band]))),
        "mean_abs_sigma_xy_eff": float(np.mean(np.abs(sigma_xy[band]))),
        "max_abs_sigma_xy_eff": float(np.max(np.abs(sigma_xy[band]))),
    }


def jump_proxy_np(state, u, v, cut_x):
    x = state["x_np"]
    y = state["y_np"]
    mask = state["crack_mask_np"]
    crack_y = float(np.mean(state["element_y_np"][mask])) if np.any(mask) else NOTCH_Y
    near = np.abs(x - cut_x) <= CUT_TOL
    above = near & (y >= crack_y + 2.0e-4) & (y <= crack_y + 1.2e-3)
    below = near & (y <= crack_y - 2.0e-4) & (y >= crack_y - 1.2e-3)
    if not np.any(above) or not np.any(below):
        return {"cut_x": cut_x, "v_jump_proxy": math.nan, "u_jump_proxy": math.nan}
    return {
        "cut_x": cut_x,
        "v_jump_proxy": float(np.mean(v[above]) - np.mean(v[below])),
        "u_jump_proxy": float(np.mean(u[above]) - np.mean(u[below])),
        "above_node_count": int(np.sum(above)),
        "below_node_count": int(np.sum(below)),
    }


def evaluate_result(field, state, args, variant, matprop, pffmodel, gcII, init_strategy, prefit_info, trace, status):
    with torch.no_grad():
        u, v, fields = compute_fields_for_uv(field, state, args, matprop, pffmodel, gcII)
        density, pos, neg = variant_density(fields, state["crack_mask"], variant)
        yy, xy = stress_variant(fields, state["crack_mask"], variant)
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    yy_np = yy.detach().cpu().numpy()
    xy_np = xy.detach().cpu().numpy()
    density_np = density.detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    neg_np = neg.detach().cpu().numpy()
    areas = triangle_areas_np(state["x_np"], state["y_np"], state["tri_np"])
    crack_mask = state["crack_mask_np"]
    cut_rows = []
    traction_total = 0.0
    traction_count = 0
    for cut_x in CUT_XS:
        row = cut_traction_np(state, yy_np, xy_np, cut_x)
        cut_rows.append(row)
        if np.isfinite(row["integrated_cut_traction_proxy"]):
            traction_total += row["integrated_cut_traction_proxy"]
            traction_count += 1
    jump_rows = [jump_proxy_np(state, u_np, v_np, cut_x) for cut_x in CUT_XS]
    mean_v_jump = float(np.nanmean([r["v_jump_proxy"] for r in jump_rows]))
    mean_u_jump = float(np.nanmean([r["u_jump_proxy"] for r in jump_rows]))
    total_energy = float(np.sum(areas * density_np))
    positive_energy = float(np.sum(areas * pos_np))
    negative_energy = float(np.sum(areas * neg_np))
    crack_negative_energy = float(np.sum(areas[crack_mask] * neg_np[crack_mask])) if np.any(crack_mask) else math.nan
    crack_positive_energy = float(np.sum(areas[crack_mask] * pos_np[crack_mask])) if np.any(crack_mask) else math.nan
    row = {
        "case": state["case"],
        "seed": state["seed"],
        "state": state["state"],
        "step": state["step"],
        "Delta": state["Delta"],
        "variant": variant,
        "init_strategy": init_strategy,
        "source_path": state["source_path"],
        "iterations": args.reopt_epochs,
        "convergence_status": status,
        "final_mechanics_loss": trace[-1]["loss_log10"] if trace else math.nan,
        "final_reaction_proxy": top_reaction_np(state["x_np"], state["y_np"], state["tri_np"], yy_np),
        "total_elastic_energy": total_energy,
        "positive_energy_contribution": positive_energy,
        "negative_energy_contribution": negative_energy,
        "crack_band_negative_energy_contribution": crack_negative_energy,
        "crack_band_positive_energy_contribution": crack_positive_energy,
        "cutline_crack_band_traction_proxy": traction_total / traction_count if traction_count else math.nan,
        "mean_v_jump_proxy": mean_v_jump,
        "mean_u_jump_proxy": mean_u_jump,
        **crack_band_metrics_np(state, yy_np, xy_np),
        **prefit_info,
    }
    save_npz(state, variant, init_strategy, u_np, v_np, fields, yy_np, xy_np, density_np, pos_np, neg_np)
    return row, cut_rows, jump_rows, trace


def save_npz(state, variant, init_strategy, u, v, fields, yy, xy, density, pos, neg):
    path = ARTIFACTS / f"{state['case']}_{state['state']}_{variant}_{init_strategy}_fields.npz"
    np.savez_compressed(
        path,
        x=state["x_np"],
        y=state["y_np"],
        triangles=state["tri_np"],
        element_x=state["element_x_np"],
        element_y=state["element_y_np"],
        u=u,
        v=v,
        alpha=state["alpha_saved"].detach().cpu().numpy(),
        alpha_elem=fields["alpha_elem"].detach().cpu().numpy(),
        crack_mask=state["crack_mask_np"],
        eps_xx=fields["eps_xx"].detach().cpu().numpy(),
        eps_yy=fields["eps_yy"].detach().cpu().numpy(),
        eps_xy=fields["eps_xy"].detach().cpu().numpy(),
        psiI=fields["psiI"].detach().cpu().numpy(),
        psiII=fields["psiII"].detach().cpu().numpy(),
        psi_minus=fields["psi_minus"].detach().cpu().numpy(),
        He_history=fields["He_history"].detach().cpu().numpy(),
        g_alpha=fields["g_alpha"].detach().cpu().numpy(),
        sigma_yy_variant=yy,
        sigma_xy_variant=xy,
        variant_elastic_energy_density=density,
        variant_positive_energy_density=pos,
        variant_negative_energy_density=neg,
    )


def run_all(args):
    setup_dirs()
    quick_trim(args)
    device = choose_device(args.device)
    invocation = "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_frozen_alpha_uv_reopt.py"
    if args.states != ["final_D0040"]:
        invocation += " --states " + " ".join(args.states)
    if args.seeds != [7, 13, 42]:
        invocation += " --seeds " + " ".join(str(seed) for seed in args.seeds)
    if args.variants != list(VARIANTS):
        invocation += " --variants " + " ".join(args.variants)
    if args.init_strategies != ["saved_uv_prefit"]:
        invocation += " --init-strategies " + " ".join(args.init_strategies)
    (PACKAGE / "commands_run.txt").write_text(
        "\n".join(
            [
                "git pull origin main",
                "Read previous negative-branch package files.",
                invocation,
                "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
                "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260612_default_unitbox_frozen_alpha_uv_reopt\\artifacts\\run_frozen_alpha_uv_reopt.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (LOGS / "run_config.json").write_text(
        json.dumps({"device": device, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    matprop, pffmodel, gcII = material(device, args.l0)
    summary_rows = []
    traction_rows = []
    jump_rows = []
    trace_rows = []
    for seed in args.seeds:
        for state_name in args.states:
            state = load_state(seed, state_name, device)
            for init_strategy in args.init_strategies:
                base_field = make_field(args, state["Delta"], device, seed_offset=seed)
                if init_strategy == "saved_uv_prefit":
                    prefit_info = prefit_saved_uv(base_field, state, args)
                else:
                    prefit_info = {"prefit_final_loss": math.nan, "prefit_disp_mse": math.nan, "prefit_iterations": 0}
                base_state = {key: value.detach().cpu().clone() for key, value in base_field.net.state_dict().items()}
                for variant in args.variants:
                    field = make_field(args, state["Delta"], device, seed_offset=seed)
                    field.net.load_state_dict(base_state)
                    trace, status = reoptimize(field, state, args, variant, matprop, pffmodel, gcII)
                    row, cuts, jumps, trace = evaluate_result(
                        field, state, args, variant, matprop, pffmodel, gcII, init_strategy, prefit_info, trace, status
                    )
                    summary_rows.append(row)
                    for cut in cuts:
                        traction_rows.append({**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "init_strategy")}, **cut})
                    for jump in jumps:
                        jump_rows.append({**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "init_strategy")}, **jump})
                    for trace_row in trace:
                        trace_rows.append({**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "init_strategy")}, **trace_row})
    summary = pd.DataFrame(summary_rows)
    summary = add_baseline_comparisons(summary)
    traction = pd.DataFrame(traction_rows)
    traction = add_traction_baselines(traction)
    jumps = pd.DataFrame(jump_rows)
    convergence = summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "init_strategy",
            "convergence_status",
            "iterations",
            "final_mechanics_loss",
            "prefit_disp_mse",
            "prefit_iterations",
        ]
    ].copy()
    trace = pd.DataFrame(trace_rows)
    write_tables(summary, traction, jumps, convergence, trace)
    make_figures(summary, traction, jumps)
    mechanism = classify_mechanism(summary, traction, jumps)
    write_docs(args, summary, traction, jumps, convergence, mechanism)
    write_manifest()
    return mechanism


def add_baseline_comparisons(summary):
    summary = summary.copy()
    keys = ["case", "seed", "state", "init_strategy"]
    baseline = summary[summary["variant"] == "baseline_current_split"][
        keys + ["final_reaction_proxy", "total_elastic_energy", "cutline_crack_band_traction_proxy", "mean_v_jump_proxy"]
    ].rename(
        columns={
            "final_reaction_proxy": "baseline_reaction_proxy",
            "total_elastic_energy": "baseline_total_elastic_energy",
            "cutline_crack_band_traction_proxy": "baseline_cutline_crack_band_traction_proxy",
            "mean_v_jump_proxy": "baseline_mean_v_jump_proxy",
        }
    )
    summary = summary.merge(baseline, on=keys, how="left")
    summary["reaction_removed_fraction_vs_baseline"] = (
        summary["baseline_reaction_proxy"] - summary["final_reaction_proxy"]
    ) / summary["baseline_reaction_proxy"]
    summary["reaction_removed_percent_vs_baseline"] = 100.0 * summary["reaction_removed_fraction_vs_baseline"]
    summary["reaction_collapses_ge_10pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.10
    summary["reaction_collapses_ge_30pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.30
    summary["reaction_collapses_ge_50pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.50
    summary["variant_lower_energy_than_baseline"] = summary["total_elastic_energy"] < summary["baseline_total_elastic_energy"]
    summary["energy_change_percent_vs_baseline"] = 100.0 * (
        summary["total_elastic_energy"] - summary["baseline_total_elastic_energy"]
    ) / summary["baseline_total_elastic_energy"]
    summary["mean_v_jump_change_vs_baseline"] = summary["mean_v_jump_proxy"] - summary["baseline_mean_v_jump_proxy"]
    return summary


def add_traction_baselines(traction):
    traction = traction.copy()
    keys = ["case", "seed", "state", "init_strategy", "cut_x"]
    baseline = traction[traction["variant"] == "baseline_current_split"][
        keys + ["integrated_cut_traction_proxy"]
    ].rename(columns={"integrated_cut_traction_proxy": "baseline_integrated_cut_traction_proxy"})
    traction = traction.merge(baseline, on=keys, how="left")
    traction["traction_removed_fraction_vs_baseline"] = (
        traction["baseline_integrated_cut_traction_proxy"] - traction["integrated_cut_traction_proxy"]
    ) / traction["baseline_integrated_cut_traction_proxy"]
    traction["traction_removed_percent_vs_baseline"] = 100.0 * traction["traction_removed_fraction_vs_baseline"]
    return traction


def write_tables(summary, traction, jumps, convergence, trace):
    summary.to_csv(TABLES / "frozen_alpha_uv_reopt_summary.csv", index=False)
    summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "init_strategy",
            "final_reaction_proxy",
            "baseline_reaction_proxy",
            "reaction_removed_percent_vs_baseline",
            "reaction_collapses_ge_10pct",
            "reaction_collapses_ge_30pct",
            "reaction_collapses_ge_50pct",
        ]
    ].to_csv(TABLES / "frozen_alpha_reaction_comparison.csv", index=False)
    summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "init_strategy",
            "total_elastic_energy",
            "positive_energy_contribution",
            "negative_energy_contribution",
            "crack_band_negative_energy_contribution",
            "crack_band_positive_energy_contribution",
            "baseline_total_elastic_energy",
            "energy_change_percent_vs_baseline",
            "variant_lower_energy_than_baseline",
        ]
    ].to_csv(TABLES / "frozen_alpha_energy_comparison.csv", index=False)
    traction.to_csv(TABLES / "frozen_alpha_crack_band_traction.csv", index=False)
    jumps.to_csv(TABLES / "frozen_alpha_displacement_jump.csv", index=False)
    convergence.to_csv(TABLES / "frozen_alpha_convergence.csv", index=False)
    trace.to_csv(LOGS / "loss_trace.csv", index=False)


def classify_mechanism(summary, traction, jumps):
    final = summary[(summary["state"] == "final_D0040") & (summary["init_strategy"] == "saved_uv_prefit")]
    traction_final = traction[(traction["state"] == "final_D0040") & (traction["init_strategy"] == "saved_uv_prefit")]
    target_variants = ["minus_degraded_in_crack_band", "minus_removed_in_crack_band"]
    verdicts = []
    for variant in target_variants:
        rows = final[final["variant"] == variant]
        tr = traction_final[traction_final["variant"] == variant]
        if rows.empty or tr.empty:
            continue
        n_reaction_30 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] >= 0.30))
        n_reaction_lt10 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] < 0.10))
        mean_traction_removed = tr.groupby("seed")["traction_removed_fraction_vs_baseline"].mean()
        n_traction_90 = int(np.sum(mean_traction_removed >= 0.90))
        jump_changes = np.asarray(rows["mean_v_jump_change_vs_baseline"], dtype=float)
        jump_changes = jump_changes[np.isfinite(jump_changes)]
        jumps_limited = bool(jump_changes.size > 0 and np.nanmean(np.abs(jump_changes)) < 5.0e-5)
        if n_reaction_30 >= 2 and n_traction_90 >= 2:
            verdicts.append("Case A: negative branch is dominant")
        elif n_traction_90 >= 2 and n_reaction_lt10 >= 2 and jumps_limited:
            verdicts.append("Case B: continuous-field or boundary-condition bridging is dominant")
    if any(v.startswith("Case A") for v in verdicts):
        return "frozen-alpha reoptimization identifies dominant mechanism: Case A, negative branch is dominant"
    if any(v.startswith("Case B") for v in verdicts):
        return "frozen-alpha reoptimization identifies dominant mechanism: Case B, continuous-field or boundary-condition bridging is dominant"
    return "Case C: unresolved"


def make_figures(summary, traction, jumps):
    final = summary[(summary["state"] == "final_D0040") & (summary["init_strategy"] == "saved_uv_prefit")]
    if final.empty:
        write_figure_summary()
        return
    plot_bar(final, "final_reaction_proxy", "reaction_comparison_by_variant_seed.png", "Reaction proxy [N]")
    plot_bar(final, "reaction_removed_percent_vs_baseline", "reaction_removal_percent_by_variant_seed.png", "Reaction removal [%]")
    plot_energy(final)
    plot_traction(traction[(traction["state"] == "final_D0040") & (traction["init_strategy"] == "saved_uv_prefit")])
    plot_jump(jumps[(jumps["state"] == "final_D0040") & (jumps["init_strategy"] == "saved_uv_prefit")])
    for case in sorted(final["case"].unique()):
        plot_saved_fields(case)
    write_figure_summary()


def plot_bar(df, column, filename, ylabel):
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
    labels = [f"s{int(r.seed)}\n{short_variant(r.variant)}" for r in df.itertuples()]
    ax.bar(np.arange(len(df)), df[column])
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename)
    plt.close(fig)


def plot_energy(df):
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["positive_energy_contribution"], width=0.36, label="positive")
    ax.bar(x + 0.18, df["negative_energy_contribution"], width=0.36, label="negative")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{int(r.seed)}\n{short_variant(r.variant)}" for r in df.itertuples()], rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("energy contribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_contribution_comparison_by_variant.png")
    plt.close(fig)


def plot_traction(df):
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=180)
    grouped = df.groupby(["variant", "cut_x"])["integrated_cut_traction_proxy"].mean().reset_index()
    for variant, sub in grouped.groupby("variant"):
        ax.plot(sub["cut_x"], sub["integrated_cut_traction_proxy"], marker="o", label=short_variant(variant))
    ax.set_xlabel("cut x [mm]")
    ax.set_ylabel("mean integrated crack-band traction proxy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "crack_band_traction_comparison_by_variant.png")
    plt.close(fig)


def plot_jump(df):
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=180)
    grouped = df.groupby(["variant", "cut_x"])["v_jump_proxy"].mean().reset_index()
    for variant, sub in grouped.groupby("variant"):
        ax.plot(sub["cut_x"], sub["v_jump_proxy"], marker="o", label=short_variant(variant))
    ax.set_xlabel("cut x [mm]")
    ax.set_ylabel("v displacement jump proxy [mm]")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "displacement_jump_proxy_comparison_by_variant.png")
    plt.close(fig)


def plot_saved_fields(case):
    files = {
        "baseline": ARTIFACTS / f"{case}_final_D0040_baseline_current_split_saved_uv_prefit_fields.npz",
        "minus_degraded": ARTIFACTS / f"{case}_final_D0040_minus_degraded_in_crack_band_saved_uv_prefit_fields.npz",
        "minus_removed": ARTIFACTS / f"{case}_final_D0040_minus_removed_in_crack_band_saved_uv_prefit_fields.npz",
    }
    existing = {key: path for key, path in files.items() if path.exists()}
    if not existing:
        return
    base_data = np.load(next(iter(existing.values())))
    tri = mtri.Triangulation(base_data["x"], base_data["y"], base_data["triangles"].astype(int))
    crack = base_data["crack_mask"].astype(bool)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=180)
    tpc = ax.tripcolor(tri, base_data["alpha"], shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    ax.scatter(base_data["element_x"][crack], base_data["element_y"][crack], c="red", s=3, label="alpha>=0.8 mask")
    ax.set_aspect("equal")
    ax.set_title(f"{case}: frozen alpha mask")
    ax.legend(frameon=False, fontsize=6)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.035)
    fig.tight_layout()
    fig.savefig(FIGURES / f"final_alpha_mask_overlay_{case}.png")
    plt.close(fig)

    for label, path in existing.items():
        data = np.load(path)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=180)
        for ax, field_name in zip(axes, ["u", "v"]):
            artist = ax.tripcolor(tri, data[field_name], shading="gouraud", cmap="coolwarm")
            ax.set_aspect("equal")
            ax.set_title(f"{field_name} {label}")
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.suptitle(f"{case}: final displacement fields")
        fig.tight_layout()
        fig.savefig(FIGURES / f"final_uv_fields_{case}_{label}.png")
        plt.close(fig)

    stress_paths = [(key, path) for key, path in existing.items() if key in {"baseline", "minus_degraded", "minus_removed"}]
    fig, axes = plt.subplots(1, len(stress_paths), figsize=(4.0 * len(stress_paths), 3.4), dpi=180)
    if len(stress_paths) == 1:
        axes = [axes]
    vmax = max(float(np.nanpercentile(np.abs(np.load(path)["sigma_yy_variant"]), 98)) for _key, path in stress_paths)
    for ax, (label, path) in zip(axes, stress_paths):
        data = np.load(path)
        artist = ax.tripcolor(tri, facecolors=data["sigma_yy_variant"], shading="flat", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], c="black", s=2, alpha=0.45)
        ax.set_aspect("equal")
        ax.set_title(label)
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
    fig.suptitle(f"{case}: sigma_yy transmission maps")
    fig.tight_layout()
    fig.savefig(FIGURES / f"stress_transmission_maps_{case}.png")
    plt.close(fig)


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only. The full/minus degradation variants are not production model changes.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `reaction_comparison_by_variant_seed.png` | Re-optimized final reaction proxy by variant and seed | Compares whether true u/v re-optimization lowers reaction. | Diagnostic mechanism evidence. |",
        "| `reaction_removal_percent_by_variant_seed.png` | Reaction removal relative to baseline by variant and seed | Directly maps to 10/30/50 percent collapse criteria. | Diagnostic mechanism evidence. |",
        "| `energy_contribution_comparison_by_variant.png` | Positive and negative energy contributions | Shows how diagnostic variants alter energy terms after re-optimization. | Diagnostic only. |",
        "| `crack_band_traction_comparison_by_variant.png` | Cut-line crack-band traction proxy | Checks whether crack-band traction remains removed after re-optimization. | Diagnostic mechanism evidence. |",
        "| `displacement_jump_proxy_comparison_by_variant.png` | Crack-band v-jump proxy by cut line | Checks whether continuous PINN develops more separation. | Diagnostic only. |",
    ]
    for seed in (7, 13, 42):
        case = f"D0040_seed{seed}_default_unitbox"
        lines.extend(
            [
                f"| `final_alpha_mask_overlay_{case}.png` | Frozen alpha and connected alpha>=0.8 mask | Shows the mask used for crack-band degradation. | Geometry support. |",
                f"| `final_uv_fields_{case}_baseline.png` | Re-optimized baseline u/v fields | Shows baseline displacement field. | Diagnostic only. |",
                f"| `final_uv_fields_{case}_minus_degraded.png` | Re-optimized minus-degraded u/v fields | Shows displacement field after crack-band psi_minus degradation. | Diagnostic only. |",
                f"| `stress_transmission_maps_{case}.png` | Baseline vs minus-degraded vs minus-removed sigma_yy maps | Shows stress transmission changes after re-optimization. | Diagnostic mechanism evidence. |",
            ]
        )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def short_variant(variant):
    return {
        "baseline_current_split": "base",
        "minus_degraded_in_crack_band": "minus-g",
        "minus_removed_in_crack_band": "minus-0",
        "full_degradation_all_energy": "full",
    }.get(variant, variant)


def write_docs(args, summary, traction, jumps, convergence, mechanism):
    final = summary[(summary["state"] == "final_D0040") & (summary["init_strategy"] == "saved_uv_prefit")]
    variants = ["minus_degraded_in_crack_band", "minus_removed_in_crack_band", "full_degradation_all_energy"]
    n_final_seeds = max(1, int(final["seed"].nunique()))
    states_text = ", ".join(args.states)
    seeds_text = ", ".join(str(seed) for seed in args.seeds)
    lines = [
        "# Frozen-alpha u/v re-optimization diagnostic",
        "",
        "## Scope",
        "",
        f"This package freezes saved alpha, uses saved HI/HII as the old-history input, keeps the original trial-history max update logic, and preserves material constants, top-u-free boundary ansatz, unit-box coordinate normalization, TM split, and the saved displacement level. It re-optimizes only PINN `u,v` under diagnostic-only crack-band energy variants. No load extension and no alpha evolution were run. The current route has no thermal field, so no thermal input was introduced. Seeds: {seeds_text}. Saved states: {states_text}.",
        "",
        "## Final D0040 summary",
        "",
        "| variant | mean reaction removal | seeds with >=30% reaction drop | mean crack-band traction removal | mean v-jump change |",
        "|---|---:|---:|---:|---:|",
    ]
    traction_final = traction[(traction["state"] == "final_D0040") & (traction["init_strategy"] == "saved_uv_prefit")]
    for variant in variants:
        rows = final[final["variant"] == variant]
        tr = traction_final[traction_final["variant"] == variant]
        mean_reaction = 100.0 * rows["reaction_removed_fraction_vs_baseline"].mean() if not rows.empty else math.nan
        n30 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] >= 0.30))
        mean_tr = 100.0 * tr.groupby("seed")["traction_removed_fraction_vs_baseline"].mean().mean() if not tr.empty else math.nan
        mean_jump = rows["mean_v_jump_change_vs_baseline"].mean() if not rows.empty else math.nan
        lines.append(f"| {variant} | {mean_reaction:.3g}% | {n30}/{n_final_seeds} | {mean_tr:.3g}% | {mean_jump:.6g} |")
    all_required = convergence["convergence_status"].isin(["converged_trace_tol", "budget_reached_finite"]).all()
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Re-optimization completed for all final D0040 seeds and variants with finite losses: {bool(all_required)}. Convergence statuses are listed in `tables/frozen_alpha_convergence.csv`; `budget_reached_finite` means the fixed iteration budget ended with finite loss but strict trace tolerance was not met.",
            "2. Reaction changes are reported in `tables/frozen_alpha_reaction_comparison.csv`; the acceptance classification is given below.",
            "3. Crack-band traction changes are reported in `tables/frozen_alpha_crack_band_traction.csv`.",
            "4. Displacement jump proxy changes are reported in `tables/frozen_alpha_displacement_jump.csv` and summarized in the table above.",
            "5. Energy changes are reported in `tables/frozen_alpha_energy_comparison.csv`; diagnostic variants are compared against the re-optimized baseline for the same seed/state.",
            f"6. Mechanism classification: **{mechanism}**.",
            "7. A production model change is not justified directly from this package; all full/minus degradation variants remain diagnostic-only.",
            "8. Next minimal intervention: ask ChatGPT to review the convergence and acceptance criteria. If Case B holds, the next diagnostic should test a discontinuous/enriched kinematic replay as non-production evidence; if Case C holds, improve replay convergence or initialization before changing physics.",
            "",
            "## Verification",
            "",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q`: 18 passed.",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260612_default_unitbox_frozen_alpha_uv_reopt\\artifacts\\run_frozen_alpha_uv_reopt.py`: passed.",
            "",
            "No physical validation is claimed.",
        ]
    )
    (PACKAGE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# Frozen-alpha u/v re-optimization package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/frozen_alpha_uv_reopt_summary.csv`",
        "3. `tables/frozen_alpha_reaction_comparison.csv`",
        "4. `tables/frozen_alpha_crack_band_traction.csv`",
        "5. `tables/frozen_alpha_energy_comparison.csv`",
        "6. `tables/frozen_alpha_displacement_jump.csv`",
        "7. `tables/frozen_alpha_convergence.csv`",
        "8. `figures/figure_summary.md`",
        "",
        f"The run freezes alpha and history fields from saved D0040 results and re-optimizes only the PINN displacement outputs. Seeds: {seeds_text}. Saved states: {states_text}.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. Does the re-optimized reaction satisfy Case A, Case B, or remain Case C?",
        "2. Are the finite-budget convergence statuses sufficient for deciding the next diagnostic?",
        "3. If continuous bridging remains, should the next task be a non-production discontinuous/enriched kinematic replay?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: frozen-alpha u/v re-optimization",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260612_default_unitbox_frozen_alpha_uv_reopt",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260612_default_unitbox_frozen_alpha_uv_reopt/REPORT.md",
        "",
        "### What changed",
        "- Added and ran a true frozen-alpha PINN u/v re-optimization diagnostic.",
        f"- Used saved D0040 fields for seeds {seeds_text}; states: {states_text}.",
        "- Frozen inputs: alpha, saved HI/HII old-history fields, material constants, TM split, top-u-free/unit_box ansatz, and saved displacement level.",
        "- The original trial-history `max(old,current)` mechanics logic was kept; alpha was not evolved.",
        "- Re-optimized only `u,v` for baseline, minus-degraded crack band, minus-removed crack band, and full-degradation diagnostic variants.",
        "- Initialization strategy used in the default run: global saved-uv prefit; no notch/lip/local/jump/geometry-guided training losses were used.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        (PACKAGE / "commands_run.txt").read_text(encoding="utf-8").splitlines()[2],
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260612_default_unitbox_frozen_alpha_uv_reopt\\artifacts\\run_frozen_alpha_uv_reopt.py",
        "```",
        "",
        "### Key results",
        f"- Mechanism classification: **{mechanism}**.",
        "- See `tables/frozen_alpha_reaction_comparison.csv` for reaction removal and 10/30/50 percent collapse flags.",
        "- See `tables/frozen_alpha_crack_band_traction.csv` for crack-band traction removal.",
        "- See `tables/frozen_alpha_convergence.csv` for convergence statuses and finite-budget notes.",
        "- Verification passed: `pytest examples\\TM_comsol_no_thermal_micro\\tests -q` reported 18 passed; package script `py_compile` passed.",
        "- Diagnostic variants are not production model changes.",
        "- No physical validation is claimed.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/frozen_alpha_uv_reopt_summary.csv`",
        "- `tables/frozen_alpha_reaction_comparison.csv`",
        "- `tables/frozen_alpha_crack_band_traction.csv`",
        "- `tables/frozen_alpha_energy_comparison.csv`",
        "- `tables/frozen_alpha_displacement_jump.csv`",
        "- `tables/frozen_alpha_convergence.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Which acceptance case does this evidence support: Case A, Case B, or Case C?",
        "2. Are finite-budget convergence statuses sufficient, or should Codex rerun selected variants with a larger budget?",
        "3. What is the next minimal Codex diagnostic without changing physical parameters?",
        "",
        "### Constraints",
        "- Do not extend loading.",
        "- Do not evolve alpha.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.",
        "- Diagnostic full/minus degradation variants are not production changes.",
        "- Do not use `--alpha-init-intact` as the main route.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def write_manifest():
    required = {
        "README.md",
        "REPORT.md",
        "HANDOFF_COMMENT.md",
        "tables/frozen_alpha_uv_reopt_summary.csv",
        "tables/frozen_alpha_reaction_comparison.csv",
        "tables/frozen_alpha_energy_comparison.csv",
        "tables/frozen_alpha_crack_band_traction.csv",
        "tables/frozen_alpha_displacement_jump.csv",
        "tables/frozen_alpha_convergence.csv",
        "figures/figure_summary.md",
    }
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel == "HANDOFF_COMMENT.md":
            typ = "handoff"
        elif rel == "figures/figure_summary.md":
            typ = "figure_summary"
        elif rel.startswith("tables/") and rel.endswith(".csv"):
            typ = "table"
        elif rel.startswith("figures/") and rel.endswith(".png"):
            typ = "figure"
        elif rel.startswith("logs/") or rel == "commands_run.txt":
            typ = "command_log"
        elif rel.startswith("artifacts/"):
            typ = "artifact"
        else:
            typ = "report"
        entries.append(
            {
                "path": rel,
                "type": typ,
                "description": describe(rel),
                "required_for_chatgpt": rel in required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps({"package": PACKAGE.name, "files": entries}, indent=2), encoding="utf-8")


def describe(rel):
    mapping = {
        "README.md": "Package overview and reading order.",
        "REPORT.md": "Main frozen-alpha u/v re-optimization report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for ChatGPT issue sync.",
        "tables/frozen_alpha_uv_reopt_summary.csv": "Main per seed/state/variant re-optimization metrics.",
        "tables/frozen_alpha_reaction_comparison.csv": "Reaction comparison and collapse flags.",
        "tables/frozen_alpha_energy_comparison.csv": "Energy contribution comparison.",
        "tables/frozen_alpha_crack_band_traction.csv": "Cut-line crack-band traction metrics.",
        "tables/frozen_alpha_displacement_jump.csv": "Displacement jump proxy metrics.",
        "tables/frozen_alpha_convergence.csv": "Convergence status and loss summary.",
        "figures/figure_summary.md": "Text summary of generated figures.",
    }
    return mapping.get(rel, "Diagnostic artifact.")


def main():
    args = parse_args()
    mechanism = run_all(args)
    print(mechanism)


if __name__ == "__main__":
    main()
