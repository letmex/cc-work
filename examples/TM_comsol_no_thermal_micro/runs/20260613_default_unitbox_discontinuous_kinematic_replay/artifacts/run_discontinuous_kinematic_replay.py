"""Diagnostic-only discontinuous/split-domain frozen-alpha kinematic replay.

This script reads the saved D0040 default unit_box fields, freezes alpha and
saved HI/HII history, and re-optimizes only displacement fields.  The split
variants use independent upper/lower displacement networks selected by a
domain label derived from the connected alpha>=0.8 crack band.  This is a
diagnostic replay, not a production training route.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
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
RUNS_ROOT = PACKAGE.parents[0]
PREV_PACKAGE = RUNS_ROOT / "20260612_default_unitbox_frozen_alpha_uv_reopt"
PREV_SCRIPT = PREV_PACKAGE / "artifacts" / "run_frozen_alpha_uv_reopt.py"
PREV_SUMMARY = PREV_PACKAGE / "tables" / "frozen_alpha_uv_reopt_summary.csv"


def _load_prev_module():
    spec = importlib.util.spec_from_file_location("frozen_alpha_uv_reopt_prev", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load previous diagnostic script: {PREV_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prev = _load_prev_module()

SPECIMEN_SIZE_MM = prev.SPECIMEN_SIZE_MM
NOTCH_X = prev.NOTCH_X
NOTCH_Y = prev.NOTCH_Y
CUT_XS = prev.CUT_XS
CUT_TOL = prev.CUT_TOL
CRACK_BAND_Y_TOL = prev.CRACK_BAND_Y_TOL
STATE_STEPS = prev.STATE_STEPS

KINEMATIC_VARIANTS = (
    "continuous_baseline",
    "split_domain_current_split",
    "split_domain_minus_degraded_crack_band",
    "split_domain_crack_band_void",
)
SPLIT_VARIANTS = {
    "split_domain_current_split",
    "split_domain_minus_degraded_crack_band",
    "split_domain_crack_band_void",
}
FINITE_STATUSES = {"converged_trace_tol", "budget_reached_finite"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run split-domain frozen-alpha kinematic replay.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    parser.add_argument("--states", nargs="+", default=["final_D0040"], choices=sorted(STATE_STEPS))
    parser.add_argument("--variants", nargs="+", default=list(KINEMATIC_VARIANTS), choices=list(KINEMATIC_VARIANTS))
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
    parser.add_argument("--baseline-tolerance-percent", type=float, default=5.0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def setup_dirs():
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def choose_device(name: str) -> str:
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
    args.hidden_layers = min(args.hidden_layers, 2)
    args.neurons = min(args.neurons, 30)
    args.prefit_epochs = min(args.prefit_epochs, 8)
    args.reopt_epochs = min(args.reopt_epochs, 8)


def make_continuous_field(args, delta, device, seed_offset):
    return prev.make_field(args, delta, device, seed_offset=seed_offset)


class SplitFieldComputation:
    """Two independent top-u-free/unit_box displacement fields selected by node label."""

    def __init__(self, args, delta, device, node_upper_mask, seed_offset):
        self.upper = prev.make_field(args, delta, device, seed_offset=seed_offset * 11 + 1)
        self.lower = prev.make_field(args, delta, device, seed_offset=seed_offset * 11 + 2)
        self.node_upper_mask = torch.as_tensor(node_upper_mask, dtype=torch.bool, device=device)

    def parameters(self):
        return list(self.upper.net.parameters()) + list(self.lower.net.parameters())

    def state_dict_pair(self):
        return {
            "upper": {key: value.detach().cpu().clone() for key, value in self.upper.net.state_dict().items()},
            "lower": {key: value.detach().cpu().clone() for key, value in self.lower.net.state_dict().items()},
        }

    def load_state_dict_pair(self, state):
        self.upper.net.load_state_dict(state["upper"])
        self.lower.net.load_state_dict(state["lower"])

    def fieldCalculation(self, inp):
        u_up, v_up, alpha_up = self.upper.fieldCalculation(inp)
        u_lo, v_lo, alpha_lo = self.lower.fieldCalculation(inp)
        mask = self.node_upper_mask
        return torch.where(mask, u_up, u_lo), torch.where(mask, v_up, v_lo), torch.where(mask, alpha_up, alpha_lo)

    def side_field_difference(self, inp):
        u_up, v_up, _ = self.upper.fieldCalculation(inp)
        u_lo, v_lo, _ = self.lower.fieldCalculation(inp)
        return u_up - u_lo, v_up - v_lo


def make_split_field(args, state, device, seed_offset):
    return SplitFieldComputation(args, state["Delta"], device, state["node_upper_np"], seed_offset)


def field_parameters(field):
    if hasattr(field, "parameters"):
        return field.parameters()
    return field.net.parameters()


def clone_field_state(field):
    if hasattr(field, "state_dict_pair"):
        return field.state_dict_pair()
    return {key: value.detach().cpu().clone() for key, value in field.net.state_dict().items()}


def load_field_state(field, state):
    if hasattr(field, "load_state_dict_pair"):
        field.load_state_dict_pair(state)
    else:
        field.net.load_state_dict(state)


def crack_path_profile(state, bins=80):
    mask = state["crack_mask_np"]
    x = state["element_x_np"][mask]
    y = state["element_y_np"][mask]
    if x.size < 2:
        return np.array([0.0, SPECIMEN_SIZE_MM]), np.array([NOTCH_Y, NOTCH_Y])
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    xs = []
    ys = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x >= left) & (x <= right)
        if np.any(in_bin):
            xs.append(0.5 * (left + right))
            ys.append(float(np.median(y[in_bin])))
    if len(xs) < 2:
        return np.array([0.0, SPECIMEN_SIZE_MM]), np.array([float(np.median(y)), float(np.median(y))])
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    order = np.argsort(xs)
    return xs[order], ys[order]


def crack_y_at(x_query, path_x, path_y):
    return np.interp(x_query, path_x, path_y, left=float(path_y[0]), right=float(path_y[-1]))


def attach_domain_split(state):
    path_x, path_y = crack_path_profile(state)
    node_path_y = crack_y_at(state["x_np"], path_x, path_y)
    elem_path_y = crack_y_at(state["element_x_np"], path_x, path_y)
    node_upper = state["y_np"] >= node_path_y
    elem_upper = state["element_y_np"] >= elem_path_y
    tri = state["tri_np"].astype(int)
    tri_upper = node_upper[tri]
    elem_mixed = np.any(tri_upper != tri_upper[:, [0]], axis=1)
    state["crack_path_x_np"] = path_x
    state["crack_path_y_np"] = path_y
    state["node_upper_np"] = node_upper
    state["elem_upper_np"] = elem_upper
    state["elem_mixed_domain_np"] = elem_mixed
    return state


def domain_split_audit_row(state):
    mask = state["crack_mask_np"]
    mixed = state["elem_mixed_domain_np"]
    upper_nodes = state["node_upper_np"]
    elem_upper = state["elem_upper_np"]
    reaches_right = bool(np.any(mask & (state["element_x_np"] >= SPECIMEN_SIZE_MM - 5.0e-4)))
    reaches_notch = bool(
        np.any(
            mask
            & (np.abs(state["element_x_np"] - NOTCH_X) <= 3.0e-4)
            & (np.abs(state["element_y_np"] - NOTCH_Y) <= 3.0e-4)
        )
    )
    if np.any(mask):
        crack_x_min = float(np.min(state["element_x_np"][mask]))
        crack_x_max = float(np.max(state["element_x_np"][mask]))
        crack_y_min = float(np.min(state["element_y_np"][mask]))
        crack_y_max = float(np.max(state["element_y_np"][mask]))
        crack_y_mean = float(np.mean(state["element_y_np"][mask]))
    else:
        crack_x_min = crack_x_max = crack_y_min = crack_y_max = crack_y_mean = math.nan
    return {
        "case": state["case"],
        "seed": state["seed"],
        "state": state["state"],
        "step": state["step"],
        "Delta": state["Delta"],
        "crack_band_element_count": int(np.sum(mask)),
        "crack_reaches_notch_window": reaches_notch,
        "crack_reaches_right_boundary": reaches_right,
        "crack_x_min": crack_x_min,
        "crack_x_max": crack_x_max,
        "crack_y_min": crack_y_min,
        "crack_y_max": crack_y_max,
        "crack_y_mean": crack_y_mean,
        "upper_node_count": int(np.sum(upper_nodes)),
        "lower_node_count": int(np.sum(~upper_nodes)),
        "upper_element_count": int(np.sum(elem_upper)),
        "lower_element_count": int(np.sum(~elem_upper)),
        "mixed_label_element_count": int(np.sum(mixed)),
        "mixed_label_inside_crack_band_count": int(np.sum(mixed & mask)),
        "mixed_label_outside_crack_band_count": int(np.sum(mixed & ~mask)),
        "path_point_count": int(len(state["crack_path_x_np"])),
        "split_y_min": float(np.min(state["crack_path_y_np"])),
        "split_y_max": float(np.max(state["crack_path_y_np"])),
    }


def load_state(seed, state_name, device):
    state = prev.load_state(seed, state_name, device)
    return attach_domain_split(state)


def compute_fields_for_field(field, state, args, matprop, pffmodel, gcII):
    u, v, _alpha_pred = field.fieldCalculation(state["inp"])
    fields = prev.compute_mixed_tm_fields(
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
    if variant in {"continuous_baseline", "split_domain_current_split"}:
        return prev.variant_density(fields, crack_mask, "baseline_current_split")
    if variant == "split_domain_minus_degraded_crack_band":
        return prev.variant_density(fields, crack_mask, "minus_degraded_in_crack_band")
    if variant == "split_domain_crack_band_void":
        density, pos, neg = prev.variant_density(fields, crack_mask, "baseline_current_split")
        density = density.clone()
        pos = pos.clone()
        neg = neg.clone()
        density[crack_mask] = 0.0
        pos[crack_mask] = 0.0
        neg[crack_mask] = 0.0
        return density, pos, neg
    raise ValueError(variant)


def stress_variant(fields, crack_mask, variant):
    if variant in {"continuous_baseline", "split_domain_current_split"}:
        return prev.stress_variant(fields, crack_mask, "baseline_current_split")
    if variant == "split_domain_minus_degraded_crack_band":
        return prev.stress_variant(fields, crack_mask, "minus_degraded_in_crack_band")
    if variant == "split_domain_crack_band_void":
        yy, xy = prev.stress_variant(fields, crack_mask, "baseline_current_split")
        yy = yy.clone()
        xy = xy.clone()
        yy[crack_mask] = 0.0
        xy[crack_mask] = 0.0
        return yy, xy
    raise ValueError(variant)


def mechanics_loss(field, state, args, variant, matprop, pffmodel, gcII):
    _u, _v, fields = compute_fields_for_field(field, state, args, matprop, pffmodel, gcII)
    density, _pos, _neg = variant_density(fields, state["crack_mask"], variant)
    energy = torch.sum(state["area"] * density)
    return torch.log10(torch.clamp(energy, min=torch.finfo(energy.dtype).tiny)), energy, fields


def prefit_saved_uv(field, state, args):
    optimizer = torch.optim.Rprop(field_parameters(field), lr=args.prefit_lr, step_sizes=(1.0e-10, 1.0))
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


def reoptimize(field, state, args, variant, matprop, pffmodel, gcII):
    optimizer = torch.optim.Rprop(field_parameters(field), lr=args.reopt_lr, step_sizes=(1.0e-10, 1.0))
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
    return trace, prev.convergence_status(trace)


def side_separation_metrics(field, state):
    if not hasattr(field, "side_field_difference"):
        return {"upper_lower_field_separation_magnitude": 0.0, "near_crack_separation_node_count": 0}
    with torch.no_grad():
        du, dv = field.side_field_difference(state["inp"])
        du_np = du.detach().cpu().numpy()
        dv_np = dv.detach().cpu().numpy()
    path_y = crack_y_at(state["x_np"], state["crack_path_x_np"], state["crack_path_y_np"])
    near = np.abs(state["y_np"] - path_y) <= 1.2e-3
    if not np.any(near):
        return {"upper_lower_field_separation_magnitude": math.nan, "near_crack_separation_node_count": 0}
    sep = np.sqrt(du_np[near] ** 2 + dv_np[near] ** 2)
    return {
        "upper_lower_field_separation_magnitude": float(np.mean(sep)),
        "near_crack_separation_node_count": int(np.sum(near)),
    }


def evaluate_result(field, state, args, variant, matprop, pffmodel, gcII, prefit_info, trace, status):
    with torch.no_grad():
        u, v, fields = compute_fields_for_field(field, state, args, matprop, pffmodel, gcII)
        density, pos, neg = variant_density(fields, state["crack_mask"], variant)
        yy, xy = stress_variant(fields, state["crack_mask"], variant)
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    yy_np = yy.detach().cpu().numpy()
    xy_np = xy.detach().cpu().numpy()
    density_np = density.detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    neg_np = neg.detach().cpu().numpy()
    areas = prev.triangle_areas_np(state["x_np"], state["y_np"], state["tri_np"])
    crack_mask = state["crack_mask_np"]
    cut_rows = []
    traction_total = 0.0
    traction_count = 0
    for cut_x in CUT_XS:
        row = prev.cut_traction_np(state, yy_np, xy_np, cut_x)
        cut_rows.append(row)
        if np.isfinite(row["integrated_cut_traction_proxy"]):
            traction_total += row["integrated_cut_traction_proxy"]
            traction_count += 1
    jump_rows = [prev.jump_proxy_np(state, u_np, v_np, cut_x) for cut_x in CUT_XS]
    mean_v_jump = float(np.nanmean([r["v_jump_proxy"] for r in jump_rows]))
    mean_u_jump = float(np.nanmean([r["u_jump_proxy"] for r in jump_rows]))
    row = {
        "case": state["case"],
        "seed": state["seed"],
        "state": state["state"],
        "step": state["step"],
        "Delta": state["Delta"],
        "variant": variant,
        "kinematic_mode": "split_domain" if variant in SPLIT_VARIANTS else "continuous",
        "source_path": state["source_path"],
        "iterations": args.reopt_epochs,
        "convergence_status": status,
        "final_mechanics_loss": trace[-1]["loss_log10"] if trace else math.nan,
        "final_reaction_proxy": prev.top_reaction_np(state["x_np"], state["y_np"], state["tri_np"], yy_np),
        "total_elastic_energy": float(np.sum(areas * density_np)),
        "positive_energy_contribution": float(np.sum(areas * pos_np)),
        "negative_energy_contribution": float(np.sum(areas * neg_np)),
        "crack_band_negative_energy_contribution": float(np.sum(areas[crack_mask] * neg_np[crack_mask])) if np.any(crack_mask) else math.nan,
        "crack_band_positive_energy_contribution": float(np.sum(areas[crack_mask] * pos_np[crack_mask])) if np.any(crack_mask) else math.nan,
        "cutline_crack_band_traction_proxy": traction_total / traction_count if traction_count else math.nan,
        "mean_v_jump_proxy": mean_v_jump,
        "mean_u_jump_proxy": mean_u_jump,
        **prev.crack_band_metrics_np(state, yy_np, xy_np),
        **side_separation_metrics(field, state),
        **prefit_info,
    }
    save_npz(state, variant, u_np, v_np, fields, yy_np, xy_np, density_np, pos_np, neg_np)
    return row, cut_rows, jump_rows, trace


def save_npz(state, variant, u, v, fields, yy, xy, density, pos, neg):
    path = ARTIFACTS / f"{state['case']}_{state['state']}_{variant}_fields.npz"
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
        node_upper=state["node_upper_np"],
        elem_upper=state["elem_upper_np"],
        elem_mixed_domain=state["elem_mixed_domain_np"],
        crack_path_x=state["crack_path_x_np"],
        crack_path_y=state["crack_path_y_np"],
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


def add_baseline_comparisons(summary):
    summary = summary.copy()
    keys = ["case", "seed", "state"]
    baseline = summary[summary["variant"] == "continuous_baseline"][
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
    summary["reaction_drop_ge_10pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.10
    summary["reaction_drop_ge_30pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.30
    summary["reaction_drop_ge_50pct"] = summary["reaction_removed_fraction_vs_baseline"] >= 0.50
    summary["optimized_energy_lower_than_continuous_baseline"] = (
        summary["total_elastic_energy"] < summary["baseline_total_elastic_energy"]
    )
    summary["energy_change_percent_vs_baseline"] = 100.0 * (
        summary["total_elastic_energy"] - summary["baseline_total_elastic_energy"]
    ) / summary["baseline_total_elastic_energy"]
    summary["mean_v_jump_change_vs_baseline"] = summary["mean_v_jump_proxy"] - summary["baseline_mean_v_jump_proxy"]
    summary["mean_u_jump_change_vs_baseline"] = summary["mean_u_jump_proxy"] - summary.groupby(keys)[
        "mean_u_jump_proxy"
    ].transform(lambda s: s.iloc[0])
    summary = add_previous_baseline_comparison(summary)
    return summary


def add_previous_baseline_comparison(summary):
    summary = summary.copy()
    if not PREV_SUMMARY.exists():
        summary["previous_frozen_alpha_baseline_reaction_proxy"] = math.nan
        summary["continuous_vs_previous_baseline_reaction_diff_percent"] = math.nan
        return summary
    prev_summary = pd.read_csv(PREV_SUMMARY)
    prev_base = prev_summary[
        (prev_summary["variant"] == "baseline_current_split")
        & (prev_summary["state"].isin(summary["state"].unique()))
    ][["case", "seed", "state", "final_reaction_proxy"]].rename(
        columns={"final_reaction_proxy": "previous_frozen_alpha_baseline_reaction_proxy"}
    )
    summary = summary.merge(prev_base, on=["case", "seed", "state"], how="left")
    mask = summary["variant"] == "continuous_baseline"
    summary["continuous_vs_previous_baseline_reaction_diff_percent"] = math.nan
    summary.loc[mask, "continuous_vs_previous_baseline_reaction_diff_percent"] = 100.0 * (
        summary.loc[mask, "final_reaction_proxy"]
        - summary.loc[mask, "previous_frozen_alpha_baseline_reaction_proxy"]
    ) / summary.loc[mask, "previous_frozen_alpha_baseline_reaction_proxy"]
    return summary


def add_traction_baselines(traction):
    traction = traction.copy()
    keys = ["case", "seed", "state", "cut_x"]
    baseline = traction[traction["variant"] == "continuous_baseline"][
        keys + ["integrated_cut_traction_proxy"]
    ].rename(columns={"integrated_cut_traction_proxy": "baseline_integrated_cut_traction_proxy"})
    traction = traction.merge(baseline, on=keys, how="left")
    traction["traction_removed_fraction_vs_baseline"] = (
        traction["baseline_integrated_cut_traction_proxy"] - traction["integrated_cut_traction_proxy"]
    ) / traction["baseline_integrated_cut_traction_proxy"]
    traction["traction_removed_percent_vs_baseline"] = 100.0 * traction["traction_removed_fraction_vs_baseline"]
    return traction


def write_tables(summary, traction, jumps, convergence, trace, audit):
    summary.to_csv(TABLES / "discontinuous_replay_summary.csv", index=False)
    summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "kinematic_mode",
            "final_reaction_proxy",
            "baseline_reaction_proxy",
            "reaction_removed_percent_vs_baseline",
            "reaction_drop_ge_10pct",
            "reaction_drop_ge_30pct",
            "reaction_drop_ge_50pct",
            "previous_frozen_alpha_baseline_reaction_proxy",
            "continuous_vs_previous_baseline_reaction_diff_percent",
        ]
    ].to_csv(TABLES / "discontinuous_reaction_comparison.csv", index=False)
    summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "kinematic_mode",
            "total_elastic_energy",
            "positive_energy_contribution",
            "negative_energy_contribution",
            "crack_band_negative_energy_contribution",
            "crack_band_positive_energy_contribution",
            "baseline_total_elastic_energy",
            "energy_change_percent_vs_baseline",
            "optimized_energy_lower_than_continuous_baseline",
        ]
    ].to_csv(TABLES / "discontinuous_energy_comparison.csv", index=False)
    traction.to_csv(TABLES / "discontinuous_crack_band_traction.csv", index=False)
    jumps.to_csv(TABLES / "discontinuous_displacement_jump.csv", index=False)
    convergence.to_csv(TABLES / "discontinuous_convergence.csv", index=False)
    audit.to_csv(TABLES / "domain_split_geometry_audit.csv", index=False)
    trace.to_csv(LOGS / "loss_trace.csv", index=False)


def classify_diagnostic(summary, traction, jumps, audit, args):
    final = summary[summary["state"] == "final_D0040"].copy()
    if final.empty:
        return "diagnostic unresolved: final_D0040 rows are missing"
    invalid_geom = audit[
        (audit["state"] == "final_D0040")
        & (~audit["crack_reaches_notch_window"] | ~audit["crack_reaches_right_boundary"])
    ]
    if not invalid_geom.empty:
        return "diagnostic unresolved: upper/lower domain labeling is geometrically invalid for at least one final seed"
    required_variants = [v for v in args.variants if v in {"continuous_baseline", "split_domain_current_split", "split_domain_minus_degraded_crack_band"}]
    finite = final[final["variant"].isin(required_variants)].groupby("variant")["convergence_status"].apply(
        lambda s: int(np.sum(s.isin(FINITE_STATUSES)))
    )
    if any(finite.get(variant, 0) < 2 for variant in required_variants):
        return "diagnostic unresolved: split-domain optimization did not converge for enough required final seeds"
    cont = final[final["variant"] == "continuous_baseline"]
    close = np.abs(cont["continuous_vs_previous_baseline_reaction_diff_percent"]) <= args.baseline_tolerance_percent
    continuous_close = bool(close.sum() >= 2)
    if not continuous_close:
        return "diagnostic unresolved: continuous baseline did not reproduce previous frozen-alpha baseline within tolerance"
    verdicts = []
    for variant in [v for v in args.variants if v in SPLIT_VARIANTS]:
        rows = final[final["variant"] == variant]
        if rows.empty:
            continue
        n_reaction_30 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] >= 0.30))
        n_reaction_10 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] >= 0.10))
        n_jump_increase = int(np.sum(rows["mean_v_jump_change_vs_baseline"] > 0.0))
        tr = traction[(traction["state"] == "final_D0040") & (traction["variant"] == variant)]
        mean_tr = tr.groupby("seed")["traction_removed_fraction_vs_baseline"].mean()
        n_traction_low = int(np.sum(mean_tr >= 0.90)) if not mean_tr.empty else 0
        if n_reaction_30 >= 2 and n_jump_increase >= 2 and n_traction_low >= 2:
            verdicts.append("confirm")
        elif n_reaction_10 < 2 or n_jump_increase < 2:
            verdicts.append("not_confirmed")
    if "confirm" in verdicts:
        return "discontinuous kinematic replay confirms continuous-field bridging"
    if "not_confirmed" in verdicts:
        return "continuous-field bridging not confirmed"
    return "diagnostic unresolved: reaction/jump/traction criteria are mixed across split variants"


def make_figures(summary, traction, jumps):
    final = summary[summary["state"] == "final_D0040"]
    if final.empty:
        write_figure_summary()
        return
    plot_bar(final, "final_reaction_proxy", "reaction_comparison_by_variant_seed.png", "Reaction proxy [N]")
    plot_bar(final, "reaction_removed_percent_vs_baseline", "reaction_removal_percent_by_variant_seed.png", "Reaction removal [%]")
    plot_bar(final, "upper_lower_field_separation_magnitude", "upper_lower_separation_by_variant_seed.png", "Upper/lower field separation [mm]")
    plot_energy(final)
    plot_traction(traction[traction["state"] == "final_D0040"])
    plot_jump(jumps[jumps["state"] == "final_D0040"])
    for case in sorted(final["case"].unique()):
        plot_saved_fields(case)
    write_figure_summary()


def short_variant(variant):
    return {
        "continuous_baseline": "cont",
        "split_domain_current_split": "split",
        "split_domain_minus_degraded_crack_band": "split-g",
        "split_domain_crack_band_void": "split-void",
    }.get(variant, variant)


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
    fig.savefig(FIGURES / "crack_opening_v_jump_map_along_crack_path.png")
    plt.close(fig)


def plot_saved_fields(case):
    files = {
        variant: ARTIFACTS / f"{case}_final_D0040_{variant}_fields.npz" for variant in KINEMATIC_VARIANTS
    }
    existing = {key: path for key, path in files.items() if path.exists()}
    if not existing:
        return
    base_data = np.load(next(iter(existing.values())))
    tri = mtri.Triangulation(base_data["x"], base_data["y"], base_data["triangles"].astype(int))
    crack = base_data["crack_mask"].astype(bool)
    elem_upper = base_data["elem_upper"].astype(bool)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=180)
    face = np.where(elem_upper, 1.0, 0.0)
    artist = ax.tripcolor(tri, facecolors=face, shading="flat", cmap="coolwarm", vmin=0, vmax=1)
    ax.scatter(base_data["element_x"][crack], base_data["element_y"][crack], c="black", s=3, label="alpha>=0.8 connected band")
    ax.plot(base_data["crack_path_x"], base_data["crack_path_y"], c="lime", lw=1.0, label="split path")
    ax.set_aspect("equal")
    ax.set_title(f"{case}: upper/lower domain labels")
    ax.legend(frameon=False, fontsize=6)
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
    fig.tight_layout()
    fig.savefig(FIGURES / f"final_alpha_mask_domain_labels_{case}.png")
    plt.close(fig)

    for variant, path in existing.items():
        data = np.load(path)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=180)
        for ax, field_name in zip(axes, ["u", "v"]):
            artist = ax.tripcolor(tri, data[field_name], shading="gouraud", cmap="coolwarm")
            ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], c="black", s=1.5, alpha=0.35)
            ax.set_aspect("equal")
            ax.set_title(f"{field_name} {short_variant(variant)}")
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.suptitle(f"{case}: displacement fields")
        fig.tight_layout()
        fig.savefig(FIGURES / f"final_uv_fields_{case}_{short_variant(variant)}.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(existing), figsize=(4.0 * len(existing), 3.4), dpi=180)
    if len(existing) == 1:
        axes = [axes]
    vmax = max(float(np.nanpercentile(np.abs(np.load(path)["sigma_yy_variant"]), 98)) for path in existing.values())
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    for ax, (variant, path) in zip(axes, existing.items()):
        data = np.load(path)
        artist = ax.tripcolor(
            tri,
            facecolors=data["sigma_yy_variant"],
            shading="flat",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], c="black", s=2, alpha=0.45)
        ax.set_aspect("equal")
        ax.set_title(short_variant(variant))
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
    fig.suptitle(f"{case}: sigma_yy transmission maps")
    fig.tight_layout()
    fig.savefig(FIGURES / f"stress_transmission_maps_{case}.png")
    plt.close(fig)


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only. The split-domain/discontinuous representation is not a production model change.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `reaction_comparison_by_variant_seed.png` | Final reaction proxy by variant and seed | Compares continuous and split-domain kinematic replays. | Diagnostic mechanism evidence. |",
        "| `reaction_removal_percent_by_variant_seed.png` | Reaction removal relative to continuous baseline | Maps directly to the 10/30/50 percent criteria. | Diagnostic mechanism evidence. |",
        "| `upper_lower_separation_by_variant_seed.png` | Mean upper/lower field separation near the crack path | Checks whether split networks actually separate. | Diagnostic only. |",
        "| `energy_contribution_comparison_by_variant.png` | Positive and negative energy contributions | Shows how split kinematics and crack-band treatment change energy. | Diagnostic only. |",
        "| `crack_band_traction_comparison_by_variant.png` | Cut-line crack-band traction proxy | Checks whether local crack-band traction is suppressed. | Diagnostic mechanism evidence. |",
        "| `crack_opening_v_jump_map_along_crack_path.png` | v-jump proxy along cut locations | Checks crack opening/jump change along the path. | Diagnostic mechanism evidence. |",
    ]
    for seed in (7, 13, 42):
        case = f"D0040_seed{seed}_default_unitbox"
        lines.append(
            f"| `final_alpha_mask_domain_labels_{case}.png` | Frozen alpha mask with upper/lower domain labels | Audits split geometry and crack-path construction. | Geometry support. |"
        )
        for variant in KINEMATIC_VARIANTS:
            lines.append(
                f"| `final_uv_fields_{case}_{short_variant(variant)}.png` | u/v fields for {short_variant(variant)} | Shows whether the replay stays continuous or separates. | Diagnostic only. |"
            )
        lines.append(
            f"| `stress_transmission_maps_{case}.png` | sigma_yy maps for continuous and split variants | Compares post-crack stress transmission routes. | Diagnostic mechanism evidence. |"
        )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docs(args, summary, traction, jumps, convergence, audit, mechanism):
    seeds_text = ", ".join(str(seed) for seed in args.seeds)
    states_text = ", ".join(args.states)
    final = summary[summary["state"] == "final_D0040"]
    split_variants = [v for v in args.variants if v in SPLIT_VARIANTS]
    n_final_seeds = max(1, int(final["seed"].nunique()))
    lines = [
        "# Discontinuous/split-domain frozen-alpha kinematic replay",
        "",
        "## Scope",
        "",
        f"This diagnostic freezes alpha and saved HI/HII history from D0040 default unit_box states and re-optimizes only displacement fields. Seeds: {seeds_text}. States: {states_text}. The split-domain variants use independent upper/lower displacement networks selected by a connected alpha>=0.8 crack-path label. The original trial-history `max(old,current)` mechanics logic, material constants, TM split, top-u-free/unit_box ansatz, displacement level, and `l0` are preserved.",
        "",
        "The split-domain representation is diagnostic-only and is not a production formulation.",
        "",
        "## Domain split construction",
        "",
        "The connected alpha>=0.8 crack band is detected from the notch-tip window by element adjacency. A median crack path y(x) is interpolated from the connected band. Nodes above this path use the upper displacement field and nodes below use the lower displacement field. Crack-band elements are audited separately in `tables/domain_split_geometry_audit.csv`.",
        "",
        "## Final D0040 summary",
        "",
        "| variant | mean reaction removal | seeds with >=30% drop | mean v-jump change | mean crack-band traction removal |",
        "|---|---:|---:|---:|---:|",
    ]
    tr_final = traction[traction["state"] == "final_D0040"]
    for variant in split_variants:
        rows = final[final["variant"] == variant]
        tr = tr_final[tr_final["variant"] == variant]
        mean_reaction = 100.0 * rows["reaction_removed_fraction_vs_baseline"].mean() if not rows.empty else math.nan
        n30 = int(np.sum(rows["reaction_removed_fraction_vs_baseline"] >= 0.30))
        mean_jump = rows["mean_v_jump_change_vs_baseline"].mean() if not rows.empty else math.nan
        mean_tr = 100.0 * tr.groupby("seed")["traction_removed_fraction_vs_baseline"].mean().mean() if not tr.empty else math.nan
        lines.append(f"| {variant} | {mean_reaction:.3g}% | {n30}/{n_final_seeds} | {mean_jump:.6g} | {mean_tr:.3g}% |")
    finite_all = bool(convergence["convergence_status"].isin(FINITE_STATUSES).all())
    continuous_close_rows = final[final["variant"] == "continuous_baseline"]
    close_count = int(
        np.sum(
            np.abs(continuous_close_rows["continuous_vs_previous_baseline_reaction_diff_percent"])
            <= args.baseline_tolerance_percent
        )
    )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Split-domain/discontinuous replay reached finite losses for all requested rows: {finite_all}. Detailed statuses are in `tables/discontinuous_convergence.csv`.",
            "2. Reaction changes relative to the continuous baseline are in `tables/discontinuous_reaction_comparison.csv`.",
            "3. Crack opening/jump proxies are in `tables/discontinuous_displacement_jump.csv`.",
            "4. Crack-band traction proxies are in `tables/discontinuous_crack_band_traction.csv`.",
            "5. Energy comparisons are in `tables/discontinuous_energy_comparison.csv`.",
            f"6. Continuous baseline rows within {args.baseline_tolerance_percent:.1f}% of the previous frozen-alpha baseline: {close_count}/{n_final_seeds}.",
            f"7. Diagnostic classification: **{mechanism}**.",
            "8. No production model change is justified directly from this diagnostic package.",
            "9. Next minimal intervention: have ChatGPT review whether the split-domain result identifies continuous-field/boundary bridging or instead points to boundary-condition/reaction definition auditing.",
            "",
            "## Cannot conclude",
            "",
            "- This package does not validate a physical crack model.",
            "- This package does not justify changing material parameters, `l0`, TM split, or history update logic.",
            "- The split-domain representation is a diagnostic replay, not a production route.",
            "",
            "## Verification",
            "",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q`: to be filled after verification.",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260613_default_unitbox_discontinuous_kinematic_replay\\artifacts\\run_discontinuous_kinematic_replay.py`: to be filled after verification.",
        ]
    )
    (PACKAGE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# Discontinuous/split-domain kinematic replay package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/discontinuous_replay_summary.csv`",
        "3. `tables/discontinuous_reaction_comparison.csv`",
        "4. `tables/discontinuous_crack_band_traction.csv`",
        "5. `tables/discontinuous_displacement_jump.csv`",
        "6. `tables/discontinuous_energy_comparison.csv`",
        "7. `tables/domain_split_geometry_audit.csv`",
        "8. `figures/figure_summary.md`",
        "",
        "This package is diagnostic-only. It freezes alpha/history and tests whether split-domain kinematics change post-crack reaction transfer.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. Does the split-domain replay confirm, fail to confirm, or leave unresolved continuous-field/boundary-condition bridging?",
        "2. Is the crack-path/domain split geometrically valid enough for this diagnostic?",
        "3. What is the next minimal intervention without changing physical parameters?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: discontinuous/split-domain frozen-alpha kinematic replay",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260613_default_unitbox_discontinuous_kinematic_replay",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260613_default_unitbox_discontinuous_kinematic_replay/REPORT.md",
        "",
        "### What changed",
        "- Added and ran a diagnostic-only split-domain/discontinuous frozen-alpha kinematic replay.",
        f"- Used saved D0040 fields for seeds {seeds_text}; states: {states_text}.",
        "- Frozen inputs: alpha, saved HI/HII old-history fields, material constants, TM split, top-u-free/unit_box ansatz, saved displacement level, and `l0`.",
        "- Re-optimized only displacement fields. Alpha was not evolved.",
        "- Constructed upper/lower domain labels from the connected alpha>=0.8 crack band.",
        "- Evaluated continuous baseline, split-domain current split, split-domain minus-degraded crack band, and split-domain crack-band void diagnostics.",
        "- No notch/lip/local/jump/geometry-guided training losses were used.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_discontinuous_kinematic_replay.py --quick",
        build_invocation(args),
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260613_default_unitbox_discontinuous_kinematic_replay\\artifacts\\run_discontinuous_kinematic_replay.py",
        "```",
        "",
        "### Key results",
        f"- Diagnostic classification: **{mechanism}**.",
        "- See `tables/domain_split_geometry_audit.csv` for split construction validity.",
        "- See `tables/discontinuous_reaction_comparison.csv` for reaction removal and 10/30/50 percent flags.",
        "- See `tables/discontinuous_displacement_jump.csv` for crack opening/jump proxies.",
        "- See `tables/discontinuous_crack_band_traction.csv` for crack-band traction suppression.",
        "- See `tables/discontinuous_energy_comparison.csv` for energy changes.",
        "- No physical validation is claimed.",
        "- No production model change is justified directly.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/discontinuous_replay_summary.csv`",
        "- `tables/discontinuous_reaction_comparison.csv`",
        "- `tables/discontinuous_crack_band_traction.csv`",
        "- `tables/discontinuous_displacement_jump.csv`",
        "- `tables/discontinuous_energy_comparison.csv`",
        "- `tables/domain_split_geometry_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this evidence confirm continuous-field/boundary-condition bridging, fail to confirm it, or remain unresolved?",
        "2. Is the split-domain geometry valid enough to interpret the reaction comparison?",
        "3. What is the next minimal Codex diagnostic without changing physical parameters?",
        "",
        "### Constraints",
        "- Do not extend loading.",
        "- Do not evolve alpha.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.",
        "- The split-domain/discontinuous representation is diagnostic-only.",
        "- Do not use `--alpha-init-intact` as the main route.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def build_invocation(args):
    cmd = "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_discontinuous_kinematic_replay.py"
    if args.states != ["final_D0040"]:
        cmd += " --states " + " ".join(args.states)
    if args.seeds != [7, 13, 42]:
        cmd += " --seeds " + " ".join(str(seed) for seed in args.seeds)
    if args.variants != list(KINEMATIC_VARIANTS):
        cmd += " --variants " + " ".join(args.variants)
    return cmd


def write_manifest():
    entries = []
    type_by_name = {
        "README.md": "report",
        "REPORT.md": "report",
        "next_questions.md": "report",
        "HANDOFF_COMMENT.md": "handoff",
        "commands_run.txt": "command_log",
        "MANIFEST.json": "artifact",
    }
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tables/"):
            ftype = "table"
            required = True
        elif rel == "figures/figure_summary.md":
            ftype = "figure_summary"
            required = True
        elif rel.startswith("figures/"):
            ftype = "figure"
            required = False
        elif rel.startswith("artifacts/"):
            ftype = "artifact"
            required = False
        elif rel.startswith("logs/"):
            ftype = "command_log"
            required = False
        else:
            ftype = type_by_name.get(path.name, "artifact")
            required = path.name in {"README.md", "REPORT.md", "HANDOFF_COMMENT.md", "MANIFEST.json"}
        entries.append(
            {
                "path": rel,
                "type": ftype,
                "description": describe_file(rel),
                "required_for_chatgpt": required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def describe_file(rel):
    descriptions = {
        "README.md": "Package reading order and diagnostic scope.",
        "REPORT.md": "Main diagnostic report and interpretation.",
        "next_questions.md": "Questions for the next ChatGPT review.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff comment for issue #1.",
        "commands_run.txt": "Commands used for this package.",
        "tables/discontinuous_replay_summary.csv": "Main per-seed/state/variant metric table.",
        "tables/discontinuous_reaction_comparison.csv": "Reaction proxy and reaction removal criteria.",
        "tables/discontinuous_energy_comparison.csv": "Energy contribution comparison.",
        "tables/discontinuous_crack_band_traction.csv": "Crack-band traction proxy by cut line.",
        "tables/discontinuous_displacement_jump.csv": "Displacement jump proxy by cut line.",
        "tables/discontinuous_convergence.csv": "Optimization convergence statuses.",
        "tables/domain_split_geometry_audit.csv": "Connected crack-band and upper/lower domain split audit.",
        "figures/figure_summary.md": "Text summary for all generated figures.",
    }
    return descriptions.get(rel, "Generated diagnostic artifact.")


def run_all(args):
    setup_dirs()
    quick_trim(args)
    device = choose_device(args.device)
    commands = [
        "git pull origin main",
        "Read previous frozen-alpha u/v re-optimization package files.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_discontinuous_kinematic_replay.py --quick",
        build_invocation(args),
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260613_default_unitbox_discontinuous_kinematic_replay\\artifacts\\run_discontinuous_kinematic_replay.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (LOGS / "run_config.json").write_text(
        json.dumps({"device": device, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    matprop, pffmodel, gcII = prev.material(device, args.l0)
    summary_rows = []
    traction_rows = []
    jump_rows = []
    trace_rows = []
    audit_rows = []
    for seed in args.seeds:
        for state_name in args.states:
            state = load_state(seed, state_name, device)
            audit_rows.append(domain_split_audit_row(state))
            for variant in args.variants:
                if variant == "continuous_baseline":
                    field = make_continuous_field(args, state["Delta"], device, seed_offset=seed)
                else:
                    field = make_split_field(args, state, device, seed_offset=seed)
                prefit_info = prefit_saved_uv(field, state, args)
                prefit_state = clone_field_state(field)
                load_field_state(field, prefit_state)
                trace, status = reoptimize(field, state, args, variant, matprop, pffmodel, gcII)
                row, cuts, jumps, trace = evaluate_result(
                    field, state, args, variant, matprop, pffmodel, gcII, prefit_info, trace, status
                )
                summary_rows.append(row)
                for cut in cuts:
                    traction_rows.append(
                        {**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "kinematic_mode")}, **cut}
                    )
                for jump in jumps:
                    jump_rows.append(
                        {**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "kinematic_mode")}, **jump}
                    )
                for trace_row in trace:
                    trace_rows.append(
                        {**{k: row[k] for k in ("case", "seed", "state", "step", "Delta", "variant", "kinematic_mode")}, **trace_row}
                    )
    summary = add_baseline_comparisons(pd.DataFrame(summary_rows))
    traction = add_traction_baselines(pd.DataFrame(traction_rows))
    jumps = pd.DataFrame(jump_rows)
    convergence = summary[
        [
            "case",
            "seed",
            "state",
            "variant",
            "kinematic_mode",
            "convergence_status",
            "iterations",
            "final_mechanics_loss",
            "prefit_disp_mse",
            "prefit_iterations",
        ]
    ].copy()
    trace = pd.DataFrame(trace_rows)
    audit = pd.DataFrame(audit_rows)
    write_tables(summary, traction, jumps, convergence, trace, audit)
    make_figures(summary, traction, jumps)
    mechanism = classify_diagnostic(summary, traction, jumps, audit, args)
    write_docs(args, summary, traction, jumps, convergence, audit, mechanism)
    write_manifest()
    return mechanism


def main():
    args = parse_args()
    mechanism = run_all(args)
    print(mechanism)


if __name__ == "__main__":
    main()
