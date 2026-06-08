"""Global prefit followed by mechanics-energy continuation.

This diagnostic deliberately avoids geometry-specific training guidance:

* no notch-lip loss,
* no notch-tip/lip masks in the training objective,
* no local displacement-jump target,
* no phase-field notch or alpha seeding.

Only global displacement targets, global element strain targets, and global
alpha-zero mechanics energy are used for optimization. Region metrics are
computed only after training for diagnosis.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy import gradients  # noqa: E402
from compute_energy_mixed_tm import compute_mixed_tm_energy  # noqa: E402
from field_computation import FieldComputation  # noqa: E402
from history_field_mixed_tm import element_areas, element_centroids  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from mixed_mode_tm import mixed_mode_ratio  # noqa: E402
from network import NeuralNet, init_xavier  # noqa: E402
from pff_model import PFFModel  # noqa: E402


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
BOUNDARY_WINDOW_MM = 5.0e-4
BOUNDARY_TOL_MM = 1.0e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run global prefit then alpha-zero mechanics-energy continuation diagnostics."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-free", type=Path, required=True)
    parser.add_argument("--target-fixed", type=Path, default=None)
    parser.add_argument("--top-u-modes", choices=["free", "fixed"], nargs="+", default=["free"])
    parser.add_argument(
        "--cases",
        choices=[
            "random_init_energy",
            "disp_global_prefit_then_energy",
            "disp_strain_global_prefit_then_energy",
            "global_curriculum",
        ],
        nargs="+",
        default=[
            "random_init_energy",
            "disp_global_prefit_then_energy",
            "disp_strain_global_prefit_then_energy",
            "global_curriculum",
        ],
    )
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--prefit-epochs", type=int, default=800)
    parser.add_argument("--energy-epochs", type=int, default=300)
    parser.add_argument("--curriculum-epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--strain-weight", type=float, default=1.0e-5)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def _prepare_output_dirs(out_dir):
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _load_target(path, device):
    with np.load(path) as data:
        target = {
            "path": str(path),
            "x": torch.tensor(data["x"], dtype=torch.float32, device=device),
            "y": torch.tensor(data["y"], dtype=torch.float32, device=device),
            "triangles": torch.tensor(data["triangles"], dtype=torch.long, device=device),
            "u": torch.tensor(data["u"], dtype=torch.float32, device=device),
            "v": torch.tensor(data["v"], dtype=torch.float32, device=device),
            "eps_xx": torch.tensor(data["eps_xx"], dtype=torch.float32, device=device),
            "eps_yy": torch.tensor(data["eps_yy"], dtype=torch.float32, device=device),
            "eps_xy": torch.tensor(data["eps_xy"], dtype=torch.float32, device=device),
            "He_current": torch.tensor(data["He_current"], dtype=torch.float32, device=device),
            "psiI": torch.tensor(data["psiI"], dtype=torch.float32, device=device),
            "psiII": torch.tensor(data["psiII"], dtype=torch.float32, device=device),
        }
    target["inp"] = torch.stack([target["x"], target["y"]], dim=1)
    target["area"] = element_areas(target["inp"], target["triangles"])
    return target


def _material(device, l0):
    matprop = MaterialProperties(
        torch.tensor(81.5, device=device),
        torch.tensor(0.38, device=device),
        torch.tensor(2.4e-6 / l0, device=device),
        torch.tensor(l0, device=device),
    )
    pffmodel = PFFModel("AT2", "volumetric", torch.tensor(5.0e-3, device=device))
    gcII = 2.0 * (1.0 + 0.38) * (0.60**2) * 2.4e-6
    return matprop, pffmodel, gcII


def _make_field(args, top_u_mode, device):
    torch.manual_seed(args.seed)
    net = NeuralNet(2, 3, args.hidden_layers, args.neurons, args.activation, args.init_coeff).to(device)
    init_xavier(net)
    return FieldComputation(
        net=net,
        domain_extrema=torch.tensor(
            [[0.0, SPECIMEN_SIZE_MM], [0.0, SPECIMEN_SIZE_MM]],
            dtype=torch.float32,
            device=device,
        ),
        lmbda=torch.tensor([args.delta], dtype=torch.float32, device=device),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32, device=device),
        alpha_constraint="nonsmooth",
        top_u_mode=top_u_mode,
        coord_normalization=getattr(args, "coord_normalization", "none"),
    )


def _compute_energy(inp, u, v, area_T, T_conn, args, device):
    matprop, pffmodel, gcII = _material(device, args.l0)
    alpha_zero = torch.zeros(inp.shape[0], dtype=inp.dtype, device=device)
    HI = torch.zeros_like(area_T)
    HII = torch.zeros_like(area_T)
    E_el, E_d, fields = compute_mixed_tm_energy(
        inp,
        u,
        v,
        alpha_zero,
        HI,
        HII,
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=1.0e-5,
        gcII=gcII,
        split_mode="tm_source",
        tm_eps_r=args.tm_eps_r,
        mechanics_mode="history",
    )
    return E_el, E_d, fields


def _global_strains(target, u, v):
    alpha_zero = torch.zeros_like(target["u"])
    eps_xx, eps_yy, eps_xy, _ga_x, _ga_y = gradients(
        target["inp"], u, v, alpha_zero, target["area"], target["triangles"]
    )
    return eps_xx, eps_yy, eps_xy


def _global_prefit_losses(target, u, v):
    eps_xx, eps_yy, eps_xy = _global_strains(target, u, v)
    disp_loss = torch.mean((u - target["u"]) ** 2 + (v - target["v"]) ** 2)
    strain_loss = torch.mean(
        (eps_xx - target["eps_xx"]) ** 2
        + (eps_yy - target["eps_yy"]) ** 2
        + (eps_xy - target["eps_xy"]) ** 2
    )
    return disp_loss, strain_loss, (eps_xx, eps_yy, eps_xy)


def _positive_log10(value):
    return torch.log10(torch.clamp(value, min=torch.finfo(value.dtype).tiny))


def _evaluate(field, target, args, device):
    u, v, _alpha_pred = field.fieldCalculation(target["inp"])
    disp_loss, strain_loss, strains = _global_prefit_losses(target, u, v)
    E_el, E_d, fields = _compute_energy(
        target["inp"], u, v, target["area"], target["triangles"], args, device
    )
    energy_loss = _positive_log10(E_el + E_d)
    return {
        "u": u,
        "v": v,
        "disp_loss": disp_loss,
        "strain_loss": strain_loss,
        "strains": strains,
        "E_el": E_el,
        "E_d": E_d,
        "fields": fields,
        "energy_loss": energy_loss,
    }


def _fit(field, target, args, device, epochs, loss_kind, case_id, trace_rows):
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    for epoch in range(epochs):
        w = 1.0
        if loss_kind == "global_curriculum":
            denom = max(1, epochs - 1)
            w = epoch / denom

        def closure():
            optimizer.zero_grad()
            state = _evaluate(field, target, args, device)
            if loss_kind == "disp_global":
                loss = state["disp_loss"]
            elif loss_kind == "disp_strain_global":
                loss = state["disp_loss"] + args.strain_weight * state["strain_loss"]
            elif loss_kind == "energy":
                loss = state["energy_loss"]
            elif loss_kind == "global_curriculum":
                prefit_loss = state["disp_loss"] + args.strain_weight * state["strain_loss"]
                loss = (1.0 - w) * prefit_loss + w * state["energy_loss"]
            else:
                raise ValueError(f"unknown loss_kind {loss_kind!r}")
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                state = _evaluate(field, target, args, device)
            trace_rows.append(
                {
                    "case_id": case_id,
                    "loss_kind": loss_kind,
                    "epoch": epoch + 1,
                    "curriculum_w": w,
                    "loss": float(loss.detach().cpu()),
                    "disp_mse": float(state["disp_loss"].detach().cpu()),
                    "strain_mse": float(state["strain_loss"].detach().cpu()),
                    "energy_log10": float(state["energy_loss"].detach().cpu()),
                    "elastic_energy": float(state["E_el"].detach().cpu()),
                }
            )


def _run_case(args, target, top_u_mode, case_name, device):
    field = _make_field(args, top_u_mode, device)
    trace_rows = []
    stage_states = []
    if case_name == "random_init_energy":
        prefit_kind = "none"
        _fit(field, target, args, device, args.energy_epochs, "energy", case_name, trace_rows)
        stage_states.append(("energy_end", _evaluate(field, target, args, device)))
    elif case_name == "disp_global_prefit_then_energy":
        prefit_kind = "disp_global"
        _fit(field, target, args, device, args.prefit_epochs, "disp_global", case_name, trace_rows)
        stage_states.append(("prefit_end", _evaluate(field, target, args, device)))
        _fit(field, target, args, device, args.energy_epochs, "energy", case_name, trace_rows)
        stage_states.append(("energy_end", _evaluate(field, target, args, device)))
    elif case_name == "disp_strain_global_prefit_then_energy":
        prefit_kind = "disp_strain_global"
        _fit(field, target, args, device, args.prefit_epochs, "disp_strain_global", case_name, trace_rows)
        stage_states.append(("prefit_end", _evaluate(field, target, args, device)))
        _fit(field, target, args, device, args.energy_epochs, "energy", case_name, trace_rows)
        stage_states.append(("energy_end", _evaluate(field, target, args, device)))
    elif case_name == "global_curriculum":
        prefit_kind = "global_curriculum"
        _fit(field, target, args, device, args.curriculum_epochs, "global_curriculum", case_name, trace_rows)
        stage_states.append(("curriculum_end", _evaluate(field, target, args, device)))
    else:
        raise ValueError(f"unknown case {case_name!r}")
    return prefit_kind, stage_states, trace_rows


def _stats(values):
    vals = values.detach()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"max": np.nan, "p95": np.nan, "mean": np.nan}
    return {
        "max": float(torch.max(vals).detach().cpu()),
        "p95": float(torch.quantile(vals, 0.95).detach().cpu()),
        "mean": float(torch.mean(vals).detach().cpu()),
    }


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return np.nan
    return float(num / den)


def _corr(a, b):
    aa = a.detach().flatten().float()
    bb = b.detach().flatten().float()
    mask = torch.isfinite(aa) & torch.isfinite(bb)
    if torch.sum(mask) < 2:
        return np.nan
    aa = aa[mask] - torch.mean(aa[mask])
    bb = bb[mask] - torch.mean(bb[mask])
    denom = torch.sqrt(torch.sum(aa**2) * torch.sum(bb**2))
    if float(denom.detach().cpu()) <= 0.0:
        return np.nan
    return float((torch.sum(aa * bb) / denom).detach().cpu())


def _region_masks(x_elem, y_elem):
    # These masks are postprocessing diagnostics only and are never used in loss functions.
    notch_tip = (
        (x_elem >= NOTCH_TIP_X_MM - TIP_HALF_WINDOW_MM)
        & (x_elem <= NOTCH_TIP_X_MM + TIP_HALF_WINDOW_MM)
        & (torch.abs(y_elem - NOTCH_CENTER_Y_MM) <= TIP_HALF_WINDOW_MM)
    )
    bottom_right = (
        (x_elem >= SPECIMEN_SIZE_MM - BOTTOM_RIGHT_WINDOW_MM)
        & (x_elem <= SPECIMEN_SIZE_MM)
        & (y_elem >= 0.0)
        & (y_elem <= BOTTOM_RIGHT_WINDOW_MM)
    )
    boundary = (
        (x_elem <= BOUNDARY_WINDOW_MM)
        | (x_elem >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM)
        | (y_elem <= BOUNDARY_WINDOW_MM)
        | (y_elem >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM)
    )
    bulk = (~notch_tip) & (~bottom_right) & (~boundary)
    return {"notch_tip": notch_tip, "bottom_right": bottom_right, "bulk": bulk}


def _classify(row):
    bulk_ratio = row["bulk_to_notch_He_current"]
    bottom_ratio = row["bottom_to_notch_He_current"]
    max_x = row["max_He_current_x"]
    max_y = row["max_He_current_y"]
    boundary_max = (
        max_x <= BOUNDARY_WINDOW_MM
        or max_x >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
        or max_y <= BOUNDARY_WINDOW_MM
        or max_y >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
    )
    if np.isfinite(bulk_ratio) and np.isfinite(bottom_ratio) and bulk_ratio < 0.35 and bottom_ratio < 0.1:
        if row["He_current_corr"] >= 0.9:
            return "target-like"
        return "notch-amplified"
    if boundary_max:
        return "boundary-dominated"
    if np.isfinite(bulk_ratio) and 0.5 <= bulk_ratio <= 2.0:
        return "broad/background"
    return "mixed/other"


def _top_reaction_force_N(inp, T_conn, sigma_yy, top_y=SPECIMEN_SIZE_MM, tol=BOUNDARY_TOL_MM):
    reaction_kN = torch.zeros((), device=sigma_yy.device, dtype=sigma_yy.dtype)
    for elem_id in range(T_conn.shape[0]):
        nodes = T_conn[elem_id]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = nodes[i]
            b = nodes[j]
            if torch.abs(inp[a, 1] - top_y) <= tol and torch.abs(inp[b, 1] - top_y) <= tol:
                reaction_kN = reaction_kN + sigma_yy[elem_id] * torch.linalg.norm(inp[a] - inp[b])
    return float((1000.0 * reaction_kN).detach().cpu())


def _metrics(case_id, top_u_mode, case_name, prefit_kind, stage, target, state, args):
    x_elem, y_elem = element_centroids(target["inp"], target["triangles"])
    masks = _region_masks(x_elem, y_elem)
    he = state["fields"]["He_current"]
    target_he = target["He_current"]
    max_idx = int(torch.argmax(he).detach().cpu())
    target_max_idx = int(torch.argmax(target_he).detach().cpu())
    notch_he = _stats(he[masks["notch_tip"]])["max"]
    bulk_he = _stats(he[masks["bulk"]])["p95"]
    bottom_he = _stats(he[masks["bottom_right"]])["max"]
    target_notch_he = _stats(target_he[masks["notch_tip"]])["max"]
    target_bulk_he = _stats(target_he[masks["bulk"]])["p95"]
    target_bottom_he = _stats(target_he[masks["bottom_right"]])["max"]
    eps_xx, eps_yy, eps_xy = state["strains"]
    strain_pred = torch.cat([eps_xx.flatten(), eps_yy.flatten(), eps_xy.flatten()])
    strain_target = torch.cat([target["eps_xx"].flatten(), target["eps_yy"].flatten(), target["eps_xy"].flatten()])
    disp_ref = torch.mean(target["u"] ** 2 + target["v"] ** 2)
    strain_ref = torch.mean(target["eps_xx"] ** 2 + target["eps_yy"] ** 2 + target["eps_xy"] ** 2)
    matprop, _pffmodel, gcII = _material(target["inp"].device, args.l0)
    row = {
        "case_id": case_id,
        "case": case_name,
        "stage": stage,
        "top_u_mode": top_u_mode,
        "prefit_kind": prefit_kind,
        "Delta": args.delta,
        "alpha_mode": "zero_fixed",
        "l0": args.l0,
        "tm_source": "split",
        "prefit_epochs": args.prefit_epochs if "prefit" in case_name else 0,
        "energy_epochs": args.energy_epochs if case_name != "global_curriculum" else 0,
        "curriculum_epochs": args.curriculum_epochs if case_name == "global_curriculum" else 0,
        "strain_weight": args.strain_weight,
        "displacement_mse": float(state["disp_loss"].detach().cpu()),
        "displacement_rel_rmse": float(torch.sqrt(state["disp_loss"] / disp_ref).detach().cpu()),
        "strain_mse": float(state["strain_loss"].detach().cpu()),
        "strain_rel_rmse": float(torch.sqrt(state["strain_loss"] / strain_ref).detach().cpu()),
        "u_corr": _corr(state["u"], target["u"]),
        "v_corr": _corr(state["v"], target["v"]),
        "strain_corr": _corr(strain_pred, strain_target),
        "He_current_corr": _corr(he, target_he),
        "mechanics_energy_log10": float(state["energy_loss"].detach().cpu()),
        "elastic_energy": float(state["E_el"].detach().cpu()),
        "fracture_energy": float(state["E_d"].detach().cpu()),
        "mixed_mode_ratio": mixed_mode_ratio(matprop, gcII=gcII),
        "reaction_N_tm_eff_proxy": _top_reaction_force_N(
            target["inp"], target["triangles"], state["fields"]["sigma_yy_tm_eff"]
        ),
        "notch_tip_He_current_max": notch_he,
        "bulk_He_current_p95": bulk_he,
        "bottom_right_He_current_max": bottom_he,
        "bulk_to_notch_He_current": _ratio(bulk_he, notch_he),
        "bottom_to_notch_He_current": _ratio(bottom_he, notch_he),
        "max_He_current": float(torch.max(he).detach().cpu()),
        "max_He_current_x": float(x_elem[max_idx].detach().cpu()),
        "max_He_current_y": float(y_elem[max_idx].detach().cpu()),
        "target_notch_tip_He_current_max": target_notch_he,
        "target_bulk_He_current_p95": target_bulk_he,
        "target_bottom_right_He_current_max": target_bottom_he,
        "target_bulk_to_notch_He_current": _ratio(target_bulk_he, target_notch_he),
        "target_bottom_to_notch_He_current": _ratio(target_bottom_he, target_notch_he),
        "target_max_He_current": float(torch.max(target_he).detach().cpu()),
        "target_max_He_current_x": float(x_elem[target_max_idx].detach().cpu()),
        "target_max_He_current_y": float(y_elem[target_max_idx].detach().cpu()),
    }
    row["classification"] = _classify(row)
    strain_row = {
        "case_id": case_id,
        "stage": stage,
        "top_u_mode": top_u_mode,
        "case": case_name,
        "prefit_kind": prefit_kind,
        "displacement_mse": row["displacement_mse"],
        "displacement_rel_rmse": row["displacement_rel_rmse"],
        "strain_mse": row["strain_mse"],
        "strain_rel_rmse": row["strain_rel_rmse"],
        "u_corr": row["u_corr"],
        "v_corr": row["v_corr"],
        "strain_corr": row["strain_corr"],
        "He_current_corr": row["He_current_corr"],
    }
    energy_row = {
        "case_id": case_id,
        "stage": stage,
        "top_u_mode": top_u_mode,
        "case": case_name,
        "prefit_kind": prefit_kind,
        "mechanics_energy_log10": row["mechanics_energy_log10"],
        "elastic_energy": row["elastic_energy"],
        "reaction_N_tm_eff_proxy": row["reaction_N_tm_eff_proxy"],
        "notch_tip_He_current_max": notch_he,
        "bulk_He_current_p95": bulk_he,
        "bottom_right_He_current_max": bottom_he,
        "bulk_to_notch_He_current": row["bulk_to_notch_He_current"],
        "bottom_to_notch_He_current": row["bottom_to_notch_He_current"],
        "max_He_current_x": row["max_He_current_x"],
        "max_He_current_y": row["max_He_current_y"],
        "classification": row["classification"],
    }
    return row, energy_row, strain_row


def _save_case_npz(path, target, state):
    x_elem, y_elem = element_centroids(target["inp"], target["triangles"])
    eps_xx, eps_yy, eps_xy = state["strains"]
    np.savez_compressed(
        path,
        x=target["x"].detach().cpu().numpy(),
        y=target["y"].detach().cpu().numpy(),
        triangles=target["triangles"].detach().cpu().numpy(),
        element_x=x_elem.detach().cpu().numpy(),
        element_y=y_elem.detach().cpu().numpy(),
        u_target=target["u"].detach().cpu().numpy(),
        v_target=target["v"].detach().cpu().numpy(),
        u_pred=state["u"].detach().cpu().numpy(),
        v_pred=state["v"].detach().cpu().numpy(),
        eps_xx_target=target["eps_xx"].detach().cpu().numpy(),
        eps_yy_target=target["eps_yy"].detach().cpu().numpy(),
        eps_xy_target=target["eps_xy"].detach().cpu().numpy(),
        eps_xx_pred=eps_xx.detach().cpu().numpy(),
        eps_yy_pred=eps_yy.detach().cpu().numpy(),
        eps_xy_pred=eps_xy.detach().cpu().numpy(),
        He_current_target=target["He_current"].detach().cpu().numpy(),
        He_current_pred=state["fields"]["He_current"].detach().cpu().numpy(),
        psiI_target=target["psiI"].detach().cpu().numpy(),
        psiII_target=target["psiII"].detach().cpu().numpy(),
        psiI_pred=state["fields"]["psiI"].detach().cpu().numpy(),
        psiII_pred=state["fields"]["psiII"].detach().cpu().numpy(),
    )


def _write_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_figures(out_dir, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [{"filename": "not_generated", "error": str(exc)}]
    generated = []
    figures = out_dir / "figures"
    representatives = [r for r in rows if r["top_u_mode"] == "free"]
    for row in representatives:
        path = out_dir / "artifacts" / f"{row['case_id']}_fields.npz"
        with np.load(path) as data:
            x_elem = data["element_x"]
            y_elem = data["element_y"]
            target_he = data["He_current_target"]
            pred_he = data["He_current_pred"]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
        for ax, values, title in (
            (axes[0], target_he, "target FE-DOF"),
            (axes[1], pred_he, "PINN after continuation"),
        ):
            sc = ax.scatter(x_elem, y_elem, c=np.log10(np.maximum(values, 1.0e-30)), s=8, cmap="viridis")
            ax.plot([0.0, NOTCH_TIP_X_MM], [NOTCH_CENTER_Y_MM, NOTCH_CENTER_Y_MM], color="white", linewidth=1.1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, label="log10(He_current)")
        fig.suptitle(row["case_id"])
        filename = f"{row['case_id']}_target_vs_after_energy_He.png"
        fig.savefig(figures / filename, dpi=170)
        plt.close(fig)
        generated.append({"filename": filename, "case_id": row["case_id"]})
    return generated


def main():
    args = parse_args()
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_paths = {"free": args.target_free, "fixed": args.target_fixed}
    commands = {"script": str(Path(__file__).resolve()), "device": device, "args": vars(args)}
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")
    rows = []
    energy_rows = []
    strain_rows = []
    trace_rows = []
    for top_u_mode in args.top_u_modes:
        target_path = target_paths.get(top_u_mode)
        if target_path is None:
            continue
        target = _load_target(target_path, device)
        for case_name in args.cases:
            base_case_id = f"{top_u_mode}_{case_name}"
            prefit_kind, stage_states, case_trace = _run_case(args, target, top_u_mode, case_name, device)
            for stage, state in stage_states:
                case_id = f"{base_case_id}_{stage}"
                metric_row, energy_row, strain_row = _metrics(
                    case_id, top_u_mode, case_name, prefit_kind, stage, target, state, args
                )
                rows.append(metric_row)
                energy_rows.append(energy_row)
                strain_rows.append(strain_row)
                _save_case_npz(args.out_dir / "artifacts" / f"{case_id}_fields.npz", target, state)
            for trace in case_trace:
                trace["case_id"] = base_case_id
                trace_rows.append(trace)
    _write_rows(args.out_dir / "tables" / "global_prefit_case_comparison.csv", rows)
    _write_rows(args.out_dir / "tables" / "energy_continuation_metrics.csv", energy_rows)
    _write_rows(args.out_dir / "tables" / "global_strain_reconstruction_metrics.csv", strain_rows)
    _write_rows(args.out_dir / "logs" / "loss_trace.csv", trace_rows)
    if not args.skip_figures:
        generated = _plot_figures(args.out_dir, rows)
        (args.out_dir / "figures" / "generated_figures.json").write_text(
            json.dumps(generated, indent=2, default=str), encoding="utf-8"
        )
    print(f"wrote {args.out_dir / 'tables' / 'global_prefit_case_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'energy_continuation_metrics.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'global_strain_reconstruction_metrics.csv'}")


if __name__ == "__main__":
    main()
