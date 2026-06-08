"""Prefit the current PINN displacement ansatz to FE-DOF mechanics targets.

This diagnostic freezes alpha to zero and trains only the current PINN
displacement ansatz against an existing FE-DOF alpha-zero displacement field.
It does not alter the coupled training path or the physical model.
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
NOTCH_LIP_HALF_GAP_MM = 5.0e-5


VARIANT_CONFIG = {
    "disp_only": {"lip_weight": 0.0, "strain_weight": 0.0},
    "disp_lip": {"lip_weight": None, "strain_weight": 0.0},
    "disp_strain": {"lip_weight": 0.0, "strain_weight": None},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prefit PINN displacement ansatz to FE-DOF mechanics fields.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-free", type=Path, required=True)
    parser.add_argument("--target-fixed", type=Path, default=None)
    parser.add_argument("--top-u-modes", choices=["free", "fixed"], nargs="+", default=["free", "fixed"])
    parser.add_argument("--variants", choices=list(VARIANT_CONFIG), nargs="+", default=list(VARIANT_CONFIG))
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lip-weight", type=float, default=1000.0)
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
    )


def _notch_lip_pair_indices(inp):
    x = inp[:, 0].detach().cpu().numpy()
    y = inp[:, 1].detach().cpu().numpy()
    upper = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM + NOTCH_LIP_HALF_GAP_MM)) <= BOUNDARY_TOL_MM)
    )[0]
    lower = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM - NOTCH_LIP_HALF_GAP_MM)) <= BOUNDARY_TOL_MM)
    )[0]
    pairs = []
    for idx in upper:
        if lower.size == 0:
            break
        j = lower[int(np.argmin(np.abs(x[lower] - x[idx])))]
        pairs.append((int(idx), int(j)))
    if not pairs:
        return None, None
    upper_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=inp.device)
    lower_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=inp.device)
    return upper_idx, lower_idx


def _lip_jumps(u, v, upper_idx, lower_idx):
    if upper_idx is None:
        empty = torch.empty(0, dtype=u.dtype, device=u.device)
        return empty, empty
    return u[upper_idx] - u[lower_idx], v[upper_idx] - v[lower_idx]


def _region_masks(x_elem, y_elem):
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
    aa = aa[mask]
    bb = bb[mask]
    aa = aa - torch.mean(aa)
    bb = bb - torch.mean(bb)
    denom = torch.sqrt(torch.sum(aa**2) * torch.sum(bb**2))
    if float(denom.detach().cpu()) <= 0.0:
        return np.nan
    return float((torch.sum(aa * bb) / denom).detach().cpu())


def _classify(row):
    bulk_ratio = row["pred_bulk_to_notch_He_current"]
    bottom_ratio = row["pred_bottom_to_notch_He_current"]
    max_x = row["pred_max_He_current_x"]
    max_y = row["pred_max_He_current_y"]
    boundary_max = (
        max_x <= BOUNDARY_WINDOW_MM
        or max_x >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
        or max_y <= BOUNDARY_WINDOW_MM
        or max_y >= SPECIMEN_SIZE_MM - BOUNDARY_WINDOW_MM
    )
    if np.isfinite(bulk_ratio) and np.isfinite(bottom_ratio) and bulk_ratio < 0.35 and bottom_ratio < 0.1:
        return "notch-amplified with boundary max" if boundary_max else "notch-amplified"
    if np.isfinite(bulk_ratio) and 0.5 <= bulk_ratio <= 2.0:
        return "broad/background"
    if boundary_max:
        return "boundary/background dominated"
    return "mixed/other"


def _variant_weights(variant, args):
    cfg = VARIANT_CONFIG[variant]
    lip_weight = args.lip_weight if cfg["lip_weight"] is None else cfg["lip_weight"]
    strain_weight = args.strain_weight if cfg["strain_weight"] is None else cfg["strain_weight"]
    return float(lip_weight), float(strain_weight)


def _evaluate_losses(u, v, target, upper_idx, lower_idx):
    alpha_zero = torch.zeros_like(target["u"])
    eps_xx, eps_yy, eps_xy, _ga_x, _ga_y = gradients(
        target["inp"], u, v, alpha_zero, target["area"], target["triangles"]
    )
    disp_loss = torch.mean((u - target["u"]) ** 2 + (v - target["v"]) ** 2)
    lip_u, lip_v = _lip_jumps(u, v, upper_idx, lower_idx)
    target_lip_u, target_lip_v = _lip_jumps(target["u"], target["v"], upper_idx, lower_idx)
    if lip_u.numel():
        lip_loss = torch.mean((lip_u - target_lip_u) ** 2 + (lip_v - target_lip_v) ** 2)
    else:
        lip_loss = torch.zeros((), dtype=u.dtype, device=u.device)
    strain_loss = torch.mean(
        (eps_xx - target["eps_xx"]) ** 2
        + (eps_yy - target["eps_yy"]) ** 2
        + (eps_xy - target["eps_xy"]) ** 2
    )
    return disp_loss, lip_loss, strain_loss, (eps_xx, eps_yy, eps_xy)


def _train_prefit(args, target, top_u_mode, variant, device):
    field = _make_field(args, top_u_mode, device)
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    upper_idx, lower_idx = _notch_lip_pair_indices(target["inp"])
    lip_weight, strain_weight = _variant_weights(variant, args)
    loss_trace = []

    for epoch in range(args.epochs):

        def closure():
            optimizer.zero_grad()
            u, v, _alpha_pred = field.fieldCalculation(target["inp"])
            disp_loss, lip_loss, strain_loss, _strains = _evaluate_losses(
                u, v, target, upper_idx, lower_idx
            )
            loss = disp_loss + lip_weight * lip_loss + strain_weight * strain_loss
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            loss_trace.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})

    u, v, _alpha_pred = field.fieldCalculation(target["inp"])
    disp_loss, lip_loss, strain_loss, strains = _evaluate_losses(u, v, target, upper_idx, lower_idx)
    E_el, E_d, fields = _compute_energy(
        target["inp"], u, v, target["area"], target["triangles"], args, device
    )
    return {
        "u": u,
        "v": v,
        "strains": strains,
        "fields": fields,
        "E_el": E_el,
        "E_d": E_d,
        "disp_loss": disp_loss,
        "lip_loss": lip_loss,
        "strain_loss": strain_loss,
        "loss_trace": loss_trace,
        "upper_idx": upper_idx,
        "lower_idx": lower_idx,
        "lip_weight": lip_weight,
        "strain_weight": strain_weight,
    }


def _max_abs(values):
    if values.numel() == 0:
        return np.nan
    return float(torch.max(torch.abs(values)).detach().cpu())


def _mse(values):
    if values.numel() == 0:
        return np.nan
    return float(torch.mean(values**2).detach().cpu())


def _case_metrics(case, target, result, args):
    x_elem, y_elem = element_centroids(target["inp"], target["triangles"])
    masks = _region_masks(x_elem, y_elem)
    pred_he = result["fields"]["He_current"]
    target_he = target["He_current"]
    pred_idx = int(torch.argmax(pred_he).detach().cpu())
    target_idx = int(torch.argmax(target_he).detach().cpu())
    pred_notch = _stats(pred_he[masks["notch_tip"]])["max"]
    pred_bulk = _stats(pred_he[masks["bulk"]])["p95"]
    pred_bottom = _stats(pred_he[masks["bottom_right"]])["max"]
    target_notch = _stats(target_he[masks["notch_tip"]])["max"]
    target_bulk = _stats(target_he[masks["bulk"]])["p95"]
    target_bottom = _stats(target_he[masks["bottom_right"]])["max"]
    pred_lip_u, pred_lip_v = _lip_jumps(result["u"], result["v"], result["upper_idx"], result["lower_idx"])
    target_lip_u, target_lip_v = _lip_jumps(target["u"], target["v"], result["upper_idx"], result["lower_idx"])
    eps_xx, eps_yy, eps_xy = result["strains"]
    strain_pred_all = torch.cat([eps_xx.flatten(), eps_yy.flatten(), eps_xy.flatten()])
    strain_target_all = torch.cat([
        target["eps_xx"].flatten(),
        target["eps_yy"].flatten(),
        target["eps_xy"].flatten(),
    ])
    disp_target_mse_ref = torch.mean(target["u"] ** 2 + target["v"] ** 2)
    strain_target_mse_ref = torch.mean(
        target["eps_xx"] ** 2 + target["eps_yy"] ** 2 + target["eps_xy"] ** 2
    )
    matprop, _pffmodel, gcII = _material(target["inp"].device, args.l0)
    row = {
        **case,
        "Delta": args.delta,
        "alpha_mode": "zero_fixed",
        "l0": args.l0,
        "tm_source": "split",
        "epochs": args.epochs,
        "lip_weight": result["lip_weight"],
        "strain_weight": result["strain_weight"],
        "final_displacement_mse": float(result["disp_loss"].detach().cpu()),
        "final_displacement_rel_rmse": float(torch.sqrt(result["disp_loss"] / disp_target_mse_ref).detach().cpu()),
        "final_lip_jump_mse": float(result["lip_loss"].detach().cpu()),
        "final_strain_mse": float(result["strain_loss"].detach().cpu()),
        "final_strain_rel_rmse": float(torch.sqrt(result["strain_loss"] / strain_target_mse_ref).detach().cpu()),
        "u_corr": _corr(result["u"], target["u"]),
        "v_corr": _corr(result["v"], target["v"]),
        "strain_corr": _corr(strain_pred_all, strain_target_all),
        "He_current_corr": _corr(pred_he, target_he),
        "elastic_energy": float(result["E_el"].detach().cpu()),
        "fracture_energy": float(result["E_d"].detach().cpu()),
        "mixed_mode_ratio": mixed_mode_ratio(matprop, gcII=gcII),
        "target_notch_lip_u_jump_abs_max": _max_abs(target_lip_u),
        "pred_notch_lip_u_jump_abs_max": _max_abs(pred_lip_u),
        "pred_to_target_lip_u_jump_ratio": _ratio(_max_abs(pred_lip_u), _max_abs(target_lip_u)),
        "target_notch_lip_v_jump_abs_max": _max_abs(target_lip_v),
        "pred_notch_lip_v_jump_abs_max": _max_abs(pred_lip_v),
        "pred_to_target_lip_v_jump_ratio": _ratio(_max_abs(pred_lip_v), _max_abs(target_lip_v)),
        "lip_u_jump_mse": _mse(pred_lip_u - target_lip_u),
        "lip_v_jump_mse": _mse(pred_lip_v - target_lip_v),
        "target_notch_He_current_max": target_notch,
        "target_bulk_He_current_p95": target_bulk,
        "target_bottom_right_He_current_max": target_bottom,
        "target_bulk_to_notch_He_current": _ratio(target_bulk, target_notch),
        "target_bottom_to_notch_He_current": _ratio(target_bottom, target_notch),
        "target_max_He_current": float(torch.max(target_he).detach().cpu()),
        "target_max_He_current_x": float(x_elem[target_idx].detach().cpu()),
        "target_max_He_current_y": float(y_elem[target_idx].detach().cpu()),
        "pred_notch_He_current_max": pred_notch,
        "pred_bulk_He_current_p95": pred_bulk,
        "pred_bottom_right_He_current_max": pred_bottom,
        "pred_bulk_to_notch_He_current": _ratio(pred_bulk, pred_notch),
        "pred_bottom_to_notch_He_current": _ratio(pred_bottom, pred_notch),
        "pred_max_He_current": float(torch.max(pred_he).detach().cpu()),
        "pred_max_He_current_x": float(x_elem[pred_idx].detach().cpu()),
        "pred_max_He_current_y": float(y_elem[pred_idx].detach().cpu()),
    }
    row["classification"] = _classify(row)
    lip_row = {
        key: row[key]
        for key in (
            "case_id",
            "top_u_mode",
            "variant",
            "target_notch_lip_u_jump_abs_max",
            "pred_notch_lip_u_jump_abs_max",
            "pred_to_target_lip_u_jump_ratio",
            "target_notch_lip_v_jump_abs_max",
            "pred_notch_lip_v_jump_abs_max",
            "pred_to_target_lip_v_jump_ratio",
            "lip_u_jump_mse",
            "lip_v_jump_mse",
            "final_lip_jump_mse",
        )
    }
    recon_row = {
        key: row[key]
        for key in (
            "case_id",
            "top_u_mode",
            "variant",
            "final_strain_mse",
            "final_strain_rel_rmse",
            "strain_corr",
            "He_current_corr",
            "target_bulk_to_notch_He_current",
            "pred_bulk_to_notch_He_current",
            "target_bottom_to_notch_He_current",
            "pred_bottom_to_notch_He_current",
            "target_max_He_current_x",
            "target_max_He_current_y",
            "pred_max_He_current_x",
            "pred_max_He_current_y",
            "classification",
        )
    }
    return row, lip_row, recon_row


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


def _save_case_npz(path, target, result):
    x_elem, y_elem = element_centroids(target["inp"], target["triangles"])
    eps_xx, eps_yy, eps_xy = result["strains"]
    np.savez_compressed(
        path,
        x=target["x"].detach().cpu().numpy(),
        y=target["y"].detach().cpu().numpy(),
        triangles=target["triangles"].detach().cpu().numpy(),
        element_x=x_elem.detach().cpu().numpy(),
        element_y=y_elem.detach().cpu().numpy(),
        u_target=target["u"].detach().cpu().numpy(),
        v_target=target["v"].detach().cpu().numpy(),
        u_pred=result["u"].detach().cpu().numpy(),
        v_pred=result["v"].detach().cpu().numpy(),
        eps_xx_target=target["eps_xx"].detach().cpu().numpy(),
        eps_yy_target=target["eps_yy"].detach().cpu().numpy(),
        eps_xy_target=target["eps_xy"].detach().cpu().numpy(),
        eps_xx_pred=eps_xx.detach().cpu().numpy(),
        eps_yy_pred=eps_yy.detach().cpu().numpy(),
        eps_xy_pred=eps_xy.detach().cpu().numpy(),
        He_current_target=target["He_current"].detach().cpu().numpy(),
        He_current_pred=result["fields"]["He_current"].detach().cpu().numpy(),
        psiI_target=target["psiI"].detach().cpu().numpy(),
        psiII_target=target["psiII"].detach().cpu().numpy(),
        psiI_pred=result["fields"]["psiI"].detach().cpu().numpy(),
        psiII_pred=result["fields"]["psiII"].detach().cpu().numpy(),
    )


def _plot_figures(out_dir, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [{"filename": "not_generated", "error": str(exc)}]

    generated = []
    figures = out_dir / "figures"
    representative = [
        r for r in rows
        if r["top_u_mode"] == "free" and r["variant"] in {"disp_only", "disp_lip", "disp_strain"}
    ]
    for row in representative:
        npz_path = out_dir / "artifacts" / f"{row['case_id']}_fields.npz"
        with np.load(npz_path) as data:
            x_elem = data["element_x"]
            y_elem = data["element_y"]
            target_he = data["He_current_target"]
            pred_he = data["He_current_pred"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
        for ax, values, title in (
            (axes[0], target_he, "target FE-DOF"),
            (axes[1], pred_he, "prefit PINN"),
        ):
            sc = ax.scatter(x_elem, y_elem, c=np.log10(np.maximum(values, 1.0e-30)), s=8, cmap="viridis")
            ax.plot([0.0, NOTCH_TIP_X_MM], [NOTCH_CENTER_Y_MM, NOTCH_CENTER_Y_MM], color="white", linewidth=1.1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, label="log10(He_current)")
        fig.suptitle(row["case_id"])
        filename = f"{row['case_id']}_target_vs_pred_He.png"
        fig.savefig(figures / filename, dpi=170)
        plt.close(fig)
        generated.append({"filename": filename, "case_id": row["case_id"]})
    return generated


def main():
    args = parse_args()
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_paths = {
        "free": args.target_free,
        "fixed": args.target_fixed,
    }
    commands = {"script": str(Path(__file__).resolve()), "device": device, "args": vars(args)}
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    rows = []
    lip_rows = []
    recon_rows = []
    for top_u_mode in args.top_u_modes:
        target_path = target_paths.get(top_u_mode)
        if target_path is None:
            continue
        target = _load_target(target_path, device)
        for variant in args.variants:
            case = {
                "case_id": f"prefit_{top_u_mode}_{variant}_e{args.epochs}",
                "top_u_mode": top_u_mode,
                "variant": variant,
                "target_path": str(target_path),
                "seed": args.seed,
                "hidden_layers": args.hidden_layers,
                "neurons": args.neurons,
            }
            result = _train_prefit(args, target, top_u_mode, variant, device)
            row, lip_row, recon_row = _case_metrics(case, target, result, args)
            rows.append(row)
            lip_rows.append(lip_row)
            recon_rows.append(recon_row)
            _save_case_npz(args.out_dir / "artifacts" / f"{case['case_id']}_fields.npz", target, result)
            _write_rows(args.out_dir / "logs" / f"{case['case_id']}_loss_trace.csv", result["loss_trace"])

    _write_rows(args.out_dir / "tables" / "prefit_case_comparison.csv", rows)
    _write_rows(args.out_dir / "tables" / "notch_lip_prefit_metrics.csv", lip_rows)
    _write_rows(args.out_dir / "tables" / "strain_he_reconstruction_metrics.csv", recon_rows)
    if not args.skip_figures:
        generated = _plot_figures(args.out_dir, rows)
        (args.out_dir / "figures" / "generated_figures.json").write_text(
            json.dumps(generated, indent=2, default=str), encoding="utf-8"
        )
    print(f"wrote {args.out_dir / 'tables' / 'prefit_case_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'notch_lip_prefit_metrics.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'strain_he_reconstruction_metrics.csv'}")


if __name__ == "__main__":
    main()
