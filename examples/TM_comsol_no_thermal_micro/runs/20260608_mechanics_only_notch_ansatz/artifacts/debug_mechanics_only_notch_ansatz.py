"""Mechanics-only notch ansatz diagnostic for the COMSOL micro-notch case.

The diagnostic freezes alpha to zero and compares the current PINN
displacement ansatz with an independent nodal-DOF mechanics baseline on the
same mesh, material constants, TM split, l0, and load step.
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

from compute_energy_mixed_tm import compute_mixed_tm_energy  # noqa: E402
from field_computation import FieldComputation  # noqa: E402
from history_field_mixed_tm import element_centroids  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from mixed_mode_tm import mixed_mode_ratio  # noqa: E402
from network import NeuralNet, init_xavier  # noqa: E402
from pff_model import PFFModel  # noqa: E402
from utils import parse_mesh  # noqa: E402


SPECIMEN_SIZE_MM = 0.01
NOTCH_TIP_X_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
TIP_HALF_WINDOW_MM = 3.0e-4
BOTTOM_RIGHT_WINDOW_MM = 5.0e-4
BOUNDARY_WINDOW_MM = 5.0e-4
BOUNDARY_TOL_MM = 1.0e-9
NOTCH_LIP_HALF_GAP_MM = 5.0e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Run alpha-zero mechanics-only notch ansatz diagnostics.")
    parser.add_argument("--mesh", type=Path, default=ROOT / "geo_coarse_with_groups_mm.msh")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--activation", default="TrainableReLU")
    parser.add_argument("--init-coeff", type=float, default=3.0)
    parser.add_argument("--tm-eps-r", type=float, default=1.0e-5)
    parser.add_argument("--l0", type=float, default=1.5e-4)
    parser.add_argument("--pinn-epochs", type=int, nargs="+", default=[0, 100, 300])
    parser.add_argument("--fedof-epochs", type=int, nargs="+", default=[300])
    parser.add_argument("--top-u-modes", choices=["fixed", "free"], nargs="+", default=["fixed", "free"])
    parser.add_argument("--loss-forms", choices=["log10_energy", "raw_energy"], nargs="+", default=["log10_energy"])
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def _prepare_output_dirs(out_dir):
    for name in ("tables", "artifacts", "figures", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


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


def _boundary_masks(inp):
    x = inp[:, 0]
    y = inp[:, 1]
    return {
        "bottom": torch.abs(y) <= BOUNDARY_TOL_MM,
        "top": torch.abs(y - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL_MM,
    }


def _apply_nodal_bc(raw_u, raw_v, inp, delta, top_u_mode):
    masks = _boundary_masks(inp)
    u_free = torch.ones_like(raw_u)
    v_free = torch.ones_like(raw_v)
    v_value = torch.zeros_like(raw_v)

    u_free[masks["bottom"]] = 0.0
    v_free[masks["bottom"]] = 0.0
    v_free[masks["top"]] = 0.0
    v_value[masks["top"]] = float(delta)
    if top_u_mode == "fixed":
        u_free[masks["top"]] = 0.0
    return raw_u * u_free, raw_v * v_free + v_value


def _loss_from_energy(E_el, E_d, loss_form):
    total = E_el + E_d
    if loss_form == "raw_energy":
        return total
    return torch.log10(total)


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


def _train_pinn(inp, area_T, T_conn, args, top_u_mode, epochs, loss_form, device):
    torch.manual_seed(args.seed)
    net = NeuralNet(2, 3, args.hidden_layers, args.neurons, args.activation, args.init_coeff).to(device)
    init_xavier(net)
    field = FieldComputation(
        net=net,
        domain_extrema=torch.tensor([[0.0, SPECIMEN_SIZE_MM], [0.0, SPECIMEN_SIZE_MM]], dtype=torch.float32, device=device),
        lmbda=torch.tensor([args.delta], dtype=torch.float32, device=device),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32, device=device),
        alpha_constraint="nonsmooth",
        top_u_mode=top_u_mode,
    )
    optimizer = torch.optim.Rprop(field.net.parameters(), lr=args.lr)
    losses = []

    def evaluate():
        u, v, _alpha_pred = field.fieldCalculation(inp)
        E_el, E_d, fields = _compute_energy(inp, u, v, area_T, T_conn, args, device)
        return u, v, E_el, E_d, fields

    with torch.enable_grad():
        u0, v0, E0, D0, _fields0 = evaluate()
        initial_loss = float(_loss_from_energy(E0, D0, loss_form).detach().cpu())
        for _ in range(epochs):

            def closure():
                optimizer.zero_grad()
                u, v, E_el, E_d, _fields = evaluate()
                loss = _loss_from_energy(E_el, E_d, loss_form)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(float(loss.detach().cpu()))
        u, v, E_el, E_d, fields = evaluate()
    return {
        "u": u,
        "v": v,
        "E_el": E_el,
        "E_d": E_d,
        "fields": fields,
        "initial_loss": initial_loss,
        "final_loss": losses[-1] if losses else initial_loss,
    }


def _train_fedof(inp, area_T, T_conn, args, top_u_mode, epochs, loss_form, device):
    torch.manual_seed(args.seed)
    raw_u = torch.nn.Parameter(torch.zeros(inp.shape[0], dtype=torch.float32, device=device))
    raw_v = torch.nn.Parameter(torch.zeros(inp.shape[0], dtype=torch.float32, device=device))
    optimizer = torch.optim.Rprop([raw_u, raw_v], lr=args.lr)
    losses = []

    def evaluate():
        u, v = _apply_nodal_bc(raw_u, raw_v, inp, args.delta, top_u_mode)
        E_el, E_d, fields = _compute_energy(inp, u, v, area_T, T_conn, args, device)
        return u, v, E_el, E_d, fields

    with torch.enable_grad():
        u0, v0, E0, D0, _fields0 = evaluate()
        initial_loss = float(_loss_from_energy(E0, D0, loss_form).detach().cpu())
        for _ in range(epochs):

            def closure():
                optimizer.zero_grad()
                u, v, E_el, E_d, _fields = evaluate()
                loss = _loss_from_energy(E_el, E_d, loss_form)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(float(loss.detach().cpu()))
        u, v, E_el, E_d, fields = evaluate()
    return {
        "u": u,
        "v": v,
        "E_el": E_el,
        "E_d": E_d,
        "fields": fields,
        "initial_loss": initial_loss,
        "final_loss": losses[-1] if losses else initial_loss,
    }


def _torch_stats(values):
    vals = values.detach()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "p95": np.nan}
    return {
        "mean": float(torch.mean(vals).cpu()),
        "std": float(torch.std(vals, unbiased=False).cpu()),
        "min": float(torch.min(vals).cpu()),
        "max": float(torch.max(vals).cpu()),
        "p95": float(torch.quantile(vals, 0.95).cpu()),
    }


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return np.nan
    return float(num / den)


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
    return {"notch_tip": notch_tip, "bottom_right": bottom_right, "bulk": bulk, "boundary": boundary}


def _nearest_node_pair_rows(inp, u, v):
    x = inp[:, 0].detach().cpu().numpy()
    y = inp[:, 1].detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    upper = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM + NOTCH_LIP_HALF_GAP_MM)) <= BOUNDARY_TOL_MM)
    )[0]
    lower = np.where(
        (x <= NOTCH_TIP_X_MM + BOUNDARY_TOL_MM)
        & (np.abs(y - (NOTCH_CENTER_Y_MM - NOTCH_LIP_HALF_GAP_MM)) <= BOUNDARY_TOL_MM)
    )[0]
    rows = []
    for idx in upper:
        if lower.size == 0:
            break
        j = lower[int(np.argmin(np.abs(x[lower] - x[idx])))]
        rows.append(
            {
                "x": float(x[idx]),
                "upper_node": int(idx),
                "lower_node": int(j),
                "gap_y": float(y[idx] - y[j]),
                "u_jump_upper_minus_lower": float(u_np[idx] - u_np[j]),
                "v_jump_upper_minus_lower": float(v_np[idx] - v_np[j]),
                "upper_u": float(u_np[idx]),
                "lower_u": float(u_np[j]),
                "upper_v": float(v_np[idx]),
                "lower_v": float(v_np[j]),
            }
        )
    return rows


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
    if np.isfinite(bulk_ratio) and np.isfinite(bottom_ratio) and bulk_ratio < 0.2 and bottom_ratio < 0.2:
        return "notch-dominated"
    if np.isfinite(bulk_ratio) and np.isfinite(bottom_ratio) and bulk_ratio < 0.35 and bottom_ratio < 0.1:
        if boundary_max:
            return "notch-amplified with boundary max"
        return "notch-amplified"
    if boundary_max and (not np.isfinite(bulk_ratio) or bulk_ratio >= 0.2):
        return "boundary/background dominated"
    if np.isfinite(bulk_ratio) and 0.5 <= bulk_ratio <= 2.0:
        return "broad/background"
    return "mixed/other"


def _summarize_case(case, inp, T_conn, result, args):
    fields = result["fields"]
    x_elem, y_elem = element_centroids(inp, T_conn)
    masks = _region_masks(x_elem, y_elem)
    he = fields["He_current"]
    mechanics = fields["mechanics_drive"]
    eps_norm = torch.sqrt(fields["eps_xx"] ** 2 + fields["eps_yy"] ** 2 + 2.0 * fields["eps_xy"] ** 2)
    max_he_idx = int(torch.argmax(he).detach().cpu())
    max_mech_idx = int(torch.argmax(mechanics).detach().cpu())
    notch_he = _torch_stats(he[masks["notch_tip"]])["max"]
    bulk_he_p95 = _torch_stats(he[masks["bulk"]])["p95"]
    bottom_he = _torch_stats(he[masks["bottom_right"]])["max"]
    notch_mech = _torch_stats(mechanics[masks["notch_tip"]])["max"]
    bulk_mech_p95 = _torch_stats(mechanics[masks["bulk"]])["p95"]
    bottom_mech = _torch_stats(mechanics[masks["bottom_right"]])["max"]
    notch_eps = _torch_stats(eps_norm[masks["notch_tip"]])["max"]
    bulk_eps_p95 = _torch_stats(eps_norm[masks["bulk"]])["p95"]
    bottom_eps = _torch_stats(eps_norm[masks["bottom_right"]])["max"]
    pairs = _nearest_node_pair_rows(inp, result["u"], result["v"])
    u_jumps = np.array([r["u_jump_upper_minus_lower"] for r in pairs], dtype=float)
    v_jumps = np.array([r["v_jump_upper_minus_lower"] for r in pairs], dtype=float)
    masks_node = _boundary_masks(inp)
    top = masks_node["top"]
    bottom = masks_node["bottom"]
    u = result["u"]
    v = result["v"]
    matprop, _pffmodel, gcII = _material(inp.device, args.l0)
    row = {
        **case,
        "Delta": args.delta,
        "alpha_mode": "zero_fixed",
        "l0": args.l0,
        "tm_source": "split",
        "tm_eps_r": args.tm_eps_r,
        "phase_field_notch": "unchanged_not_active_in_alpha_zero_mechanics_only",
        "history_fields": "HI=0,HII=0",
        "elastic_energy": float(result["E_el"].detach().cpu()),
        "fracture_energy": float(result["E_d"].detach().cpu()),
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "mixed_mode_ratio": mixed_mode_ratio(matprop, gcII=gcII),
        "He_current_mean": _torch_stats(he)["mean"],
        "He_current_std": _torch_stats(he)["std"],
        "He_current_max": _torch_stats(he)["max"],
        "max_He_current_x": float(x_elem[max_he_idx].detach().cpu()),
        "max_He_current_y": float(y_elem[max_he_idx].detach().cpu()),
        "mechanics_drive_max": _torch_stats(mechanics)["max"],
        "max_mechanics_drive_x": float(x_elem[max_mech_idx].detach().cpu()),
        "max_mechanics_drive_y": float(y_elem[max_mech_idx].detach().cpu()),
        "notch_tip_He_current_max": notch_he,
        "bulk_He_current_p95": bulk_he_p95,
        "bottom_right_He_current_max": bottom_he,
        "bulk_to_notch_He_current": _ratio(bulk_he_p95, notch_he),
        "bottom_to_notch_He_current": _ratio(bottom_he, notch_he),
        "notch_tip_mechanics_drive_max": notch_mech,
        "bulk_mechanics_drive_p95": bulk_mech_p95,
        "bottom_right_mechanics_drive_max": bottom_mech,
        "bulk_to_notch_mechanics_drive": _ratio(bulk_mech_p95, notch_mech),
        "bottom_to_notch_mechanics_drive": _ratio(bottom_mech, notch_mech),
        "notch_tip_eps_norm_max": notch_eps,
        "bulk_eps_norm_p95": bulk_eps_p95,
        "bottom_right_eps_norm_max": bottom_eps,
        "bulk_to_notch_eps_norm": _ratio(bulk_eps_p95, notch_eps),
        "bottom_to_notch_eps_norm": _ratio(bottom_eps, notch_eps),
        "notch_lip_pairs": len(pairs),
        "notch_lip_u_jump_abs_max": float(np.max(np.abs(u_jumps))) if u_jumps.size else np.nan,
        "notch_lip_v_jump_abs_max": float(np.max(np.abs(v_jumps))) if v_jumps.size else np.nan,
        "notch_lip_u_jump_std": float(np.std(u_jumps)) if u_jumps.size else np.nan,
        "notch_lip_v_jump_std": float(np.std(v_jumps)) if v_jumps.size else np.nan,
        "top_u_abs_max": float(torch.max(torch.abs(u[top])).detach().cpu()) if torch.any(top) else np.nan,
        "top_v_error_max": float(torch.max(torch.abs(v[top] - args.delta)).detach().cpu()) if torch.any(top) else np.nan,
        "bottom_u_abs_max": float(torch.max(torch.abs(u[bottom])).detach().cpu()) if torch.any(bottom) else np.nan,
        "bottom_v_abs_max": float(torch.max(torch.abs(v[bottom])).detach().cpu()) if torch.any(bottom) else np.nan,
    }
    row["classification"] = _classify(row)
    pair_rows = [{**case, **pair} for pair in pairs]
    return row, pair_rows


def _write_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_npz(path, inp, T_conn, result):
    fields = result["fields"]
    x_elem, y_elem = element_centroids(inp, T_conn)
    np.savez_compressed(
        path,
        x=inp[:, 0].detach().cpu().numpy(),
        y=inp[:, 1].detach().cpu().numpy(),
        triangles=T_conn.detach().cpu().numpy(),
        element_x=x_elem.detach().cpu().numpy(),
        element_y=y_elem.detach().cpu().numpy(),
        u=result["u"].detach().cpu().numpy(),
        v=result["v"].detach().cpu().numpy(),
        alpha=np.zeros(inp.shape[0], dtype=np.float32),
        alpha_elem=fields["alpha_elem"].detach().cpu().numpy(),
        He_current=fields["He_current"].detach().cpu().numpy(),
        mechanics_drive=fields["mechanics_drive"].detach().cpu().numpy(),
        psiI=fields["psiI"].detach().cpu().numpy(),
        psiII=fields["psiII"].detach().cpu().numpy(),
        eps_xx=fields["eps_xx"].detach().cpu().numpy(),
        eps_yy=fields["eps_yy"].detach().cpu().numpy(),
        eps_xy=fields["eps_xy"].detach().cpu().numpy(),
    )


def _plot_representative(figures_dir, npz_records):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - diagnostic environment fallback
        return [{"filename": "not_generated", "error": str(exc)}]

    generated = []
    for record in npz_records:
        if not record["representative"]:
            continue
        with np.load(record["path"]) as data:
            x_elem = data["element_x"]
            y_elem = data["element_y"]
            he = data["He_current"]
        fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
        scatter = ax.scatter(x_elem, y_elem, c=np.log10(np.maximum(he, 1.0e-30)), s=8, cmap="viridis")
        ax.plot([0.0, NOTCH_TIP_X_MM], [NOTCH_CENTER_Y_MM, NOTCH_CENTER_Y_MM], color="white", linewidth=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(record["case_id"])
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("log10(He_current)")
        out = figures_dir / f"{record['case_id']}_He_current.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        generated.append({"filename": out.name, "case_id": record["case_id"]})
    return generated


def main():
    args = parse_args()
    _prepare_output_dirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, tri, area = parse_mesh(str(args.mesh), gradient_type="numerical")
    inp = torch.tensor(np.column_stack([x, y]), dtype=torch.float32, device=device)
    T_conn = torch.tensor(tri, dtype=torch.long, device=device)
    area_T = torch.tensor(area, dtype=torch.float32, device=device)

    rows = []
    pair_rows = []
    npz_records = []
    commands = {
        "script": str(Path(__file__).resolve()),
        "device": device,
        "args": vars(args),
    }
    (args.out_dir / "commands_run.json").write_text(json.dumps(commands, indent=2, default=str), encoding="utf-8")

    for top_u_mode in args.top_u_modes:
        for loss_form in args.loss_forms:
            for epochs in args.pinn_epochs:
                case = {
                    "case_id": f"pinn_{top_u_mode}_{loss_form}_e{epochs}",
                    "method": "PINN",
                    "top_u_mode": top_u_mode,
                    "loss_form": loss_form,
                    "epochs": epochs,
                    "hidden_layers": args.hidden_layers,
                    "neurons": args.neurons,
                    "seed": args.seed,
                }
                result = _train_pinn(inp, area_T, T_conn, args, top_u_mode, epochs, loss_form, device)
                row, pairs = _summarize_case(case, inp, T_conn, result, args)
                rows.append(row)
                pair_rows.extend(pairs)
                npz_path = args.out_dir / "artifacts" / f"{case['case_id']}_fields.npz"
                _save_npz(npz_path, inp, T_conn, result)
                npz_records.append({"case_id": case["case_id"], "path": npz_path, "representative": loss_form == "log10_energy" and epochs == max(args.pinn_epochs)})

            for epochs in args.fedof_epochs:
                case = {
                    "case_id": f"fedof_{top_u_mode}_{loss_form}_e{epochs}",
                    "method": "FE_DOF",
                    "top_u_mode": top_u_mode,
                    "loss_form": loss_form,
                    "epochs": epochs,
                    "hidden_layers": np.nan,
                    "neurons": np.nan,
                    "seed": args.seed,
                }
                result = _train_fedof(inp, area_T, T_conn, args, top_u_mode, epochs, loss_form, device)
                row, pairs = _summarize_case(case, inp, T_conn, result, args)
                rows.append(row)
                pair_rows.extend(pairs)
                npz_path = args.out_dir / "artifacts" / f"{case['case_id']}_fields.npz"
                _save_npz(npz_path, inp, T_conn, result)
                npz_records.append({"case_id": case["case_id"], "path": npz_path, "representative": loss_form == "log10_energy"})

    _write_rows(args.out_dir / "tables" / "mechanics_only_comparison.csv", rows)
    lip_summary = [
        {
            "case_id": row["case_id"],
            "method": row["method"],
            "top_u_mode": row["top_u_mode"],
            "loss_form": row["loss_form"],
            "epochs": row["epochs"],
            "notch_lip_pairs": row["notch_lip_pairs"],
            "notch_lip_u_jump_abs_max": row["notch_lip_u_jump_abs_max"],
            "notch_lip_v_jump_abs_max": row["notch_lip_v_jump_abs_max"],
            "notch_lip_u_jump_std": row["notch_lip_u_jump_std"],
            "notch_lip_v_jump_std": row["notch_lip_v_jump_std"],
            "notch_tip_He_current_max": row["notch_tip_He_current_max"],
            "bulk_to_notch_He_current": row["bulk_to_notch_He_current"],
            "bottom_to_notch_He_current": row["bottom_to_notch_He_current"],
            "classification": row["classification"],
        }
        for row in rows
    ]
    _write_rows(args.out_dir / "tables" / "notch_lip_comparison.csv", lip_summary)
    _write_rows(args.out_dir / "tables" / "notch_lip_node_pairs.csv", pair_rows)
    if not args.skip_figures:
        generated = _plot_representative(args.out_dir / "figures", npz_records)
        (args.out_dir / "figures" / "generated_figures.json").write_text(
            json.dumps(generated, indent=2, default=str), encoding="utf-8"
        )
    print(f"wrote {args.out_dir / 'tables' / 'mechanics_only_comparison.csv'}")
    print(f"wrote {args.out_dir / 'tables' / 'notch_lip_comparison.csv'}")


if __name__ == "__main__":
    main()
