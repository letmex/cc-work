"""Postprocess TM checkpoint results into reaction and stress-strain outputs.

The normal reaction metric is the energy-conjugate derivative dPi/dDelta
computed from saved per-step checkpoints. If checkpoints are unavailable, the
module writes a table that marks the reaction metric as unavailable rather than
substituting a boundary-stress curve.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_energy_mixed_tm import compute_mixed_tm_energy  # noqa: E402
from construct_model import construct_model  # noqa: E402
from field_computation import FieldComputation  # noqa: E402
from input_data_from_mesh import prep_input_data  # noqa: E402


SPECIMEN_SIZE_MM = 0.01
REFERENCE_LENGTH_MM = 0.01
REFERENCE_AREA_MM2 = 0.01
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
RIGHT_BOUNDARY_X = 0.01
RIGHT_BOUNDARY_BAND = 2.5e-4

STRESS_STRAIN_TABLE_NAME = "stress_strain_by_step.csv"
REACTION_TABLE_NAME = "reaction_by_step.csv"
REACTION_AVAILABILITY_NAME = "reaction_metric_availability.csv"


def default_curve_dir(result_dir: Path) -> Path:
    return Path(result_dir) / "curves"


def default_figure_dir(result_dir: Path) -> Path:
    return Path(result_dir) / "figures"


def parse_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not path.exists():
        return settings
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        settings[key.strip()] = value.strip()
    return settings


def to_float(settings: dict[str, str], key: str, default: float = math.nan) -> float:
    value = settings.get(key)
    if value is None or value == "None" or value == "":
        return default
    return float(value)


def _as_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def checkpoint_paths(model_dir: Path) -> list[Path]:
    ckpt_dir = Path(model_dir) / "best_models" / "step_checkpoints"
    return sorted(ckpt_dir.glob("checkpoint_mixedH_TM_step_*.pt"), key=_as_step)


def load_checkpoint_payload(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def field_paths(result_dir: Path) -> list[Path]:
    return sorted(Path(result_dir).glob("fields_mixed_tm_step_*.npz"), key=_as_step)


def _read_displacements(result_dir: Path) -> list[tuple[int, float]]:
    csv_path = Path(result_dir) / "displacement_list.csv"
    if csv_path.exists():
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row.get("step", len(rows))))
                delta = float(row.get("displacement_mm", row.get("Delta", "nan")))
                rows.append((step, delta))
        return rows
    npy_path = Path(result_dir) / "displacement_list.npy"
    if npy_path.exists():
        values = np.load(npy_path)
        return [(idx, float(value)) for idx, value in enumerate(values)]
    rows = []
    for path in field_paths(result_dir):
        with np.load(path) as data:
            rows.append((_as_step(path), float(data["displacement_mm"])))
    return rows


def build_stress_strain_curve(
    reactions: pd.DataFrame,
    reference_length_mm: float = REFERENCE_LENGTH_MM,
    reference_area_mm2: float = REFERENCE_AREA_MM2,
    load_case: str | None = None,
) -> pd.DataFrame:
    data = reactions.copy()
    if data.empty:
        raise RuntimeError("Reaction table is empty")
    if "reaction_N_energy" not in data.columns:
        raise RuntimeError("Reaction table lacks reaction_N_energy")
    if "Delta" not in data.columns and "Delta_mm" in data.columns:
        data["Delta"] = data["Delta_mm"]
    if "reaction_N_virtual_work" not in data.columns:
        data["reaction_N_virtual_work"] = math.nan
    if "alpha0p8_through_crack" not in data.columns:
        data["alpha0p8_through_crack"] = False
    resolved_case = load_case
    if resolved_case is None and "load_case" in data.columns:
        cases = [value for value in data["load_case"].dropna().unique()]
        resolved_case = str(cases[0]) if cases else "tension"
    resolved_case = resolved_case or "tension"
    data["reference_length_mm"] = float(reference_length_mm)
    data["reference_area_mm2"] = float(reference_area_mm2)
    data["load_case"] = resolved_case
    if resolved_case == "shear":
        if "Delta_s" not in data.columns:
            data["Delta_s"] = data["Delta"]
        data["engineering_shear_strain"] = data["Delta_s"].astype(float) / float(reference_length_mm)
        data["nominal_shear_stress_energy_MPa"] = (
            data["reaction_N_energy"].astype(float) / float(reference_area_mm2)
        )
        data["nominal_shear_stress_virtual_work_MPa"] = (
            data["reaction_N_virtual_work"].astype(float) / float(reference_area_mm2)
        )
        data["stress_strain_primary_metric"] = "nominal_shear_stress_energy_MPa"
    else:
        data["nominal_strain"] = data["Delta"].astype(float) / float(reference_length_mm)
        data["nominal_stress_energy_MPa"] = data["reaction_N_energy"].astype(float) / float(reference_area_mm2)
        data["nominal_stress_virtual_work_MPa"] = (
            data["reaction_N_virtual_work"].astype(float) / float(reference_area_mm2)
        )
        data["stress_strain_primary_metric"] = "nominal_stress_energy_MPa"
    data["reaction_metric_status"] = "energy_conjugate"
    data["is_energy_conjugate"] = True
    if resolved_case == "shear":
        preferred = [
            "seed",
            "step",
            "load_case",
            "Delta_s",
            "engineering_shear_strain",
            "reference_length_mm",
            "reference_area_mm2",
            "stress_strain_primary_metric",
            "reaction_metric_status",
            "is_energy_conjugate",
            "nominal_shear_stress_energy_MPa",
            "nominal_shear_stress_virtual_work_MPa",
            "reaction_N_energy",
            "reaction_N_virtual_work",
            "alpha0p8_through_crack",
        ]
    else:
        preferred = [
            "seed",
            "step",
            "load_case",
            "Delta",
            "nominal_strain",
            "reference_length_mm",
            "reference_area_mm2",
            "stress_strain_primary_metric",
            "reaction_metric_status",
            "is_energy_conjugate",
            "nominal_stress_energy_MPa",
            "nominal_stress_virtual_work_MPa",
            "reaction_N_energy",
            "reaction_N_virtual_work",
            "alpha0p8_through_crack",
        ]
    columns = [column for column in preferred if column in data.columns]
    columns.extend([column for column in data.columns if column not in columns])
    return data[columns].sort_values(["seed", "step"] if "seed" in data.columns else ["step"])


def write_unavailable_stress_strain_curve(
    result_dir: Path,
    out_path: Path,
    reason: str,
    reference_length_mm: float = REFERENCE_LENGTH_MM,
    reference_area_mm2: float = REFERENCE_AREA_MM2,
    seed: int | None = None,
    load_case: str = "tension",
) -> Path:
    rows = []
    for step, delta in _read_displacements(result_dir):
        row = {
            "step": int(step),
            "Delta": float(delta),
            "load_case": load_case,
            "reference_length_mm": float(reference_length_mm),
            "reference_area_mm2": float(reference_area_mm2),
            "stress_strain_primary_metric": "reaction_metric_unavailable",
            "reaction_metric_status": str(reason),
            "is_energy_conjugate": False,
            "reaction_N_energy": math.nan,
            "reaction_N_virtual_work": math.nan,
            "alpha0p8_through_crack": False,
            "unavailable_reason": str(reason),
        }
        if load_case == "shear":
            row["Delta_s"] = float(delta)
            row["engineering_shear_strain"] = float(delta) / float(reference_length_mm)
            row["nominal_shear_stress_energy_MPa"] = math.nan
            row["nominal_shear_stress_virtual_work_MPa"] = math.nan
        else:
            row["nominal_strain"] = float(delta) / float(reference_length_mm)
            row["nominal_stress_energy_MPa"] = math.nan
            row["nominal_stress_virtual_work_MPa"] = math.nan
        if seed is not None:
            row["seed"] = int(seed)
        rows.append(row)
    if not rows:
        rows.append(
            {
                "step": math.nan,
                "Delta": math.nan,
                "load_case": load_case,
                "reference_length_mm": float(reference_length_mm),
                "reference_area_mm2": float(reference_area_mm2),
                "stress_strain_primary_metric": "reaction_metric_unavailable",
                "reaction_metric_status": str(reason),
                "is_energy_conjugate": False,
                "reaction_N_energy": math.nan,
                "alpha0p8_through_crack": False,
                "unavailable_reason": str(reason),
            }
        )
        if load_case == "shear":
            rows[-1]["Delta_s"] = math.nan
            rows[-1]["engineering_shear_strain"] = math.nan
            rows[-1]["nominal_shear_stress_energy_MPa"] = math.nan
        else:
            rows[-1]["nominal_strain"] = math.nan
            rows[-1]["nominal_stress_energy_MPa"] = math.nan
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def edge_map(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for elem, nodes in enumerate(triangles.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges[tuple(sorted((int(a), int(b))))].append(elem)
    return edges


def element_adjacency(triangles: np.ndarray) -> list[list[int]]:
    edges = edge_map(triangles)
    adjacency = [[] for _ in range(len(triangles))]
    for elems in edges.values():
        if len(elems) == 2:
            a, b = elems
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def connected_component(mask: np.ndarray, adjacency: list[list[int]], seed_mask: np.ndarray) -> np.ndarray:
    seeds = np.flatnonzero(mask & seed_mask)
    visited = np.zeros(mask.shape[0], dtype=bool)
    queue: deque[int] = deque()
    for idx in seeds:
        visited[idx] = True
        queue.append(int(idx))
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    return visited


def through_metrics(data: dict[str, np.ndarray], threshold: float = 0.8) -> dict[str, object]:
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    mask = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    comp = connected_component(mask, element_adjacency(data["triangles"]), seed_mask)
    comp_x = x[comp]
    comp_y = y[comp]
    if comp_x.size:
        reaches_right = bool(np.any(comp_x >= RIGHT_BOUNDARY_X - RIGHT_BOUNDARY_BAND))
        return {
            "alpha0p8_through_crack": reaches_right,
            "alpha0p8_connected_count": int(np.sum(comp)),
            "alpha0p8_connected_min_x": float(np.min(comp_x)),
            "alpha0p8_connected_max_x": float(np.max(comp_x)),
            "alpha0p8_connected_mean_y": float(np.mean(comp_y)),
            "alpha0p8_connected_x_span": float(np.max(comp_x) - np.min(comp_x)),
        }
    return {
        "alpha0p8_through_crack": False,
        "alpha0p8_connected_count": 0,
        "alpha0p8_connected_min_x": math.nan,
        "alpha0p8_connected_max_x": math.nan,
        "alpha0p8_connected_mean_y": math.nan,
        "alpha0p8_connected_x_span": 0.0,
    }


def _tensor_to_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _element_centroids_np(inp: torch.Tensor, t_conn: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    pts = _tensor_to_np(inp)
    tri = _tensor_to_np(t_conn).astype(int)
    centers = pts[tri].mean(axis=1)
    return centers[:, 0], centers[:, 1]


def _seed_from_settings(settings: dict[str, str]) -> int | None:
    value = settings.get("seed")
    return int(value) if value not in {None, ""} else None


def _build_field(settings: dict[str, str], device: torch.device):
    domain_extrema = torch.tensor([[0.0, SPECIMEN_SIZE_MM], [0.0, SPECIMEN_SIZE_MM]], device=device)
    pff_dict = {
        "PFF_model": settings.get("PFF_model", "AT2"),
        "se_split": settings.get("se_split", "volumetric"),
        "tol_ir": 5.0e-3,
    }
    mat_dict = {
        "mat_E": to_float(settings, "mat_E_kN_per_mm2"),
        "mat_nu": to_float(settings, "mat_nu"),
        "w1": to_float(settings, "w1_kN_per_mm2"),
        "l0": to_float(settings, "l0_mm"),
    }
    net_dict = {
        "hidden_layers": int(settings["hidden_layers"]),
        "neurons": int(settings["neurons"]),
        "seed": int(settings["seed"]),
        "activation": settings["activation"],
        "init_coeff": float(settings["coeff"]),
    }
    pffmodel, matprop, network = construct_model(pff_dict, mat_dict, net_dict, domain_extrema, device)
    field = FieldComputation(
        net=network,
        domain_extrema=domain_extrema,
        lmbda=torch.tensor([0.0], device=device),
        theta=torch.tensor([np.pi / 2], device=device),
        alpha_constraint="nonsmooth",
        top_u_mode=settings.get("top_u_mode", "free"),
        coord_normalization=settings.get("coord_normalization", "unit_box"),
        load_case=settings.get("load_case", "tension"),
    )
    field.net = field.net.to(device)
    crack_dict = {"x_init": [0.0], "y_init": [NOTCH_Y], "L_crack": [0.0], "angle_crack": [0.0]}
    numr_dict = {"alpha_constraint": "nonsmooth", "gradient_type": settings.get("gradient_type", "numerical")}
    mesh_file = settings.get("fine_mesh_file") or str(ROOT / "geo_coarse_with_groups_mm.msh")
    inp, t_conn, area_t, _ = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file, device)
    return field, matprop, pffmodel, inp, t_conn, area_t


def _history_for_step(step: int, payloads: dict[int, dict], area_t: torch.Tensor, device: torch.device):
    if step <= 0 or (step - 1) not in payloads:
        zeros = torch.zeros_like(area_t, device=device)
        return zeros, zeros.clone(), "zero_initial_history"
    prev = payloads[step - 1]
    return (
        prev["history"]["HI"].to(device=device, dtype=area_t.dtype),
        prev["history"]["HII"].to(device=device, dtype=area_t.dtype),
        f"checkpoint_step_{step - 1:04d}_post_history",
    )


def _alpha_old_for_step(step: int, result_dir: Path, device: torch.device):
    if step <= 0:
        return None
    prev_field = Path(result_dir) / f"fields_mixed_tm_step_{step - 1:04d}.npz"
    if not prev_field.exists():
        return None
    with np.load(prev_field) as data:
        return torch.as_tensor(data["alpha"], dtype=torch.float32, device=device)


def _energy_and_fields(
    field: FieldComputation,
    state_dict: dict,
    delta_value: float,
    hi: torch.Tensor,
    hii: torch.Tensor,
    matprop,
    pffmodel,
    inp: torch.Tensor,
    t_conn: torch.Tensor,
    area_t: torch.Tensor,
    settings: dict[str, str],
    alpha_old=None,
    create_delta_grad: bool = False,
):
    field.net.load_state_dict(state_dict)
    delta = torch.tensor(float(delta_value), dtype=inp.dtype, device=inp.device, requires_grad=create_delta_grad)
    field.lmbda = delta
    u, v, alpha = field.fieldCalculation(inp)
    e_el, e_d, fields = compute_mixed_tm_energy(
        inp,
        u,
        v,
        alpha,
        hi,
        hii,
        matprop,
        pffmodel,
        area_t,
        t_conn,
        eta_residual=to_float(settings, "eta_residual", 1.0e-5),
        gcII=to_float(settings, "GcII_kN_per_mm", math.nan),
        gcII_factor=to_float(settings, "GcII_factor", 1.0),
        tm_eps_r=to_float(settings, "tm_eps_r", 1.0e-5),
        alpha_old=alpha_old,
    )
    return delta, e_el, e_d, fields, u, v, alpha


def _recomputed_data(inp, t_conn, fields, alpha) -> dict[str, np.ndarray]:
    x_elem, y_elem = _element_centroids_np(inp, t_conn)
    return {
        "x": _tensor_to_np(inp[:, 0]),
        "y": _tensor_to_np(inp[:, 1]),
        "triangles": _tensor_to_np(t_conn).astype(int),
        "element_x": x_elem,
        "element_y": y_elem,
        "alpha": _tensor_to_np(alpha),
        "alpha_elem": _tensor_to_np(fields["alpha_elem"]),
    }


def compute_exact_reaction_rows(model_dir: Path, result_dir: Path, device: str | torch.device = "cpu") -> pd.DataFrame:
    device = torch.device(device)
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    settings = parse_settings(model_dir / "model_settings.txt")
    load_case = settings.get("load_case", "tension")
    if not settings:
        raise RuntimeError(f"Missing model settings: {model_dir / 'model_settings.txt'}")
    ckpts = checkpoint_paths(model_dir)
    if not ckpts:
        raise RuntimeError("no_step_checkpoints")
    field, matprop, pffmodel, inp, t_conn, area_t = _build_field(settings, device)
    payloads = {int(_as_step(path)): load_checkpoint_payload(path, device) for path in ckpts}
    seed = _seed_from_settings(settings)
    rows = []
    for ckpt_path in ckpts:
        step = _as_step(ckpt_path)
        payload = payloads[step]
        delta_value = float(payload["Delta"])
        hi, hii, history_source = _history_for_step(step, payloads, area_t, device)
        alpha_old = _alpha_old_for_step(step, result_dir, device)
        delta, e_el, e_d, fields, _u, _v, alpha = _energy_and_fields(
            field,
            payload["model_state_dict"],
            delta_value,
            hi,
            hii,
            matprop,
            pffmodel,
            inp,
            t_conn,
            area_t,
            settings,
            alpha_old=alpha_old,
            create_delta_grad=True,
        )
        pi = e_el + e_d
        grad = torch.autograd.grad(pi, delta, retain_graph=False, allow_unused=False)[0]
        data = _recomputed_data(inp, t_conn, fields, alpha)
        row = {
            "step": int(step),
            "load_case": load_case,
            "Delta": delta_value,
            "strain": delta_value / SPECIMEN_SIZE_MM,
            "checkpoint": str(ckpt_path),
            "history_source": history_source,
            "reaction_N_energy": 1000.0 * float(grad.detach().cpu()),
            "reaction_N_virtual_work": math.nan,
            "Pi_total_kNmm": float(pi.detach().cpu()),
            "elastic_energy_kNmm": float(e_el.detach().cpu()),
            "fracture_energy_kNmm": float(e_d.detach().cpu()),
            "reaction_metric_status": "energy_conjugate",
            "is_energy_conjugate": True,
            **through_metrics(data),
        }
        if load_case == "shear":
            row["Delta_s"] = delta_value
            row["engineering_shear_strain"] = delta_value / SPECIMEN_SIZE_MM
        if seed is not None:
            row["seed"] = int(seed)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "step"] if seed is not None else ["step"])


def _write_availability(out_dir: Path, row: dict[str, object]) -> Path:
    path = Path(out_dir) / REACTION_AVAILABILITY_NAME
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def generate_figures_for_stress_strain_curve(
    result_dir: Path,
    stress_strain_table: Path,
    figure_dir: Path | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    dpi: int = 300,
) -> list[Path]:
    from plot_results import make_result_figures

    result_dir = Path(result_dir)
    stress_strain_table = Path(stress_strain_table)
    figure_dir = Path(figure_dir) if figure_dir is not None else default_figure_dir(result_dir)
    return make_result_figures(
        result_dir,
        figure_dir,
        run_label=run_label,
        stress_strain_csv=stress_strain_table,
        seed=seed,
        dpi=dpi,
    )


def _attach_figure_outputs(
    result: dict[str, object],
    result_dir: Path,
    stress_strain_table: Path,
    generate_figures: bool,
    figure_dir: Path | None,
    run_label: str | None,
    seed: int | None,
    figure_dpi: int,
    fail_on_figure_error: bool,
) -> dict[str, object]:
    if not generate_figures:
        result["figure_status"] = "disabled"
        return result
    try:
        generated = generate_figures_for_stress_strain_curve(
            result_dir,
            stress_strain_table,
            figure_dir=figure_dir,
            run_label=run_label,
            seed=seed,
            dpi=figure_dpi,
        )
    except Exception as exc:
        result["figure_status"] = f"failed_{type(exc).__name__}"
        result["figure_error"] = repr(exc)
        if fail_on_figure_error:
            raise
        return result
    result["figure_status"] = "generated"
    result["figure_dir"] = Path(figure_dir) if figure_dir is not None else default_figure_dir(result_dir)
    result["figures"] = generated
    return result


def run_results_postprocess(
    model_dir: Path,
    result_dir: Path,
    out_dir: Path | None = None,
    device: str | torch.device = "cpu",
    generate_figures: bool = True,
    figure_dir: Path | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    figure_dpi: int = 300,
    fail_on_figure_error: bool = False,
) -> dict[str, object]:
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    out_dir = Path(out_dir) if out_dir is not None else default_curve_dir(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / STRESS_STRAIN_TABLE_NAME
    reaction_path = out_dir / REACTION_TABLE_NAME
    settings = parse_settings(model_dir / "model_settings.txt")
    settings_seed = _seed_from_settings(settings) if settings else None
    load_case = settings.get("load_case", "tension") if settings else "tension"
    selected_seed = seed if seed is not None else settings_seed
    ckpts = checkpoint_paths(model_dir)
    if not ckpts:
        write_unavailable_stress_strain_curve(
            result_dir,
            curve_path,
            "no_step_checkpoints",
            seed=settings_seed,
            load_case=load_case,
        )
        _write_availability(
            out_dir,
            {
                "status": "no_step_checkpoints",
                "model_dir": str(model_dir),
                "result_dir": str(result_dir),
                "checkpoint_count": 0,
                "stress_strain_table": str(curve_path),
                "exact_reaction_computable": False,
            },
        )
        result = {"status": "no_step_checkpoints", "stress_strain_table": curve_path, "exact_reaction_computable": False}
        return _attach_figure_outputs(
            result,
            result_dir,
            curve_path,
            generate_figures,
            figure_dir,
            run_label,
            selected_seed,
            figure_dpi,
            fail_on_figure_error,
        )
    try:
        reactions = compute_exact_reaction_rows(model_dir, result_dir, device=device)
        reactions.to_csv(reaction_path, index=False)
        curve = build_stress_strain_curve(reactions, load_case=load_case)
        curve.to_csv(curve_path, index=False)
        _write_availability(
            out_dir,
            {
                "status": "energy_conjugate",
                "model_dir": str(model_dir),
                "result_dir": str(result_dir),
                "checkpoint_count": len(ckpts),
                "reaction_table": str(reaction_path),
                "stress_strain_table": str(curve_path),
                "exact_reaction_computable": True,
            },
        )
        result = {
            "status": "energy_conjugate",
            "reaction_table": reaction_path,
            "stress_strain_table": curve_path,
            "exact_reaction_computable": True,
        }
        return _attach_figure_outputs(
            result,
            result_dir,
            curve_path,
            generate_figures,
            figure_dir,
            run_label,
            selected_seed,
            figure_dpi,
            fail_on_figure_error,
        )
    except Exception as exc:
        reason = f"exact_reaction_failed_{type(exc).__name__}"
        write_unavailable_stress_strain_curve(
            result_dir,
            curve_path,
            reason,
            seed=settings_seed,
            load_case=load_case,
        )
        _write_availability(
            out_dir,
            {
                "status": reason,
                "model_dir": str(model_dir),
                "result_dir": str(result_dir),
                "checkpoint_count": len(ckpts),
                "stress_strain_table": str(curve_path),
                "exact_reaction_computable": False,
                "error": repr(exc),
            },
        )
        result = {"status": reason, "stress_strain_table": curve_path, "exact_reaction_computable": False}
        return _attach_figure_outputs(
            result,
            result_dir,
            curve_path,
            generate_figures,
            figure_dir,
            run_label,
            selected_seed,
            figure_dpi,
            fail_on_figure_error,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TM reaction/stress-strain tables and figures from step checkpoints."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--fail-on-figure-error", action="store_true")
    args = parser.parse_args()
    result = run_results_postprocess(
        args.model_dir,
        args.result_dir,
        args.out_dir,
        device=args.device,
        generate_figures=not args.no_figures,
        figure_dir=args.figure_dir,
        run_label=args.run_label,
        seed=args.seed,
        figure_dpi=args.figure_dpi,
        fail_on_figure_error=args.fail_on_figure_error,
    )
    print(result)


if __name__ == "__main__":
    main()
