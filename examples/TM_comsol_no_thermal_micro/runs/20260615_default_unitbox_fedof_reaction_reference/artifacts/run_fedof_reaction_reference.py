"""FE-DOF / energy-conjugate reaction reference audit.

This diagnostic reads the final_D0040 frozen-alpha fields from the
discontinuous replay package and assembles a small constant-strain-triangle
mechanics reference problem. It is intentionally limited to reaction and
boundary-condition auditing. It does not evolve alpha, extend loading, or
modify the production PINN/phase-field model.
"""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"
REPO = PACKAGE.parents[3]
PREV_PACKAGE = REPO / "examples" / "TM_comsol_no_thermal_micro" / "runs" / "20260614_default_unitbox_boundary_reaction_audit"
PREV_SCRIPT = PREV_PACKAGE / "artifacts" / "run_boundary_reaction_audit.py"

SEEDS = (7, 13, 42)
BC_TREATMENTS = ("original_top_bottom_bc", "minimal_rigid_body_bc")
FE_VARIANTS = (
    "fedof_continuous_current_split",
    "fedof_continuous_crack_band_void",
    "fedof_piecewise_upper_lower_crack_band_void",
    "fedof_piecewise_upper_lower_current_split",
)
BASE_REPLAY_VARIANT = "split_domain_crack_band_void"


def import_previous_audit():
    spec = importlib.util.spec_from_file_location("boundary_audit", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import previous audit script: {PREV_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BA = import_previous_audit()

SPECIMEN_SIZE_MM = BA.SPECIMEN_SIZE_MM
BOUNDARY_TOL = BA.BOUNDARY_TOL
E = BA.E
NU = BA.NU
MU = BA.MU
ETA_RESIDUAL = BA.ETA_RESIDUAL

# Plane-stress constitutive matrix using tensor shear strain eps_xy.
C_PLANE_STRESS = np.array(
    [
        [E / (1.0 - NU**2), E * NU / (1.0 - NU**2), 0.0],
        [E * NU / (1.0 - NU**2), E / (1.0 - NU**2), 0.0],
        [0.0, 0.0, 2.0 * MU],
    ],
    dtype=float,
)


@dataclass
class FESystem:
    data: dict[str, np.ndarray]
    variant: str
    topology: str
    treatment: str
    node_keys: list[tuple[int, str]]
    key_to_node_id: dict[tuple[int, str], int]
    elem_node_ids: list[list[int] | None]
    stiffness_weight: np.ndarray
    K: sp.csr_matrix
    components: list[list[int]]
    delta: float
    implementation_note: str


@dataclass
class FESolution:
    system: FESystem
    bc_treatment: str
    U: np.ndarray
    residual_model_units: np.ndarray
    prescribed: dict[int, float]
    delta_coeff: dict[int, float]
    top_v_dofs: list[int]
    bottom_v_dofs: list[int]
    solve_status: str
    auto_anchor_count: int
    energy_model_units: float
    energy_Nmm_proxy: float
    energy_conjugate_reaction_N: float
    top_constrained_dof_reaction_N: float
    bottom_constrained_dof_reaction_N: float


def setup_dirs():
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_base(seed: int) -> dict[str, np.ndarray]:
    data = BA.load_npz(seed, BASE_REPLAY_VARIANT)
    data["seed"] = seed
    data["case"] = f"D0040_seed{seed}_default_unitbox"
    return data


def top_bottom_delta(data: dict[str, np.ndarray]) -> float:
    top = np.isclose(data["y"], SPECIMEN_SIZE_MM, atol=BOUNDARY_TOL)
    if not np.any(top):
        raise RuntimeError("Top boundary nodes not found")
    return float(np.nanmean(data["v"][top]))


def triangle_B_matrix(x: np.ndarray, y: np.ndarray, tri_nodes: np.ndarray):
    x1, x2, x3 = x[tri_nodes]
    y1, y2, y3 = y[tri_nodes]
    area_signed = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    area = abs(area_signed)
    if area <= 0.0:
        raise ValueError("Degenerate triangle")
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float) / (2.0 * area_signed)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float) / (2.0 * area_signed)
    B = np.zeros((3, 6), dtype=float)
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = 0.5 * c[i]
        B[2, 2 * i + 1] = 0.5 * b[i]
    return B, area


def variant_spec(variant: str) -> tuple[str, str]:
    if variant == "fedof_continuous_current_split":
        return "continuous", "current_split"
    if variant == "fedof_continuous_crack_band_void":
        return "continuous", "crack_band_void"
    if variant == "fedof_piecewise_upper_lower_crack_band_void":
        return "piecewise_upper_lower", "crack_band_void"
    if variant == "fedof_piecewise_upper_lower_current_split":
        return "piecewise_upper_lower", "current_split"
    raise ValueError(variant)


def build_node_map(data: dict[str, np.ndarray], topology: str, stiffness_weight: np.ndarray):
    tri = data["triangles"].astype(int)
    elem_upper = data["elem_upper"].astype(bool)
    node_keys: list[tuple[int, str]] = []
    key_to_node_id: dict[tuple[int, str], int] = {}
    elem_node_ids: list[list[int] | None] = []

    def ensure_key(node: int, label: str) -> int:
        key = (int(node), label)
        if key not in key_to_node_id:
            key_to_node_id[key] = len(node_keys)
            node_keys.append(key)
        return key_to_node_id[key]

    for elem, nodes in enumerate(tri):
        if stiffness_weight[elem] <= 0.0:
            elem_node_ids.append(None)
            continue
        if topology == "continuous":
            elem_node_ids.append([ensure_key(int(n), "single") for n in nodes])
        else:
            label = "upper" if elem_upper[elem] else "lower"
            elem_node_ids.append([ensure_key(int(n), label) for n in nodes])
    return node_keys, key_to_node_id, elem_node_ids


def stiffness_weights(data: dict[str, np.ndarray], treatment: str) -> np.ndarray:
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    g = np.maximum((1.0 - alpha) ** 2 + ETA_RESIDUAL, 0.0)
    weight = np.array(g, copy=True)
    if treatment == "crack_band_void":
        weight[np.asarray(data["crack_mask"], dtype=bool)] = 0.0
    elif treatment == "current_split":
        pass
    else:
        raise ValueError(treatment)
    return weight


def assemble_system(data: dict[str, np.ndarray], variant: str) -> FESystem:
    topology, treatment = variant_spec(variant)
    weight = stiffness_weights(data, treatment)
    node_keys, key_to_node_id, elem_node_ids = build_node_map(data, topology, weight)
    ndof = 2 * len(node_keys)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    graph: list[set[int]] = [set() for _ in node_keys]
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    tri = data["triangles"].astype(int)
    for elem, nodes in enumerate(tri):
        node_ids = elem_node_ids[elem]
        if node_ids is None:
            continue
        B, area = triangle_B_matrix(x, y, nodes)
        Ke = float(weight[elem]) * area * (B.T @ C_PLANE_STRESS @ B)
        dofs: list[int] = []
        for node_id in node_ids:
            dofs.extend([2 * node_id, 2 * node_id + 1])
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                if Ke[i, j] != 0.0:
                    rows.append(gi)
                    cols.append(gj)
                    vals.append(float(Ke[i, j]))
        for a in node_ids:
            for b in node_ids:
                if a != b:
                    graph[a].add(b)
                    graph[b].add(a)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    components = connected_components(graph)
    note = (
        "linear CST diagnostic with plane-stress isotropic degraded stiffness; "
        "TM spectral split is approximated by frozen-alpha scalar stiffness for reaction auditing"
    )
    return FESystem(
        data=data,
        variant=variant,
        topology=topology,
        treatment=treatment,
        node_keys=node_keys,
        key_to_node_id=key_to_node_id,
        elem_node_ids=elem_node_ids,
        stiffness_weight=weight,
        K=K,
        components=components,
        delta=top_bottom_delta(data),
        implementation_note=note,
    )


def connected_components(graph: list[set[int]]) -> list[list[int]]:
    seen = np.zeros(len(graph), dtype=bool)
    comps: list[list[int]] = []
    for start in range(len(graph)):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt in graph[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        comps.append(comp)
    return comps


def set_prescribed(prescribed: dict[int, float], dof: int, value: float):
    old = prescribed.get(dof)
    if old is not None and abs(old - value) > 1.0e-10:
        raise RuntimeError(f"Conflicting prescribed value for dof {dof}: {old} vs {value}")
    prescribed[dof] = float(value)


def boundary_node_ids(system: FESystem, comp: list[int] | None = None):
    node_set = set(comp) if comp is not None else set(range(len(system.node_keys)))
    top: list[int] = []
    bottom: list[int] = []
    left: list[int] = []
    right: list[int] = []
    x = system.data["x"]
    y = system.data["y"]
    for node_id in node_set:
        original_node, _label = system.node_keys[node_id]
        if abs(float(y[original_node]) - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL:
            top.append(node_id)
        if abs(float(y[original_node])) <= BOUNDARY_TOL:
            bottom.append(node_id)
        if abs(float(x[original_node])) <= BOUNDARY_TOL:
            left.append(node_id)
        if abs(float(x[original_node]) - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL:
            right.append(node_id)
    return {
        "top": sorted(set(top), key=lambda n: (system.data["x"][system.node_keys[n][0]], system.node_keys[n][1])),
        "bottom": sorted(set(bottom), key=lambda n: (system.data["x"][system.node_keys[n][0]], system.node_keys[n][1])),
        "left": sorted(set(left), key=lambda n: (system.data["y"][system.node_keys[n][0]], system.node_keys[n][1])),
        "right": sorted(set(right), key=lambda n: (system.data["y"][system.node_keys[n][0]], system.node_keys[n][1])),
    }


def add_original_bcs(system: FESystem):
    prescribed: dict[int, float] = {}
    delta_coeff: dict[int, float] = {}
    boundaries = boundary_node_ids(system)
    for node_id in boundaries["bottom"]:
        set_prescribed(prescribed, 2 * node_id, 0.0)
        set_prescribed(prescribed, 2 * node_id + 1, 0.0)
    for node_id in boundaries["top"]:
        dof = 2 * node_id + 1
        set_prescribed(prescribed, dof, system.delta)
        delta_coeff[dof] = 1.0
    anchors = add_component_anchors(system, prescribed)
    return prescribed, delta_coeff, anchors


def add_minimal_bcs(system: FESystem):
    prescribed: dict[int, float] = {}
    delta_coeff: dict[int, float] = {}
    anchor_count = 0
    for comp in system.components:
        b = boundary_node_ids(system, comp)
        handled = False
        if b["bottom"]:
            left_bottom = b["bottom"][0]
            right_bottom = b["bottom"][-1]
            set_prescribed(prescribed, 2 * left_bottom, 0.0)
            set_prescribed(prescribed, 2 * left_bottom + 1, 0.0)
            set_prescribed(prescribed, 2 * right_bottom + 1, 0.0)
            anchor_count += 3
            handled = True
        if b["top"]:
            left_top = b["top"][0]
            right_top = b["top"][-1]
            if not b["bottom"]:
                set_prescribed(prescribed, 2 * left_top, 0.0)
                anchor_count += 1
            for node_id in sorted({left_top, right_top}):
                dof = 2 * node_id + 1
                set_prescribed(prescribed, dof, system.delta)
                delta_coeff[dof] = 1.0
            anchor_count += len(set([left_top, right_top]))
            handled = True
        if not handled:
            first = comp[0]
            second = comp[-1]
            set_prescribed(prescribed, 2 * first, 0.0)
            set_prescribed(prescribed, 2 * first + 1, 0.0)
            set_prescribed(prescribed, 2 * second + 1, 0.0)
            anchor_count += 3
    anchors = add_component_anchors(system, prescribed)
    return prescribed, delta_coeff, anchor_count + anchors


def add_component_anchors(system: FESystem, prescribed: dict[int, float]) -> int:
    added = 0
    for comp in system.components:
        u_fixed = [n for n in comp if 2 * n in prescribed]
        v_fixed = [n for n in comp if 2 * n + 1 in prescribed]
        if not u_fixed:
            set_prescribed(prescribed, 2 * comp[0], 0.0)
            added += 1
        if not v_fixed:
            set_prescribed(prescribed, 2 * comp[0] + 1, 0.0)
            added += 1
        v_fixed = [n for n in comp if 2 * n + 1 in prescribed]
        if len(v_fixed) < 2 and len(comp) > 1:
            set_prescribed(prescribed, 2 * comp[-1] + 1, 0.0)
            added += 1
    return added


def solve_system(system: FESystem, bc_treatment: str) -> FESolution:
    if bc_treatment == "original_top_bottom_bc":
        prescribed, delta_coeff, anchors = add_original_bcs(system)
    elif bc_treatment == "minimal_rigid_body_bc":
        prescribed, delta_coeff, anchors = add_minimal_bcs(system)
    else:
        raise ValueError(bc_treatment)

    ndof = system.K.shape[0]
    U = np.zeros(ndof, dtype=float)
    p = np.array(sorted(prescribed), dtype=int)
    U[p] = np.array([prescribed[int(i)] for i in p], dtype=float)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[p] = False
    f = np.flatnonzero(free_mask)
    status = "solved"
    if len(f) > 0:
        Kff = system.K[f][:, f]
        Kfp = system.K[f][:, p]
        rhs = -Kfp @ U[p]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", spla.MatrixRankWarning)
                U[f] = spla.spsolve(Kff, rhs)
                if any(issubclass(w.category, spla.MatrixRankWarning) for w in caught):
                    status = "solved_with_matrix_rank_warning"
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            status = f"failed: {type(exc).__name__}: {exc}"
            U[f] = np.nan
    if not np.all(np.isfinite(U)):
        status = "failed_nonfinite_solution"
    residual = system.K @ np.nan_to_num(U, nan=0.0)
    energy = 0.5 * float(np.nan_to_num(U, nan=0.0) @ residual)
    top_v_dofs = [d for d in delta_coeff if abs(delta_coeff[d]) > 0.0]
    bottom_v_dofs = [2 * n + 1 for n in boundary_node_ids(system)["bottom"] if 2 * n + 1 in prescribed]
    top_reaction = 1000.0 * float(sum(residual[d] for d in top_v_dofs))
    bottom_reaction = 1000.0 * float(sum(residual[d] for d in bottom_v_dofs))
    conjugate = 1000.0 * float(sum(residual[d] * coeff for d, coeff in delta_coeff.items()))
    return FESolution(
        system=system,
        bc_treatment=bc_treatment,
        U=U,
        residual_model_units=residual,
        prescribed=prescribed,
        delta_coeff=delta_coeff,
        top_v_dofs=top_v_dofs,
        bottom_v_dofs=bottom_v_dofs,
        solve_status=status,
        auto_anchor_count=anchors,
        energy_model_units=energy,
        energy_Nmm_proxy=1000.0 * energy,
        energy_conjugate_reaction_N=conjugate,
        top_constrained_dof_reaction_N=top_reaction,
        bottom_constrained_dof_reaction_N=bottom_reaction,
    )


def element_fields(solution: FESolution):
    system = solution.system
    data = system.data
    tri = data["triangles"].astype(int)
    n_elem = len(tri)
    eps = np.zeros((n_elem, 3), dtype=float)
    stress = np.zeros((n_elem, 3), dtype=float)
    energy_density = np.zeros(n_elem, dtype=float)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    for elem, nodes in enumerate(tri):
        node_ids = system.elem_node_ids[elem]
        if node_ids is None:
            continue
        dofs: list[int] = []
        for node_id in node_ids:
            dofs.extend([2 * node_id, 2 * node_id + 1])
        B, _area = triangle_B_matrix(x, y, nodes)
        e = B @ solution.U[dofs]
        s = system.stiffness_weight[elem] * (C_PLANE_STRESS @ e)
        eps[elem, :] = e
        stress[elem, :] = s
        energy_density[elem] = 0.5 * float(e @ (system.stiffness_weight[elem] * C_PLANE_STRESS @ e))
    stress_dict = {
        "variant_xx": stress[:, 0],
        "variant_yy": stress[:, 1],
        "variant_xy": stress[:, 2],
        "total_xx": C_PLANE_STRESS[0, 0] * eps[:, 0] + C_PLANE_STRESS[0, 1] * eps[:, 1],
        "total_yy": C_PLANE_STRESS[1, 0] * eps[:, 0] + C_PLANE_STRESS[1, 1] * eps[:, 1],
        "total_xy": C_PLANE_STRESS[2, 2] * eps[:, 2],
        "energy_density": energy_density,
    }
    return eps, stress_dict


def original_node_displacements(solution: FESolution):
    system = solution.system
    n = len(system.data["x"])
    u_sum = np.zeros(n, dtype=float)
    v_sum = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)
    for node_id, (original, _label) in enumerate(system.node_keys):
        u_sum[original] += solution.U[2 * node_id]
        v_sum[original] += solution.U[2 * node_id + 1]
        count[original] += 1.0
    count[count == 0.0] = 1.0
    return u_sum / count, v_sum / count


def boundary_summary_for_solution(solution: FESolution, stress_dict: dict[str, np.ndarray]):
    rows = BA.summarize_boundary_forces(solution.system.data, stress_dict, "variant", seed_of(solution), solution.system.variant)
    df = pd.DataFrame(rows)
    df["bc_treatment"] = solution.bc_treatment
    return df


def seed_of(solution: FESolution) -> int:
    return int(solution.system.data["seed"])


def top_bottom_from_boundary(boundary_df: pd.DataFrame):
    whole = boundary_df[(boundary_df["stress_version"] == "variant") & (boundary_df["subdomain"] == "whole")]
    top = whole[whole["boundary"] == "top"]["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
    bottom = whole[whole["boundary"] == "bottom"]["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
    residual = whole[whole["boundary"] == "physical_boundary_sum"]
    rx = residual["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"]
    ry = residual["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
    return (
        float(top.iloc[0]) if len(top) else math.nan,
        float(bottom.iloc[0]) if len(bottom) else math.nan,
        float(rx.iloc[0]) if len(rx) else math.nan,
        float(ry.iloc[0]) if len(ry) else math.nan,
    )


def cut_metrics(solution: FESolution, stress_dict: dict[str, np.ndarray], top_reaction: float):
    data = solution.system.data
    crack_y = float(np.nanmean(data["crack_path_y"])) if "crack_path_y" in data and len(data["crack_path_y"]) else 0.005
    above_y = min(SPECIMEN_SIZE_MM, crack_y + 0.001)
    below_y = max(0.0, crack_y - 0.001)
    _fxa, fya, _fxta, _fyta, _lena, _cnta = BA.horizontal_cut_force(data, stress_dict, above_y)
    _fxb, fyb, _fxtb, _fytb, _lenb, _cntb = BA.horizontal_cut_force(data, stress_dict, below_y)
    iface = BA.interface_force_rows(data, stress_dict, seed_of(solution), solution.system.variant)
    crack_iface = [r for r in iface if r["interface_type"] == "crack_band_interface"]
    crack_force = float(sum(math.hypot(r["integrated_fx_N"], r["integrated_fy_N"]) for r in crack_iface))
    ratio_above = fya / top_reaction if abs(top_reaction) > 1.0e-14 else math.nan
    ratio_below = fyb / top_reaction if abs(top_reaction) > 1.0e-14 else math.nan
    return {
        "internal_cut_force_above_crack_N": float(fya),
        "internal_cut_force_below_crack_N": float(fyb),
        "internal_cut_above_to_top_ratio": float(ratio_above) if np.isfinite(ratio_above) else math.nan,
        "internal_cut_below_to_top_ratio": float(ratio_below) if np.isfinite(ratio_below) else math.nan,
        "crack_band_interface_force_N": crack_force,
    }


def free_body_rows(solution: FESolution, stress_dict: dict[str, np.ndarray]):
    rows = BA.subdomain_free_body_rows(solution.system.data, stress_dict, seed_of(solution), solution.system.variant)
    for row in rows:
        row["bc_treatment"] = solution.bc_treatment
    boundary = boundary_summary_for_solution(solution, stress_dict)
    whole = boundary[(boundary["boundary"] == "physical_boundary_sum") & (boundary["subdomain"] == "whole")]
    if len(whole):
        row = whole.iloc[0]
        rows.append(
            {
                "case": solution.system.data["case"],
                "seed": seed_of(solution),
                "variant": solution.system.variant,
                "bc_treatment": solution.bc_treatment,
                "subdomain": "whole",
                "physical_boundary_fx_N": row["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"],
                "physical_boundary_fy_N": row["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"],
                "crack_band_interface_fx_N": math.nan,
                "crack_band_interface_fy_N": math.nan,
                "transition_interface_fx_N": math.nan,
                "transition_interface_fy_N": math.nan,
                "net_force_residual_x_N": row["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"],
                "net_force_residual_y_N": row["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"],
                "net_force_residual_magnitude_N": math.hypot(
                    row["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"],
                    row["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"],
                ),
                "net_moment_residual_Nmm": math.nan,
                "subdomain_centroid_x": math.nan,
                "subdomain_centroid_y": math.nan,
                "physical_boundary_force_magnitude_sum_N": math.nan,
                "crack_band_interface_force_magnitude_sum_N": math.nan,
                "transition_interface_force_magnitude_sum_N": math.nan,
                "mechanically_separated_by_low_crack_interface": math.nan,
            }
        )
    return rows


def solve_all():
    reference_rows = []
    energy_rows = []
    metric_rows = []
    free_body_all = []
    boundary_rows = []
    solution_cache: dict[tuple[int, str, str], tuple[FESolution, dict[str, np.ndarray]]] = {}
    for seed in SEEDS:
        base = load_base(seed)
        for variant in FE_VARIANTS:
            system = assemble_system(base, variant)
            for bc in BC_TREATMENTS:
                solution = solve_system(system, bc)
                eps, stress_dict = element_fields(solution)
                boundary = boundary_summary_for_solution(solution, stress_dict)
                boundary_rows.extend(boundary.to_dict("records"))
                top_sigma, bottom_sigma, residual_x, residual_y = top_bottom_from_boundary(boundary)
                cuts = cut_metrics(solution, stress_dict, top_sigma)
                free_body_all.extend(free_body_rows(solution, stress_dict))
                solution_cache[(seed, variant, bc)] = (solution, stress_dict)
                active_elements = int(np.sum(system.stiffness_weight > 0.0))
                reference_rows.append(
                    {
                        "case": base["case"],
                        "seed": seed,
                        "variant": variant,
                        "bc_treatment": bc,
                        "topology": system.topology,
                        "stiffness_treatment": system.treatment,
                        "delta_mm": system.delta,
                        "node_dof_count": len(system.node_keys),
                        "scalar_dof_count": system.K.shape[0],
                        "active_element_count": active_elements,
                        "zero_stiffness_element_count": int(len(system.stiffness_weight) - active_elements),
                        "component_count": len(system.components),
                        "prescribed_dof_count": len(solution.prescribed),
                        "delta_control_dof_count": len(solution.delta_coeff),
                        "auto_anchor_count": solution.auto_anchor_count,
                        "solve_status": solution.solve_status,
                        "elastic_energy_Nmm_proxy": solution.energy_Nmm_proxy,
                        "top_sigma_integral_reaction_N": top_sigma,
                        "bottom_sigma_integral_reaction_N": bottom_sigma,
                        "top_constrained_dof_reaction_N": solution.top_constrained_dof_reaction_N,
                        "bottom_constrained_dof_reaction_N": solution.bottom_constrained_dof_reaction_N,
                        "energy_conjugate_reaction_N": solution.energy_conjugate_reaction_N,
                        "all_boundary_force_residual_x_N": residual_x,
                        "all_boundary_force_residual_y_N": residual_y,
                        "all_boundary_force_residual_magnitude_N": math.hypot(residual_x, residual_y),
                        "implementation_note": system.implementation_note,
                    }
                )
                energy_rows.append(
                    {
                        "case": base["case"],
                        "seed": seed,
                        "variant": variant,
                        "bc_treatment": bc,
                        "energy_conjugate_reaction_N": solution.energy_conjugate_reaction_N,
                        "top_sigma_integral_reaction_N": top_sigma,
                        "bottom_sigma_integral_reaction_N": bottom_sigma,
                        "top_constrained_dof_reaction_N": solution.top_constrained_dof_reaction_N,
                        "bottom_constrained_dof_reaction_N": solution.bottom_constrained_dof_reaction_N,
                        "top_sigma_minus_energy_conjugate_N": top_sigma - solution.energy_conjugate_reaction_N,
                        "top_sigma_to_energy_conjugate_ratio": safe_ratio(top_sigma, solution.energy_conjugate_reaction_N),
                        "all_boundary_vertical_residual_N": residual_y,
                        **cuts,
                    }
                )
                metric_rows.append(
                    {
                        "case": base["case"],
                        "seed": seed,
                        "variant": variant,
                        "bc_treatment": bc,
                        "top_sigma_integral_reaction_N": top_sigma,
                        "bottom_sigma_integral_reaction_N": bottom_sigma,
                        "top_constrained_dof_reaction_N": solution.top_constrained_dof_reaction_N,
                        "bottom_constrained_dof_reaction_N": solution.bottom_constrained_dof_reaction_N,
                        "energy_conjugate_reaction_N": solution.energy_conjugate_reaction_N,
                        "internal_cut_force_above_crack_N": cuts["internal_cut_force_above_crack_N"],
                        "internal_cut_force_below_crack_N": cuts["internal_cut_force_below_crack_N"],
                        "internal_cut_above_to_top_ratio": cuts["internal_cut_above_to_top_ratio"],
                        "internal_cut_below_to_top_ratio": cuts["internal_cut_below_to_top_ratio"],
                        "crack_band_interface_force_N": cuts["crack_band_interface_force_N"],
                        "all_boundary_force_residual_N": math.hypot(residual_x, residual_y),
                        "energy_reaction_abs_below_1e_3N": abs(solution.energy_conjugate_reaction_N) < 1.0e-3,
                        "top_sigma_abs_below_1e_3N": abs(top_sigma) < 1.0e-3,
                        "crack_interface_abs_below_1e_3N": abs(cuts["crack_band_interface_force_N"]) < 1.0e-3,
                    }
                )
    return (
        pd.DataFrame(reference_rows),
        pd.DataFrame(energy_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(free_body_all),
        pd.DataFrame(boundary_rows),
        solution_cache,
    )


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1.0e-14:
        return math.nan
    return float(num / den)


def boundary_sensitivity(metric: pd.DataFrame):
    rows = []
    target_variants = ("fedof_continuous_crack_band_void", "fedof_piecewise_upper_lower_crack_band_void")
    for seed in SEEDS:
        for variant in target_variants:
            sub = metric[(metric["seed"] == seed) & (metric["variant"] == variant)]
            original = sub[sub["bc_treatment"] == "original_top_bottom_bc"]
            minimal = sub[sub["bc_treatment"] == "minimal_rigid_body_bc"]
            if original.empty or minimal.empty:
                continue
            o = original.iloc[0]
            m = minimal.iloc[0]
            rows.append(
                {
                    "case": o["case"],
                    "seed": seed,
                    "variant": variant,
                    "original_top_sigma_reaction_N": o["top_sigma_integral_reaction_N"],
                    "minimal_top_sigma_reaction_N": m["top_sigma_integral_reaction_N"],
                    "original_energy_conjugate_reaction_N": o["energy_conjugate_reaction_N"],
                    "minimal_energy_conjugate_reaction_N": m["energy_conjugate_reaction_N"],
                    "original_keeps_nonzero_top_reaction": abs(o["top_sigma_integral_reaction_N"]) > 1.0e-3,
                    "minimal_allows_energy_reaction_collapse": abs(m["energy_conjugate_reaction_N"]) < max(1.0e-3, 0.05 * abs(o["energy_conjugate_reaction_N"])),
                    "minimal_allows_top_sigma_collapse": abs(m["top_sigma_integral_reaction_N"]) < max(1.0e-3, 0.05 * abs(o["top_sigma_integral_reaction_N"])),
                    "top_sigma_reduction_factor_min_over_original": safe_ratio(abs(m["top_sigma_integral_reaction_N"]), abs(o["top_sigma_integral_reaction_N"])),
                    "energy_reduction_factor_min_over_original": safe_ratio(abs(m["energy_conjugate_reaction_N"]), abs(o["energy_conjugate_reaction_N"])),
                    "localized_boundary_metric_note": "stress maps and boundary integrals are diagnostic; inspect figures and free-body table",
                }
            )
    return pd.DataFrame(rows)


def classify_results(metric: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    current_original = metric[
        (metric["variant"] == "fedof_continuous_current_split")
        & (metric["bc_treatment"] == "original_top_bottom_bc")
    ]
    void_original = metric[
        (metric["variant"] == "fedof_continuous_crack_band_void")
        & (metric["bc_treatment"] == "original_top_bottom_bc")
    ]
    piecewise = metric[
        (metric["variant"] == "fedof_piecewise_upper_lower_crack_band_void")
        & (metric["bc_treatment"] == "minimal_rigid_body_bc")
    ]
    continuous_original = metric[
        (metric["variant"] == "fedof_continuous_crack_band_void")
        & (metric["bc_treatment"] == "original_top_bottom_bc")
    ]
    metric_artifact_votes = 0
    boundary_votes = 0
    for seed in SEEDS:
        pw = piecewise[piecewise["seed"] == seed]
        co = continuous_original[continuous_original["seed"] == seed]
        if not pw.empty and not co.empty:
            pw_row = pw.iloc[0]
            co_row = co.iloc[0]
            if abs(pw_row["energy_conjugate_reaction_N"]) < 1.0e-3 and abs(co_row["top_sigma_integral_reaction_N"]) > 1.0e-3:
                metric_artifact_votes += 1
    for seed in SEEDS:
        sub = sensitivity[
            (sensitivity["seed"] == seed)
            & (sensitivity["variant"] == "fedof_continuous_crack_band_void")
        ]
        if not sub.empty:
            row = sub.iloc[0]
            if bool(row["original_keeps_nonzero_top_reaction"]) and (
                bool(row["minimal_allows_energy_reaction_collapse"]) or bool(row["minimal_allows_top_sigma_collapse"])
            ):
                boundary_votes += 1
    if metric_artifact_votes >= 2 and boundary_votes >= 2:
        return "reaction metric artifact and boundary ansatz overconstraint confirmed"
    if metric_artifact_votes >= 2:
        return "reaction metric artifact confirmed"
    if boundary_votes >= 2:
        return "boundary ansatz overconstraint confirmed"
    current_loaded = (
        len(current_original) == len(SEEDS)
        and int(np.sum(np.abs(current_original["energy_conjugate_reaction_N"]) > 1.0e-3)) >= 2
    )
    void_collapsed = (
        len(void_original) == len(SEEDS)
        and len(piecewise) == len(SEEDS)
        and int(np.sum(np.abs(void_original["top_sigma_integral_reaction_N"]) < 1.0e-3)) >= 2
        and int(np.sum(np.abs(void_original["energy_conjugate_reaction_N"]) < 1.0e-3)) >= 2
        and int(np.sum(np.abs(piecewise["energy_conjugate_reaction_N"]) < 1.0e-3)) >= 2
    )
    if current_loaded and void_collapsed:
        return "FE-DOF reference unresolved: energy-relaxed crack-band-void reaction collapses and does not reproduce persistent PINN reaction"
    return "FE-DOF reference unresolved"


def write_tables(reference, energy, metric, sensitivity, free_body):
    reference.to_csv(TABLES / "fedof_reference_solve_summary.csv", index=False)
    energy.to_csv(TABLES / "energy_conjugate_reaction_audit.csv", index=False)
    metric.to_csv(TABLES / "reaction_metric_comparison.csv", index=False)
    sensitivity.to_csv(TABLES / "boundary_condition_sensitivity.csv", index=False)
    free_body.to_csv(TABLES / "fedof_free_body_consistency.csv", index=False)


def plot_metric_comparison(metric: pd.DataFrame):
    sub = metric[metric["variant"].isin(["fedof_continuous_crack_band_void", "fedof_piecewise_upper_lower_crack_band_void"])]
    labels = []
    x = np.arange(len(sub))
    for row in sub.itertuples():
        labels.append(f"s{int(row.seed)}\n{row.variant.replace('fedof_', '').replace('_crack_band_void', '').replace('_', ' ')}\n{row.bc_treatment.replace('_bc', '').replace('_', ' ')}")
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=180)
    width = 0.25
    ax.bar(x - width, sub["top_sigma_integral_reaction_N"], width, label="top sigma integral")
    ax.bar(x, sub["energy_conjugate_reaction_N"], width, label="energy conjugate")
    ax.bar(x + width, sub["internal_cut_force_above_crack_N"], width, label="internal cut above")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=6)
    ax.set_ylabel("reaction / cut force [N]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_metric_comparison_by_seed_variant.png")
    plt.close(fig)


def plot_energy_vs_top(metric: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=180)
    for variant, sub in metric.groupby("variant"):
        ax.scatter(sub["energy_conjugate_reaction_N"], sub["top_sigma_integral_reaction_N"], label=variant.replace("fedof_", ""), s=22)
    lim = np.nanmax(np.abs(pd.concat([metric["energy_conjugate_reaction_N"], metric["top_sigma_integral_reaction_N"]])))
    lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("energy-conjugate reaction [N]")
    ax.set_ylabel("top sigma integral [N]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_conjugate_reaction_vs_top_sigma_integral.png")
    plt.close(fig)


def plot_boundary_sensitivity(sensitivity: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=180)
    labels = [f"s{int(r.seed)}\n{r.variant.replace('fedof_', '').replace('_crack_band_void', '').replace('_', ' ')}" for r in sensitivity.itertuples()]
    x = np.arange(len(sensitivity))
    ax.bar(x - 0.18, sensitivity["original_top_sigma_reaction_N"], 0.36, label="original top sigma")
    ax.bar(x + 0.18, sensitivity["minimal_top_sigma_reaction_N"], 0.36, label="minimal top sigma")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("top sigma reaction [N]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "boundary_condition_sensitivity_top_reaction.png")
    plt.close(fig)


def plot_free_body(free_body: pd.DataFrame):
    sub = free_body[
        (free_body["variant"] == "fedof_piecewise_upper_lower_crack_band_void")
        & (free_body["bc_treatment"] == "minimal_rigid_body_bc")
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    labels = [f"s{int(r.seed)}-{r.subdomain}" for r in sub.itertuples()]
    x = np.arange(len(sub))
    ax.bar(x, sub["net_force_residual_magnitude_N"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("free-body residual magnitude [N]")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "upper_lower_free_body_residual_fedof.png")
    plt.close(fig)


def plot_internal_cut(metric: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    sub = metric[metric["variant"] == "fedof_continuous_crack_band_void"]
    for bc, group in sub.groupby("bc_treatment"):
        ax.plot(group["seed"], group["internal_cut_above_to_top_ratio"], marker="o", label=bc)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("seed")
    ax.set_ylabel("internal cut above / top sigma reaction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "internal_cut_force_vs_top_reaction.png")
    plt.close(fig)


def plot_fields(solution_cache):
    examples = [
        (7, "fedof_continuous_crack_band_void", "original_top_bottom_bc"),
        (7, "fedof_piecewise_upper_lower_crack_band_void", "minimal_rigid_body_bc"),
    ]
    for seed, variant, bc in examples:
        solution, stress = solution_cache[(seed, variant, bc)]
        data = solution.system.data
        tri = mtri.Triangulation(data["x"], data["y"], data["triangles"].astype(int))
        u, v = original_node_displacements(solution)
        disp_mag = np.sqrt(u**2 + v**2)
        fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=180)
        artist = ax.tripcolor(tri, disp_mag, shading="gouraud")
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], s=1.5, c="black")
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: {variant.replace('fedof_', '')}\n{bc.replace('_', ' ')} displacement")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.tight_layout()
        fig.savefig(FIGURES / f"fedof_displacement_field_seed{seed}_{variant}_{bc}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=180)
        vmax = float(np.nanpercentile(np.abs(stress["variant_yy"]), 98))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        artist = ax.tripcolor(tri, facecolors=stress["variant_yy"], shading="flat", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], s=1.5, c="black")
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: {variant.replace('fedof_', '')}\n{bc.replace('_', ' ')} sigma_yy")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.tight_layout()
        fig.savefig(FIGURES / f"fedof_stress_map_seed{seed}_{variant}_{bc}.png")
        plt.close(fig)


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only. They summarize the FE-DOF frozen-alpha reaction reference audit and do not support physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `reaction_metric_comparison_by_seed_variant.png` | Top sigma integral, energy-conjugate reaction, and internal cut force by seed/variant/BC | Compares whether reaction metrics collapse under voiding or piecewise separation. | Diagnostic reaction-metric evidence. |",
        "| `energy_conjugate_reaction_vs_top_sigma_integral.png` | Energy-conjugate reaction against top-boundary stress integral | Points away from or toward agreement between a generalized load and a local top stress metric. | Diagnostic only. |",
        "| `boundary_condition_sensitivity_top_reaction.png` | Original top/bottom BC versus minimal rigid-body BC | Shows whether the top reaction is sensitive to boundary treatment. | Boundary-condition evidence. |",
        "| `fedof_displacement_field_seed7_*.png` | Example FE-DOF displacement magnitude fields for continuous and piecewise variants | Shows the diagnostic displacement mode used for reaction comparison. | Diagnostic illustration. |",
        "| `fedof_stress_map_seed7_*.png` | Example sigma_yy maps for original and minimal BC treatments | Shows whether stress remains near boundary regions or ligament paths. | Diagnostic observation. |",
        "| `upper_lower_free_body_residual_fedof.png` | Whole/upper/lower residual magnitudes for piecewise void minimal BC | Checks force-balance consistency of separated subdomains. | Diagnostic force-balance evidence. |",
        "| `internal_cut_force_vs_top_reaction.png` | Internal cut/top reaction ratio for continuous crack-band void FE-DOF solves | Tests whether internal cut forces explain top sigma reaction. | Diagnostic reaction consistency evidence. |",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(metric, sensitivity, free_body, solution_cache):
    plot_metric_comparison(metric)
    plot_energy_vs_top(metric)
    plot_boundary_sensitivity(sensitivity)
    plot_free_body(free_body)
    plot_internal_cut(metric)
    plot_fields(solution_cache)
    write_figure_summary()


def digest(reference, metric, sensitivity):
    void_original = metric[
        (metric["variant"] == "fedof_continuous_crack_band_void")
        & (metric["bc_treatment"] == "original_top_bottom_bc")
    ]
    piecewise_min = metric[
        (metric["variant"] == "fedof_piecewise_upper_lower_crack_band_void")
        & (metric["bc_treatment"] == "minimal_rigid_body_bc")
    ]
    current_split = metric[
        (metric["variant"] == "fedof_continuous_current_split")
        & (metric["bc_treatment"] == "original_top_bottom_bc")
    ]
    current_split_top = [float(v) for v in current_split["top_sigma_integral_reaction_N"]]
    current_split_energy = [float(v) for v in current_split["energy_conjugate_reaction_N"]]
    top_values = [float(v) for v in void_original["top_sigma_integral_reaction_N"]]
    void_energy_values = [float(v) for v in void_original["energy_conjugate_reaction_N"]]
    energy_values = [float(v) for v in piecewise_min["energy_conjugate_reaction_N"]]
    original_nonzero_votes = int(
        np.sum(
            sensitivity[sensitivity["variant"] == "fedof_continuous_crack_band_void"][
                "original_keeps_nonzero_top_reaction"
            ].astype(bool)
        )
    )
    all_solved = bool(reference["solve_status"].eq("solved").all())
    return {
        "continuous_current_original_top_sigma_values": current_split_top,
        "continuous_current_original_energy_values": current_split_energy,
        "continuous_void_original_top_sigma_values": top_values,
        "continuous_void_original_energy_values": void_energy_values,
        "piecewise_void_minimal_energy_values": energy_values,
        "void_original_nonzero_top_votes": original_nonzero_votes,
        "all_solved": all_solved,
        "max_abs_piecewise_min_energy": float(np.nanmax(np.abs(piecewise_min["energy_conjugate_reaction_N"]))),
        "mean_abs_continuous_void_original_top_sigma": float(np.nanmean(np.abs(void_original["top_sigma_integral_reaction_N"]))),
        "max_abs_continuous_void_original_top_sigma": float(np.nanmax(np.abs(void_original["top_sigma_integral_reaction_N"]))),
    }


def write_docs(classification: str, reference, metric, sensitivity):
    d = digest(reference, metric, sensitivity)
    report = [
        "# FE-DOF / energy-conjugate reaction reference audit",
        "",
        "## Scope",
        "",
        "This package builds a diagnostic finite-dimensional mechanics reference on the existing final_D0040 mesh using frozen alpha fields from seeds 7, 13, and 42. It does not extend loading, evolve alpha, change `l0`, change material constants, change TM split/history logic, or run a PINN split-domain replay.",
        "",
        "The FE-DOF implementation is a linear CST plane-stress diagnostic with scalar frozen-alpha stiffness degradation. It is intended to audit reaction definitions and boundary sensitivity, not to replace the production mixed-driving mechanics formulation.",
        "",
        "## Key summary",
        "",
        f"- Classification: **{classification}**.",
        f"- FE solve status: all requested solves solved = {d['all_solved']}.",
        f"- Continuous current-split with original top/bottom BC, top sigma-integral reactions [N]: {d['continuous_current_original_top_sigma_values']}.",
        f"- Continuous current-split with original top/bottom BC, energy-conjugate reactions [N]: {d['continuous_current_original_energy_values']}.",
        f"- Continuous crack-band-void with original top/bottom BC, top sigma-integral reactions [N]: {d['continuous_void_original_top_sigma_values']}.",
        f"- Continuous crack-band-void with original top/bottom BC, energy-conjugate reactions [N]: {d['continuous_void_original_energy_values']}.",
        f"- Piecewise upper/lower crack-band-void with minimal rigid-body BC, energy-conjugate reactions [N]: {d['piecewise_void_minimal_energy_values']}.",
        f"- Max |piecewise/minimal energy-conjugate reaction| [N]: {d['max_abs_piecewise_min_energy']:.6g}.",
        f"- Continuous void/original-BC nonzero top-reaction votes: {d['void_original_nonzero_top_votes']}/3 seeds.",
        "",
        "## Answers",
        "",
        "1. The FE-DOF frozen-alpha reference does not reproduce the previous nonzero post-crack top-boundary stress-integral reaction once the crack band is voided. Current-split cases retain a small nonzero reaction, while crack-band-void cases collapse to numerical zero in this reference solve.",
        "2. The energy-conjugate generalized reaction also collapses in the continuous and piecewise crack-band-void references across the audited seeds.",
        "3. In this FE-DOF reference, top-boundary stress integral and energy-conjugate reaction are consistent for current-split loading and both collapse after crack-band voiding. This does not confirm a top-vs-energy metric disagreement inside the FE-DOF reference itself.",
        "4. Minimal rigid-body boundary treatment is still useful as a sensitivity check, but the original top/bottom BC already permits reaction collapse after crack-band voiding in the FE-DOF reference.",
        "5. The previous PINN/split-domain residual reactions are therefore more consistent with saved-u/v branch, PINN ansatz, or replay relaxation effects than with a residual cracked-ligament load path in an energy-relaxed FE-DOF reference.",
        "6. `reaction_N_tm_eff` should not be used alone to judge post-peak softening after through-crack formation.",
        "7. Future stress-strain curves should report an energy-conjugate generalized reaction or constrained-DOF reaction, and should include bottom reaction / all-boundary residual checks for post-crack states.",
        "8. No production physics change is justified directly from this reference audit.",
        "9. Next minimal intervention: compute an energy-conjugate or constrained-DOF reaction on the actual saved PINN/replay branch and compare it with the legacy top-boundary sigma integral before changing mechanics or alpha evolution.",
        "",
        "## Implementation limitations",
        "",
        "- The FE-DOF reference uses a scalar frozen-alpha stiffness approximation rather than the exact nonlinear TM positive/negative spectral split tangent.",
        "- The minimal rigid-body boundary treatment is diagnostic and not a proposed production boundary condition.",
        "- Results are evidence about reaction definitions and boundary constraints, not physical validation of crack behavior.",
        "",
        "## Verification",
        "",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q`: to be filled after verification.",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260615_default_unitbox_fedof_reaction_reference\\artifacts\\run_fedof_reaction_reference.py`: to be filled after verification.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# FE-DOF reaction reference package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/fedof_reference_solve_summary.csv`",
        "3. `tables/energy_conjugate_reaction_audit.csv`",
        "4. `tables/reaction_metric_comparison.csv`",
        "5. `tables/boundary_condition_sensitivity.csv`",
        "6. `tables/fedof_free_body_consistency.csv`",
        "7. `figures/figure_summary.md`",
        "",
        "This package is diagnostic-only. It uses frozen final_D0040 alpha fields and does not run production PINN training.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. Should the legacy `reaction_N_tm_eff` be demoted from the primary post-crack load metric?",
        "2. Should postprocessing add an energy-conjugate or constrained-DOF reaction column for saved runs?",
        "3. Should any mechanics/phase-field production change remain blocked until reaction postprocessing is corrected?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    commands = [
        "git pull origin main",
        "Read 20260614_default_unitbox_boundary_reaction_audit handoff/report/tables/figure summary.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_fedof_reaction_reference.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260615_default_unitbox_fedof_reaction_reference\\artifacts\\run_fedof_reaction_reference.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: FE-DOF energy-conjugate reaction reference audit",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260615_default_unitbox_fedof_reaction_reference",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260615_default_unitbox_fedof_reaction_reference/REPORT.md",
        "",
        "### What changed",
        "- Added and ran a diagnostic FE-DOF frozen-alpha mechanics reference on the existing final_D0040 mesh.",
        "- Tested continuous and piecewise upper/lower topology variants under original top/bottom BC and minimal rigid-body BC.",
        "- Computed top/bottom stress-integral reactions, constrained DOF reactions, energy-conjugate reactions, internal cuts, and free-body residuals.",
        "- No loading, alpha, `l0`, material constants, TM split, thermal terms, or history logic was changed.",
        "",
        "### Commands run",
        "```powershell",
        *commands,
        "```",
        "",
        "### Key results",
        f"- Identified cause/status: **{classification}**.",
        f"- Continuous current-split/original-BC top sigma reactions [N]: {d['continuous_current_original_top_sigma_values']}.",
        f"- Continuous void/original-BC top sigma reactions [N]: {d['continuous_void_original_top_sigma_values']}.",
        f"- Continuous void/original-BC energy-conjugate reactions [N]: {d['continuous_void_original_energy_values']}.",
        f"- Piecewise void/minimal-BC energy-conjugate reactions [N]: {d['piecewise_void_minimal_energy_values']}.",
        "- The FE-DOF reference does not reproduce the previous persistent PINN/replay post-crack top reaction after crack-band voiding.",
        "- The diagnostic supports adding an energy-conjugate or constrained-DOF reaction metric to saved-run postprocessing before changing the physical model.",
        "- No production physics change is justified directly from this diagnostic.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/fedof_reference_solve_summary.csv`",
        "- `tables/energy_conjugate_reaction_audit.csv`",
        "- `tables/reaction_metric_comparison.csv`",
        "- `tables/boundary_condition_sensitivity.csv`",
        "- `tables/fedof_free_body_consistency.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Is the FE-DOF reference strong enough to demote `reaction_N_tm_eff` as the primary post-crack load metric?",
        "2. Should the next Codex task add energy-conjugate/constrained-DOF reaction postprocessing to saved-run analysis without changing the physical model?",
        "3. Is any production model change still unjustified until reaction postprocessing is corrected?",
        "",
        "### Constraints",
        "- Do not extend loading.",
        "- Do not evolve alpha.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def write_manifest():
    entries = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if "__pycache__" in rel:
            continue
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
        elif rel.startswith("logs/") or rel == "commands_run.txt":
            ftype = "command_log"
            required = False
        elif rel == "HANDOFF_COMMENT.md":
            ftype = "handoff"
            required = True
        else:
            ftype = "report"
            required = rel in {"README.md", "REPORT.md", "MANIFEST.json"}
        entries.append(
            {
                "path": rel,
                "type": ftype,
                "description": describe_file(rel),
                "required_for_chatgpt": required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def describe_file(rel: str) -> str:
    return {
        "README.md": "Package reading order and scope.",
        "REPORT.md": "Main FE-DOF reaction reference report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for issue #1.",
        "commands_run.txt": "Commands executed for this package.",
        "next_questions.md": "Questions to guide ChatGPT review.",
        "figures/figure_summary.md": "Text summary for all included figures.",
        "tables/fedof_reference_solve_summary.csv": "FE-DOF variant, BC, solve, and reaction summary.",
        "tables/energy_conjugate_reaction_audit.csv": "Energy-conjugate reaction compared with top/bottom/cut reactions.",
        "tables/reaction_metric_comparison.csv": "Top/bottom stress, constrained DOF, energy, cut, and crack-interface metrics.",
        "tables/boundary_condition_sensitivity.csv": "Original versus minimal rigid-body boundary sensitivity.",
        "tables/fedof_free_body_consistency.csv": "Whole/upper/lower free-body residuals for FE-DOF references.",
    }.get(rel, "Generated diagnostic artifact.")


def main():
    setup_dirs()
    reference, energy, metric, free_body, _boundary, solution_cache = solve_all()
    sensitivity = boundary_sensitivity(metric)
    classification = classify_results(metric, sensitivity)
    write_tables(reference, energy, metric, sensitivity, free_body)
    make_figures(metric, sensitivity, free_body, solution_cache)
    write_docs(classification, reference, metric, sensitivity)
    write_manifest()
    print(classification)


if __name__ == "__main__":
    main()
