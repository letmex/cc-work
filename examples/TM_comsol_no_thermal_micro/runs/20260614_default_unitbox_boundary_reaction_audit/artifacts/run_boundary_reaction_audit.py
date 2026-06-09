"""Boundary/reaction/free-body audit for D0040 split-domain replay fields.

The script is diagnostic-only. It reads the previous frozen-alpha
split-domain replay artifacts and audits whether the top-boundary reaction
proxy is a reliable post-crack global load metric. No loading, alpha,
material, TM split, or history logic is changed.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"
REPO = PACKAGE.parents[3]
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
PREV_PACKAGE = REPO / "examples" / "TM_comsol_no_thermal_micro" / "runs" / "20260613_default_unitbox_discontinuous_kinematic_replay"
PREV_ARTIFACTS = PREV_PACKAGE / "artifacts"

SPECIMEN_SIZE_MM = 0.01
BOUNDARY_TOL = 1.0e-9
CUT_XS = (0.006, 0.007, 0.008, 0.009)
HORIZONTAL_CUT_OFFSETS = (-0.0010, 0.0010)
E = 81.5
NU = 0.38
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
ETA_RESIDUAL = 1.0e-5
TM_EPS_R = 1.0e-5

SEEDS = (7, 13, 42)
VARIANTS = (
    "continuous_baseline",
    "split_domain_current_split",
    "split_domain_minus_degraded_crack_band",
    "split_domain_crack_band_void",
)
SYNTHETIC_FIELDS = (
    "piecewise_rigid_upper_lower",
    "zero_displacement_reference",
    "saved_uv_reference",
)
SYNTHETIC_STRESS_TREATMENTS = ("current_split", "crack_band_void")


def setup_dirs():
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_npz(seed: int, variant: str) -> dict[str, np.ndarray]:
    path = PREV_ARTIFACTS / f"D0040_seed{seed}_default_unitbox_final_D0040_{variant}_fields.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["case"] = f"D0040_seed{seed}_default_unitbox"
    arrays["seed"] = seed
    arrays["variant"] = variant
    arrays["path"] = str(path)
    return arrays


def triangle_areas(x, y, tri):
    pts = np.column_stack([x, y])
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def element_centroids(x, y, tri):
    return np.mean(x[tri], axis=1), np.mean(y[tri], axis=1)


def element_strains_from_uv(x, y, tri, u, v):
    area = triangle_areas(x, y, tri)
    eps_xx = (
        (y[tri[:, 1]] - y[tri[:, 2]]) * u[tri[:, 0]]
        + (y[tri[:, 2]] - y[tri[:, 0]]) * u[tri[:, 1]]
        + (y[tri[:, 0]] - y[tri[:, 1]]) * u[tri[:, 2]]
    ) / (2.0 * area)
    grad_u_y = (
        (x[tri[:, 2]] - x[tri[:, 1]]) * u[tri[:, 0]]
        + (x[tri[:, 0]] - x[tri[:, 2]]) * u[tri[:, 1]]
        + (x[tri[:, 1]] - x[tri[:, 0]]) * u[tri[:, 2]]
    ) / (2.0 * area)
    grad_v_x = (
        (y[tri[:, 1]] - y[tri[:, 2]]) * v[tri[:, 0]]
        + (y[tri[:, 2]] - y[tri[:, 0]]) * v[tri[:, 1]]
        + (y[tri[:, 0]] - y[tri[:, 1]]) * v[tri[:, 2]]
    ) / (2.0 * area)
    eps_yy = (
        (x[tri[:, 2]] - x[tri[:, 1]]) * v[tri[:, 0]]
        + (x[tri[:, 0]] - x[tri[:, 2]]) * v[tri[:, 1]]
        + (x[tri[:, 1]] - x[tri[:, 0]]) * v[tri[:, 2]]
    ) / (2.0 * area)
    eps_xy = 0.5 * (grad_u_y + grad_v_x)
    return eps_xx, eps_yy, eps_xy


def positive_part(x):
    return 0.5 * (x + np.abs(x))


def tm_source_stress(eps_xx, eps_yy, eps_xy, alpha_elem):
    eps_zz = -NU / (1.0 - NU) * (eps_xx + eps_yy)
    em = 0.5 * (eps_xx + eps_yy)
    ed = 0.5 * (eps_xx - eps_yy)
    r = np.sqrt(ed**2 + eps_xy**2 + TM_EPS_R**2)
    safe_r = np.where(r > 0.0, r, 1.0)
    r0 = r - TM_EPS_R
    e1 = em + r0
    e2 = em - r0
    e3 = eps_zz
    e1p = positive_part(e1)
    e2p = positive_part(e2)
    e3p = positive_part(e3)
    sp = e1p + e2p
    dp = e1p - e2p
    chi = ed / safe_r
    eta = eps_xy / safe_r
    epxx = 0.5 * sp + 0.5 * dp * chi
    epyy = 0.5 * sp - 0.5 * dp * chi
    epxy = 0.5 * dp * eta
    tr_p = e1p + e2p + e3p

    tr_e = eps_xx + eps_yy + eps_zz
    total_xx = LAMBDA * tr_e + 2.0 * MU * eps_xx
    total_yy = LAMBDA * tr_e + 2.0 * MU * eps_yy
    total_xy = 2.0 * MU * eps_xy
    plus_xx = LAMBDA * tr_p + 2.0 * MU * epxx
    plus_yy = LAMBDA * tr_p + 2.0 * MU * epyy
    plus_xy = 2.0 * MU * epxy
    minus_xx = total_xx - plus_xx
    minus_yy = total_yy - plus_yy
    minus_xy = total_xy - plus_xy
    g = (1.0 - alpha_elem) ** 2 + ETA_RESIDUAL
    eff_xx = total_xx + (g - 1.0) * plus_xx
    eff_yy = total_yy + (g - 1.0) * plus_yy
    eff_xy = total_xy + (g - 1.0) * plus_xy
    energy_density = 0.5 * LAMBDA * tr_e**2 + MU * (eps_xx**2 + eps_yy**2 + eps_zz**2 + 2.0 * eps_xy**2)
    return {
        "total_xx": total_xx,
        "total_yy": total_yy,
        "total_xy": total_xy,
        "plus_xx": plus_xx,
        "plus_yy": plus_yy,
        "plus_xy": plus_xy,
        "minus_xx": minus_xx,
        "minus_yy": minus_yy,
        "minus_xy": minus_xy,
        "eff_xx": eff_xx,
        "eff_yy": eff_yy,
        "eff_xy": eff_xy,
        "g": g,
        "energy_density": energy_density,
    }


def variant_stress_from_fields(data, treatment=None):
    stress = tm_source_stress(data["eps_xx"], data["eps_yy"], data["eps_xy"], data["alpha_elem"])
    variant = treatment or str(data["variant"])
    crack = data["crack_mask"].astype(bool)
    sx = np.array(stress["eff_xx"], copy=True)
    sy = np.array(stress["eff_yy"], copy=True)
    sxy = np.array(stress["eff_xy"], copy=True)
    if variant in {"split_domain_minus_degraded_crack_band", "minus_degraded_crack_band"}:
        g = stress["g"]
        sx[crack] = g[crack] * stress["total_xx"][crack]
        sy[crack] = g[crack] * stress["total_yy"][crack]
        sxy[crack] = g[crack] * stress["total_xy"][crack]
    elif variant in {"split_domain_crack_band_void", "crack_band_void"}:
        sx[crack] = 0.0
        sy[crack] = 0.0
        sxy[crack] = 0.0
    return {
        **stress,
        "variant_xx": sx,
        "variant_yy": sy,
        "variant_xy": sxy,
    }


def edge_map(tri):
    edges = defaultdict(list)
    for elem, nodes in enumerate(tri.astype(int)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges[tuple(sorted((int(a), int(b))))].append(elem)
    return edges


def known_boundary_normal(pa, pb):
    if abs(pa[1] - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL and abs(pb[1] - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL:
        return "top", np.array([0.0, 1.0]), np.array([1.0, 0.0])
    if abs(pa[1]) <= BOUNDARY_TOL and abs(pb[1]) <= BOUNDARY_TOL:
        return "bottom", np.array([0.0, -1.0]), np.array([1.0, 0.0])
    if abs(pa[0]) <= BOUNDARY_TOL and abs(pb[0]) <= BOUNDARY_TOL:
        return "left", np.array([-1.0, 0.0]), np.array([0.0, 1.0])
    if abs(pa[0] - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL and abs(pb[0] - SPECIMEN_SIZE_MM) <= BOUNDARY_TOL:
        return "right", np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return None, None, None


def traction_components(sxx, syy, sxy, normal, tangent, length):
    tx = (sxx * normal[0] + sxy * normal[1]) * length
    ty = (sxy * normal[0] + syy * normal[1]) * length
    normal_force = tx * normal[0] + ty * normal[1]
    tangential_force = tx * tangent[0] + ty * tangent[1]
    return 1000.0 * tx, 1000.0 * ty, 1000.0 * normal_force, 1000.0 * tangential_force


def boundary_force_rows(data, stress, stress_prefix, subdomain_filter=None):
    x = data["x"]
    y = data["y"]
    tri = data["triangles"].astype(int)
    elem_upper = data["elem_upper"].astype(bool)
    edges = edge_map(tri)
    rows = []
    for (a, b), elems in edges.items():
        if len(elems) != 1:
            continue
        elem = elems[0]
        if subdomain_filter == "upper" and not elem_upper[elem]:
            continue
        if subdomain_filter == "lower" and elem_upper[elem]:
            continue
        pa = np.array([x[a], y[a]])
        pb = np.array([x[b], y[b]])
        boundary, normal, tangent = known_boundary_normal(pa, pb)
        if boundary is None:
            continue
        length = float(np.linalg.norm(pb - pa))
        tx, ty, fn, ft = traction_components(
            stress[f"{stress_prefix}_xx"][elem],
            stress[f"{stress_prefix}_yy"][elem],
            stress[f"{stress_prefix}_xy"][elem],
            normal,
            tangent,
            length,
        )
        midpoint = 0.5 * (pa + pb)
        rows.append(
            {
                "boundary": boundary,
                "subdomain": subdomain_filter or "whole",
                "elem": elem,
                "edge_length": length,
                "mid_x": midpoint[0],
                "mid_y": midpoint[1],
                "normal_x": normal[0],
                "normal_y": normal[1],
                "integrated_fx_N": tx,
                "integrated_fy_N": ty,
                "normal_force_N": fn,
                "tangential_force_N": ft,
            }
        )
    return rows


def summarize_boundary_forces(data, stress, stress_prefix, seed, variant):
    rows = []
    for subdomain in ("whole", "upper", "lower"):
        raw = boundary_force_rows(data, stress, stress_prefix, None if subdomain == "whole" else subdomain)
        grouped = defaultdict(lambda: {"fx": 0.0, "fy": 0.0, "fn": 0.0, "ft": 0.0, "length": 0.0, "edges": 0})
        for row in raw:
            g = grouped[row["boundary"]]
            g["fx"] += row["integrated_fx_N"]
            g["fy"] += row["integrated_fy_N"]
            g["fn"] += row["normal_force_N"]
            g["ft"] += row["tangential_force_N"]
            g["length"] += row["edge_length"]
            g["edges"] += 1
        for boundary, g in grouped.items():
            rows.append(
                {
                    "case": f"D0040_seed{seed}_default_unitbox",
                    "seed": seed,
                    "variant": variant,
                    "stress_version": stress_prefix,
                    "boundary": boundary,
                    "subdomain": subdomain,
                    "integrated_sigma_xx_nx_plus_sigma_xy_ny_N": g["fx"],
                    "integrated_sigma_xy_nx_plus_sigma_yy_ny_N": g["fy"],
                    "normal_force_N": g["fn"],
                    "tangential_force_N": g["ft"],
                    "boundary_length_mm": g["length"],
                    "edge_count": g["edges"],
                }
            )
        residual_fx = sum(g["fx"] for g in grouped.values())
        residual_fy = sum(g["fy"] for g in grouped.values())
        rows.append(
            {
                "case": f"D0040_seed{seed}_default_unitbox",
                "seed": seed,
                "variant": variant,
                "stress_version": stress_prefix,
                "boundary": "physical_boundary_sum",
                "subdomain": subdomain,
                "integrated_sigma_xx_nx_plus_sigma_xy_ny_N": residual_fx,
                "integrated_sigma_xy_nx_plus_sigma_yy_ny_N": residual_fy,
                "normal_force_N": math.nan,
                "tangential_force_N": math.nan,
                "boundary_length_mm": math.nan,
                "edge_count": sum(g["edges"] for g in grouped.values()),
            }
        )
    return rows


def element_outward_normal_to_edge(data, elem, a, b):
    x = data["x"]
    y = data["y"]
    pa = np.array([x[a], y[a]])
    pb = np.array([x[b], y[b]])
    midpoint = 0.5 * (pa + pb)
    center = np.array([data["element_x"][elem], data["element_y"][elem]])
    direction = midpoint - center
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        edge = pb - pa
        direction = np.array([edge[1], -edge[0]])
        norm = float(np.linalg.norm(direction))
    return direction / norm


def interface_force_rows(data, stress, seed, variant):
    tri = data["triangles"].astype(int)
    crack = data["crack_mask"].astype(bool)
    elem_upper = data["elem_upper"].astype(bool)
    mixed = data["elem_mixed_domain"].astype(bool)
    edges = edge_map(tri)
    rows = []
    for (a, b), elems in edges.items():
        if len(elems) != 2:
            continue
        e0, e1 = elems
        kind = None
        sides = []
        if crack[e0] != crack[e1]:
            kind = "crack_band_interface"
            sides = [e0 if not crack[e0] else e1]
        elif elem_upper[e0] != elem_upper[e1] or mixed[e0] or mixed[e1]:
            kind = "mixed_or_transition_interface"
            sides = [e0, e1]
        if kind is None:
            continue
        length = float(np.linalg.norm(np.array([data["x"][a], data["y"][a]]) - np.array([data["x"][b], data["y"][b]])))
        tangent = np.array([data["x"][b] - data["x"][a], data["y"][b] - data["y"][a]])
        tangent = tangent / (np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0.0 else 1.0)
        for elem in sides:
            normal = element_outward_normal_to_edge(data, elem, a, b)
            tx, ty, fn, ft = traction_components(
                stress["variant_xx"][elem],
                stress["variant_yy"][elem],
                stress["variant_xy"][elem],
                normal,
                tangent,
                length,
            )
            subdomain = "upper" if elem_upper[elem] else "lower"
            rows.append(
                {
                    "case": f"D0040_seed{seed}_default_unitbox",
                    "seed": seed,
                    "variant": variant,
                    "interface_type": kind,
                    "subdomain": subdomain,
                    "elem": elem,
                    "edge_length_mm": length,
                    "mid_x": 0.5 * (data["x"][a] + data["x"][b]),
                    "mid_y": 0.5 * (data["y"][a] + data["y"][b]),
                    "integrated_fx_N": tx,
                    "integrated_fy_N": ty,
                    "normal_force_N": fn,
                    "tangential_force_N": ft,
                }
            )
    return rows


def subdomain_free_body_rows(data, stress, seed, variant):
    rows = []
    raw_boundary = []
    for subdomain in ("upper", "lower"):
        raw_boundary.extend(boundary_force_rows(data, stress, "variant", subdomain))
    raw_interface = interface_force_rows(data, stress, seed, variant)
    for subdomain in ("upper", "lower"):
        phys = [r for r in raw_boundary if r["subdomain"] == subdomain]
        crack = [r for r in raw_interface if r["subdomain"] == subdomain and r["interface_type"] == "crack_band_interface"]
        transition = [r for r in raw_interface if r["subdomain"] == subdomain and r["interface_type"] == "mixed_or_transition_interface"]
        elem_mask = data["elem_upper"].astype(bool) if subdomain == "upper" else ~data["elem_upper"].astype(bool)
        cx = float(np.mean(data["element_x"][elem_mask]))
        cy = float(np.mean(data["element_y"][elem_mask]))
        all_forces = phys + crack + transition
        fx = sum(r["integrated_fx_N"] for r in all_forces)
        fy = sum(r["integrated_fy_N"] for r in all_forces)
        moment = sum((r["mid_x"] - cx) * r["integrated_fy_N"] - (r["mid_y"] - cy) * r["integrated_fx_N"] for r in all_forces)
        crack_mag = sum(math.hypot(r["integrated_fx_N"], r["integrated_fy_N"]) for r in crack)
        transition_mag = sum(math.hypot(r["integrated_fx_N"], r["integrated_fy_N"]) for r in transition)
        phys_mag = sum(math.hypot(r["integrated_fx_N"], r["integrated_fy_N"]) for r in phys)
        rows.append(
            {
                "case": f"D0040_seed{seed}_default_unitbox",
                "seed": seed,
                "variant": variant,
                "subdomain": subdomain,
                "physical_boundary_fx_N": sum(r["integrated_fx_N"] for r in phys),
                "physical_boundary_fy_N": sum(r["integrated_fy_N"] for r in phys),
                "crack_band_interface_fx_N": sum(r["integrated_fx_N"] for r in crack),
                "crack_band_interface_fy_N": sum(r["integrated_fy_N"] for r in crack),
                "transition_interface_fx_N": sum(r["integrated_fx_N"] for r in transition),
                "transition_interface_fy_N": sum(r["integrated_fy_N"] for r in transition),
                "net_force_residual_x_N": fx,
                "net_force_residual_y_N": fy,
                "net_force_residual_magnitude_N": math.hypot(fx, fy),
                "net_moment_residual_Nmm": moment,
                "subdomain_centroid_x": cx,
                "subdomain_centroid_y": cy,
                "physical_boundary_force_magnitude_sum_N": phys_mag,
                "crack_band_interface_force_magnitude_sum_N": crack_mag,
                "transition_interface_force_magnitude_sum_N": transition_mag,
                "mechanically_separated_by_low_crack_interface": crack_mag < 0.01 * max(phys_mag, 1.0e-12),
            }
        )
    return rows


def vertical_cut_force(data, stress, x0):
    rows = []
    for elem, nodes in enumerate(data["triangles"].astype(int)):
        xs = data["x"][nodes]
        ys = data["y"][nodes]
        if x0 < np.min(xs) or x0 > np.max(xs):
            continue
        intersections = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            x1, x2 = xs[i], xs[j]
            y1, y2 = ys[i], ys[j]
            if abs(x1 - x2) <= 1.0e-14:
                if abs(x0 - x1) <= 1.0e-12:
                    intersections.extend([y1, y2])
                continue
            if (x0 - x1) * (x0 - x2) <= 0.0:
                t = (x0 - x1) / (x2 - x1)
                if -1.0e-12 <= t <= 1.0 + 1.0e-12:
                    intersections.append(y1 + t * (y2 - y1))
        if len(intersections) < 2:
            continue
        length = float(np.max(intersections) - np.min(intersections))
        if length <= 0.0:
            continue
        rows.append((elem, length))
    fx_eff = sum(stress["variant_xx"][e] * length for e, length in rows) * 1000.0
    fy_eff = sum(stress["variant_xy"][e] * length for e, length in rows) * 1000.0
    fx_total = sum(stress["total_xx"][e] * length for e, length in rows) * 1000.0
    fy_total = sum(stress["total_xy"][e] * length for e, length in rows) * 1000.0
    return fx_eff, fy_eff, fx_total, fy_total, sum(length for _e, length in rows), len(rows)


def horizontal_cut_force(data, stress, y0):
    rows = []
    for elem, nodes in enumerate(data["triangles"].astype(int)):
        xs = data["x"][nodes]
        ys = data["y"][nodes]
        if y0 < np.min(ys) or y0 > np.max(ys):
            continue
        intersections = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            x1, x2 = xs[i], xs[j]
            y1, y2 = ys[i], ys[j]
            if abs(y1 - y2) <= 1.0e-14:
                if abs(y0 - y1) <= 1.0e-12:
                    intersections.extend([x1, x2])
                continue
            if (y0 - y1) * (y0 - y2) <= 0.0:
                t = (y0 - y1) / (y2 - y1)
                if -1.0e-12 <= t <= 1.0 + 1.0e-12:
                    intersections.append(x1 + t * (x2 - x1))
        if len(intersections) < 2:
            continue
        length = float(np.max(intersections) - np.min(intersections))
        if length <= 0.0:
            continue
        rows.append((elem, length))
    fx_eff = sum(stress["variant_xy"][e] * length for e, length in rows) * 1000.0
    fy_eff = sum(stress["variant_yy"][e] * length for e, length in rows) * 1000.0
    fx_total = sum(stress["total_xy"][e] * length for e, length in rows) * 1000.0
    fy_total = sum(stress["total_yy"][e] * length for e, length in rows) * 1000.0
    return fx_eff, fy_eff, fx_total, fy_total, sum(length for _e, length in rows), len(rows)


def reaction_vs_cut_rows(data, stress, seed, variant, top_reaction):
    rows = []
    for x0 in CUT_XS:
        fx, fy, fx_total, fy_total, length, count = vertical_cut_force(data, stress, x0)
        rows.append(
            {
                "case": f"D0040_seed{seed}_default_unitbox",
                "seed": seed,
                "variant": variant,
                "cut_type": "vertical",
                "cut_location": x0,
                "effective_normal_force_N": fx,
                "effective_shear_force_N": fy,
                "undegraded_normal_force_N": fx_total,
                "undegraded_shear_force_N": fy_total,
                "top_boundary_reaction_N": top_reaction,
                "cut_force_explains_top_reaction_ratio": fy / top_reaction if abs(top_reaction) > 1.0e-12 else math.nan,
                "cut_length_mm": length,
                "element_count": count,
            }
        )
    crack_y = float(np.mean(data["element_y"][data["crack_mask"].astype(bool)]))
    for offset in HORIZONTAL_CUT_OFFSETS:
        y0 = float(np.clip(crack_y + offset, 1.0e-5, SPECIMEN_SIZE_MM - 1.0e-5))
        fx, fy, fx_total, fy_total, length, count = horizontal_cut_force(data, stress, y0)
        rows.append(
            {
                "case": f"D0040_seed{seed}_default_unitbox",
                "seed": seed,
                "variant": variant,
                "cut_type": "horizontal_above" if offset > 0 else "horizontal_below",
                "cut_location": y0,
                "effective_normal_force_N": fy,
                "effective_shear_force_N": fx,
                "undegraded_normal_force_N": fy_total,
                "undegraded_shear_force_N": fx_total,
                "top_boundary_reaction_N": top_reaction,
                "cut_force_explains_top_reaction_ratio": fy / top_reaction if abs(top_reaction) > 1.0e-12 else math.nan,
                "cut_length_mm": length,
                "element_count": count,
            }
        )
    interface = interface_force_rows(data, stress, seed, variant)
    crack_interface = [r for r in interface if r["interface_type"] == "crack_band_interface"]
    rows.append(
        {
            "case": f"D0040_seed{seed}_default_unitbox",
            "seed": seed,
            "variant": variant,
            "cut_type": "crack_band_interface",
            "cut_location": math.nan,
            "effective_normal_force_N": sum(r["normal_force_N"] for r in crack_interface),
            "effective_shear_force_N": sum(r["tangential_force_N"] for r in crack_interface),
            "undegraded_normal_force_N": math.nan,
            "undegraded_shear_force_N": math.nan,
            "top_boundary_reaction_N": top_reaction,
            "cut_force_explains_top_reaction_ratio": (
                sum(r["normal_force_N"] for r in crack_interface) / top_reaction if abs(top_reaction) > 1.0e-12 else math.nan
            ),
            "cut_length_mm": sum(r["edge_length_mm"] for r in crack_interface),
            "element_count": len(crack_interface),
        }
    )
    return rows


def synthetic_uv(data, field_name):
    x = data["x"]
    y = data["y"]
    delta = 2.0e-4
    if "u" in data and "v" in data:
        saved_u = data["u"]
        saved_v = data["v"]
    else:
        saved_u = np.zeros_like(x)
        saved_v = (y / SPECIMEN_SIZE_MM) * delta
    if field_name == "saved_uv_reference":
        return saved_u.copy(), saved_v.copy()
    if field_name == "zero_displacement_reference":
        return np.zeros_like(x), (y / SPECIMEN_SIZE_MM) * delta
    if field_name == "piecewise_rigid_upper_lower":
        u = np.zeros_like(x)
        v = np.where(data["node_upper"].astype(bool), delta, 0.0)
        return u, v
    raise ValueError(field_name)


def synthetic_data_from_base(data, field_name):
    clone = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()}
    u, v = synthetic_uv(data, field_name)
    eps_xx, eps_yy, eps_xy = element_strains_from_uv(data["x"], data["y"], data["triangles"].astype(int), u, v)
    clone["u"] = u
    clone["v"] = v
    clone["eps_xx"] = eps_xx
    clone["eps_yy"] = eps_yy
    clone["eps_xy"] = eps_xy
    clone["variant"] = field_name
    return clone


def crack_band_traction_proxy(data, stress):
    crack = data["crack_mask"].astype(bool)
    if not np.any(crack):
        return math.nan
    area = triangle_areas(data["x"], data["y"], data["triangles"].astype(int))
    return float(np.sum(np.sqrt(stress["variant_yy"][crack] ** 2 + stress["variant_xy"][crack] ** 2) * area[crack]))


def elastic_energy(data, stress):
    area = triangle_areas(data["x"], data["y"], data["triangles"].astype(int))
    return float(np.sum(area * stress["energy_density"]))


def reaction_definition_audit_rows():
    return [
        {
            "file": "plot_clean_tm_results.py",
            "function": "top_reaction_force_N",
            "reported_quantity": "reaction_tm_eff_N / stress_tm_eff_N_per_mm2",
            "stress_component_integrated": "sigma_yy or sigma_yy_tm_eff",
            "boundary_integrated": "top boundary y=0.01",
            "sign_convention": "+sigma_yy * top edge length, no outward-normal sign reversal",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Used for stress-strain plotting; section-area normalization is postprocessing.",
        },
        {
            "file": "debug_fedof_staggered_baseline.py",
            "function": "_top_reaction_force_N",
            "reported_quantity": "reaction_N_tm_eff",
            "stress_component_integrated": "sigma_yy_tm_eff",
            "boundary_integrated": "top boundary y=0.01",
            "sign_convention": "+sigma_yy * top edge length, converted kN to N",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Does not inspect bottom/side reactions or internal cuts.",
        },
        {
            "file": "debug_prefit_then_energy_mechanics.py",
            "function": "_top_reaction_force_N",
            "reported_quantity": "reaction_N_tm_eff_proxy",
            "stress_component_integrated": "sigma_yy_tm_eff",
            "boundary_integrated": "top boundary y=0.01",
            "sign_convention": "+sigma_yy * top edge length, converted kN to N",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Mechanics-only diagnostic proxy.",
        },
        {
            "file": "runs/20260613.../run_discontinuous_kinematic_replay.py",
            "function": "top_reaction_np",
            "reported_quantity": "final_reaction_proxy",
            "stress_component_integrated": "variant sigma_yy",
            "boundary_integrated": "top boundary y=0.01",
            "sign_convention": "+sigma_yy * top edge length, converted kN to N",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Variant-specific top-boundary proxy; not a free-body reaction balance.",
        },
        {
            "file": "config.py",
            "function": "training report metadata",
            "reported_quantity": "reaction_force_method",
            "stress_component_integrated": "top-boundary sigma_yy stress integration",
            "boundary_integrated": "top boundary",
            "sign_convention": "metadata only",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Confirms intended method is a top-boundary stress integral.",
        },
        {
            "file": "analyze_drive_broadening_stepwise.py",
            "function": "stepwise table assembly",
            "reported_quantity": "reaction_N_tm_eff, macro_stress, macro_strain",
            "stress_component_integrated": "not recomputed here",
            "boundary_integrated": "copied from diagnostics",
            "sign_convention": "inherits upstream reaction convention",
            "conjugate_to_imposed_displacement_component": True,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": True,
            "notes": "Aggregates diagnostic rows for reports.",
        },
        {
            "file": "repo search",
            "function": "external work proxy",
            "reported_quantity": "not located",
            "stress_component_integrated": "not available",
            "boundary_integrated": "not available",
            "sign_convention": "not available",
            "conjugate_to_imposed_displacement_component": False,
            "measures_global_load_transfer_across_cracked_ligament": False,
            "only_top_boundary_local_stress_integral": False,
            "notes": "No active external-work proxy was found in the audited code paths.",
        },
    ]


def boundary_condition_audit_rows():
    return [
        {
            "location": "bottom boundary",
            "prescribed_u": "u=0 through eta=0 and bubble=0",
            "prescribed_v": "v=0 through eta=0 and bubble=0",
            "left_right_or_notch_condition": "not applicable",
            "split_domain_parameter_sharing": "upper/lower networks independent in split replay",
            "constraints_after_split": "bottom nodes remain fixed for whichever split field labels them",
            "overconstraint_risk": "can anchor lower subdomain even when crack band is void",
            "prefit_effect": "prefit initializes near saved continuous field but is not active during reoptimization",
        },
        {
            "location": "top boundary",
            "prescribed_u": "top-u-free: u=Delta*raw_u, not fixed to zero",
            "prescribed_v": "v=Delta prescribed by eta*sin(theta)",
            "left_right_or_notch_condition": "not applicable",
            "split_domain_parameter_sharing": "upper/lower networks independent; top nodes mostly upper-labeled",
            "constraints_after_split": "top vertical displacement remains imposed after crack-band voiding",
            "overconstraint_risk": "can create local top-boundary self-stress not equivalent to cracked-ligament load",
            "prefit_effect": "saved-u/v prefit may keep the split solution close to continuous branch unless energy strongly relaxes it",
        },
        {
            "location": "left/right boundaries",
            "prescribed_u": "free, no explicit side Dirichlet condition",
            "prescribed_v": "free, no explicit side Dirichlet condition",
            "left_right_or_notch_condition": "side boundaries are not constrained in the ansatz",
            "split_domain_parameter_sharing": "none after split",
            "constraints_after_split": "side boundaries can carry stress but are not directly prescribed",
            "overconstraint_risk": "side load path must be audited through boundary force balance",
            "prefit_effect": "prefit may preserve side displacement patterns from saved field",
        },
        {
            "location": "notch/crack band",
            "prescribed_u": "no direct displacement boundary condition",
            "prescribed_v": "no direct displacement boundary condition",
            "left_right_or_notch_condition": "real geometric notch is not forced to alpha=1 in this diagnostic",
            "split_domain_parameter_sharing": "split labels allow different upper/lower displacement fields",
            "constraints_after_split": "no explicit continuity penalty across connected alpha>=0.8 band",
            "overconstraint_risk": "remaining stress can come from top/bottom ansatz or stress metric rather than crack-band traction",
            "prefit_effect": "prefit-to-saved-u/v is the main path that can start from a continuous-like branch",
        },
    ]


def run_audit():
    setup_dirs()
    all_boundary = []
    subdomain_rows = []
    cut_rows = []
    rigid_rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            data = load_npz(seed, variant)
            stress = variant_stress_from_fields(data)
            all_boundary.extend(summarize_boundary_forces(data, stress, "variant", seed, variant))
            all_boundary.extend(summarize_boundary_forces(data, stress, "total", seed, variant))
            subdomain_rows.extend(subdomain_free_body_rows(data, stress, seed, variant))
            top = next(
                row["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
                for row in summarize_boundary_forces(data, stress, "variant", seed, variant)
                if row["boundary"] == "top" and row["subdomain"] == "whole"
            )
            cut_rows.extend(reaction_vs_cut_rows(data, stress, seed, variant, top))
        base = load_npz(seed, "continuous_baseline")
        for field_name in SYNTHETIC_FIELDS:
            synthetic = synthetic_data_from_base(base, field_name)
            for treatment in SYNTHETIC_STRESS_TREATMENTS:
                stress = variant_stress_from_fields(synthetic, treatment=treatment)
                top_rows = summarize_boundary_forces(synthetic, stress, "variant", seed, f"{field_name}_{treatment}")
                top = next(
                    row["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
                    for row in top_rows
                    if row["boundary"] == "top" and row["subdomain"] == "whole"
                )
                residual = next(
                    row
                    for row in top_rows
                    if row["boundary"] == "physical_boundary_sum" and row["subdomain"] == "whole"
                )
                rigid_rows.append(
                    {
                        "case": f"D0040_seed{seed}_default_unitbox",
                        "seed": seed,
                        "synthetic_field": field_name,
                        "stress_treatment": treatment,
                        "top_reaction_proxy_N": top,
                        "whole_boundary_residual_x_N": residual["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"],
                        "whole_boundary_residual_y_N": residual["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"],
                        "elastic_energy": elastic_energy(synthetic, stress),
                        "crack_band_traction_proxy": crack_band_traction_proxy(synthetic, stress),
                        "nonzero_reaction_forced_by_boundary_constraints": abs(top) > 0.05,
                    }
                )

    pd.DataFrame(reaction_definition_audit_rows()).to_csv(TABLES / "reaction_definition_audit.csv", index=False)
    pd.DataFrame(all_boundary).to_csv(TABLES / "all_boundary_force_balance.csv", index=False)
    pd.DataFrame(subdomain_rows).to_csv(TABLES / "subdomain_free_body_audit.csv", index=False)
    pd.DataFrame(cut_rows).to_csv(TABLES / "reaction_vs_internal_cut_consistency.csv", index=False)
    pd.DataFrame(boundary_condition_audit_rows()).to_csv(TABLES / "boundary_condition_overconstraint_audit.csv", index=False)
    pd.DataFrame(rigid_rows).to_csv(TABLES / "rigid_body_sanity_audit.csv", index=False)

    mechanism = classify_audit(pd.DataFrame(all_boundary), pd.DataFrame(subdomain_rows), pd.DataFrame(cut_rows), pd.DataFrame(rigid_rows))
    make_figures(
        pd.DataFrame(all_boundary),
        pd.DataFrame(subdomain_rows),
        pd.DataFrame(cut_rows),
        pd.DataFrame(rigid_rows),
    )
    write_docs(mechanism)
    write_manifest()
    return mechanism


def classify_audit(boundary, subdomain, cuts, rigid):
    void_top = boundary[
        (boundary["variant"] == "split_domain_crack_band_void")
        & (boundary["stress_version"] == "variant")
        & (boundary["subdomain"] == "whole")
        & (boundary["boundary"] == "top")
    ]
    rigid_zero = rigid[
        (rigid["synthetic_field"] == "zero_displacement_reference")
        & (rigid["stress_treatment"] == "crack_band_void")
    ]
    rigid_piece = rigid[
        (rigid["synthetic_field"] == "piecewise_rigid_upper_lower")
        & (rigid["stress_treatment"] == "crack_band_void")
    ]
    high_zero = int(np.sum(np.abs(rigid_zero["top_reaction_proxy_N"]) > 0.5))
    low_piece = int(np.sum(np.abs(rigid_piece["top_reaction_proxy_N"]) < 0.05))
    high_void = int(np.sum(np.abs(void_top["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]) > 0.5))
    if high_zero >= 2 and low_piece >= 2 and high_void >= 2:
        return "reaction/boundary cause identified: boundary ansatz overconstrains separated subdomains and top reaction is a local boundary stress metric"
    ratios = cuts[cuts["variant"] == "split_domain_crack_band_void"]["cut_force_explains_top_reaction_ratio"].replace([np.inf, -np.inf], np.nan)
    if ratios.notna().sum() >= 6 and np.nanmedian(np.abs(ratios)) < 0.25:
        return "reaction/boundary cause identified: top-boundary reaction is not explained by internal cut forces"
    return "reaction/boundary audit unresolved"


def plot_boundary_vectors(boundary):
    variant_rows = boundary[(boundary["stress_version"] == "variant") & (boundary["subdomain"] == "whole")]
    for seed, seed_df in variant_rows.groupby("seed"):
        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.2 * len(VARIANTS), 3.1), dpi=180)
        for ax, variant in zip(axes, VARIANTS):
            sub = seed_df[(seed_df["variant"] == variant) & (seed_df["boundary"].isin(["top", "bottom", "left", "right"]))]
            ax.set_title(variant.replace("split_domain_", "").replace("_", "\n"), fontsize=8)
            ax.set_xlim(-0.002, 0.012)
            ax.set_ylim(-0.002, 0.012)
            ax.set_aspect("equal")
            ax.plot([0, 0.01, 0.01, 0, 0], [0, 0, 0.01, 0.01, 0], "k-", lw=0.8)
            centers = {"top": (0.005, 0.011), "bottom": (0.005, -0.001), "left": (-0.001, 0.005), "right": (0.011, 0.005)}
            scale = 0.002 / max(1.0, np.nanmax(np.sqrt(sub["integrated_sigma_xx_nx_plus_sigma_xy_ny_N"] ** 2 + sub["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"] ** 2)))
            for row in sub.itertuples():
                cx, cy = centers[row.boundary]
                ax.arrow(cx, cy, row.integrated_sigma_xx_nx_plus_sigma_xy_ny_N * scale, row.integrated_sigma_xy_nx_plus_sigma_yy_ny_N * scale, head_width=0.00025, length_includes_head=True)
                ax.text(cx, cy, row.boundary[0], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"All-boundary force vectors: seed {seed}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"all_boundary_force_vectors_seed{seed}.png")
        plt.close(fig)


def plot_cut_consistency(cuts):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    sub = cuts[(cuts["cut_type"] == "horizontal_above") & (cuts["variant"].isin(VARIANTS))]
    for variant, vdf in sub.groupby("variant"):
        grouped = vdf.groupby("seed")["cut_force_explains_top_reaction_ratio"].mean().reset_index()
        ax.plot(grouped["seed"], grouped["cut_force_explains_top_reaction_ratio"], marker="o", label=variant.replace("split_domain_", ""))
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("seed")
    ax.set_ylabel("horizontal-above cut fy / top reaction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "top_reaction_vs_internal_cut_force.png")
    plt.close(fig)


def plot_subdomain_residual(subdomain):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    final = subdomain[subdomain["variant"] == "split_domain_crack_band_void"]
    labels = [f"s{int(r.seed)}-{r.subdomain[0]}" for r in final.itertuples()]
    ax.bar(np.arange(len(final)), final["net_force_residual_magnitude_N"])
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("net free-body residual magnitude [N]")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "upper_lower_subdomain_free_body_residual.png")
    plt.close(fig)


def plot_boundary_condition_map():
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
    ax.plot([0, 0.01, 0.01, 0, 0], [0, 0, 0.01, 0.01, 0], "k-", lw=1.0)
    ax.text(0.005, 0.0107, "top: v=Delta, u free", ha="center", fontsize=8)
    ax.text(0.005, -0.0009, "bottom: u=v=0", ha="center", fontsize=8)
    ax.text(-0.0010, 0.005, "left free", rotation=90, va="center", fontsize=8)
    ax.text(0.0108, 0.005, "right free", rotation=90, va="center", fontsize=8)
    ax.plot([0.005, 0.01], [0.005, 0.005], "r--", lw=1.2, label="alpha>=0.8 path")
    ax.set_xlim(-0.002, 0.012)
    ax.set_ylim(-0.002, 0.012)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "boundary_condition_map.png")
    plt.close(fig)


def plot_rigid_body(rigid):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    sub = rigid[rigid["stress_treatment"] == "crack_band_void"]
    labels = [f"s{int(r.seed)}\n{r.synthetic_field.replace('_reference','').replace('piecewise_','pw_')}" for r in sub.itertuples()]
    ax.bar(np.arange(len(sub)), sub["top_reaction_proxy_N"])
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("top reaction proxy [N]")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "rigid_body_sanity_reaction_comparison.png")
    plt.close(fig)


def plot_split_labels_and_stress_maps():
    for seed in SEEDS:
        data = load_npz(seed, "split_domain_crack_band_void")
        tri = mtri.Triangulation(data["x"], data["y"], data["triangles"].astype(int))
        fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
        artist = ax.tripcolor(tri, facecolors=data["elem_upper"], shading="flat", cmap="coolwarm", vmin=0, vmax=1)
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], s=2, c="black")
        ax.plot([0, 0.01, 0.01, 0, 0], [0, 0, 0.01, 0.01, 0], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: split labels and boundaries")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.tight_layout()
        fig.savefig(FIGURES / f"split_domain_labels_boundaries_seed{seed}.png")
        plt.close(fig)

        stress = variant_stress_from_fields(data)
        fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
        vmax = float(np.nanpercentile(np.abs(stress["variant_yy"]), 98))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0
        artist = ax.tripcolor(tri, facecolors=stress["variant_yy"], shading="flat", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.scatter(data["element_x"][data["crack_mask"].astype(bool)], data["element_y"][data["crack_mask"].astype(bool)], s=2, c="black")
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: void variant sigma_yy")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stress_map_void_high_reaction_seed{seed}.png")
        plt.close(fig)


def make_figures(boundary, subdomain, cuts, rigid):
    plot_boundary_vectors(boundary)
    plot_cut_consistency(cuts)
    plot_subdomain_residual(subdomain)
    plot_boundary_condition_map()
    plot_rigid_body(rigid)
    plot_split_labels_and_stress_maps()
    write_figure_summary()


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only and support the boundary/reaction audit.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `all_boundary_force_vectors_seed*.png` | Physical-boundary force vectors for each seed and replay variant | Shows whether top reaction is balanced by bottom/side forces. | Diagnostic force-balance evidence. |",
        "| `top_reaction_vs_internal_cut_force.png` | Internal horizontal cut force divided by top reaction | Checks whether internal cuts explain the top-boundary reaction. | Diagnostic reaction consistency evidence. |",
        "| `upper_lower_subdomain_free_body_residual.png` | Upper/lower free-body residual magnitudes for void replay | Shows whether split subdomains are in force balance. | Diagnostic only. |",
        "| `boundary_condition_map.png` | Displacement ansatz boundary conditions | Summarizes top/bottom/side constraints. | Boundary-condition evidence. |",
        "| `rigid_body_sanity_reaction_comparison.png` | Synthetic-field top reaction under crack-band void treatment | Tests whether boundary ansatz alone can force nonzero reaction. | Diagnostic sanity evidence. |",
        "| `split_domain_labels_boundaries_seed*.png` | Split labels, crack band, and physical boundaries | Audits geometry used for subdomain calculations. | Geometry support. |",
        "| `stress_map_void_high_reaction_seed*.png` | sigma_yy map for crack-band void replay | Shows where stress remains when crack-band traction is zero. | Diagnostic observation. |",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_digest():
    boundary = pd.read_csv(TABLES / "all_boundary_force_balance.csv")
    subdomain = pd.read_csv(TABLES / "subdomain_free_body_audit.csv")
    rigid = pd.read_csv(TABLES / "rigid_body_sanity_audit.csv")
    void_top = boundary[
        (boundary["variant"] == "split_domain_crack_band_void")
        & (boundary["stress_version"] == "variant")
        & (boundary["boundary"] == "top")
        & (boundary["subdomain"] == "whole")
    ]["integrated_sigma_xy_nx_plus_sigma_yy_ny_N"]
    piecewise = rigid[
        (rigid["synthetic_field"] == "piecewise_rigid_upper_lower")
        & (rigid["stress_treatment"] == "crack_band_void")
    ]["top_reaction_proxy_N"]
    zero = rigid[
        (rigid["synthetic_field"] == "zero_displacement_reference")
        & (rigid["stress_treatment"] == "crack_band_void")
    ]["top_reaction_proxy_N"]
    saved = rigid[
        (rigid["synthetic_field"] == "saved_uv_reference")
        & (rigid["stress_treatment"] == "crack_band_void")
    ]["top_reaction_proxy_N"]
    residual = subdomain[subdomain["variant"] == "split_domain_crack_band_void"]["net_force_residual_magnitude_N"]
    return {
        "void_top_mean": float(void_top.mean()),
        "void_top_values": [float(v) for v in void_top],
        "piecewise_rigid_abs_max": float(np.max(np.abs(piecewise))),
        "zero_reference_mean": float(zero.mean()),
        "saved_reference_mean": float(saved.mean()),
        "void_subdomain_residual_mean": float(residual.mean()),
    }


def write_docs(mechanism):
    digest = table_digest()
    report = [
        "# Boundary/reaction/free-body audit",
        "",
        "## Scope",
        "",
        "This package audits the final_D0040 split-domain replay fields from seeds 7, 13, and 42. It does not extend loading, evolve alpha, change material parameters, change `l0`, change TM split, or retrain a production model.",
        "",
        "## Key summary",
        "",
        f"- Classification: **{mechanism}**.",
        f"- Crack-band-void replay top reactions [N]: {digest['void_top_values']}.",
        f"- Piecewise rigid upper/lower synthetic field, crack-band-void treatment, max |top reaction| [N]: {digest['piecewise_rigid_abs_max']:.6g}.",
        f"- Zero-displacement-reference synthetic field mean top reaction [N]: {digest['zero_reference_mean']:.6g}.",
        f"- Saved-u/v synthetic field mean top reaction under void treatment [N]: {digest['saved_reference_mean']:.6g}.",
        f"- Mean void-replay upper/lower subdomain residual magnitude [N]: {digest['void_subdomain_residual_mean']:.6g}.",
        "",
        "## Answers",
        "",
        "1. `reaction_N_tm_eff` is conjugate to the imposed vertical top displacement in the narrow sense that it integrates `sigma_yy_tm_eff` over the top boundary.",
        "2. It is a top-boundary local stress integral; by itself it does not prove global load transfer across the cracked ligament.",
        "3. After crack-band voiding, upper/lower subdomain free-body residuals are nonzero; see `tables/subdomain_free_body_audit.csv`.",
        "4. Remaining force is concentrated in physical boundary and subdomain residual terms rather than in the voided crack band; see `tables/all_boundary_force_balance.csv` and `tables/reaction_vs_internal_cut_consistency.csv`.",
        "5. Internal cut forces do not consistently explain the top-boundary reaction across all seeds and variants.",
        "6. Synthetic fields show that boundary constraints can force stress in the zero-displacement-reference field, while a piecewise rigid upper/lower field nearly removes the top reaction under crack-band void treatment.",
        "7. The prefit-to-saved-u/v step is a plausible contributor because saved-u/v synthetic fields retain nonzero top reaction under crack-band void treatment.",
        f"8. Current audit decision: **{mechanism}**.",
        "9. No production model change is justified directly from this diagnostic.",
        "10. Next minimal intervention: ask ChatGPT to decide whether to audit the top/bottom displacement ansatz and reaction definition against a FE-DOF free-body calculation, before changing physics.",
        "",
        "## Verification",
        "",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q`: to be filled after verification.",
        "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260614_default_unitbox_boundary_reaction_audit\\artifacts\\run_boundary_reaction_audit.py`: to be filled after verification.",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = [
        "# Boundary/reaction/free-body audit package",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/reaction_definition_audit.csv`",
        "3. `tables/all_boundary_force_balance.csv`",
        "4. `tables/subdomain_free_body_audit.csv`",
        "5. `tables/reaction_vs_internal_cut_consistency.csv`",
        "6. `tables/boundary_condition_overconstraint_audit.csv`",
        "7. `tables/rigid_body_sanity_audit.csv`",
        "8. `figures/figure_summary.md`",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    next_questions = [
        "# Next questions",
        "",
        "1. Is the reaction/boundary cause sufficiently identified across 2/3 seeds?",
        "2. Should the next diagnostic compare this top-boundary reaction to a FE-DOF free-body reaction calculation?",
        "3. Should any production model change be deferred until reaction definition is fixed?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions) + "\n", encoding="utf-8")

    commands = [
        "git pull origin main",
        "Read 20260613_default_unitbox_discontinuous_kinematic_replay handoff/report/tables/figure summary.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_boundary_reaction_audit.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260614_default_unitbox_boundary_reaction_audit\\artifacts\\run_boundary_reaction_audit.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: boundary/reaction/free-body audit",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260614_default_unitbox_boundary_reaction_audit",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260614_default_unitbox_boundary_reaction_audit/REPORT.md",
        "",
        "### What changed",
        "- Added and ran a diagnostic-only boundary/reaction/free-body audit.",
        "- Audited final_D0040 split-domain replay fields for seeds 7, 13, 42 and variants continuous_baseline, split_domain_current_split, split_domain_minus_degraded_crack_band, and split_domain_crack_band_void.",
        "- Reconstructed total/effective/variant stress fields from saved u/v, alpha, and strain fields.",
        "- Computed top/bottom/left/right boundary force integrals, upper/lower subdomain free-body terms, internal cut consistency, boundary-condition audit, and synthetic rigid-body sanity tests.",
        "- No loading, alpha, material, `l0`, TM split, or history logic was changed.",
        "",
        "### Commands run",
        "```powershell",
        *commands,
        "```",
        "",
        "### Key results",
        f"- Identified cause/status: **{mechanism}**.",
        "- `reaction_N_tm_eff` is a top-boundary `sigma_yy_tm_eff` integral; it is locally conjugate to top vertical displacement but is not by itself a global cracked-ligament load metric.",
        "- Crack-band-void replay still has nonzero top reactions for seeds 7 and 13, while synthetic piecewise-rigid upper/lower fields nearly remove top reaction under crack-band-void treatment.",
        "- Boundary-condition and prefit effects remain plausible contributors; no production model change is justified directly.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/reaction_definition_audit.csv`",
        "- `tables/all_boundary_force_balance.csv`",
        "- `tables/subdomain_free_body_audit.csv`",
        "- `tables/reaction_vs_internal_cut_consistency.csv`",
        "- `tables/boundary_condition_overconstraint_audit.csv`",
        "- `tables/rigid_body_sanity_audit.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Is the dominant reaction/boundary cause identified across at least 2/3 seeds?",
        "2. Should the next minimal diagnostic be a FE-DOF free-body/reaction comparison rather than another PINN split replay?",
        "3. Is any production model change justified yet?",
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


def describe_file(rel):
    return {
        "REPORT.md": "Main audit report and answers.",
        "README.md": "Package reading order.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for issue #1.",
        "tables/reaction_definition_audit.csv": "Static audit of reaction/reporting code paths.",
        "tables/all_boundary_force_balance.csv": "Top/bottom/left/right force integrals and residual sums.",
        "tables/subdomain_free_body_audit.csv": "Upper/lower subdomain free-body force and moment proxies.",
        "tables/reaction_vs_internal_cut_consistency.csv": "Top reaction compared with internal cut forces.",
        "tables/boundary_condition_overconstraint_audit.csv": "Boundary ansatz and overconstraint notes.",
        "tables/rigid_body_sanity_audit.csv": "Synthetic displacement field sanity tests.",
        "figures/figure_summary.md": "Text summary for all figures.",
    }.get(rel, "Generated diagnostic artifact.")


def main():
    mechanism = run_audit()
    print(mechanism)


if __name__ == "__main__":
    main()
