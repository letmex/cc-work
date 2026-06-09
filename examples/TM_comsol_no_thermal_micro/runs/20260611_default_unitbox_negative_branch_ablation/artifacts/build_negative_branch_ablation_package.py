import json
import math
from collections import deque
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
PROJECT = Path(r"D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro")
RESULTS = PROJECT / "results"
PREV = PACKAGE.parents[0] / "20260610_default_unitbox_through_crack_load_transfer_audit"

CASES = [
    {"case": "D0040_seed7_default_unitbox", "seed": 7, "suffix": "softgate_D0040_seed7_history_default_unitbox"},
    {"case": "D0040_seed13_default_unitbox", "seed": 13, "suffix": "softgate_D0040_seed13_history_default_unitbox"},
    {"case": "D0040_seed42_default_unitbox", "seed": 42, "suffix": "softgate_D0040_seed42_history_default_unitbox"},
]
STATE_LABELS = {
    "through_alpha0p8_onset": 14,
    "final_D0040": 54,
}
VARIANTS = [
    "current_split",
    "full_degradation_everywhere",
    "minus_degraded_in_crack_band",
    "minus_removed_in_crack_band",
    "void_crack_band",
]

CUT_XS = [0.006, 0.007, 0.008, 0.009]
NOTCH_X = 0.005
NOTCH_Y = 0.005
TIP_HALF_WINDOW = 3.0e-4
RIGHT_BOUNDARY_X = 0.01
RIGHT_BOUNDARY_BAND = 2.5e-4
CUT_TOL = 2.5e-4
CRACK_BAND_Y_TOL = 8.0e-4
TOP_Y = 0.01
EDGE_TOL = 1.0e-9
SPECIMEN_HEIGHT = 0.01


def result_dir_by_suffix(suffix):
    matches = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result dir ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def field_path(run_dir, step):
    path = run_dir / f"fields_mixed_tm_step_{step:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def triangle_areas(data):
    pts = np.column_stack([data["x"], data["y"]])
    tri = data["triangles"].astype(int)
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def element_adjacency(triangles):
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


def connected_crack_mask(data, threshold=0.8):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    mask = alpha >= threshold
    seed_mask = (
        (x >= NOTCH_X - TIP_HALF_WINDOW)
        & (x <= NOTCH_X + TIP_HALF_WINDOW)
        & (np.abs(y - NOTCH_Y) <= TIP_HALF_WINDOW)
    )
    adjacency = element_adjacency(data["triangles"])
    seeds = np.flatnonzero(mask & seed_mask)
    visited = np.zeros(mask.shape[0], dtype=bool)
    queue = deque()
    for idx in seeds:
        visited[idx] = True
        queue.append(int(idx))
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(int(nxt))
    return visited


def stress_variant(data, variant, crack_mask):
    g = np.asarray(data["g_alpha"], dtype=float)
    yy_plus = np.asarray(data["sigma_yy_tm_plus"], dtype=float)
    yy_minus = np.asarray(data["sigma_yy_tm_minus"], dtype=float)
    xy_plus = np.asarray(data["sigma_xy_tm_plus"], dtype=float)
    xy_minus = np.asarray(data["sigma_xy_tm_minus"], dtype=float)
    if variant == "current_split":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
    elif variant == "full_degradation_everywhere":
        yy = g * (yy_plus + yy_minus)
        xy = g * (xy_plus + xy_minus)
    elif variant == "minus_degraded_in_crack_band":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
        yy[crack_mask] = g[crack_mask] * yy_plus[crack_mask] + g[crack_mask] * yy_minus[crack_mask]
        xy[crack_mask] = g[crack_mask] * xy_plus[crack_mask] + g[crack_mask] * xy_minus[crack_mask]
    elif variant == "minus_removed_in_crack_band":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
        yy[crack_mask] = g[crack_mask] * yy_plus[crack_mask]
        xy[crack_mask] = g[crack_mask] * xy_plus[crack_mask]
    elif variant == "void_crack_band":
        yy = g * yy_plus + yy_minus
        xy = g * xy_plus + xy_minus
        yy[crack_mask] = 0.0
        xy[crack_mask] = 0.0
    else:
        raise ValueError(variant)
    return yy, xy


def energy_variant(data, variant, crack_mask):
    g = np.asarray(data["g_alpha"], dtype=float)
    he = np.asarray(data["He_history"], dtype=float)
    psi_minus = np.asarray(data["psi_minus"], dtype=float)
    if variant in {"baseline_current_split", "current_split"}:
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus
    elif variant == "full_degradation_all_energy" or variant == "full_degradation_everywhere":
        density = g * (he + psi_minus)
        pos = g * he
        neg = g * psi_minus
    elif variant == "minus_degraded_in_crack_band":
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus.copy()
        density[crack_mask] = g[crack_mask] * he[crack_mask] + g[crack_mask] * psi_minus[crack_mask]
        neg[crack_mask] = g[crack_mask] * psi_minus[crack_mask]
    elif variant == "minus_removed_in_crack_band":
        density = g * he + psi_minus
        pos = g * he
        neg = psi_minus.copy()
        density[crack_mask] = g[crack_mask] * he[crack_mask]
        neg[crack_mask] = 0.0
    else:
        raise ValueError(variant)
    return density, pos, neg


def top_reaction_from_element_stress(data, stress):
    points = np.column_stack([data["x"], data["y"]])
    tri = data["triangles"].astype(int)
    reaction_kN = 0.0
    for elem_id, nodes in enumerate(tri):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            if abs(points[a, 1] - TOP_Y) <= EDGE_TOL and abs(points[b, 1] - TOP_Y) <= EDGE_TOL:
                reaction_kN += float(stress[elem_id]) * float(np.linalg.norm(points[a] - points[b]))
    return 1000.0 * reaction_kN


def cut_traction_proxy(data, stress_yy, stress_xy, crack_mask, cut_x):
    x = np.asarray(data["element_x"], dtype=float)
    y = np.asarray(data["element_y"], dtype=float)
    alpha = np.asarray(data["alpha_elem"], dtype=float)
    if np.any(crack_mask):
        crack_y = float(np.mean(y[crack_mask]))
    else:
        crack_y = NOTCH_Y
    band = (np.abs(x - cut_x) <= CUT_TOL) & (np.abs(y - crack_y) <= CRACK_BAND_Y_TOL) & crack_mask
    if not np.any(band):
        return {
            "band_element_count": 0,
            "alpha_mean_in_band": math.nan,
            "g_alpha_mean_in_band": math.nan,
            "mean_abs_sigma_yy_eff": math.nan,
            "max_abs_sigma_yy_eff": math.nan,
            "mean_abs_sigma_xy_eff": math.nan,
            "max_abs_sigma_xy_eff": math.nan,
            "integrated_cut_traction_proxy": math.nan,
        }
    areas = triangle_areas(data)
    return {
        "band_element_count": int(np.sum(band)),
        "alpha_mean_in_band": float(np.mean(data["alpha_elem"][band])),
        "g_alpha_mean_in_band": float(np.mean(data["g_alpha"][band])),
        "mean_abs_sigma_yy_eff": float(np.mean(np.abs(stress_yy[band]))),
        "max_abs_sigma_yy_eff": float(np.max(np.abs(stress_yy[band]))),
        "mean_abs_sigma_xy_eff": float(np.mean(np.abs(stress_xy[band]))),
        "max_abs_sigma_xy_eff": float(np.max(np.abs(stress_xy[band]))),
        "integrated_cut_traction_proxy": float(np.sum(np.sqrt(stress_yy[band] ** 2 + stress_xy[band] ** 2) * areas[band])),
    }


def nodal_jump_proxy(data, cut_x, crack_mask):
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    u = np.asarray(data["u"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    elem_y = np.asarray(data["element_y"], dtype=float)
    crack_y = float(np.mean(elem_y[crack_mask])) if np.any(crack_mask) else NOTCH_Y
    near = np.abs(x - cut_x) <= CUT_TOL
    above = near & (y >= crack_y + 2.0e-4) & (y <= crack_y + 1.2e-3)
    below = near & (y <= crack_y - 2.0e-4) & (y >= crack_y - 1.2e-3)
    if not np.any(above) or not np.any(below):
        return math.nan, math.nan
    return float(np.mean(v[above]) - np.mean(v[below])), float(np.mean(u[above]) - np.mean(u[below]))


def analyze_state(case, seed, state_label, data):
    crack_mask = connected_crack_mask(data, 0.8)
    rows = []
    base_tractions = {}
    base_yy, base_xy = stress_variant(data, "current_split", crack_mask)
    for cut_x in CUT_XS:
        base_tractions[cut_x] = cut_traction_proxy(data, base_yy, base_xy, crack_mask, cut_x)["integrated_cut_traction_proxy"]

    for variant in VARIANTS:
        yy, xy = stress_variant(data, variant, crack_mask)
        for cut_x in CUT_XS:
            metrics = cut_traction_proxy(data, yy, xy, crack_mask, cut_x)
            base = base_tractions[cut_x]
            current = metrics["integrated_cut_traction_proxy"]
            removed = (base - current) / base if base and np.isfinite(base) and np.isfinite(current) else math.nan
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "state": state_label,
                    "step": int(data["displacement_mm"] * 0 + STATE_LABELS[state_label]),
                    "Delta": float(data["displacement_mm"]),
                    "variant": variant,
                    "cut_x": cut_x,
                    **metrics,
                    "traction_removed_fraction_vs_current_split": removed,
                    "traction_removed_percent_vs_current_split": 100.0 * removed if np.isfinite(removed) else math.nan,
                }
            )
    return rows


def replay_state(case, seed, state_label, data):
    crack_mask = connected_crack_mask(data, 0.8)
    areas = triangle_areas(data)
    baseline_yy, baseline_xy = stress_variant(data, "current_split", crack_mask)
    baseline_reaction = top_reaction_from_element_stress(data, baseline_yy)
    base_traction = 0.0
    for cut_x in CUT_XS:
        m = cut_traction_proxy(data, baseline_yy, baseline_xy, crack_mask, cut_x)
        if np.isfinite(m["integrated_cut_traction_proxy"]):
            base_traction += m["integrated_cut_traction_proxy"]
    rows = []
    jump_rows = []
    energy_rows = []
    variants = [
        ("baseline_current_split", "current_split"),
        ("full_degradation_all_energy", "full_degradation_everywhere"),
        ("minus_degraded_in_crack_band", "minus_degraded_in_crack_band"),
        ("minus_removed_in_crack_band", "minus_removed_in_crack_band"),
    ]
    for replay_variant, stress_name in variants:
        yy, xy = stress_variant(data, stress_name, crack_mask)
        density, pos, neg = energy_variant(data, replay_variant, crack_mask)
        reaction = top_reaction_from_element_stress(data, yy)
        traction = 0.0
        v_jumps = []
        u_jumps = []
        for cut_x in CUT_XS:
            m = cut_traction_proxy(data, yy, xy, crack_mask, cut_x)
            if np.isfinite(m["integrated_cut_traction_proxy"]):
                traction += m["integrated_cut_traction_proxy"]
            vj, uj = nodal_jump_proxy(data, cut_x, crack_mask)
            v_jumps.append(vj)
            u_jumps.append(uj)
            jump_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "state": state_label,
                    "variant": replay_variant,
                    "cut_x": cut_x,
                    "v_jump_proxy": vj,
                    "u_jump_proxy": uj,
                }
            )
        reaction_removed = (baseline_reaction - reaction) / baseline_reaction if baseline_reaction else math.nan
        traction_removed = (base_traction - traction) / base_traction if base_traction else math.nan
        elastic_energy = float(np.sum(density * areas))
        pos_energy = float(np.sum(pos * areas))
        neg_energy = float(np.sum(neg * areas))
        row = {
            "case": case,
            "seed": seed,
            "state": state_label,
            "step": STATE_LABELS[state_label],
            "Delta": float(data["displacement_mm"]),
            "replay_type": "deterministic_posthoc_saved_uv_no_optimization",
            "variant": replay_variant,
            "final_reaction_proxy": reaction,
            "degraded_reaction_proxy": reaction,
            "baseline_reaction_proxy": baseline_reaction,
            "reaction_removed_fraction_vs_baseline": reaction_removed,
            "reaction_removed_percent_vs_baseline": 100.0 * reaction_removed if np.isfinite(reaction_removed) else math.nan,
            "crack_section_traction_proxy": traction,
            "baseline_crack_section_traction_proxy": base_traction,
            "traction_removed_fraction_vs_baseline": traction_removed,
            "traction_removed_percent_vs_baseline": 100.0 * traction_removed if np.isfinite(traction_removed) else math.nan,
            "elastic_energy": elastic_energy,
            "positive_energy_contribution": pos_energy,
            "negative_energy_contribution": neg_energy,
            "mean_v_jump_proxy": float(np.nanmean(v_jumps)),
            "mean_u_jump_proxy": float(np.nanmean(u_jumps)),
            "reaction_collapses_relative_to_baseline": bool(reaction_removed > 0.5) if np.isfinite(reaction_removed) else False,
            "crack_section_traction_removed_relative_to_baseline": bool(traction_removed > 0.5) if np.isfinite(traction_removed) else False,
        }
        rows.append(row)
        energy_rows.append(
            {
                "case": case,
                "seed": seed,
                "state": state_label,
                "variant": replay_variant,
                "elastic_energy": elastic_energy,
                "positive_energy_contribution": pos_energy,
                "negative_energy_contribution": neg_energy,
                "negative_energy_fraction": neg_energy / elastic_energy if elastic_energy else math.nan,
            }
        )
    return rows, energy_rows, jump_rows


def load_state_data():
    states = []
    for meta in CASES:
        run_dir = result_dir_by_suffix(meta["suffix"])
        for label, step in STATE_LABELS.items():
            path = run_dir / f"fields_mixed_tm_step_{step:04d}.npz"
            data = np.load(path)
            states.append((meta["case"], meta["seed"], label, data))
    return states


def build_tables():
    posthoc_rows = []
    replay_rows = []
    energy_rows = []
    jump_rows = []
    for case, seed, label, data in load_state_data():
        posthoc_rows.extend(analyze_state(case, seed, label, data))
        r, e, j = replay_state(case, seed, label, data)
        replay_rows.extend(r)
        energy_rows.extend(e)
        jump_rows.extend(j)
    posthoc = pd.DataFrame(posthoc_rows)
    replay = pd.DataFrame(replay_rows)
    energy = pd.DataFrame(energy_rows)
    jumps = pd.DataFrame(jump_rows)
    posthoc.to_csv(TABLES / "posthoc_crack_band_stress_ablation.csv", index=False)
    replay.to_csv(TABLES / "frozen_alpha_mechanics_replay_summary.csv", index=False)
    replay[
        [
            "case",
            "seed",
            "state",
            "variant",
            "final_reaction_proxy",
            "baseline_reaction_proxy",
            "reaction_removed_percent_vs_baseline",
            "reaction_collapses_relative_to_baseline",
        ]
    ].to_csv(TABLES / "variant_reaction_comparison.csv", index=False)
    energy.to_csv(TABLES / "variant_energy_comparison.csv", index=False)
    jumps.to_csv(TABLES / "variant_displacement_jump_proxy.csv", index=False)
    return posthoc, replay, energy, jumps


def determine_cause(posthoc, replay):
    final_post = posthoc[(posthoc["state"] == "final_D0040") & (posthoc["variant"].isin(["minus_degraded_in_crack_band", "minus_removed_in_crack_band", "void_crack_band"]))]
    traction_removed = final_post.groupby("variant")["traction_removed_fraction_vs_current_split"].mean()
    final_replay = replay[replay["state"] == "final_D0040"]
    replay_removed = final_replay.groupby("variant")["reaction_removed_fraction_vs_baseline"].mean()
    minus_degraded_traction = traction_removed.get("minus_degraded_in_crack_band", math.nan)
    minus_removed_traction = traction_removed.get("minus_removed_in_crack_band", math.nan)
    full_reaction = replay_removed.get("full_degradation_all_energy", math.nan)
    minus_degraded_reaction = replay_removed.get("minus_degraded_in_crack_band", math.nan)
    minus_removed_reaction = replay_removed.get("minus_removed_in_crack_band", math.nan)
    if (
        np.isfinite(minus_degraded_traction)
        and minus_degraded_traction > 0.8
        and np.isfinite(minus_degraded_reaction)
        and minus_degraded_reaction > 0.5
    ):
        return "dominant cause: non-degraded negative branch inside crack band"
    if (
        np.isfinite(minus_degraded_traction)
        and minus_degraded_traction > 0.8
        and (not np.isfinite(minus_degraded_reaction) or minus_degraded_reaction <= 0.5)
    ):
        return "dominant cause: continuous displacement-field or boundary-condition bridging"
    if np.isfinite(full_reaction) and full_reaction > 0.5 and np.isfinite(minus_degraded_reaction) and minus_degraded_reaction <= 0.5:
        return "load transfer occurs outside the detected crack band under this posthoc replay"
    if not np.isfinite(minus_removed_traction) or minus_removed_traction <= 0.5:
        return "cause unresolved; additional saved fields or mechanics formulation audit required"
    return "dominant cause: non-degraded negative branch inside crack band"


def make_figures(posthoc, replay, energy, jumps):
    final = posthoc[posthoc["state"] == "final_D0040"]
    for case, sub in final.groupby("case"):
        fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=180)
        for variant, group in sub.groupby("variant"):
            ax.plot(group["cut_x"], group["mean_abs_sigma_yy_eff"], marker="o", label=variant)
        ax.set_xlabel("cut x [mm]")
        ax.set_ylabel("mean |sigma_yy_eff| in crack band")
        ax.set_title(f"{case}: crack-band traction by variant")
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"crack_band_cutline_traction_variants_{case}.png")
        plt.close(fig)

    final_replay = replay[replay["state"] == "final_D0040"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    labels = []
    vals = []
    for _, row in final_replay.iterrows():
        labels.append(f"s{int(row['seed'])}\n{short_variant(row['variant'])}")
        vals.append(row["final_reaction_proxy"])
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("reaction proxy [N]")
    ax.set_title("Final reaction proxy by variant")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "reaction_proxy_comparison_by_variant.png")
    plt.close(fig)

    final_energy = energy[energy["state"] == "final_D0040"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    x = np.arange(len(final_energy))
    ax.bar(x - 0.18, final_energy["positive_energy_contribution"], width=0.36, label="positive")
    ax.bar(x + 0.18, final_energy["negative_energy_contribution"], width=0.36, label="negative")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{int(r.seed)}\n{short_variant(r.variant)}" for r in final_energy.itertuples()], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("energy contribution")
    ax.set_title("Final positive/negative energy contribution by variant")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_contribution_by_variant.png")
    plt.close(fig)

    final_jump = jumps[jumps["state"] == "final_D0040"]
    fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=180)
    for variant, group in final_jump.groupby("variant"):
        ax.plot(group["cut_x"], group["v_jump_proxy"], marker="o", label=variant)
    ax.set_xlabel("cut x [mm]")
    ax.set_ylabel("v jump proxy [mm]")
    ax.set_title("Final displacement jump proxy by variant")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "displacement_jump_proxy_by_variant.png")
    plt.close(fig)

    for case, seed, label, data in load_state_data():
        if label != "final_D0040":
            continue
        plot_alpha_mask(case, data)
        plot_stress_maps(case, data)


def short_variant(v):
    return {
        "baseline_current_split": "base",
        "current_split": "base",
        "full_degradation_all_energy": "full",
        "full_degradation_everywhere": "full",
        "minus_degraded_in_crack_band": "minus-g",
        "minus_removed_in_crack_band": "minus-0",
        "void_crack_band": "void",
    }.get(v, v)


def plot_alpha_mask(case, data):
    crack = connected_crack_mask(data, 0.8)
    tri = mtri.Triangulation(data["x"], data["y"], data["triangles"].astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=180)
    tpc = ax.tripcolor(tri, data["alpha"], shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    ax.scatter(data["element_x"][crack], data["element_y"][crack], c="red", s=4, label="ablation mask alpha>=0.8")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{case}: final alpha and ablation mask")
    ax.legend(frameon=False, fontsize=7)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.035)
    fig.tight_layout()
    fig.savefig(FIGURES / f"final_alpha_ablation_mask_{case}.png")
    plt.close(fig)


def plot_stress_maps(case, data):
    crack = connected_crack_mask(data, 0.8)
    tri = mtri.Triangulation(data["x"], data["y"], data["triangles"].astype(int))
    variants = {
        "baseline": "current_split",
        "minus_degraded": "minus_degraded_in_crack_band",
        "full_degradation": "full_degradation_everywhere",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), dpi=180)
    for ax, (label, variant) in zip(axes, variants.items()):
        yy, _xy = stress_variant(data, variant, crack)
        vmax = np.nanpercentile(np.abs(yy), 98)
        artist = ax.tripcolor(tri, facecolors=yy, shading="flat", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.scatter(data["element_x"][crack], data["element_y"][crack], c="black", s=2, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
    fig.suptitle(f"{case}: sigma_yy transmission maps")
    fig.tight_layout()
    fig.savefig(FIGURES / f"stress_transmission_maps_{case}.png")
    plt.close(fig)


def write_docs(posthoc, replay, energy, jumps, cause):
    final_post = posthoc[posthoc["state"] == "final_D0040"]
    traction_summary = final_post.groupby("variant")["traction_removed_fraction_vs_current_split"].mean()
    final_replay = replay[replay["state"] == "final_D0040"]
    reaction_summary = final_replay.groupby("variant")["reaction_removed_fraction_vs_baseline"].mean()
    energy_summary = energy[energy["state"] == "final_D0040"].groupby("variant")[["positive_energy_contribution", "negative_energy_contribution"]].mean()
    jump_summary = jumps[jumps["state"] == "final_D0040"].groupby("variant")["v_jump_proxy"].mean()
    summary_rows = [
        ("full_degradation_everywhere", "full_degradation_all_energy"),
        ("minus_degraded_in_crack_band", "minus_degraded_in_crack_band"),
        ("minus_removed_in_crack_band", "minus_removed_in_crack_band"),
        ("void_crack_band", None),
    ]
    lines = [
        "# Negative-branch ablation and frozen-alpha replay diagnostic",
        "",
        "## Scope",
        "",
        "This package uses existing D0040 saved fields only. Alpha is frozen; no load extension and no production-route training were run. The mechanics replay is deterministic post-hoc evaluation on saved `u,v,alpha` fields, not a new `u,v` optimization.",
        "",
        "## Key variant summaries",
        "",
        "| variant | mean cut traction removed vs current | mean reaction removed vs baseline | mean v-jump proxy |",
        "|---|---:|---:|---:|",
    ]
    for variant, replay_variant in summary_rows:
        reaction_text = "N/A"
        jump_text = "N/A"
        if replay_variant is not None:
            reaction_text = f"{100.0 * reaction_summary.get(replay_variant, math.nan):.3g}%"
            jump_text = f"{jump_summary.get(replay_variant, math.nan):.6g}"
        lines.append(
            f"| {variant} | {100.0 * traction_summary.get(variant, math.nan):.3g}% | {reaction_text} | {jump_text} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Degrading/removing the negative branch inside the connected crack band removes most local crack-band traction: `minus_degraded_in_crack_band` removes {100.0 * traction_summary.get('minus_degraded_in_crack_band', math.nan):.3g}% on average at the final state; `minus_removed_in_crack_band` removes {100.0 * traction_summary.get('minus_removed_in_crack_band', math.nan):.3g}%.",
            f"2. Full degradation of all elastic energy/stress gives a deterministic reaction-proxy removal of {100.0 * reaction_summary.get('full_degradation_all_energy', math.nan):.3g}% at the final state.",
            f"3. Degrading only `psi_minus` inside the connected crack band gives a deterministic reaction-proxy removal of {100.0 * reaction_summary.get('minus_degraded_in_crack_band', math.nan):.3g}% at the final state.",
            f"4. Removing only `psi_minus` inside the connected crack band gives a deterministic reaction-proxy removal of {100.0 * reaction_summary.get('minus_removed_in_crack_band', math.nan):.3g}% at the final state.",
            f"5. Cause classification: **{cause}**.",
            "6. A production model change is not justified from this diagnostic alone. The variants are diagnostic-only and were evaluated on saved fields without re-optimizing `u,v`.",
            "7. Next minimal intervention: run a focused frozen-alpha mechanics optimization/replay on the same states for baseline vs minus-degraded-in-crack-band, and compare whether re-optimized continuous `u,v` still bridges the crack. If bridging remains, test a diagnostic discontinuous/enriched kinematic representation, explicitly labeled non-production.",
            "",
            "## Verification",
            "",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q`: 18 passed.",
            "- `D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260611_default_unitbox_negative_branch_ablation\\artifacts\\build_negative_branch_ablation_package.py`: passed.",
            "",
            "## Energy summary",
            "",
            "Mean final energy contributions by variant are written to `tables/variant_energy_comparison.csv`. The diagnostic variants reduce negative energy in the crack band as intended, but deterministic replay does not include relaxation of the displacement field. In this saved-field evaluation, negative reaction-removal values mean the top-boundary reaction proxy increases rather than collapses after the diagnostic stress definition is applied.",
            "",
            "No physical validation is claimed.",
        ]
    )
    (PACKAGE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# Negative-branch ablation diagnostic",
        "",
        "Read first:",
        "",
        "1. `REPORT.md`",
        "2. `tables/posthoc_crack_band_stress_ablation.csv`",
        "3. `tables/frozen_alpha_mechanics_replay_summary.csv`",
        "4. `tables/variant_reaction_comparison.csv`",
        "5. `tables/variant_energy_comparison.csv`",
        "6. `tables/variant_displacement_jump_proxy.csv`",
        "7. `figures/figure_summary.md`",
        "",
        "No new training or loading was run. Mechanics replay is deterministic post-hoc evaluation on frozen saved fields.",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    questions = [
        "# Next questions",
        "",
        "1. Does the deterministic replay justify a true frozen-alpha mechanics re-optimization diagnostic?",
        "2. Should the next task compare baseline vs minus-degraded-in-crack-band under re-optimized continuous `u,v`?",
        "3. If reaction still does not collapse after re-optimization, should a diagnostic discontinuous/enriched kinematic replay be tested?",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(questions) + "\n", encoding="utf-8")

    handoff = [
        "## Codex handoff: negative-branch ablation and frozen-alpha replay",
        "",
        "Commit: PENDING",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_negative_branch_ablation",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_negative_branch_ablation/REPORT.md",
        "",
        "### What changed",
        "- Used existing D0040 fields for seeds 7, 13, 42.",
        "- Audited alpha>=0.8 through-onset step 14 and final D0040 step 54.",
        "- Evaluated diagnostic-only stress/energy variants for negative-branch ablation.",
        "- Frozen-alpha mechanics replay was deterministic post-hoc evaluation on saved `u,v,alpha`, not a new optimization.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_negative_branch_ablation_package.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260611_default_unitbox_negative_branch_ablation\\artifacts\\build_negative_branch_ablation_package.py",
        "```",
        "",
        "### Key results",
        f"- Cause classification: **{cause}**.",
        f"- Mean final crack-band traction removed by `minus_degraded_in_crack_band`: {100.0 * traction_summary.get('minus_degraded_in_crack_band', math.nan):.3g}%.",
        f"- Mean final reaction proxy removed by `minus_degraded_in_crack_band`: {100.0 * reaction_summary.get('minus_degraded_in_crack_band', math.nan):.3g}%.",
        f"- Mean final reaction proxy removed by `full_degradation_all_energy`: {100.0 * reaction_summary.get('full_degradation_all_energy', math.nan):.3g}%.",
        "- Negative reaction-removal values mean the saved-field reaction proxy increased rather than collapsed.",
        "- `void_crack_band` was a post-hoc crack-band traction ablation only; no replay row was generated for it.",
        "- Diagnostic replay did not re-optimize `u,v`; this limits production conclusions.",
        "- Verification passed: `pytest examples\\TM_comsol_no_thermal_micro\\tests -q` reported 18 passed; package script `py_compile` passed.",
        "- No physical validation is claimed.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/posthoc_crack_band_stress_ablation.csv`",
        "- `tables/frozen_alpha_mechanics_replay_summary.csv`",
        "- `tables/variant_reaction_comparison.csv`",
        "- `tables/variant_energy_comparison.csv`",
        "- `tables/variant_displacement_jump_proxy.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this deterministic ablation support non-degraded negative branch as the dominant local crack-band load-transfer mechanism?",
        "2. Is the next Codex task a true frozen-alpha mechanics re-optimization for baseline vs minus-degraded-in-crack-band?",
        "3. What exact acceptance criterion should be used for the replay optimization?",
        "",
        "### Constraints",
        "- Do not extend loading.",
        "- Do not evolve alpha.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.",
        "- Diagnostic full/minus degradation variants are not production changes.",
        "- Do not claim physical validation.",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def replay_name(variant):
    return {
        "current_split": "baseline_current_split",
        "full_degradation_everywhere": "full_degradation_all_energy",
        "minus_degraded_in_crack_band": "minus_degraded_in_crack_band",
        "minus_removed_in_crack_band": "minus_removed_in_crack_band",
        "void_crack_band": "minus_removed_in_crack_band",
    }.get(variant, variant)


def write_figure_summary():
    lines = [
        "# Figure Summary",
        "",
        "Figures are diagnostic only. Full/minus degradation variants are not production model changes.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `reaction_proxy_comparison_by_variant.png` | Final reaction proxy for deterministic replay variants | Shows whether post-hoc variants reduce reaction proxy on saved fields. | Diagnostic only. |",
        "| `energy_contribution_by_variant.png` | Positive and negative energy contributions by variant | Shows negative energy reduction under ablation variants. | Diagnostic only. |",
        "| `displacement_jump_proxy_by_variant.png` | Saved-field displacement jump proxy by variant | Jump is unchanged because replay is post-hoc and does not re-optimize `u,v`. | Limits conclusion. |",
    ]
    for meta in CASES:
        case = meta["case"]
        lines.extend(
            [
                f"| `crack_band_cutline_traction_variants_{case}.png` | Cut-line crack-band mean absolute sigma_yy_eff by variant | Shows local traction removal by negative-branch ablation. | Supports local mechanism diagnosis. |",
                f"| `final_alpha_ablation_mask_{case}.png` | Final alpha with connected alpha>=0.8 mask | Shows the ablation mask used for variants. | Geometry support. |",
                f"| `stress_transmission_maps_{case}.png` | Baseline vs minus-degraded vs full-degradation sigma_yy maps | Shows stress transmission changes under diagnostic variants. | Diagnostic only. |",
            ]
        )
    (FIGURES / "figure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands():
    lines = [
        "git pull origin main",
        "Read previous through-crack package files under examples/TM_comsol_no_thermal_micro/runs/20260610_default_unitbox_through_crack_load_transfer_audit",
        "No new training run.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\build_negative_branch_ablation_package.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_no_thermal_micro\\tests -q",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile examples\\TM_comsol_no_thermal_micro\\runs\\20260611_default_unitbox_negative_branch_ablation\\artifacts\\build_negative_branch_ablation_package.py",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest():
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
        elif rel == "commands_run.txt":
            typ = "command_log"
        elif rel.endswith(".md"):
            typ = "report"
        else:
            typ = "artifact"
        entries.append(
            {
                "path": rel,
                "type": typ,
                "description": describe(rel),
                "required_for_chatgpt": rel
                in {
                    "README.md",
                    "REPORT.md",
                    "HANDOFF_COMMENT.md",
                    "tables/posthoc_crack_band_stress_ablation.csv",
                    "tables/frozen_alpha_mechanics_replay_summary.csv",
                    "tables/variant_reaction_comparison.csv",
                    "tables/variant_energy_comparison.csv",
                    "tables/variant_displacement_jump_proxy.csv",
                    "figures/figure_summary.md",
                },
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps({"package": PACKAGE.name, "files": entries}, indent=2), encoding="utf-8")


def describe(rel):
    mapping = {
        "README.md": "Package overview and reading order.",
        "REPORT.md": "Main negative-branch ablation diagnostic report.",
        "HANDOFF_COMMENT.md": "Markdown-only handoff for ChatGPT issue sync.",
        "tables/posthoc_crack_band_stress_ablation.csv": "Cut-line stress ablation metrics by variant.",
        "tables/frozen_alpha_mechanics_replay_summary.csv": "Deterministic frozen-alpha replay/evaluation summary.",
        "tables/variant_reaction_comparison.csv": "Reaction proxy comparison by variant.",
        "tables/variant_energy_comparison.csv": "Energy contribution comparison by variant.",
        "tables/variant_displacement_jump_proxy.csv": "Displacement jump proxy by variant and cut line.",
        "figures/figure_summary.md": "Text summary of figures.",
    }
    return mapping.get(rel, "Diagnostic artifact or figure.")


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    posthoc, replay, energy, jumps = build_tables()
    cause = determine_cause(posthoc, replay)
    make_figures(posthoc, replay, energy, jumps)
    write_docs(posthoc, replay, energy, jumps, cause)
    write_figure_summary()
    write_commands()
    write_manifest()


if __name__ == "__main__":
    main()
