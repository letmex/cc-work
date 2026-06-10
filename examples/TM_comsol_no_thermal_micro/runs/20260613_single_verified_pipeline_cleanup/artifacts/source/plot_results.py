import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


FIELD_SPECS = (
    {"key": "alpha", "label": "Damage alpha", "filename": "alpha", "nodal": True, "cmap": "viridis", "vmin": 0.0, "vmax": 1.0},
    {"key": "u", "label": "u displacement [mm]", "filename": "u", "nodal": True, "cmap": "coolwarm"},
    {"key": "v", "label": "v displacement [mm]", "filename": "v", "nodal": True, "cmap": "coolwarm"},
    {"key": "disp", "label": "Displacement magnitude [mm]", "filename": "disp", "nodal": True, "cmap": "viridis", "vmin": 0.0},
    {"key": "HI", "label": "HI", "filename": "HI", "nodal": False, "cmap": "magma"},
    {"key": "HII", "label": "HII", "filename": "HII", "nodal": False, "cmap": "magma"},
    {"key": "He", "label": "He", "filename": "He", "nodal": False, "cmap": "magma"},
    {"key": "He_history", "label": "He history", "filename": "He_history", "nodal": False, "cmap": "magma"},
    {"key": "He_current", "label": "He current", "filename": "He_current", "nodal": False, "cmap": "magma"},
    {"key": "mechanics_drive", "label": "Mechanics drive", "filename": "mechanics_drive", "nodal": False, "cmap": "magma"},
)

DOMAIN_X_MM = (0.0, 0.01)
DOMAIN_Y_MM = (0.0, 0.01)
SPECIMEN_HEIGHT_MM = 0.01
COMSOL_SECTION_AREA_MM2 = 1.0e-5


def default_plot_out_dir(label):
    return ROOT / "outputs" / "figures" / _safe_label(label)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate COMSOL micro no-thermal TM result plots.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result-dir", type=Path, help="Directory containing fields_mixed_tm_step_*.npz files.")
    group.add_argument("--suffix", help="Unique suffix of a result directory under ./outputs/results.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--specimen-height", type=float, default=SPECIMEN_HEIGHT_MM)
    parser.add_argument("--section-area", type=float, default=COMSOL_SECTION_AREA_MM2)
    parser.add_argument("--stress-strain-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_result_dir(suffix):
    matches = []
    for root in (ROOT / "outputs" / "results", ROOT / "results"):
        if root.exists():
            matches.extend([p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)])
    matches = sorted(matches)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result dir ending with {suffix!r}, found {len(matches)}.")
    return matches[0]


def field_paths(result_dir):
    paths = sorted(Path(result_dir).glob("fields_mixed_tm_step_*.npz"))
    if paths:
        return paths
    raise RuntimeError(f"No supported field npz files found in {result_dir}.")


def infer_short_run_label(path):
    text = Path(path).name
    seed = _infer_seed_from_name(text)
    load = None
    for candidate in ("D0020", "D0040"):
        if candidate in text:
            load = candidate
            break
    if seed is not None and load is not None:
        return f"seed{seed}_{load}"
    if seed is not None:
        return f"seed{seed}"
    suffix = text.split("_gradient_numerical_")[-1] if "_gradient_numerical_" in text else text
    return suffix[:64]


def _safe_label(value):
    text = str(value).strip()
    text = "_".join(part for part in re.split(r"[_\s]+", text) if part) or "results"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _ascii_metadata_path(path):
    text = str(path)
    if all(ord(ch) < 128 for ch in text):
        return text
    name = Path(path).name
    safe_name = name if all(ord(ch) < 128 for ch in name) else "<non-ascii filename>"
    return f"{safe_name} (non-ASCII parent path omitted)"


def _to_float_array(data, key):
    return np.asarray(data[key], dtype=float)


def field_values(data, key):
    if key == "disp":
        return np.sqrt(_to_float_array(data, "u") ** 2 + _to_float_array(data, "v") ** 2)
    return _to_float_array(data, key)


def _field_limits(values, vmin=None, vmax=None):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return vmin, vmax
    lo = float(np.nanmin(finite)) if vmin is None else vmin
    hi = float(np.nanmax(finite)) if vmax is None else vmax
    if np.isclose(lo, hi):
        pad = 1.0 if np.isclose(lo, 0.0) else 0.05 * abs(lo)
        lo -= pad
        hi += pad
    return lo, hi


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def _draw_domain(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*DOMAIN_X_MM)
    ax.set_ylim(*DOMAIN_Y_MM)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_field(data, spec, out_path, run_label="", dpi=300, ax=None, add_colorbar=True):
    triang = mtri.Triangulation(data["x"], data["y"], data["triangles"].astype(int))
    values = field_values(data, spec["key"])
    vmin, vmax = _field_limits(values, spec.get("vmin"), spec.get("vmax"))
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(4.4, 3.9), dpi=dpi)
    else:
        fig = ax.figure
    if spec["nodal"]:
        artist = ax.tripcolor(
            triang,
            values,
            shading="gouraud",
            cmap=spec["cmap"],
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
    else:
        artist = ax.tripcolor(
            triang,
            facecolors=values,
            shading="flat",
            cmap=spec["cmap"],
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
    title = spec["label"] if not run_label else f"{spec['label']}: {run_label}"
    ax.set_title(title)
    _draw_domain(ax)
    if add_colorbar:
        cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
        cbar.ax.tick_params(length=2.5, width=0.7)
    if created:
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return artist


def plot_field_panel(data, out_path, run_label="", dpi=300):
    specs = [spec for spec in FIELD_SPECS if spec["key"] == "disp" or spec["key"] in data]
    n_cols = min(4, max(1, len(specs)))
    n_rows = int(np.ceil(len(specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.35 * n_cols, 3.2 * n_rows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, spec in zip(axes, specs):
        plot_field(data, spec, out_path=None, run_label="", dpi=dpi, ax=ax, add_colorbar=True)
        ax.set_title(spec["label"])
    for ax in axes[len(specs):]:
        fig.delaxes(ax)
    fig.suptitle(run_label, y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _as_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _infer_seed_from_name(path):
    text = str(path)
    patterns = (
        r"(?:^|[_\-])Seed[_\-]?(\d+)(?:[_\-]|$)",
        r"(?:^|[_\-])seed[_\-]?(\d+)(?:[_\-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _candidate_stress_strain_csvs(result_dir, out_dir):
    roots = [
        Path(result_dir),
        Path(result_dir) / "curves",
        Path(result_dir) / "tables",
        Path(out_dir),
        Path(out_dir) / "curves",
        Path(out_dir) / "tables",
    ]
    seen = set()
    for root in roots:
        candidate = root / "stress_strain_by_step.csv"
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            yield candidate


def _read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalise_stress_strain_rows(rows):
    normalised = []
    for row in rows:
        item = dict(row)
        if "Delta_mm" not in item and "Delta" in item:
            item["Delta_mm"] = item["Delta"]
        if "strain" not in item and "nominal_strain" in item:
            item["strain"] = item["nominal_strain"]
        item["stress_strain_primary_metric"] = item.get(
            "stress_strain_primary_metric", "nominal_stress_energy_MPa"
        )
        item["reaction_metric_status"] = item.get("reaction_metric_status", "energy_conjugate")
        normalised.append(item)
    return sorted(normalised, key=lambda row: (_as_int(row.get("step"), 0), _as_float(row.get("Delta_mm"), 0.0)))


def load_stress_strain_rows(path, result_dir, seed=None):
    rows = _read_csv_dicts(path)
    if not rows:
        raise RuntimeError(f"Stress-strain CSV is empty: {path}")
    seeds = sorted({row_seed for row_seed in (_as_int(row.get("seed")) for row in rows) if row_seed is not None})
    selected_seed = seed if seed is not None else _infer_seed_from_name(result_dir)
    if selected_seed is not None and seeds:
        rows = [row for row in rows if _as_int(row.get("seed")) == selected_seed]
        if not rows:
            raise RuntimeError(f"No rows for seed {selected_seed} in stress-strain CSV: {path}")
    elif len(seeds) > 1:
        raise RuntimeError(
            "Stress-strain CSV contains multiple seeds; pass --seed "
            "or use a result directory name containing Seed_<n>."
        )
    return _normalise_stress_strain_rows(rows), selected_seed


def resolve_stress_strain_table(result_dir, out_dir, stress_strain_csv=None, seed=None):
    if stress_strain_csv is not None:
        path = Path(stress_strain_csv)
        if not path.exists():
            raise RuntimeError(f"Stress-strain CSV does not exist: {path}")
        rows, selected_seed = load_stress_strain_rows(path, result_dir, seed=seed)
        return rows, path, selected_seed
    for path in _candidate_stress_strain_csvs(result_dir, out_dir):
        rows, selected_seed = load_stress_strain_rows(path, result_dir, seed=seed)
        return rows, path, selected_seed
    return None, None, seed


def unavailable_reaction_rows(paths, specimen_height=1.0):
    rows = []
    for path in paths:
        data = np.load(path)
        delta = float(data["displacement_mm"])
        strain = delta / specimen_height if specimen_height != 0.0 else np.nan
        rows.append(
            {
                "step": int(path.stem.split("_")[-1]),
                "Delta_mm": delta,
                "strain": strain,
                "stress_strain_primary_metric": "reaction_metric_unavailable",
                "reaction_metric_status": "reaction_metric_unavailable",
                "is_energy_conjugate": False,
                "reaction_N_energy": np.nan,
                "nominal_stress_energy_MPa": np.nan,
            }
        )
    return rows


def write_stress_strain_csv(rows, out_path):
    preferred = [
        "seed",
        "step",
        "Delta",
        "Delta_mm",
        "nominal_strain",
        "strain",
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
    keys = []
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _row_series(rows, *keys):
    values = []
    for row in rows:
        value = np.nan
        for key in keys:
            if key in row:
                value = _as_float(row.get(key))
                break
        values.append(value)
    return np.asarray(values, dtype=float)


def _has_finite(values):
    return bool(np.any(np.isfinite(values)))


def plot_stress_strain(rows, out_path, dpi=300):
    strain = _row_series(rows, "nominal_strain", "strain")
    stress_primary = _row_series(rows, "nominal_stress_energy_MPa")
    stress_virtual = _row_series(rows, "nominal_stress_virtual_work_MPa")
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=dpi)
    if _has_finite(stress_primary):
        ax.plot(
            strain,
            stress_primary,
            color="#D55E00",
            marker="s",
            markersize=3,
            linewidth=1.5,
            label="Energy-conjugate",
        )
    else:
        ax.set_title("Reaction metric unavailable")
    if _has_finite(stress_virtual):
        ax.plot(
            strain,
            stress_virtual,
            color="#CC79A7",
            linestyle=":",
            linewidth=1.2,
            label="Virtual-work check",
        )
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Engineering stress [MPa]")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_reaction_strain(rows, out_path, dpi=300):
    strain = _row_series(rows, "nominal_strain", "strain")
    reaction_primary = _row_series(rows, "reaction_N_energy")
    reaction_virtual = _row_series(rows, "reaction_N_virtual_work")
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=dpi)
    if _has_finite(reaction_primary):
        ax.plot(
            strain,
            reaction_primary,
            color="#D55E00",
            marker="s",
            markersize=3,
            linewidth=1.5,
            label="Energy-conjugate",
        )
    else:
        ax.set_title("Reaction metric unavailable")
    if _has_finite(reaction_virtual):
        ax.plot(
            strain,
            reaction_virtual,
            color="#CC79A7",
            linestyle=":",
            linewidth=1.2,
            label="Virtual-work check",
        )
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Reaction [N]")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_result_figures(
    result_dir,
    out_dir,
    run_label=None,
    specimen_height=1.0,
    section_area=1.0,
    stress_strain_csv=None,
    seed=None,
    dpi=300,
):
    del section_area
    apply_publication_style()
    result_dir = Path(result_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = field_paths(result_dir)
    final_data = np.load(paths[-1])
    label = _safe_label(run_label or infer_short_run_label(result_dir))

    generated = []
    for spec in FIELD_SPECS:
        if spec["key"] == "disp" or spec["key"] in final_data:
            out_path = out_dir / f"final_{spec['filename']}_{label}.png"
            plot_field(final_data, spec, out_path, run_label=run_label or result_dir.name, dpi=dpi)
            generated.append(out_path)

    panel_path = out_dir / f"final_fields_panel_{label}.png"
    plot_field_panel(final_data, panel_path, run_label=run_label or result_dir.name, dpi=dpi)
    generated.append(panel_path)

    table_rows, table_csv, selected_seed = resolve_stress_strain_table(
        result_dir,
        out_dir,
        stress_strain_csv=stress_strain_csv,
        seed=seed,
    )
    rows = table_rows if table_rows is not None else unavailable_reaction_rows(paths, specimen_height=specimen_height)
    csv_path = out_dir / f"stress_strain_data_{label}.csv"
    write_stress_strain_csv(rows, csv_path)
    generated.append(csv_path)
    if table_csv is not None:
        metadata_path = out_dir / f"stress_strain_source_{label}.txt"
        metadata_path.write_text(
            "\n".join(
                [
                    "source=energy_conjugate",
                    f"stress_strain_csv={_ascii_metadata_path(table_csv)}",
                    f"selected_seed={selected_seed}",
                    "primary_metric=nominal_stress_energy_MPa",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        generated.append(metadata_path)

    stress_path = out_dir / f"stress_strain_{label}.png"
    plot_stress_strain(rows, stress_path, dpi=dpi)
    generated.append(stress_path)

    reaction_path = out_dir / f"reaction_strain_{label}.png"
    plot_reaction_strain(rows, reaction_path, dpi=dpi)
    generated.append(reaction_path)
    return generated


def main():
    args = parse_args()
    result_dir = args.result_dir if args.result_dir is not None else find_result_dir(args.suffix)
    label = _safe_label(args.run_label or infer_short_run_label(result_dir))
    out_dir = args.out_dir or default_plot_out_dir(label)
    generated = make_result_figures(
        result_dir,
        out_dir,
        run_label=args.run_label,
        specimen_height=args.specimen_height,
        section_area=args.section_area,
        stress_strain_csv=args.stress_strain_csv,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"result_dir={result_dir}")
    print(f"out_dir={out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
