from pathlib import Path

import numpy as np


CRITICAL_LOAD_SCHEDULE = np.array(
    [
        1.0e-6,
        5.0e-6,
        1.0e-5,
        1.5e-5,
        2.0e-5,
        2.5e-5,
        3.0e-5,
        3.5e-5,
        4.0e-5,
        4.5e-5,
        4.7e-5,
        4.9e-5,
        5.1e-5,
        5.3e-5,
        5.5e-5,
        5.7e-5,
        5.9e-5,
        6.1e-5,
        6.3e-5,
        6.5e-5,
        6.7e-5,
        6.9e-5,
        7.1e-5,
        7.3e-5,
        7.5e-5,
        7.7e-5,
        8.0e-5,
        8.5e-5,
        9.0e-5,
        9.5e-5,
        1.0e-4,
    ],
    dtype=float,
)


def default_displacement_schedule():
    disp = np.concatenate(
        [
            np.linspace(1.0e-6, 4.0e-5, 9),
            np.linspace(4.2e-5, 8.0e-5, 20),
            np.linspace(8.5e-5, 1.0e-4, 4),
        ]
    )
    disp = np.unique(np.round(disp, 10))
    disp.sort()
    return disp


def _read_schedule_file(path):
    values = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.replace(",", " ").split():
            values.append(float(part))
    if not values:
        raise ValueError(f"Load schedule file is empty: {path}")
    disp = np.asarray(values, dtype=float)
    disp = np.unique(np.round(disp, 10))
    disp.sort()
    return disp


def build_displacement_schedule(args, auto_smoke=False):
    if getattr(args, "load_schedule_file", None):
        disp = _read_schedule_file(args.load_schedule_file)
        label = "file"
    elif getattr(args, "custom_load_schedule", None) == "critical":
        disp = CRITICAL_LOAD_SCHEDULE.copy()
        label = "critical"
    elif getattr(args, "delta_max", None) is not None:
        steps = args.max_steps if args.max_steps is not None else 24
        if steps < 1:
            raise ValueError("--max-steps must be positive when --delta-max is used")
        disp = np.linspace(1.0e-6, float(args.delta_max), steps)
        disp = np.unique(np.round(disp, 10))
        disp.sort()
        label = "delta_max_linear"
    else:
        disp = default_displacement_schedule()
        label = "default"
        if args.max_steps is not None:
            disp = disp[: args.max_steps]
        elif auto_smoke:
            disp = disp[:1]

    if label != "default" and args.max_steps is not None and len(disp) > args.max_steps:
        disp = disp[: args.max_steps]
    return disp, label
