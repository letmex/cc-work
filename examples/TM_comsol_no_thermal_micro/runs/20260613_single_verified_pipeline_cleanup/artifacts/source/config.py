import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from load_schedule import build_displacement_schedule


SPECIMEN_SIZE_MM = 0.01
NOTCH_LENGTH_MM = 0.005
NOTCH_CENTER_Y_MM = 0.005
NOTCH_HALF_WIDTH = 5.0e-5
MIDDLE_LOWER_Y_MM = 0.003
MIDDLE_UPPER_Y_MM = 0.007
REFINEMENT_STRATEGY = "comsol_mphtxt_physical_groups"
SURFACE_BLOCKS = ("lower", "middle", "upper")
MESH_SPECS = {
    "smoke": {"filename": "geo_coarse_with_groups_mm.msh"},
    "coarse": {"filename": "geo_coarse_with_groups_mm.msh"},
    "fine": {"filename": "geo_coarse_with_groups_mm.msh"},
}

DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_CHECKPOINT_ROOT = "outputs/checkpoints"
DEFAULT_RESULTS_ROOT = "outputs/results"
DEFAULT_FIGURE_ROOT = "outputs/figures"
DEFAULT_CURVE_ROOT = "outputs/curves"
DEFAULT_LOG_ROOT = "outputs/logs"
DEFAULT_DEBUG_ROOT = "outputs/debug"


def _resolve_path(value, base_dir):
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _parse_args():
    def _str_to_bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")

    parser = argparse.ArgumentParser(
        description="COMSOL no-thermal micro TM route with history mechanics and tm_source split."
    )
    parser.add_argument("hidden_layers", nargs="?", type=int, default=8)
    parser.add_argument("neurons", nargs="?", type=int, default=400)
    parser.add_argument("seed", nargs="?", type=int, default=1)
    parser.add_argument("activation", nargs="?", default="TrainableReLU")
    parser.add_argument("init_coeff", nargs="?", type=float, default=3.0)
    parser.add_argument("--l0", type=float, default=float(os.getenv("TM_COMSOL_MICRO_L0", 1.5e-4)))
    parser.add_argument("--n-rprop", type=int, default=None)
    parser.add_argument("--n-lbfgs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--delta-max", type=float, default=None)
    parser.add_argument(
        "--custom-load-schedule",
        choices=["critical"],
        default=None,
        help="Named displacement schedule. 'critical' densifies 0.0045-0.0080.",
    )
    parser.add_argument(
        "--load-schedule-file",
        default=None,
        help="Text/CSV file containing displacement values.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--GcII",
        type=float,
        default=None,
        help="Optional mode-II toughness. Defaults to COMSOL 2*(1+nu)*(0.60)^2*Gf0.",
    )
    parser.add_argument(
        "--GcII-factor",
        type=float,
        default=None,
        help="Multiplier for GcII relative to COMSOL Gf0 when --GcII is not set.",
    )
    parser.add_argument(
        "--tm-eps-r",
        type=float,
        default=float(os.getenv("TM_COMSOL_MICRO_TM_EPS_R", 1.0e-5)),
        help="Regularization eps_r used by tm_source split.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to the run directory.",
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("TM_COMSOL_MICRO_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        help="Managed output root. Relative paths resolve under the example directory.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Advanced override for checkpoint/model output directory.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Advanced override for field/result output directory.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Advanced override for TensorBoard log directory.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Advanced override for postprocess figure directory.",
    )
    parser.add_argument(
        "--curve-dir",
        default=None,
        help="Advanced override for postprocess curve/table directory.",
    )
    parser.add_argument(
        "--top-u-mode",
        choices=["fixed", "free"],
        default=os.getenv("TM_COMSOL_MICRO_TOP_U_MODE", "free"),
        help="Advanced displacement ansatz override. Default free is the verified TM route.",
    )
    parser.add_argument(
        "--coord-normalization",
        choices=["none", "unit_box"],
        default=os.getenv("TM_COMSOL_MICRO_COORD_NORMALIZATION", "unit_box"),
        help="Advanced NN coordinate-input override. Default unit_box maps x,y to [-1,1].",
    )
    parser.add_argument(
        "--save-step-checkpoints",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="Save composite per-loading-step checkpoints. Default true; pass false to disable.",
    )
    parser.add_argument(
        "--checkpoint-every-step",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="When --save-step-checkpoints is true, save every loading step.",
    )
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()

PATH_ROOT = Path(__file__).resolve().parent
LOCAL_SOURCE = PATH_ROOT / "source"
if str(LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

COMSOL_E_KN_PER_MM2 = 81.5
COMSOL_NU = 0.38
Gc = 2.4e-6
GcI = Gc
if args.GcII is not None:
    GcII = float(args.GcII)
    GcII_factor = GcII / Gc
else:
    GcII_factor = (
        float(args.GcII_factor)
        if args.GcII_factor is not None
        else 2.0 * (1.0 + COMSOL_NU) * (0.60**2)
    )
    GcII = Gc * GcII_factor
if GcII <= 0.0:
    raise ValueError("--GcII or --GcII-factor must produce a positive GcII")
if args.tm_eps_r < 0.0:
    raise ValueError("--tm-eps-r must be non-negative")

history_mode = "mixedH_TM"
mixed_split_mode = "tm_source"
mixed_mechanics_mode = "history"
geometry_mode = "comsol_micro_gap"
initial_phase_field_crack_enabled = False
initial_phase_field_crack_length = 0.0
strict_reproduction = True
phasefield_notch_enabled = False
notch_profile = "source_exponential"
legacy_alpha_irreversibility = False

small_network = args.hidden_layers <= 2 and args.neurons <= 50
explicit_multi_step = (
    (args.max_steps is not None and args.max_steps > 1)
    or args.delta_max is not None
    or args.custom_load_schedule is not None
    or args.load_schedule_file is not None
)
auto_smoke = (
    args.smoke
    or os.getenv("TM_COMSOL_MICRO_SMOKE", "0") == "1"
    or (not args.full and not explicit_multi_step and small_network)
)

network_dict = {
    "model_type": "MLP",
    "hidden_layers": args.hidden_layers,
    "neurons": args.neurons,
    "seed": args.seed,
    "activation": args.activation,
    "init_coeff": args.init_coeff,
}

optimizer_dict = {
    "weight_decay": 1e-5,
    "n_epochs_RPROP": args.n_rprop if args.n_rprop is not None else (10 if auto_smoke else 10000),
    "n_epochs_LBFGS": args.n_lbfgs if args.n_lbfgs is not None else (0 if auto_smoke else 1),
    "optim_rel_tol_pretrain": 1e-6,
    "optim_rel_tol": 5e-7,
}

training_dict = {
    "save_model_every_n": 100,
    "notch_half_width": NOTCH_HALF_WIDTH,
    "eta_residual": 1.0e-5,
    "history_mode": history_mode,
    "GcII": GcII,
    "GcII_factor": GcII_factor,
    "mixed_mode_ratio": Gc / GcII,
    "mixed_split_mode": mixed_split_mode,
    "mixed_mechanics_mode": mixed_mechanics_mode,
    "tm_eps_r": float(args.tm_eps_r),
    "top_u_mode": args.top_u_mode,
    "coord_normalization": args.coord_normalization,
    "save_step_checkpoints": bool(args.save_step_checkpoints),
    "checkpoint_every_step": bool(args.checkpoint_every_step),
}

numr_dict = {"alpha_constraint": "nonsmooth", "gradient_type": "numerical"}
PFF_model_dict = {"PFF_model": "AT2", "se_split": "volumetric", "tol_ir": 5e-3}

mat_prop_dict = {
    "mat_E": COMSOL_E_KN_PER_MM2,
    "mat_nu": COMSOL_NU,
    "w1": Gc / args.l0,
    "l0": args.l0,
}

domain_extrema = torch.tensor([[0.0, SPECIMEN_SIZE_MM], [0.0, SPECIMEN_SIZE_MM]])
loading_angle = torch.tensor([np.pi / 2])
crack_dict = {
    "x_init": [0.0],
    "y_init": [NOTCH_CENTER_Y_MM],
    "L_crack": [0.0],
    "angle_crack": [0.0],
}

disp, load_schedule_label = build_displacement_schedule(args, auto_smoke=auto_smoke)

if auto_smoke:
    coarse_mesh_file = str(PATH_ROOT / MESH_SPECS["smoke"]["filename"])
    fine_mesh_file = str(PATH_ROOT / MESH_SPECS["smoke"]["filename"])
else:
    coarse_mesh_file = str(PATH_ROOT / MESH_SPECS["coarse"]["filename"])
    fine_mesh_file = str(PATH_ROOT / MESH_SPECS["fine"]["filename"])

if args.full:
    mode_label = "full"
elif auto_smoke:
    mode_label = "smoke"
else:
    mode_label = "medium"

l0_label = f"{args.l0:g}"
tm_eps_r_label = f"_tmEpsR_{args.tm_eps_r:g}".replace(".", "p").replace("-", "m")
top_u_label = "" if args.top_u_mode == "free" else "_topUfixed"
coord_norm_label = "" if args.coord_normalization == "unit_box" else "_coordRaw"
optimizer_label = ""
if args.n_rprop is not None or args.n_lbfgs is not None:
    optimizer_label = (
        f"_rprop_{optimizer_dict['n_epochs_RPROP']}"
        f"_lbfgs_{optimizer_dict['n_epochs_LBFGS']}"
    )
run_suffix = args.run_suffix.strip().replace("-", "_").replace(" ", "_")
run_suffix_label = f"_{run_suffix}" if run_suffix else ""

model_name = (
    f"{mode_label}_hl_{network_dict['hidden_layers']}"
    f"_Neurons_{network_dict['neurons']}"
    f"_activation_{network_dict['activation']}"
    f"_coeff_{network_dict['init_coeff']}"
    f"_Seed_{network_dict['seed']}"
    f"_PFFmodel_AT2"
    f"_l0_{l0_label}"
    f"_comsolMicroNoThermal_TM_verified"
    f"{tm_eps_r_label}"
    f"{top_u_label}"
    f"{coord_norm_label}"
    f"{optimizer_label}"
    f"_gradient_{numr_dict['gradient_type']}"
    f"{run_suffix_label}"
)
run_id = run_suffix if run_suffix else model_name
output_root_path = _resolve_path(args.output_root, PATH_ROOT)
checkpoint_root_path = output_root_path / "checkpoints"
results_root_path = output_root_path / "results"
log_root_path = output_root_path / "logs"
figure_root_path = output_root_path / "figures"
curve_root_path = output_root_path / "curves"
debug_root_path = output_root_path / "debug"

model_path = _resolve_path(args.model_dir, PATH_ROOT) if args.model_dir else checkpoint_root_path / run_id
model_path.mkdir(parents=True, exist_ok=True)
trainedModel_path = model_path / Path("best_models")
trainedModel_path.mkdir(parents=True, exist_ok=True)
intermediateModel_path = model_path / Path("intermediate_models")
intermediateModel_path.mkdir(parents=True, exist_ok=True)
results_path = _resolve_path(args.result_dir, PATH_ROOT) if args.result_dir else results_root_path / run_id
results_path.mkdir(parents=True, exist_ok=True)
log_path = _resolve_path(args.log_dir, PATH_ROOT) if args.log_dir else log_root_path / run_id
log_path.mkdir(parents=True, exist_ok=True)
postprocess_figure_dir = _resolve_path(args.figure_dir, PATH_ROOT) if args.figure_dir else results_path / "figures"
postprocess_curve_dir = _resolve_path(args.curve_dir, PATH_ROOT) if args.curve_dir else results_path / "curves"
training_dict["results_path"] = results_path
training_dict["run_id"] = run_id
training_dict["output_root"] = str(output_root_path)
np.save(results_path / "displacement_list.npy", disp)
np.savetxt(
    results_path / "displacement_list.csv",
    np.column_stack([np.arange(len(disp)), disp]),
    delimiter=",",
    header="step,displacement_mm",
    comments="",
)

with open(model_path / Path("model_settings.txt"), "w", encoding="utf-8") as file:
    file.write("case: TM_comsol_no_thermal_micro")
    file.write(f"\nmode: {mode_label}")
    file.write(f"\nrun_id: {run_id}")
    file.write(f"\nmodel_name: {model_name}")
    file.write(f"\nrun_suffix: {run_suffix}")
    file.write("\nroute: mixedH_TM + tm_source + history")
    file.write(f"\nstrict_reproduction: {strict_reproduction}")
    file.write(f"\ngeometry_mode: {geometry_mode}")
    file.write("\nnotch_geometry: COMSOL explicit thin geometric notch gap")
    file.write(f"\nnotch_half_width_mm: {NOTCH_HALF_WIDTH:g}")
    file.write(f"\nnotch_length_mm: {NOTCH_LENGTH_MM:g}")
    file.write(f"\nnotch_center_y_mm: {NOTCH_CENTER_Y_MM:g}")
    file.write(f"\nhistory_mode: {history_mode}")
    file.write(f"\nmixed_split_mode: {mixed_split_mode}")
    file.write(f"\nmixed_mechanics_mode: {mixed_mechanics_mode}")
    file.write(f"\ntm_eps_r: {training_dict['tm_eps_r']}")
    file.write(f"\ntop_u_mode: {training_dict['top_u_mode']}")
    file.write(f"\ncoord_normalization: {training_dict['coord_normalization']}")
    file.write(f"\nsave_step_checkpoints: {training_dict['save_step_checkpoints']}")
    file.write(f"\ncheckpoint_every_step: {training_dict['checkpoint_every_step']}")
    file.write(f"\neta_residual: {training_dict['eta_residual']}")
    file.write(f"\nGcII_arg: {args.GcII}")
    file.write(f"\nGcII_factor: {GcII_factor}")
    file.write(f"\nmixed_mode_ratio_Gf0_over_GcII: {training_dict['mixed_mode_ratio']}")
    file.write(f"\nload_schedule: {load_schedule_label}")
    file.write(f"\ndelta_max_arg: {args.delta_max}")
    file.write(f"\ncustom_load_schedule: {args.custom_load_schedule}")
    file.write(f"\nload_schedule_file: {args.load_schedule_file}")
    file.write(f"\ndisplacement_list: {','.join(format(float(x), '.7g') for x in disp)}")
    file.write(f"\nmesh_refinement_strategy: {REFINEMENT_STRATEGY}")
    file.write(f"\nsurface_blocks: {','.join(SURFACE_BLOCKS)}")
    file.write(f"\nmiddle_lower_y_mm: {MIDDLE_LOWER_Y_MM:g}")
    file.write(f"\nmiddle_upper_y_mm: {MIDDLE_UPPER_Y_MM:g}")
    file.write("\nunit_strategy: COMSOL no-thermal physical kN-mm units; mesh converted from m to mm")
    file.write(f"\nhidden_layers: {network_dict['hidden_layers']}")
    file.write(f"\nneurons: {network_dict['neurons']}")
    file.write(f"\nseed: {network_dict['seed']}")
    file.write(f"\nactivation: {network_dict['activation']}")
    file.write(f"\ncoeff: {network_dict['init_coeff']}")
    file.write("\nPFF_model: AT2")
    file.write(f"\nse_split: {PFF_model_dict['se_split']}")
    file.write(f"\ngradient_type: {numr_dict['gradient_type']}")
    file.write(f"\nn_epochs_RPROP: {optimizer_dict['n_epochs_RPROP']}")
    file.write(f"\nn_epochs_LBFGS: {optimizer_dict['n_epochs_LBFGS']}")
    file.write(f"\nmat_E_kN_per_mm2: {mat_prop_dict['mat_E']}")
    file.write(f"\nmat_nu: {mat_prop_dict['mat_nu']}")
    file.write(f"\nGc_kN_per_mm: {Gc}")
    file.write(f"\nGcI_kN_per_mm: {GcI}")
    file.write(f"\nGcII_kN_per_mm: {GcII}")
    file.write(f"\nl0_mm: {mat_prop_dict['l0']}")
    file.write(f"\nw1_kN_per_mm2: {mat_prop_dict['w1']}")
    file.write(f"\ncoarse_mesh_file: {coarse_mesh_file}")
    file.write(f"\nfine_mesh_file: {fine_mesh_file}")
    file.write(f"\nload_steps: {len(disp)}")
    file.write(f"\noutput_root: {output_root_path}")
    file.write(f"\ncheckpoint_root: {checkpoint_root_path}")
    file.write(f"\nresults_root: {results_root_path}")
    file.write(f"\nfigure_root: {figure_root_path}")
    file.write(f"\ncurve_root: {curve_root_path}")
    file.write(f"\nlog_root: {log_root_path}")
    file.write(f"\ndebug_root: {debug_root_path}")
    file.write(f"\nmodel_path: {model_path}")
    file.write(f"\nresults_path: {results_path}")
    file.write(f"\npostprocess_curve_dir: {postprocess_curve_dir}")
    file.write(f"\npostprocess_figure_dir: {postprocess_figure_dir}")
    file.write("\nreaction_force_method: energy-conjugate checkpoint reaction")
    file.write("\nreaction_force_unit_thickness: 1 mm")
    file.write(f"\ndevice: {device}")

writer = SummaryWriter(str(log_path))
