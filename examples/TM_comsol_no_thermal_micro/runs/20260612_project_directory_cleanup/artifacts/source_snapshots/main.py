from config import *

from field_computation import FieldComputation
from alpha_initialization import apply_alpha_init_intact
from construct_model import construct_model
from train_mixed_tm import train_mixed_tm
from postprocess_results import run_results_postprocess


pffmodel, matprop, network = construct_model(
    PFF_model_dict,
    mat_prop_dict,
    network_dict,
    domain_extrema,
    device,
)
if training_dict.get("alpha_init_intact", False):
    apply_alpha_init_intact(network)
field_comp = FieldComputation(
    net=network,
    domain_extrema=domain_extrema,
    lmbda=torch.tensor([0.0], device=device),
    theta=loading_angle,
    alpha_constraint=numr_dict["alpha_constraint"],
    top_u_mode=training_dict.get("top_u_mode", "free"),
    coord_normalization=training_dict.get("coord_normalization", "unit_box"),
)
field_comp.net = field_comp.net.to(device)
field_comp.domain_extrema = field_comp.domain_extrema.to(device)
field_comp.theta = field_comp.theta.to(device)


if __name__ == "__main__":
    train_mixed_tm(
        field_comp,
        disp,
        pffmodel,
        matprop,
        crack_dict,
        numr_dict,
        optimizer_dict,
        training_dict,
        coarse_mesh_file,
        fine_mesh_file,
        device,
        trainedModel_path,
        intermediateModel_path,
        writer,
    )
    run_results_postprocess(
        model_path,
        results_path,
        out_dir=postprocess_curve_dir,
        figure_dir=postprocess_figure_dir,
        run_label=run_id,
        device="cpu",
    )
