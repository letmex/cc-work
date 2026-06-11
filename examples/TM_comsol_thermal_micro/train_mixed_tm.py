import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from compute_energy_mixed_tm import compute_mixed_tm_energy
from history_field_mixed_tm import (
    append_mixed_tm_summary,
    commit_mixed_tm_history_from_model,
    initialize_mixed_history_fields,
    save_mixed_tm_step_fields,
)
from input_data_from_mesh import prep_input_data
from optim import get_optimizer


def _thermal_energy_kwargs(training_dict):
    training_dict = training_dict or {}
    return {
        "thermal_temperature": training_dict.get("thermal_temperature", None),
        "thermal_delta_T": training_dict.get("thermal_delta_T", None),
        "thermal_mode": training_dict.get("thermal_mode", "off"),
        "thermal_delta_T0": training_dict.get("thermal_delta_T0", 0.0),
        "thermal_grad_y": training_dict.get("thermal_grad_y", 0.0),
        "thermal_y0": training_dict.get("thermal_y0", 0.0),
        "thermal_alpha_T": training_dict.get("thermal_alpha_T", 18.9e-6),
        "thermal_Tref": training_dict.get("thermal_Tref", 273.15),
    }


class EarlyStopping:
    def __init__(self, tol_steps=10, min_delta=1e-3, device="cpu"):
        self.tol_steps = torch.tensor([tol_steps], dtype=torch.int, device=device)
        self.min_delta = torch.tensor([min_delta], dtype=torch.float, device=device)
        self.counter = torch.tensor([0], dtype=torch.int, device=device)
        self.early_stop = False

    def __call__(self, train_loss, train_loss_prev):
        delta = torch.abs(train_loss - train_loss_prev) / (
            torch.abs(train_loss_prev) + np.finfo(float).eps
        )
        if delta > self.min_delta:
            self.counter = self.counter * 0
        else:
            self.counter += 1
            if self.counter >= self.tol_steps:
                self.early_stop = True


def fit_mixed_tm(
    field_comp,
    training_set_collocation,
    T_conn,
    area_T,
    history_old,
    matprop,
    pffmodel,
    weight_decay,
    num_epochs,
    optimizer,
    intermediateModel_path=None,
    writer=None,
    training_dict=None,
):
    training_dict = training_dict or {}
    eta_residual = training_dict.get("eta_residual", 1.0e-8)
    gcII = training_dict.get("GcII", None)
    gcII_factor = training_dict.get("GcII_factor", 1.0)
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    thermal_kwargs = _thermal_energy_kwargs(training_dict)
    loss_data = []

    for epoch in range(num_epochs):
        loop = tqdm(training_set_collocation, miniters=25)
        for _, (inp_train, _) in enumerate(loop):

            def closure():
                optimizer.zero_grad()
                if T_conn is None:
                    inp_train.requires_grad = True
                u, v, alpha = field_comp.fieldCalculation(inp_train)
                loss_E_el, loss_E_d, _ = compute_mixed_tm_energy(
                    inp_train,
                    u,
                    v,
                    alpha,
                    history_old["HI"],
                    history_old["HII"],
                    matprop,
                    pffmodel,
                    area_T,
                    T_conn,
                    eta_residual=eta_residual,
                    gcII=gcII,
                    gcII_factor=gcII_factor,
                    tm_eps_r=tm_eps_r,
                    alpha_old=history_old.get("alpha_old"),
                    **thermal_kwargs,
                )
                loss_var = torch.log10(loss_E_el + loss_E_d)

                loss_reg = 0.0
                if weight_decay != 0:
                    for name, param in field_comp.net.named_parameters():
                        if "weight" in name:
                            loss_reg += torch.sum(param**2)

                loss = loss_var + weight_decay * loss_reg
                if writer is not None:
                    writer.add_scalars(
                        "mixedH_TM_U_p_" + str(field_comp.lmbda.item()),
                        {"loss": loss.item(), "loss_E": loss_var.item()},
                        epoch,
                    )
                loop.set_description(
                    f"mixedH_TM U_p: {field_comp.lmbda}, Epoch [{epoch}/{num_epochs}]"
                )
                loop.set_postfix(loss=loss.item(), loss_E=loss_var.item())
                loss_data.append(loss.item())

                if intermediateModel_path is not None:
                    idx = len(loss_data)
                    steps = training_dict.get("save_model_every_n", 0)
                    if steps > 0 and idx >= steps and idx % steps == 0:
                        intermModel_path = intermediateModel_path / Path(
                            "intermediate_mixedH_TM_1NN_"
                            + str(int(field_comp.lmbda * 1000000))
                            + "by1000000_"
                            + str(idx)
                            + ".pt"
                        )
                        torch.save(field_comp.net.state_dict(), intermModel_path)

                loss.backward()
                return loss

            optimizer.step(closure=closure)

    return loss_data


def fit_mixed_tm_with_early_stopping(
    field_comp,
    training_set_collocation,
    T_conn,
    area_T,
    history_old,
    matprop,
    pffmodel,
    weight_decay,
    num_epochs,
    optimizer,
    min_delta,
    intermediateModel_path=None,
    writer=None,
    training_dict=None,
):
    training_dict = training_dict or {}
    eta_residual = training_dict.get("eta_residual", 1.0e-8)
    gcII = training_dict.get("GcII", None)
    gcII_factor = training_dict.get("GcII_factor", 1.0)
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    thermal_kwargs = _thermal_energy_kwargs(training_dict)
    loss_data = []
    early_stopping = EarlyStopping(tol_steps=10, min_delta=min_delta, device=area_T.device)
    loss_prev = torch.tensor([0.0], device=area_T.device)

    for epoch in range(num_epochs):
        loop = tqdm(training_set_collocation, miniters=25)
        for _, (inp_train, _) in enumerate(loop):
            optimizer.zero_grad()
            if T_conn is None:
                inp_train.requires_grad = True
            u, v, alpha = field_comp.fieldCalculation(inp_train)
            loss_E_el, loss_E_d, _ = compute_mixed_tm_energy(
                inp_train,
                u,
                v,
                alpha,
                history_old["HI"],
                history_old["HII"],
                matprop,
                pffmodel,
                area_T,
                T_conn,
                eta_residual=eta_residual,
                gcII=gcII,
                gcII_factor=gcII_factor,
                tm_eps_r=tm_eps_r,
                alpha_old=history_old.get("alpha_old"),
                **thermal_kwargs,
            )
            loss_var = torch.log10(loss_E_el + loss_E_d)

            loss_reg = 0.0
            if weight_decay != 0:
                for name, param in field_comp.net.named_parameters():
                    if "weight" in name:
                        loss_reg += torch.sum(param**2)

            loss = loss_var + weight_decay * loss_reg
            if writer is not None:
                writer.add_scalars(
                    "mixedH_TM_U_p_" + str(field_comp.lmbda.item()),
                    {"loss": loss.item(), "loss_E": loss_var.item()},
                    epoch,
                )
            loop.set_description(
                f"mixedH_TM U_p: {field_comp.lmbda}, Epoch [{epoch}/{num_epochs}]"
            )
            loop.set_postfix(loss=loss.item(), loss_E=loss_var.item())
            loss_data.append(loss.item())

            if intermediateModel_path is not None:
                idx = len(loss_data)
                steps = training_dict.get("save_model_every_n", 0)
                if steps > 0 and idx >= steps and idx % steps == 0:
                    intermModel_path = intermediateModel_path / Path(
                        "intermediate_mixedH_TM_1NN_"
                        + str(int(field_comp.lmbda * 1000000))
                        + "by1000000_"
                        + str(idx)
                        + ".pt"
                    )
                    torch.save(field_comp.net.state_dict(), intermModel_path)

            loss.backward()
            optimizer.step()

        early_stopping(loss, loss_prev)
        if early_stopping.early_stop:
            break
        loss_prev = loss

    return loss_data


def _make_training_set(inp, device):
    outp = torch.zeros(inp.shape[0], 1).to(device)
    return DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)


def save_mixed_tm_step_checkpoint(
    trainedModel_path,
    step,
    displacement,
    field_comp,
    history_old,
    optimizer_dict,
    training_dict,
):
    checkpoint_dir = trainedModel_path / "step_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        key: value
        for key, value in training_dict.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    payload = {
        "checkpoint_format": "TM_comsol_no_thermal_micro_mixedH_TM_step_checkpoint_v1",
        "step_index": int(step),
        "Delta": float(displacement),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in field_comp.net.state_dict().items()
        },
        "HI_old": history_old["HI"].detach().cpu().clone(),
        "HII_old": history_old["HII"].detach().cpu().clone(),
        "He": history_old["He"].detach().cpu().clone(),
        "history": {
            "HI": history_old["HI"].detach().cpu().clone(),
            "HII": history_old["HII"].detach().cpu().clone(),
            "He": history_old["He"].detach().cpu().clone(),
        },
        "optimizer_state_dict": None,
        "optimizer_state_available": False,
        "optimizer_config": dict(optimizer_dict),
        "config_snapshot": config_snapshot,
        "random_seeds": {
            "torch_initial_seed": int(torch.initial_seed()),
            "numpy_seed_state_head": int(np.random.get_state()[1][0]),
        },
        "history_mode": "mixedH_TM",
        "tm_source_split": "tm_source",
        "mechanics_objective": "history",
    }
    path = checkpoint_dir / f"checkpoint_mixedH_TM_step_{int(step):04d}.pt"
    torch.save(payload, path)
    return path


def should_save_mixed_tm_step_checkpoint(training_dict):
    return bool(training_dict.get("save_step_checkpoints", False)) and bool(
        training_dict.get("checkpoint_every_step", True)
    )


def _run_mixed_tm_step(
    field_comp,
    training_set,
    T_conn,
    area_T,
    history_old,
    matprop,
    pffmodel,
    optimizer_dict,
    training_dict,
    intermediateModel_path,
    writer,
):
    loss_data = []
    if optimizer_dict["n_epochs_LBFGS"] > 0:
        optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
        loss_data += fit_mixed_tm(
            field_comp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            max(optimizer_dict["n_epochs_LBFGS"], 1),
            optimizer,
            intermediateModel_path=None,
            writer=writer,
            training_dict=training_dict,
        )
    if optimizer_dict["n_epochs_RPROP"] > 0:
        optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
        loss_data += fit_mixed_tm_with_early_stopping(
            field_comp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            optimizer_dict["n_epochs_RPROP"],
            optimizer,
            optimizer_dict["optim_rel_tol"],
            intermediateModel_path=intermediateModel_path,
            writer=writer,
            training_dict=training_dict,
        )
    return loss_data


def train_mixed_tm(
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
):
    eta_residual = training_dict.get("eta_residual", 1.0e-8)
    gcII = training_dict.get("GcII", None)
    gcII_factor = training_dict.get("GcII_factor", 1.0)
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    thermal_kwargs = _thermal_energy_kwargs(training_dict)
    results_path = training_dict["results_path"]

    inp, T_conn, area_T, _ = prep_input_data(
        matprop, pffmodel, crack_dict, numr_dict, mesh_file=coarse_mesh_file, device=device
    )
    training_set = _make_training_set(inp, device)
    history_old = initialize_mixed_history_fields(area_T)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)

    pretrain_loss = []
    if optimizer_dict["n_epochs_LBFGS"] > 0:
        optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
        pretrain_loss += fit_mixed_tm(
            field_comp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            max(optimizer_dict["n_epochs_LBFGS"], 1),
            optimizer,
            intermediateModel_path=None,
            writer=writer,
            training_dict=training_dict,
        )
    if optimizer_dict["n_epochs_RPROP"] > 0:
        optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
        pretrain_loss += fit_mixed_tm_with_early_stopping(
            field_comp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            optimizer_dict["n_epochs_RPROP"],
            optimizer,
            optimizer_dict["optim_rel_tol_pretrain"],
            intermediateModel_path=None,
            writer=writer,
            training_dict=training_dict,
        )

    torch.save(field_comp.net.state_dict(), trainedModel_path / Path("trained_1NN_initTraining.pt"))
    with open(trainedModel_path / Path("trainLoss_1NN_initTraining.npy"), "wb") as file:
        np.save(file, np.asarray(pretrain_loss))

    inp, T_conn, area_T, _ = prep_input_data(
        matprop, pffmodel, crack_dict, numr_dict, mesh_file=fine_mesh_file, device=device
    )
    training_set = _make_training_set(inp, device)
    history_old = initialize_mixed_history_fields(area_T)
    summary_path = results_path / "diagnostics_mixed_tm_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    for j, disp_i in enumerate(disp):
        field_comp.lmbda = torch.tensor(disp_i).to(device)
        print(f"mixedH_TM idx: {j}; displacement: {field_comp.lmbda}")
        loss_data = _run_mixed_tm_step(
            field_comp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict,
            training_dict,
            intermediateModel_path,
            writer,
        )

        history_old, alpha, u, v, fields = commit_mixed_tm_history_from_model(
            field_comp,
            inp,
            history_old,
            matprop,
            pffmodel,
            area_T,
            T_conn,
            eta_residual=eta_residual,
            gcII=gcII,
            gcII_factor=gcII_factor,
            tm_eps_r=tm_eps_r,
            **thermal_kwargs,
        )
        history_old["alpha_old"] = alpha.detach().clone()
        save_mixed_tm_step_fields(results_path, j, disp_i, inp, T_conn, alpha, u, v, fields)
        append_mixed_tm_summary(
            summary_path,
            j,
            disp_i,
            inp,
            T_conn,
            fields,
            u=u,
            v=v,
            top_u_mode=training_dict.get("top_u_mode", "fixed"),
            coord_mapping=field_comp.coord_mapping_diagnostics(inp),
        )
        np.savez_compressed(
            trainedModel_path / f"history_mixedH_TM_step_{j:04d}.npz",
            HI=history_old["HI"].detach().cpu().numpy(),
            HII=history_old["HII"].detach().cpu().numpy(),
            He=history_old["He"].detach().cpu().numpy(),
        )

        torch.save(field_comp.net.state_dict(), trainedModel_path / Path("trained_1NN_" + str(j) + ".pt"))
        if should_save_mixed_tm_step_checkpoint(training_dict):
            save_mixed_tm_step_checkpoint(
                trainedModel_path,
                j,
                disp_i,
                field_comp,
                history_old,
                optimizer_dict,
                training_dict,
            )
        with open(trainedModel_path / Path("trainLoss_1NN_" + str(j) + ".npy"), "wb") as file:
            np.save(file, np.asarray(loss_data))
