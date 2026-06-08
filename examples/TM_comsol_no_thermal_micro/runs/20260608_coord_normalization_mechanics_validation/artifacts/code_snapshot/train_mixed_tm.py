import csv
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from compute_energy_mixed_tm import compute_mixed_tm_energy, compute_mixed_tm_fields
from history_field_mixed_tm import (
    append_mixed_tm_summary,
    commit_mixed_history_from_fields,
    commit_mixed_tm_history_from_model,
    element_centroids,
    initialize_mixed_history_fields,
    save_mixed_tm_step_fields,
)
from input_data_from_mesh import prep_input_data
from optim import get_optimizer


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
    split_mode = training_dict.get("mixed_split_mode", "current")
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    mechanics_mode = training_dict.get("mixed_mechanics_mode", "history")
    phase_proximal_mode = training_dict.get("phase_proximal_mode", "none")
    eta_eff = training_dict.get("eta_eff", 0.0)
    dt = training_dict.get("dt", 1.0)
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
                    split_mode=split_mode,
                    tm_eps_r=tm_eps_r,
                    mechanics_mode=mechanics_mode,
                    alpha_old=history_old.get("alpha_old"),
                    phase_proximal_mode=phase_proximal_mode,
                    eta_eff=eta_eff,
                    dt=dt,
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
    split_mode = training_dict.get("mixed_split_mode", "current")
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    mechanics_mode = training_dict.get("mixed_mechanics_mode", "history")
    phase_proximal_mode = training_dict.get("phase_proximal_mode", "none")
    eta_eff = training_dict.get("eta_eff", 0.0)
    dt = training_dict.get("dt", 1.0)
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
                split_mode=split_mode,
                tm_eps_r=tm_eps_r,
                mechanics_mode=mechanics_mode,
                alpha_old=history_old.get("alpha_old"),
                phase_proximal_mode=phase_proximal_mode,
                eta_eff=eta_eff,
                dt=dt,
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


def _positive_log10_energy(energy):
    tiny = torch.finfo(energy.dtype).tiny
    return torch.log10(torch.clamp(energy, min=tiny))


def _weight_decay_loss(field_comp, weight_decay):
    if weight_decay == 0:
        return 0.0
    loss_reg = 0.0
    for name, param in field_comp.net.named_parameters():
        if "weight" in name:
            loss_reg += torch.sum(param**2)
    return weight_decay * loss_reg


def _alpha_old_nodal_or_zero(history_old, inp):
    alpha_old = history_old.get("alpha_old")
    if alpha_old is None:
        return torch.zeros(inp.shape[0], dtype=inp.dtype, device=inp.device)
    alpha_old = alpha_old.to(device=inp.device, dtype=inp.dtype)
    if alpha_old.shape[0] != inp.shape[0]:
        raise ValueError(
            "staggered solve requires nodal alpha_old with one value per mesh node; "
            f"got shape {tuple(alpha_old.shape)} for {inp.shape[0]} nodes"
        )
    return alpha_old


def _max_location(values, x_elem, y_elem):
    idx = int(torch.argmax(values).detach().cpu())
    return (
        float(torch.max(values).detach().cpu()),
        float(x_elem[idx].detach().cpu()),
        float(y_elem[idx].detach().cpu()),
    )


def _append_staggered_substep_diagnostics(
    path,
    step,
    displacement,
    stagger_iter,
    substep,
    mechanics_substep_loss,
    phase_substep_loss,
    inp,
    T_conn,
    fields,
):
    if path is None:
        return
    x_elem, y_elem = element_centroids(inp, T_conn)
    he_current, he_current_x, he_current_y = _max_location(fields["He_current"], x_elem, y_elem)
    he_history, he_history_x, he_history_y = _max_location(fields["He_history"], x_elem, y_elem)
    mechanics_drive, mechanics_drive_x, mechanics_drive_y = _max_location(
        fields["mechanics_drive"], x_elem, y_elem
    )
    row = {
        "solve_scheme": "staggered",
        "step": int(step),
        "Delta": float(displacement),
        "stagger_iter": int(stagger_iter),
        "substep": substep,
        "mechanics_substep_loss": float(mechanics_substep_loss)
        if np.isfinite(mechanics_substep_loss)
        else np.nan,
        "phase_substep_loss": float(phase_substep_loss)
        if np.isfinite(phase_substep_loss)
        else np.nan,
        "alpha_mean": float(torch.mean(fields["alpha_elem"]).detach().cpu()),
        "alpha_max": float(torch.max(fields["alpha_elem"]).detach().cpu()),
        "alpha_min": float(torch.min(fields["alpha_elem"]).detach().cpu()),
        "max_He_current": he_current,
        "max_He_current_x": he_current_x,
        "max_He_current_y": he_current_y,
        "max_He_history": he_history,
        "max_He_history_x": he_history_x,
        "max_He_history_y": he_history_y,
        "max_mechanics_drive": mechanics_drive,
        "max_mechanics_drive_x": mechanics_drive_x,
        "max_mechanics_drive_y": mechanics_drive_y,
    }
    columns = list(row.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
        "split_mode": training_dict.get("mixed_split_mode", "current"),
        "mechanics_mode": training_dict.get("mixed_mechanics_mode", "history"),
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


def _staggered_common_kwargs(training_dict):
    return {
        "eta_residual": training_dict.get("eta_residual", 1.0e-8),
        "gcII": training_dict.get("GcII", None),
        "gcII_factor": training_dict.get("GcII_factor", 1.0),
        "split_mode": training_dict.get("mixed_split_mode", "current"),
        "tm_eps_r": training_dict.get("tm_eps_r", 0.0),
        "phase_proximal_mode": training_dict.get("phase_proximal_mode", "none"),
        "eta_eff": training_dict.get("eta_eff", 0.0),
        "dt": training_dict.get("dt", 1.0),
    }


def _staggered_mechanics_loss(
    field_comp,
    inp_train,
    T_conn,
    area_T,
    history_old,
    alpha_fixed,
    matprop,
    pffmodel,
    weight_decay,
    training_dict,
):
    if T_conn is None:
        inp_train.requires_grad = True
    u, v, _alpha_nn = field_comp.fieldCalculation(inp_train)
    kwargs = _staggered_common_kwargs(training_dict)
    loss_E_el, _loss_E_d, _fields = compute_mixed_tm_energy(
        inp_train,
        u,
        v,
        alpha_fixed.detach(),
        history_old["HI"],
        history_old["HII"],
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=kwargs["eta_residual"],
        gcII=kwargs["gcII"],
        gcII_factor=kwargs["gcII_factor"],
        split_mode=kwargs["split_mode"],
        tm_eps_r=kwargs["tm_eps_r"],
        mechanics_mode="current_split",
    )
    return _positive_log10_energy(loss_E_el) + _weight_decay_loss(field_comp, weight_decay)


def _staggered_phase_loss(
    field_comp,
    inp_train,
    T_conn,
    area_T,
    history_trial,
    history_old,
    u_fixed,
    v_fixed,
    matprop,
    pffmodel,
    weight_decay,
    training_dict,
):
    if T_conn is None:
        inp_train.requires_grad = True
    _u_nn, _v_nn, alpha = field_comp.fieldCalculation(inp_train)
    kwargs = _staggered_common_kwargs(training_dict)
    fields = compute_mixed_tm_fields(
        inp_train,
        u_fixed.detach(),
        v_fixed.detach(),
        alpha,
        history_trial["HI"],
        history_trial["HII"],
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=kwargs["eta_residual"],
        gcII=kwargs["gcII"],
        gcII_factor=kwargs["gcII_factor"],
        split_mode=kwargs["split_mode"],
        tm_eps_r=kwargs["tm_eps_r"],
        mechanics_mode="history",
        alpha_old=history_old.get("alpha_old"),
        phase_proximal_mode=kwargs["phase_proximal_mode"],
        eta_eff=kwargs["eta_eff"],
        dt=kwargs["dt"],
    )
    phase_energy = torch.sum(area_T * fields["phase_history_total_density"])
    return _positive_log10_energy(phase_energy) + _weight_decay_loss(field_comp, weight_decay)


def _fit_staggered_substep(
    substep,
    field_comp,
    training_set,
    T_conn,
    area_T,
    history_old,
    matprop,
    pffmodel,
    weight_decay,
    num_epochs,
    optimizer,
    training_dict,
    alpha_fixed=None,
    history_trial=None,
    u_fixed=None,
    v_fixed=None,
    min_delta=None,
):
    loss_data = []
    is_lbfgs = optimizer.__class__.__name__.upper() == "LBFGS"
    early_stopping = None
    loss_prev = None
    if (not is_lbfgs) and min_delta is not None:
        early_stopping = EarlyStopping(tol_steps=10, min_delta=min_delta, device=area_T.device)
        loss_prev = torch.tensor([0.0], device=area_T.device)
    for epoch in range(num_epochs):
        last_loss = None
        loop = tqdm(training_set, miniters=25)
        for _, (inp_train, _) in enumerate(loop):
            if substep == "mechanics":
                loss_fn = lambda: _staggered_mechanics_loss(
                    field_comp,
                    inp_train,
                    T_conn,
                    area_T,
                    history_old,
                    alpha_fixed,
                    matprop,
                    pffmodel,
                    weight_decay,
                    training_dict,
                )
            elif substep == "phase":
                loss_fn = lambda: _staggered_phase_loss(
                    field_comp,
                    inp_train,
                    T_conn,
                    area_T,
                    history_trial,
                    history_old,
                    u_fixed,
                    v_fixed,
                    matprop,
                    pffmodel,
                    weight_decay,
                    training_dict,
                )
            else:
                raise ValueError(f"Unknown staggered substep {substep!r}")

            if is_lbfgs:

                def closure():
                    optimizer.zero_grad()
                    loss = loss_fn()
                    loss.backward()
                    loss_data.append(float(loss.detach().cpu()))
                    return loss

                loss = optimizer.step(closure=closure)
                loss_value = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
            else:
                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                last_loss = loss.detach()
                loss_data.append(loss_value)

            loop.set_description(
                f"staggered {substep} U_p: {field_comp.lmbda}, Epoch [{epoch}/{num_epochs}]"
            )
            loop.set_postfix(loss=loss_value)
        if early_stopping is not None and last_loss is not None:
            early_stopping(last_loss, loss_prev)
            if early_stopping.early_stop:
                break
            loss_prev = last_loss
    return loss_data


def _evaluate_staggered_mechanics_trial(
    field_comp,
    inp,
    T_conn,
    area_T,
    history_old,
    alpha_fixed,
    matprop,
    pffmodel,
    training_dict,
):
    kwargs = _staggered_common_kwargs(training_dict)
    with torch.no_grad():
        u, v, _alpha_nn = field_comp.fieldCalculation(inp)
    fields = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha_fixed.detach(),
        history_old["HI"],
        history_old["HII"],
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=kwargs["eta_residual"],
        gcII=kwargs["gcII"],
        gcII_factor=kwargs["gcII_factor"],
        split_mode=kwargs["split_mode"],
        tm_eps_r=kwargs["tm_eps_r"],
        mechanics_mode="current_split",
        alpha_old=history_old.get("alpha_old"),
    )
    ratio = float(fields["mixed_mode_ratio"][0].detach().cpu()) if fields["mixed_mode_ratio"].numel() else 1.0
    history_trial = commit_mixed_history_from_fields(
        history_old["HI"], history_old["HII"], fields["psiI"], fields["psiII"], ratio=ratio
    )
    fields["HI"] = history_trial["HI"]
    fields["HII"] = history_trial["HII"]
    fields["He"] = history_trial["He"]
    fields["He_history"] = history_trial["He"]
    return history_trial, alpha_fixed.detach(), u.detach(), v.detach(), fields


def _evaluate_staggered_phase_final(
    field_comp,
    inp,
    T_conn,
    area_T,
    history_trial,
    history_old,
    u_fixed,
    v_fixed,
    matprop,
    pffmodel,
    training_dict,
):
    kwargs = _staggered_common_kwargs(training_dict)
    with torch.no_grad():
        _u_nn, _v_nn, alpha = field_comp.fieldCalculation(inp)
    fields = compute_mixed_tm_fields(
        inp,
        u_fixed.detach(),
        v_fixed.detach(),
        alpha,
        history_trial["HI"],
        history_trial["HII"],
        matprop,
        pffmodel,
        area_T,
        T_conn,
        eta_residual=kwargs["eta_residual"],
        gcII=kwargs["gcII"],
        gcII_factor=kwargs["gcII_factor"],
        split_mode=kwargs["split_mode"],
        tm_eps_r=kwargs["tm_eps_r"],
        mechanics_mode="history",
        alpha_old=history_old.get("alpha_old"),
        phase_proximal_mode=kwargs["phase_proximal_mode"],
        eta_eff=kwargs["eta_eff"],
        dt=kwargs["dt"],
    )
    history_new = {
        "HI": history_trial["HI"].detach(),
        "HII": history_trial["HII"].detach(),
        "He": history_trial["He"].detach(),
        "alpha_old": alpha.detach().clone(),
    }
    fields["HI"] = history_new["HI"]
    fields["HII"] = history_new["HII"]
    fields["He"] = history_new["He"]
    fields["He_history"] = history_new["He"]
    return history_new, alpha.detach(), u_fixed.detach(), v_fixed.detach(), fields


def _run_mixed_tm_step_staggered(
    field_comp,
    inp,
    training_set,
    T_conn,
    area_T,
    history_old,
    matprop,
    pffmodel,
    optimizer_dict,
    training_dict,
    step,
    displacement,
    substep_summary_path=None,
):
    loss_data = []
    history_work = {
        "HI": history_old["HI"],
        "HII": history_old["HII"],
        "He": history_old["He"],
    }
    if "alpha_old" in history_old:
        history_work["alpha_old"] = history_old["alpha_old"]

    alpha_fixed = _alpha_old_nodal_or_zero(history_work, inp).detach()
    final_alpha = alpha_fixed
    final_u = torch.zeros_like(inp[:, 0])
    final_v = torch.zeros_like(inp[:, 1])
    final_fields = None
    stagger_iters = int(training_dict.get("stagger_iters", 1))
    rprop_min_delta = (
        optimizer_dict["optim_rel_tol_pretrain"] if int(step) < 0 else optimizer_dict["optim_rel_tol"]
    )

    for stagger_iter in range(stagger_iters):
        mechanics_losses = []
        if optimizer_dict["n_epochs_LBFGS"] > 0:
            optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
            mechanics_losses += _fit_staggered_substep(
                "mechanics",
                field_comp,
                training_set,
                T_conn,
                area_T,
                history_work,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                max(optimizer_dict["n_epochs_LBFGS"], 1),
                optimizer,
                training_dict,
                alpha_fixed=alpha_fixed,
                min_delta=rprop_min_delta,
            )
        if optimizer_dict["n_epochs_RPROP"] > 0:
            optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
            mechanics_losses += _fit_staggered_substep(
                "mechanics",
                field_comp,
                training_set,
                T_conn,
                area_T,
                history_work,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                optimizer_dict["n_epochs_RPROP"],
                optimizer,
                training_dict,
                alpha_fixed=alpha_fixed,
            )
        loss_data += mechanics_losses
        mechanics_loss = mechanics_losses[-1] if mechanics_losses else np.nan
        history_trial, _alpha_mech, u_mech, v_mech, mechanics_fields = _evaluate_staggered_mechanics_trial(
            field_comp,
            inp,
            T_conn,
            area_T,
            history_work,
            alpha_fixed,
            matprop,
            pffmodel,
            training_dict,
        )
        _append_staggered_substep_diagnostics(
            substep_summary_path,
            step,
            displacement,
            stagger_iter,
            "mechanics",
            mechanics_loss,
            np.nan,
            inp,
            T_conn,
            mechanics_fields,
        )

        phase_losses = []
        if optimizer_dict["n_epochs_LBFGS"] > 0:
            optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
            phase_losses += _fit_staggered_substep(
                "phase",
                field_comp,
                training_set,
                T_conn,
                area_T,
                history_work,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                max(optimizer_dict["n_epochs_LBFGS"], 1),
                optimizer,
                training_dict,
                history_trial=history_trial,
                u_fixed=u_mech,
                v_fixed=v_mech,
                min_delta=rprop_min_delta,
            )
        if optimizer_dict["n_epochs_RPROP"] > 0:
            optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
            phase_losses += _fit_staggered_substep(
                "phase",
                field_comp,
                training_set,
                T_conn,
                area_T,
                history_work,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                optimizer_dict["n_epochs_RPROP"],
                optimizer,
                training_dict,
                history_trial=history_trial,
                u_fixed=u_mech,
                v_fixed=v_mech,
            )
        loss_data += phase_losses
        phase_loss = phase_losses[-1] if phase_losses else np.nan
        history_work, final_alpha, final_u, final_v, final_fields = _evaluate_staggered_phase_final(
            field_comp,
            inp,
            T_conn,
            area_T,
            history_trial,
            history_work,
            u_mech,
            v_mech,
            matprop,
            pffmodel,
            training_dict,
        )
        _append_staggered_substep_diagnostics(
            substep_summary_path,
            step,
            displacement,
            stagger_iter,
            "phase",
            mechanics_loss,
            phase_loss,
            inp,
            T_conn,
            final_fields,
        )
        alpha_fixed = final_alpha.detach().clone()

    return loss_data, history_work, final_alpha, final_u, final_v, final_fields


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
    split_mode = training_dict.get("mixed_split_mode", "current")
    tm_eps_r = training_dict.get("tm_eps_r", 0.0)
    mechanics_mode = training_dict.get("mixed_mechanics_mode", "history")
    phase_proximal_mode = training_dict.get("phase_proximal_mode", "none")
    eta_eff = training_dict.get("eta_eff", 0.0)
    dt = training_dict.get("dt", 1.0)
    solve_scheme = training_dict.get("solve_scheme", "coupled")
    results_path = training_dict["results_path"]

    inp, T_conn, area_T, _ = prep_input_data(
        matprop, pffmodel, crack_dict, numr_dict, mesh_file=coarse_mesh_file, device=device
    )
    training_set = _make_training_set(inp, device)
    history_old = initialize_mixed_history_fields(area_T)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)

    pretrain_loss = []
    if solve_scheme == "staggered":
        pretrain_loss, _history_pre, _alpha_pre, _u_pre, _v_pre, _fields_pre = _run_mixed_tm_step_staggered(
            field_comp,
            inp,
            training_set,
            T_conn,
            area_T,
            history_old,
            matprop,
            pffmodel,
            optimizer_dict,
            training_dict,
            step=-1,
            displacement=float(disp[0]),
            substep_summary_path=None,
        )
    else:
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
    staggered_substep_summary_path = results_path / "diagnostics_staggered_substeps.csv"
    if solve_scheme == "staggered" and staggered_substep_summary_path.exists():
        staggered_substep_summary_path.unlink()

    for j, disp_i in enumerate(disp):
        field_comp.lmbda = torch.tensor(disp_i).to(device)
        print(f"mixedH_TM idx: {j}; displacement: {field_comp.lmbda}")
        if solve_scheme == "staggered":
            loss_data, history_old, alpha, u, v, fields = _run_mixed_tm_step_staggered(
                field_comp,
                inp,
                training_set,
                T_conn,
                area_T,
                history_old,
                matprop,
                pffmodel,
                optimizer_dict,
                training_dict,
                step=j,
                displacement=disp_i,
                substep_summary_path=staggered_substep_summary_path,
            )
        else:
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
                split_mode=split_mode,
                tm_eps_r=tm_eps_r,
                mechanics_mode=mechanics_mode,
                phase_proximal_mode=phase_proximal_mode,
                eta_eff=eta_eff,
                dt=dt,
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
