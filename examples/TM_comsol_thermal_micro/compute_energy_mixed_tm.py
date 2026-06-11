import torch

from compute_energy import gradients, stress
from mixed_mode_tm import (
    mixed_mode_energy_split,
    mixed_mode_ratio,
    tm_source_effective_stress_fields,
)
from thermal_prescribed import (
    DEFAULT_ALPHA_T,
    DEFAULT_TREF_K,
    apply_thermal_strain,
    delta_T_from_temperature,
    prescribed_delta_T,
)


TM_STRESS_FIELD_KEYS = (
    "sigma_xx_tm_total",
    "sigma_yy_tm_total",
    "sigma_xy_tm_total",
    "sigma_xx_tm_plus",
    "sigma_yy_tm_plus",
    "sigma_xy_tm_plus",
    "sigma_xx_tm_minus",
    "sigma_yy_tm_minus",
    "sigma_xy_tm_minus",
    "sigma_xx_tm_eff",
    "sigma_yy_tm_eff",
    "sigma_xy_tm_eff",
)


def _alpha_on_elements(alpha, T_conn):
    if T_conn is None:
        return alpha
    return (alpha[T_conn[:, 0]] + alpha[T_conn[:, 1]] + alpha[T_conn[:, 2]]) / 3.0


def _optional_alpha_old_on_elements(alpha_old, alpha_elem, T_conn):
    if alpha_old is None:
        return torch.zeros_like(alpha_elem)
    if alpha_old.shape == alpha_elem.shape:
        return alpha_old.to(device=alpha_elem.device, dtype=alpha_elem.dtype)
    if T_conn is None:
        return alpha_old.to(device=alpha_elem.device, dtype=alpha_elem.dtype)
    alpha_old = alpha_old.to(device=alpha_elem.device, dtype=alpha_elem.dtype)
    return (alpha_old[T_conn[:, 0]] + alpha_old[T_conn[:, 1]] + alpha_old[T_conn[:, 2]]) / 3.0


def _element_centroids(inp, T_conn):
    if T_conn is None:
        return inp[:, 0], inp[:, 1]
    x = (inp[T_conn[:, 0], 0] + inp[T_conn[:, 1], 0] + inp[T_conn[:, 2], 0]) / 3.0
    y = (inp[T_conn[:, 0], 1] + inp[T_conn[:, 1], 1] + inp[T_conn[:, 2], 1]) / 3.0
    return x, y


def _thermal_value_on_elements(value, reference, T_conn, name):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0 or value.numel() == 1:
        return torch.zeros_like(reference) + value.reshape(())
    if value.shape == reference.shape:
        return value
    if T_conn is not None and value.ndim == 1 and value.numel() > int(torch.max(T_conn).detach().cpu()):
        return (value[T_conn[:, 0]] + value[T_conn[:, 1]] + value[T_conn[:, 2]]) / 3.0
    raise ValueError(f"{name} must be scalar, element-sized, or nodal-sized")


def _resolve_delta_T(
    inp,
    strain_11,
    T_conn,
    thermal_temperature=None,
    thermal_delta_T=None,
    thermal_mode="off",
    thermal_delta_T0=0.0,
    thermal_grad_y=0.0,
    thermal_y0=0.0,
    thermal_Tref=DEFAULT_TREF_K,
):
    if thermal_temperature is not None and thermal_delta_T is not None:
        raise ValueError("Specify either thermal_temperature or thermal_delta_T, not both")
    if thermal_temperature is not None:
        temperature = torch.as_tensor(thermal_temperature, device=strain_11.device, dtype=strain_11.dtype)
        delta_T = delta_T_from_temperature(
            temperature,
            Tref=torch.as_tensor(thermal_Tref, device=strain_11.device, dtype=strain_11.dtype),
        )
        return _thermal_value_on_elements(delta_T, strain_11, T_conn, "thermal_temperature")
    if thermal_delta_T is not None:
        return _thermal_value_on_elements(thermal_delta_T, strain_11, T_conn, "thermal_delta_T")
    if thermal_mode in {None, "off"}:
        return torch.zeros_like(strain_11)
    x_elem, y_elem = _element_centroids(inp, T_conn)
    delta_T = prescribed_delta_T(
        mode=thermal_mode,
        x=x_elem,
        y=y_elem,
        delta_T0=thermal_delta_T0,
        grad_y=thermal_grad_y,
        y0=thermal_y0,
    )
    return _thermal_value_on_elements(delta_T, strain_11, None, "prescribed thermal delta_T")


def compute_mixed_tm_fields(
    inp,
    u,
    v,
    alpha,
    HI_old,
    HII_old,
    matprop,
    pffmodel,
    area_elem,
    T_conn=None,
    eta_residual=1.0e-8,
    gcII=None,
    gcII_factor=1.0,
    tm_eps_r=0.0,
    alpha_old=None,
    thermal_temperature=None,
    thermal_delta_T=None,
    thermal_mode="off",
    thermal_delta_T0=0.0,
    thermal_grad_y=0.0,
    thermal_y0=0.0,
    thermal_alpha_T=DEFAULT_ALPHA_T,
    thermal_Tref=DEFAULT_TREF_K,
):
    if HI_old.shape != area_elem.shape:
        raise ValueError(
            f"HI_old shape {tuple(HI_old.shape)} must match area_elem shape {tuple(area_elem.shape)}"
        )
    if HII_old.shape != area_elem.shape:
        raise ValueError(
            f"HII_old shape {tuple(HII_old.shape)} must match area_elem shape {tuple(area_elem.shape)}"
        )

    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(
        inp, u, v, alpha, area_elem, T_conn
    )
    alpha_elem = _alpha_on_elements(alpha, T_conn)
    delta_T = _resolve_delta_T(
        inp,
        strain_11,
        T_conn,
        thermal_temperature=thermal_temperature,
        thermal_delta_T=thermal_delta_T,
        thermal_mode=thermal_mode,
        thermal_delta_T0=thermal_delta_T0,
        thermal_grad_y=thermal_grad_y,
        thermal_y0=thermal_y0,
        thermal_Tref=thermal_Tref,
    )
    elastic_strain = apply_thermal_strain(
        strain_11,
        strain_22,
        strain_12,
        delta_T=delta_T,
        alpha_T=thermal_alpha_T,
        Tref=thermal_Tref,
    )
    eps_xx_elastic = elastic_strain["eps_xx"]
    eps_yy_elastic = elastic_strain["eps_yy"]
    eps_xy_elastic = elastic_strain["eps_xy"]
    split = mixed_mode_energy_split(
        eps_xx_elastic,
        eps_yy_elastic,
        eps_xy_elastic,
        matprop,
        eps_r=tm_eps_r,
    )
    HI_trial = torch.maximum(HI_old.detach(), split["psiI"])
    HII_trial = torch.maximum(HII_old.detach(), split["psiII"])
    ratio = mixed_mode_ratio(matprop, gcII=gcII, gcII_factor=gcII_factor)
    He_trial = HI_trial + ratio * HII_trial
    He_current = split["psiI"] + ratio * split["psiII"]
    mechanics_drive = He_trial

    damage_fn, _, c_w = pffmodel.damageFun(alpha_elem)
    alpha_old_elem = _optional_alpha_old_on_elements(alpha_old, alpha_elem, T_conn)
    alpha_step_change = alpha_elem - alpha_old_elem
    grad_alpha_norm_sq = grad_alpha_x**2 + grad_alpha_y**2
    grad_alpha_norm = torch.sqrt(grad_alpha_norm_sq)
    g_alpha = (1.0 - alpha_elem) ** 2 + eta_residual
    fracture_energy_density = matprop.w1 / c_w * (
        damage_fn + matprop.l0**2 * grad_alpha_norm_sq
    )
    mechanics_current_energy_density = g_alpha * He_current + split["psi_minus"]
    history_elastic_energy_density = g_alpha * He_trial + split["psi_minus"]
    phase_history_energy_density = g_alpha * He_trial
    phase_history_total_density = phase_history_energy_density + fracture_energy_density
    elastic_energy_density = history_elastic_energy_density

    max_principal_strain = 0.5 * (strain_11 + strain_22) + torch.sqrt(
        (0.5 * (strain_11 - strain_22)) ** 2 + strain_12**2
    )
    _, sigma_yy, _ = stress(eps_xx_elastic, eps_yy_elastic, eps_xy_elastic, alpha_elem, matprop, pffmodel)
    tm_stress = tm_source_effective_stress_fields(
        eps_xx_elastic,
        eps_yy_elastic,
        eps_xy_elastic,
        alpha_elem,
        matprop,
        eta_residual=eta_residual,
        eps_r=tm_eps_r,
    )
    stress_fields = {key: tm_stress[key] for key in TM_STRESS_FIELD_KEYS}

    fields = {
        "alpha_elem": alpha_elem,
        "strain_11": strain_11,
        "strain_22": strain_22,
        "strain_12": strain_12,
        "eps_xx": strain_11,
        "eps_yy": strain_22,
        "eps_xy": strain_12,
        "eps_xx_total": strain_11,
        "eps_yy_total": strain_22,
        "eps_xy_total": strain_12,
        "eps_xx_elastic": eps_xx_elastic,
        "eps_yy_elastic": eps_yy_elastic,
        "eps_xy_elastic": eps_xy_elastic,
        "delta_T": delta_T,
        "thermal_delta_T": delta_T,
        "thermal_eps_xx": elastic_strain["thermal_eps_xx"],
        "thermal_eps_yy": elastic_strain["thermal_eps_yy"],
        "thermal_eps_xy": elastic_strain["thermal_eps_xy"],
        "thermal_alpha_T": torch.full_like(delta_T, float(thermal_alpha_T)),
        "thermal_Tref": torch.full_like(delta_T, float(thermal_Tref)),
        "thermal_active": torch.full_like(delta_T, float(torch.any(delta_T != 0.0).detach().cpu())),
        "eps_zz": split["eps_zz"],
        "grad_alpha_x": grad_alpha_x,
        "grad_alpha_y": grad_alpha_y,
        "grad_alpha_norm": grad_alpha_norm,
        "alpha_old_elem": alpha_old_elem,
        "alpha_step_change": alpha_step_change,
        "psiI": split["psiI"],
        "psiII": split["psiII"],
        "psi_minus": split["psi_minus"],
        "psi_residual_raw": split["psi_residual_raw"],
        "psi_total": split["psi_total"],
        "tm_eps_r": torch.full_like(split["psiI"], float(tm_eps_r)),
        "HI": HI_trial,
        "HII": HII_trial,
        "He": He_trial,
        "He_history": He_trial,
        "He_current": He_current,
        "mechanics_drive": mechanics_drive,
        "phase_history_drive": He_trial,
        "mixed_mode_ratio": torch.full_like(He_trial, float(ratio)),
        "g_alpha": g_alpha,
        "max_principal_strain": max_principal_strain,
        "sigma_yy": sigma_yy,
        **stress_fields,
        "mechanics_current_energy_density": mechanics_current_energy_density,
        "history_elastic_energy_density": history_elastic_energy_density,
        "phase_history_energy_density": phase_history_energy_density,
        "phase_history_total_density": phase_history_total_density,
        "elastic_energy_density": elastic_energy_density,
        "fracture_energy_density": fracture_energy_density,
        "alpha_step_change_mean": torch.mean(torch.abs(alpha_step_change)),
        "alpha_step_change_max": torch.max(torch.abs(alpha_step_change)),
    }
    return fields


def compute_mixed_tm_energy(
    inp,
    u,
    v,
    alpha,
    HI_old,
    HII_old,
    matprop,
    pffmodel,
    area_elem,
    T_conn=None,
    eta_residual=1.0e-8,
    gcII=None,
    gcII_factor=1.0,
    tm_eps_r=0.0,
    alpha_old=None,
    thermal_temperature=None,
    thermal_delta_T=None,
    thermal_mode="off",
    thermal_delta_T0=0.0,
    thermal_grad_y=0.0,
    thermal_y0=0.0,
    thermal_alpha_T=DEFAULT_ALPHA_T,
    thermal_Tref=DEFAULT_TREF_K,
):
    fields = compute_mixed_tm_fields(
        inp,
        u,
        v,
        alpha,
        HI_old,
        HII_old,
        matprop,
        pffmodel,
        area_elem,
        T_conn,
        eta_residual=eta_residual,
        gcII=gcII,
        gcII_factor=gcII_factor,
        tm_eps_r=tm_eps_r,
        alpha_old=alpha_old,
        thermal_temperature=thermal_temperature,
        thermal_delta_T=thermal_delta_T,
        thermal_mode=thermal_mode,
        thermal_delta_T0=thermal_delta_T0,
        thermal_grad_y=thermal_grad_y,
        thermal_y0=thermal_y0,
        thermal_alpha_T=thermal_alpha_T,
        thermal_Tref=thermal_Tref,
    )
    E_el = torch.sum(area_elem * fields["elastic_energy_density"])
    E_d = torch.sum(area_elem * fields["fracture_energy_density"])
    return E_el, E_d, fields
