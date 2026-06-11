import torch


DEFAULT_ALPHA_T = 18.9e-6
DEFAULT_TREF_K = 273.15
DEFAULT_T0_C = 0.0


def thermal_constants(alpha_T=DEFAULT_ALPHA_T, Tref=DEFAULT_TREF_K, T0_c=DEFAULT_T0_C):
    return {
        "alpha_T": float(alpha_T),
        "Tref_K": float(Tref),
        "T0_C": float(T0_c),
    }


def _as_like(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def delta_T_from_temperature(temperature, Tref=DEFAULT_TREF_K):
    temperature = torch.as_tensor(temperature)
    return temperature - _as_like(Tref, temperature)


def thermal_strain_2d(delta_T, alpha_T=DEFAULT_ALPHA_T):
    delta_T = torch.as_tensor(delta_T)
    thermal_normal = _as_like(alpha_T, delta_T) * delta_T
    return {
        "eps_xx": thermal_normal,
        "eps_yy": thermal_normal,
        "eps_xy": torch.zeros_like(delta_T),
    }


def apply_thermal_strain(
    exx,
    eyy,
    exy,
    temperature=None,
    delta_T=None,
    alpha_T=DEFAULT_ALPHA_T,
    Tref=DEFAULT_TREF_K,
):
    if temperature is not None and delta_T is not None:
        raise ValueError("Specify either temperature or delta_T, not both")
    exx = torch.as_tensor(exx)
    eyy = torch.as_tensor(eyy, device=exx.device, dtype=exx.dtype)
    exy = torch.as_tensor(exy, device=exx.device, dtype=exx.dtype)
    if temperature is not None:
        temperature = torch.as_tensor(temperature, device=exx.device, dtype=exx.dtype)
        delta_T = delta_T_from_temperature(temperature, Tref=_as_like(Tref, exx))
    elif delta_T is None:
        delta_T = torch.zeros_like(exx)
    else:
        delta_T = torch.as_tensor(delta_T, device=exx.device, dtype=exx.dtype)
    thermal_normal = _as_like(alpha_T, exx) * delta_T
    return {
        "eps_xx": exx - thermal_normal,
        "eps_yy": eyy - thermal_normal,
        "eps_xy": exy,
        "delta_T": delta_T,
        "thermal_eps_xx": thermal_normal,
        "thermal_eps_yy": thermal_normal,
        "thermal_eps_xy": torch.zeros_like(exy),
    }


def prescribed_delta_T(
    mode="off",
    x=None,
    y=None,
    delta_T0=0.0,
    grad_y=0.0,
    y0=0.0,
):
    if mode == "off":
        reference = y if y is not None else x
        if reference is None:
            return torch.as_tensor(0.0)
        return torch.zeros_like(reference)
    if mode == "uniform":
        reference = y if y is not None else x
        if reference is None:
            return torch.as_tensor(float(delta_T0))
        return torch.zeros_like(reference) + float(delta_T0)
    if mode == "linear_y":
        if y is None:
            raise ValueError("linear_y prescribed thermal mode requires y")
        return float(delta_T0) + float(grad_y) * (y - _as_like(y0, y))
    raise ValueError("mode must be 'off', 'uniform', or 'linear_y'")
