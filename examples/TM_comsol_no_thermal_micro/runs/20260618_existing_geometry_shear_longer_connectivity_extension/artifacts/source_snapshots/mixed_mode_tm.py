import torch


def positive_part(x):
    return 0.5 * (x + torch.abs(x))


def resolve_gc(matprop):
    return matprop.w1 * matprop.l0


def resolve_gcII(matprop, gcII=None, gcII_factor=1.0):
    if gcII is not None:
        return torch.as_tensor(gcII, device=matprop.w1.device, dtype=matprop.w1.dtype)
    return resolve_gc(matprop) * float(gcII_factor)


def mixed_mode_ratio(matprop, gcII=None, gcII_factor=1.0):
    gc = resolve_gc(matprop)
    gcII_value = resolve_gcII(matprop, gcII=gcII, gcII_factor=gcII_factor)
    return float((gc / gcII_value).detach().cpu())


def _material_tensors(eps_xx, matprop):
    lam = torch.as_tensor(matprop.mat_lmbda, device=eps_xx.device, dtype=eps_xx.dtype)
    mu = torch.as_tensor(matprop.mat_mu, device=eps_xx.device, dtype=eps_xx.dtype)
    nu = torch.as_tensor(matprop.mat_nu, device=eps_xx.device, dtype=eps_xx.dtype)
    return lam, mu, nu


def _total_energy_density(eps_xx, eps_yy, eps_xy, eps_zz, lam, mu):
    tr = eps_xx + eps_yy + eps_zz
    eps_contract = eps_xx**2 + eps_yy**2 + eps_zz**2 + 2.0 * eps_xy**2
    return 0.5 * lam * tr**2 + mu * eps_contract


def _finish_split(eps_xx, eps_yy, eps_xy, eps_zz, psiI, psiII, lam, mu, extra=None):
    psi_total = _total_energy_density(eps_xx, eps_yy, eps_xy, eps_zz, lam, mu)
    psi_residual_raw = psi_total - psiI - psiII
    psi_minus = torch.clamp(psi_residual_raw, min=0.0)
    fields = {
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
        "eps_zz": eps_zz,
        "psiI": psiI,
        "psiII": psiII,
        "He": psiI + psiII,
        "psi_minus": psi_minus,
        "psi_residual_raw": psi_residual_raw,
        "psi_total": psi_total,
    }
    if extra:
        fields.update(extra)
    return fields


def _tm_source_split(eps_xx, eps_yy, eps_xy, matprop, eps_r=0.0):
    lam, mu, nu = _material_tensors(eps_xx, matprop)
    eps_r = torch.as_tensor(eps_r, device=eps_xx.device, dtype=eps_xx.dtype)
    eps_zz = -nu / (1.0 - nu) * (eps_xx + eps_yy)
    em = 0.5 * (eps_xx + eps_yy)
    ed = 0.5 * (eps_xx - eps_yy)
    r = torch.sqrt(ed**2 + eps_xy**2 + eps_r**2)
    safe_r = torch.where(r > 0.0, r, torch.ones_like(r))
    r0 = r - eps_r
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
    epzz = e3p
    ep2 = epxx**2 + epyy**2 + epzz**2 + 2.0 * epxy**2
    tr_p = e1p + e2p + e3p
    psiI = 0.5 * lam * tr_p**2
    psiII = mu * ep2

    return _finish_split(
        eps_xx,
        eps_yy,
        eps_xy,
        eps_zz,
        psiI,
        psiII,
        lam,
        mu,
        {
            "em": em,
            "ed": ed,
            "r": r,
            "r0": r0,
            "eps_r": torch.full_like(eps_xx, float(eps_r.detach().cpu())),
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e1p": e1p,
            "e2p": e2p,
            "e3p": e3p,
            "epxx": epxx,
            "epyy": epyy,
            "epxy": epxy,
            "epzz": epzz,
            "ep2": ep2,
            "tr_pos": tr_p,
            "chi": chi,
            "eta": eta,
        },
    )


def tm_source_effective_stress_fields(
    eps_xx,
    eps_yy,
    eps_xy,
    alpha,
    matprop,
    eta_residual=1.0e-8,
    eps_r=0.0,
):
    """
    COMSOL-aligned no-thermal stress postprocessing for the verified TM source split.

    The Java model applies ExternalStress=(g_d-1)*sigma_plus to the base
    linear-elastic stress. The corresponding effective stress is
    sigma_total + (g_alpha - 1) * sigma_plus.
    """
    split = _tm_source_split(eps_xx, eps_yy, eps_xy, matprop, eps_r=eps_r)
    lam, mu, _nu = _material_tensors(eps_xx, matprop)
    g_alpha = (1.0 - alpha) ** 2 + float(eta_residual)

    tr_e = eps_xx + eps_yy + split["eps_zz"]
    sigma_xx_total = lam * tr_e + 2.0 * mu * eps_xx
    sigma_yy_total = lam * tr_e + 2.0 * mu * eps_yy
    sigma_xy_total = 2.0 * mu * eps_xy

    sigma_xx_plus = lam * split["tr_pos"] + 2.0 * mu * split["epxx"]
    sigma_yy_plus = lam * split["tr_pos"] + 2.0 * mu * split["epyy"]
    sigma_xy_plus = 2.0 * mu * split["epxy"]

    sigma_xx_minus = sigma_xx_total - sigma_xx_plus
    sigma_yy_minus = sigma_yy_total - sigma_yy_plus
    sigma_xy_minus = sigma_xy_total - sigma_xy_plus

    split.update(
        {
            "sigma_xx_tm_total": sigma_xx_total,
            "sigma_yy_tm_total": sigma_yy_total,
            "sigma_xy_tm_total": sigma_xy_total,
            "sigma_xx_tm_plus": sigma_xx_plus,
            "sigma_yy_tm_plus": sigma_yy_plus,
            "sigma_xy_tm_plus": sigma_xy_plus,
            "sigma_xx_tm_minus": sigma_xx_minus,
            "sigma_yy_tm_minus": sigma_yy_minus,
            "sigma_xy_tm_minus": sigma_xy_minus,
            "sigma_xx_tm_eff": sigma_xx_total + (g_alpha - 1.0) * sigma_xx_plus,
            "sigma_yy_tm_eff": sigma_yy_total + (g_alpha - 1.0) * sigma_yy_plus,
            "sigma_xy_tm_eff": sigma_xy_total + (g_alpha - 1.0) * sigma_xy_plus,
        }
    )
    return split


def mixed_mode_energy_split(eps_xx, eps_yy, eps_xy, matprop, eps_r=0.0):
    """No-thermal mixed-mode TM source split for the COMSOL micro-notch route."""
    return _tm_source_split(eps_xx, eps_yy, eps_xy, matprop, eps_r=eps_r)
