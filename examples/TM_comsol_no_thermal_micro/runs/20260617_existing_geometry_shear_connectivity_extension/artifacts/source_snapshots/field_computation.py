import torch
import torch.nn as nn


class FieldComputation:
    """
    Constructs displacement and phase fields from NN outputs.

    Dirichlet boundary conditions are baked in with a y-only bubble ansatz:
    the bottom edge is fixed, the top edge has prescribed loading, and the
    left/right sides remain free. The bubble is dimensionless so the network
    correction does not vanish when the COMSOL micro geometry is only 0.01 mm
    high.
    """

    def __init__(
        self,
        net,
        domain_extrema,
        lmbda,
        theta,
        alpha_constraint="nonsmooth",
        top_u_mode="fixed",
        coord_normalization="none",
        load_case="tension",
    ):
        self.net = net
        self.domain_extrema = domain_extrema
        self.theta = theta
        self.lmbda = lmbda
        if top_u_mode not in {"fixed", "free"}:
            raise ValueError("top_u_mode must be 'fixed' or 'free'")
        self.top_u_mode = top_u_mode
        if coord_normalization not in {"none", "unit_box"}:
            raise ValueError("coord_normalization must be 'none' or 'unit_box'")
        self.coord_normalization = coord_normalization
        if load_case not in {"tension", "shear"}:
            raise ValueError("load_case must be 'tension' or 'shear'")
        self.load_case = load_case
        if alpha_constraint == "smooth":
            self.alpha_constraint = torch.sigmoid
        else:
            self.alpha_constraint = NonsmoothSigmoid(2.0, 1e-3)

    def network_input(self, inp):
        if self.coord_normalization == "none":
            return inp
        extrema = self.domain_extrema.to(dtype=inp.dtype, device=inp.device)
        mins = extrema[:, 0]
        spans = extrema[:, 1] - extrema[:, 0]
        if torch.any(spans <= 0.0):
            raise ValueError("domain_extrema must have positive ranges for coordinate normalization")
        return 2.0 * (inp - mins) / spans - 1.0

    def coord_mapping_diagnostics(self, inp):
        with torch.no_grad():
            mapped = self.network_input(inp)
            return {
                "coord_normalization": self.coord_normalization,
                "x_hat_min": float(torch.min(mapped[:, 0]).detach().cpu()),
                "x_hat_max": float(torch.max(mapped[:, 0]).detach().cpu()),
                "y_hat_min": float(torch.min(mapped[:, 1]).detach().cpu()),
                "y_hat_max": float(torch.max(mapped[:, 1]).detach().cpu()),
                "t3_gradients_use_physical_xy": True,
            }

    def fieldCalculation(self, inp):
        out = self.net(self.network_input(inp))
        raw_u = out[:, 0]
        raw_v = out[:, 1]
        alpha = self.alpha_constraint(out[:, 2])

        y = inp[:, -1]
        y0 = self.domain_extrema[1, 0]
        yL = self.domain_extrema[1, 1]
        eta = (y - y0) / (yL - y0)
        bubble = eta * (1.0 - eta)
        theta = torch.as_tensor(self.theta, dtype=inp.dtype, device=inp.device)

        if self.load_case == "shear":
            free_top_shape = eta + bubble
            u = self.lmbda * (eta + bubble * raw_u)
            v = self.lmbda * free_top_shape * raw_v
        else:
            if self.top_u_mode == "free":
                top_u_raw = raw_u
                u_shape = eta * top_u_raw + bubble * raw_u
            else:
                top_u_raw = torch.zeros_like(raw_u)
                u_shape = bubble * raw_u + eta * torch.cos(theta)
            u = u_shape * self.lmbda
            v = (bubble * raw_v + eta * torch.sin(theta)) * self.lmbda
        return u, v, alpha

    def update_hist_alpha(self, inp):
        _, _, pred_alpha = self.fieldCalculation(inp)
        return pred_alpha.detach()


class NonsmoothSigmoid(nn.Module):
    """
    Continuous piecewise linear increasing function with a central ramp.
    """

    def __init__(self, support=2.0, coeff=1e-3):
        super(NonsmoothSigmoid, self).__init__()
        self.support = support
        self.coeff = coeff

    def forward(self, x):
        a = x > self.support
        b = x < -self.support
        c = torch.logical_not(torch.logical_or(a, b))
        out = (
            a * (self.coeff * (x - self.support) + 1.0)
            + b * (self.coeff * (x + self.support))
            + c * (x / 2.0 / self.support + 0.5)
        )
        return out


def phase_field_bounds_stats(alpha):
    detached = alpha.detach()
    return {
        "min_d": float(torch.min(detached).cpu()),
        "max_d": float(torch.max(detached).cpu()),
        "mean_d": float(torch.mean(detached).cpu()),
        "n_d_lt_0": int(torch.sum(detached < 0.0).cpu()),
        "n_d_gt_1": int(torch.sum(detached > 1.0).cpu()),
    }
