from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from compute_energy import field_grads  # noqa: E402
from field_computation import FieldComputation  # noqa: E402


class SpyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, inp):
        self.seen = inp.detach().clone()
        return torch.zeros((inp.shape[0], 3), dtype=inp.dtype, device=inp.device)


def _field(net, coord_normalization):
    return FieldComputation(
        net=net,
        domain_extrema=torch.tensor([[0.0, 0.01], [0.0, 0.01]], dtype=torch.float32),
        lmbda=torch.tensor([1.0e-6], dtype=torch.float32),
        theta=torch.tensor([torch.pi / 2.0], dtype=torch.float32),
        alpha_constraint="nonsmooth",
        top_u_mode="free",
        coord_normalization=coord_normalization,
    )


def test_coord_normalization_none_preserves_raw_network_input():
    net = SpyNet()
    field = _field(net, "none")
    inp = torch.tensor([[0.0025, 0.0075], [0.0100, 0.0000]], dtype=torch.float32)

    field.fieldCalculation(inp)

    assert torch.allclose(net.seen, inp)


def test_coord_normalization_unit_box_maps_domain_to_minus_one_one():
    net = SpyNet()
    field = _field(net, "unit_box")
    inp = torch.tensor(
        [[0.0, 0.0], [0.01, 0.01], [0.005, 0.005]],
        dtype=torch.float32,
    )

    field.fieldCalculation(inp)

    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(net.seen, expected)


def test_unit_box_keeps_boundary_ansatz_on_physical_y_coordinates():
    net = SpyNet()
    field = _field(net, "unit_box")
    inp = torch.tensor(
        [[0.002, 0.0], [0.004, 0.005], [0.006, 0.01]],
        dtype=torch.float32,
    )

    _u, v, _alpha = field.fieldCalculation(inp)

    expected_v = torch.tensor([0.0, 0.5e-6, 1.0e-6], dtype=torch.float32)
    assert torch.allclose(v, expected_v, atol=1.0e-12)


def test_t3_field_grads_use_physical_coordinates():
    inp = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.0, 0.02]], dtype=torch.float32)
    field = 2.0 * inp[:, 0] + 3.0 * inp[:, 1]
    area = torch.tensor([1.0e-4], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)

    grad_x, grad_y = field_grads(inp, field, area, tri)

    assert torch.allclose(grad_x, torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(grad_y, torch.tensor([3.0], dtype=torch.float32))


def test_coord_normalization_cli_and_no_local_geometry_loss_tokens():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    field_text = (ROOT / "field_computation.py").read_text(encoding="utf-8")
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    train_text = (ROOT / "train_mixed_tm.py").read_text(encoding="utf-8")

    assert "--coord-normalization" in config_text
    assert "coord_normalization" in config_text
    assert "coord_normalization" in field_text
    assert "coord_normalization=training_dict.get" in main_text
    assert "coord_mapping=field_comp.coord_mapping_diagnostics(inp)" in train_text

    for forbidden in ["notch_lip_loss", "lip_weight", "local_weight", "displacement_jump_target"]:
        assert forbidden not in config_text
        assert forbidden not in train_text
