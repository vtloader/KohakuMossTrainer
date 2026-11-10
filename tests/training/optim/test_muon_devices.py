"""Muon on every device: the Newton-Schulz iteration's device independence, the
optimiser's step / step_masked semantics, and the fused up/gate splits. Model-free."""

import pytest
import torch

from kohakumoss.training.optim.groups import (
    build_param_groups,
    muon_eligible,
    muon_splits,
)
from kohakumoss.training.optim.optimizers.muon import (
    NS_COEFFS,
    MuonHybrid,
    newton_schulz,
)
from tests._util.devices import DEVICES
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (128, 128)])
def test_newton_schulz_is_device_independent(device, shape):
    """The bf16 iteration must give the same orthogonal factor on the host and every
    accelerator (it is elementwise + matmul, no device-specific rounding)."""
    torch.manual_seed(0)
    g = torch.randn(*shape)
    cpu = newton_schulz(g.clone(), 5)
    other = newton_schulz(g.to(device), 5).cpu()
    cos = torch.nn.functional.cosine_similarity(cpu.flatten(), other.flatten(), dim=0)
    assert cos.item() > 1 - 1e-5, f"{device}: cosine {cos.item()}"
    rel = (cpu - other).norm().item() / max(cpu.norm().item(), 1e-12)
    assert rel < 1e-2, f"{device}: relative {rel}"


@pytest.mark.parametrize("device", DEVICES)
def test_muon_and_adamw_groups_step(device):
    """A Muon group (2-D) and an AdamW group (1-D) both step on `device`."""
    torch.manual_seed(0)
    muon_p = torch.nn.Parameter(torch.randn(16, 8, device=device))
    adam_p = torch.nn.Parameter(torch.randn(16, device=device))
    groups = [
        {
            "params": [muon_p],
            "lr": 1e-2,
            "weight_decay": 0.1,
            "use_muon": True,
            "splits": [1],
        },
        {
            "params": [adam_p],
            "lr": 1e-2,
            "weight_decay": 0.1,
            "use_muon": False,
            "splits": [1],
        },
    ]
    opt = MuonHybrid(groups)
    muon_p.grad = torch.randn_like(muon_p)
    adam_p.grad = torch.randn_like(adam_p)
    before = (muon_p.detach().clone(), adam_p.detach().clone())
    opt.step()
    assert not torch.equal(muon_p.detach(), before[0]), "the Muon group did not step"
    assert not torch.equal(adam_p.detach(), before[1]), "the AdamW group did not step"
    assert torch.isfinite(muon_p.detach()).all() and torch.isfinite(adam_p.detach()).all()


@pytest.mark.parametrize("device", DEVICES)
def test_a_flagged_step_changes_nothing(device):
    """`step_masked(found_inf=1)` leaves the parameter and the momentum untouched (the
    fused-step bundle's non-finite skip)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16, device=device))
    groups = [
        {"params": [p], "lr": 1e-2, "weight_decay": 0.1, "use_muon": True, "splits": [1]}
    ]
    opt = MuonHybrid(groups)
    p.grad = torch.randn_like(p)
    opt.step()
    after_step = p.detach().clone()
    momentum = opt.state[p]["momentum_buffer"].clone()
    p.grad = torch.full_like(p, float("inf"))
    opt.step_masked(torch.tensor(1.0, device=device))
    assert torch.equal(p.detach(), after_step), "a flagged step moved the parameter"
    assert torch.equal(
        opt.state[p]["momentum_buffer"], momentum
    ), "a flagged step moved the momentum"
    must_fail(
        lambda: _equal(torch.equal(p.detach(), after_step), False), "the step was skipped"
    )


def _equal(a, b) -> None:
    assert a == b


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_qkv_splits(device, dtype):
    """A fused qkv weight trains with its three chunks orthogonalised separately."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3 * 16, 8, device=device, dtype=dtype))
    groups = [
        {"params": [p], "lr": 1e-2, "weight_decay": 0.0, "use_muon": True, "splits": [3]}
    ]
    opt = MuonHybrid(groups)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert not torch.equal(p.detach(), before)
    assert torch.isfinite(p.detach()).all()
    assert p.dtype == dtype


def test_muon_eligibility():
    """Only 2-D matrices (embeddings and the head excluded) train under Muon; a fused
    qkv / up-gate weight carries its chunk count."""
    qkv = torch.nn.Parameter(torch.zeros(384, 128))
    norm = torch.nn.Parameter(torch.zeros(128))
    assert muon_eligible("layers.0.proj.qkv.weight", qkv)
    assert not muon_eligible("layers.0.norm1.weight", norm)
    assert muon_splits("layers.0.proj.qkv.weight") == 3
    assert muon_splits("layers.0.proj.ffn_up_gate.weight") == 2
    assert muon_splits("layers.0.proj.q.weight") == 1


def test_groups_split_muon_from_adamw():
    """The parameter groups route the matrices to Muon and everything else to AdamW."""
    named = [
        ("layers.0.proj.qkv.weight", torch.nn.Parameter(torch.zeros(384, 128))),
        ("layers.0.proj.o.weight", torch.nn.Parameter(torch.zeros(128, 128))),
        ("layers.0.norm1.weight", torch.nn.Parameter(torch.zeros(128))),
    ]
    groups = build_param_groups(named, 1e-2, 0.1, "moss", use_muon=True)
    muon = [g for g in groups if g["use_muon"]]
    adamw = [g for g in groups if not g["use_muon"]]
    assert len(muon) == 1 and muon[0]["splits"] == [3, 1], muon
    assert len(adamw) == 1 and len(adamw[0]["params"]) == 1, adamw


def test_ns_coefficients():
    """The iteration's coefficients are the published quintic (3.4445, -4.775, 2.0315)."""
    assert NS_COEFFS == (3.4445, -4.7750, 2.0315)
