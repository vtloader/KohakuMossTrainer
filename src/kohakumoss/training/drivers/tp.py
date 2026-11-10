"""Checkpoint parts under tensor parallelism: the sharded projections gathered back
into the full weights, the replicated entries taken from rank 0, the optimiser state
written per rank.

A TP rank's `trainable` state holds its shard of the sharded projections and the full
value of the replicated parameters, so the gather is per entry: a sharded entry is
concatenated along the axis it was cut on (in rank order) and a replicated entry is
taken from rank 0. See docs/guides/multi-gpu.md (tensor parallel).
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn

from kohakumoss.training.drivers.common import Parts
from kohakumoss.training.state import TrainState
from kohakumoss_common.log import get_logger

log = get_logger(__name__)


def tp_group():
    return dist.group.WORLD


def shard_kinds(plan) -> dict[str, str]:
    """Every sharded entry's full state-dict name -> how it was cut: `column` / `row`
    (the plain splits), `gate_up` (a fused up/gate weight, its two halves cut
    separately) or `bias` (a column-parallel bias, dim 0). Only entries that were
    really cut appear."""
    out = {}
    for linear, kind in plan.sharded:
        name = _module_name(plan.backbone, linear)
        if name is None:
            continue
        out[f"{name}.weight"] = (
            "gate_up" if getattr(linear, "_tp_gate_up", False) else kind
        )
        if getattr(linear, "bias", None) is not None and kind == "column":
            out[f"{name}.bias"] = "bias"
        lyco = linear._modules.get("lyco")
        if lyco is not None:
            # only the matrix the plan cut; the other keeps its full shape
            matrix = "lora_up" if kind == "column" else "lora_down"
            if f"{matrix}.weight" in dict(lyco.named_parameters()):
                cut_kind = "gate_up" if (
                    kind == "column" and getattr(linear, "_tp_gate_up", False)
                ) else kind
                out[f"{name}.lyco.{matrix}.weight"] = cut_kind
    return out


def _module_name(root: nn.Module, module: nn.Module) -> str | None:
    for name, m in root.named_modules():
        if m is module:
            return name
    return None


GATHER_DIM = {"column": 0, "row": 1, "bias": 0}


def _gather_entry(t: Tensor, kind: str, world: int) -> Tensor:
    """`t`'s full value: every rank's shard concatenated along the axis it was cut on
    (a fused up/gate weight: its two halves separately)."""
    if world <= 1:
        return t
    if kind == "gate_up":
        half = t.shape[0] // 2
        up = _gather_entry(t[:half], "column", world)
        gate = _gather_entry(t[half:], "column", world)
        return torch.cat((up, gate), dim=0)
    dim = GATHER_DIM.get(kind)
    if dim is None:
        return t
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous(), group=tp_group())
    return torch.cat(gathered, dim=dim)


def gather_state(state: dict, kinds: dict[str, str], world: int) -> dict:
    """Full tensors of `state`: sharded entries concatenated, the rest from rank 0."""
    out = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    for name, t in state.items():
        kind = kinds.get(name)
        if kind is None:
            # replicated: rank 0's copy
            if rank == 0:
                out[name] = t.detach().cpu()
            continue
        full = _gather_entry(t.detach(), kind, world)
        if rank == 0:
            out[name] = full.cpu()
    return out


def place_state(state: dict, kinds: dict[str, str], world: int, rank: int) -> dict:
    """`state` (full tensors) -> this rank's slice of every sharded entry."""
    out = dict(state)
    for name, kind in kinds.items():
        if name not in state or world <= 1:
            continue
        out[name] = _shard_entry(state[name], kind, rank, world)
    return out


def _shard_entry(t: Tensor, kind: str, rank: int, world: int) -> Tensor:
    """`t`'s `rank` slice of `world`, cut the way its TP kind was (a fused up/gate
    weight: its two halves separately)."""
    if kind == "gate_up":
        half = t.shape[0] // 2
        up = t[:half].chunk(world, dim=0)[rank]
        gate = t[half:].chunk(world, dim=0)[rank]
        return torch.cat((up, gate), dim=0).contiguous()
    dim = GATHER_DIM.get(kind)
    if dim is None:
        return t
    return t.chunk(world, dim=dim)[rank].contiguous()


class TpParts(Parts):
    """Parts of a TP-sharded backbone: rank 0 writes the full trainable / EMA state and
    the full optimiser state is summed by name, every rank takes part (collectives)."""

    def __init__(self, *args, plan, model, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = plan
        self.model = model
        self.kinds = shard_kinds(plan)

    def full_trainable(self) -> dict:
        return gather_state(self.trainable.state_dict(), self.kinds, self.world_of())

    def world_of(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def __call__(self, state: TrainState) -> dict:
        parts = super().__call__(state)
        parts["trainable"] = self.full_trainable()
        parts["ema"] = gather_state(
            self.ema.state_dict(), self.kinds, self.world_of()
        )
        # every rank writes its own optimiser shard
        parts["optim_rank"] = parts["optim"]
        parts["optim"] = {}
        return parts

    def before_export(self) -> None:
        """Load the full trainable state on rank 0, so the export sees whole weights
        (collective: every rank calls it)."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        full = self.full_trainable()
        if dist.get_rank() == 0:
            self.trainable.load_state_dict(full)

    def restore(self, ckpt: dict, state: TrainState) -> None:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            super().restore(ckpt, state)
            return
        # every rank reads the full state (rank 0 wrote it) and slices its shard
        world, rank = self.world_of(), dist.get_rank()
        ckpt = dict(ckpt)
        ckpt["trainable"] = place_state(ckpt["trainable"], self.kinds, world, rank)
        ckpt["ema"] = place_state(ckpt["ema"], self.kinds, world, rank)
        ckpt["optim"] = ckpt.get("optim_rank", ckpt.get("optim", {}))
        super().restore(ckpt, state)
