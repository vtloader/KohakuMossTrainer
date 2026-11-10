"""Tensor parallelism (TP): the sharded linears of a MOSS block, their autograd
all-reduces, and the build-time sharding of the model's own parameter objects.

A TP rank holds one slice of every sharded weight and runs the same tokens as every
other rank: TP splits the weights, not the batch, so the loss and the token counts are
identical everywhere and need no reduction.

The split follows the block's data flow:

    x -> norm (replicated) -> q/k/v (column: by head) -> attention (local heads)
      -> o (row: all-reduce) -> residual
      -> norm (replicated) -> ffn_up_gate (column: by ffn column) -> SiTU
      -> ffn_down (row: all-reduce) -> residual

Norms, the embedders, the conditioning linears and the heads stay replicated; their
gradients are summed once per step (`all_reduce_grads`). A full finetune also shards the
replicated modules' *weights* is NOT done: they are small next to the blocks.
See docs/guides/multi-gpu.md (tensor parallel).
"""

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn

from kohakumoss_common.log import get_logger

log = get_logger(__name__)

COLUMN = "column"
ROW = "row"
REPLICATE = "replicate"
SPLIT_KINDS = (COLUMN, ROW)


def split_dim(kind: str) -> int | None:
    """The weight axis a `kind` shards: 0 (out features) for column parallel, 1 (in
    features) for row parallel, None for a replicated one."""
    match kind:
        case "column":
            return 0
        case "row":
            return 1
        case _:
            return None


def tp_group():
    """The default process group (the TP ranks)."""
    return dist.group.WORLD if dist.is_initialized() else None


def tp_size() -> int:
    """Ranks the weights are split over (1 without a process group)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def tp_rank() -> int:
    """This process's TP rank (0 without a process group)."""
    return dist.get_rank() if dist.is_initialized() else 0


def valid_width(size: int, world: int) -> bool:
    """`size` splits evenly over `world` ranks."""
    return world <= 1 or size % world == 0


def shard_slice(t: Tensor, kind: str, rank: int, world: int) -> Tensor:
    """`t`'s slice for `rank` of `world`: contiguous along `split_dim(kind)`, `t` whole
    when replicated, `world` is 1 or `t` is empty. A 1-D tensor (a bias) splits along
    dim 0."""
    dim = split_dim(kind)
    if dim is None or world <= 1 or t.numel() == 0:
        return t
    if t.dim() < 2:
        dim = 0
    size = t.shape[dim]
    if size % world:
        raise ValueError(
            f"TP={world}: weight dim {dim} of {tuple(t.shape)} does not divide by "
            f"{world}"
        )
    return t.chunk(world, dim=dim)[rank].contiguous()


def shard_parameter(p: nn.Parameter, kind: str, rank: int, world: int) -> None:
    """Replace `p`'s data with its shard in place (the parameter object, hence the
    optimiser's identity, stays). A 1-D bias shards along dim 0 for both kinds: a
    column-parallel layer's bias has the output width, a row-parallel layer's has none
    (MOSS's row-parallel projections are bias-less, so only the former occurs)."""
    if split_dim(kind) is None or world <= 1 or p.numel() == 0:
        return
    p.data = shard_slice(p.data, kind, rank, world)


class _CopyToParallel(torch.autograd.Function):
    """Identity forward; backward all-reduces the gradient: a column-parallel layer's
    input is replicated, so every rank's input gradient is a partial sum."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def backward(ctx, g: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        return g


class _ReduceFromParallel(torch.autograd.Function):
    """All-reduce forward (a row-parallel layer's partial outputs sum); identity
    backward (every rank gets the full output gradient)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, g: Tensor) -> Tensor:
        return g


def copy_to_parallel(x: Tensor) -> Tensor:
    """The replicated input of a column-parallel linear (backward sums the gradient)."""
    return _CopyToParallel.apply(x)


def reduce_from_parallel(x: Tensor) -> Tensor:
    """Sum a row-parallel linear's partial output over the TP ranks."""
    return _ReduceFromParallel.apply(x)


class TPLinear(nn.Module):
    """The TP view of one linear: the local matmul plus the communication the layer's
    `kind` needs. `linear` is the model's own module, so its parameters (its shards)
    stay the ones the optimiser and the adapters see."""

    def __init__(self, linear: nn.Module, kind: str = REPLICATE):
        super().__init__()
        self.linear = linear
        self.kind = kind

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return getattr(self.linear, "bias", None)

    @property
    def in_features(self) -> int:
        """The local input width."""
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        """The local output width."""
        return self.weight.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        match self.kind:
            case "column":
                return self.linear(copy_to_parallel(x))
            case "row":
                return reduce_from_parallel(self.linear(x))
            case _:
                return self.linear(x)


def sum_grads(params: Iterable[nn.Parameter], group=None) -> None:
    """SUM the `.grad` of every parameter in `params` over the TP ranks, in place:
    each rank holds a different slice of the step's output (a sharded weight, or an
    adapter matrix cut with it), so the ranks' gradients are partial sums."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    grads = [p.grad for p in params if p is not None and p.grad is not None]
    if not grads:
        return
    dist.all_reduce_coalesced(grads, op=dist.ReduceOp.SUM, group=group)


def all_reduce_grads(params: Iterable[nn.Parameter], group=None) -> None:
    """Average the `.grad` of every replicated parameter over the TP ranks, in place:
    each rank computed the same partial gradient from the replicated module, so the
    mean is the full gradient (a SUM would scale it by the world size)."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    grads = [p.grad for p in params if p is not None and p.grad is not None]
    if not grads:
        return
    world = dist.get_world_size(group)
    if world <= 1:
        return
    dist.all_reduce_coalesced(grads, op=dist.ReduceOp.SUM, group=group)
    for g in grads:
        g.div_(world)
