"""The TP sharding plan of a MOSS backbone: which linear is column / row parallel and
how the fused block ops are re-bound once the weights are cut.

Split (per joint block and per text-refiner block):

    q, k, v          column (out features = heads * head_dim)
    o                row    (in features = heads * head_dim)
    ffn_up_gate      column (out features = 2 * ffn_dim)
    ffn_down         row    (in features = ffn_dim)

The blocks' fused ops (`linear_qk_rope`, `linear_situ`, `linear_gated_residual`) take
the weight themselves; they cannot see a shard, so the TP plan drops them and the blocks
bind the composed path (`MossLinear.forward` -> the TP wrapper). See docs/guides/
multi-gpu.md (tensor parallel).
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from kohakumoss.parallel.tp import (
    COLUMN,
    REPLICATE,
    ROW,
    copy_to_parallel,
    reduce_from_parallel,
    shard_parameter,
    tp_rank,
    tp_size,
    valid_width,
)
from kohakumoss_common.log import get_logger

log = get_logger(__name__)

# projection name -> TP kind (the attention / FFN names of JointBlock.proj)
PROJ_KIND = {
    "q": COLUMN,
    "k": COLUMN,
    "v": COLUMN,
    "o": ROW,
    "ffn_up_gate": COLUMN,
    "ffn_down": ROW,
}
# The text refiner is replicated whole: its qkv is one fused weight, so cutting it along
# the output axis would split q, k and v at different points (its 1-2 blocks are a small
# part of the parameter count). Every name maps to REPLICATE.
TEXT_PROJ_KIND = {
    "qkv": REPLICATE,
    "proj": REPLICATE,
    "ffn_up": REPLICATE,
    "ffn_gate": REPLICATE,
    "ffn_down": REPLICATE,
}


def shard_gate_up(p: Tensor, rank: int, world: int) -> Tensor:
    """A fused `[up | gate]` weight's shard: the up half's slice, then the gate half's
    (a plain chunk of the whole would give one rank all of up and the other all of
    gate, which `situ_glu`'s chunk(2) cannot read)."""
    if world <= 1 or p.numel() == 0:
        return p
    half = p.shape[0] // 2
    if half % world:
        raise ValueError(
            f"TP={world}: fused up/gate half {half} does not divide by {world}"
        )
    up = p[:half].chunk(world, dim=0)[rank]
    gate = p[half:].chunk(world, dim=0)[rank]
    return torch.cat((up, gate), dim=0).contiguous()


def shard_gate_up_parameter(p: nn.Parameter, rank: int, world: int) -> None:
    """Replace a fused up/gate parameter's data with its shard in place."""
    if world <= 1 or p.numel() == 0:
        return
    p.data = shard_gate_up(p.data, rank, world)


def _shard_adapter(linear: nn.Module, kind: str, rank: int, world: int) -> None:
    """Cut the LyCORIS adapter on `linear` the way the base was cut, so the adapter's
    matrices keep matching the (now local) weight:

    - column parallel (out features cut): `lora_up` [out, r] cuts dim 0;
    - row parallel (in features cut): `lora_down` [r, in] cuts dim 1.

    The other matrix keeps its shape (the axis it reads/writes was not cut). A
    `lora_mid` (LoCon, from a conv 3x3 reshape) needs the same treatment as the base's
    own axis and is cut with it; a DoRA magnitude follows the output axis.
    """
    lyco = linear._modules.get("lyco")
    if lyco is None:
        return []
    params = dict(lyco.named_parameters())
    targets: list[tuple[nn.Parameter, int]] = []
    if kind == COLUMN:
        for name in ("lora_up.weight", "dora_scale"):
            if name in params:
                targets.append((params[name], 0))
    elif kind == ROW:
        for name in ("lora_down.weight",):
            if name in params:
                targets.append((params[name], 1))
    cut = []
    gate_up = getattr(linear, "_tp_gate_up", False)
    for p, dim in targets:
        if p.numel() == 0 or p.shape[dim] % world:
            raise ValueError(
                f"TP={world}: adapter {p.shape} dim {dim} does not divide"
            )
        if gate_up and dim == 0:
            # the fused up/gate base cuts its two halves separately: do the same here
            p.data = shard_gate_up(p.data, rank, world)
        else:
            p.data = p.data.chunk(world, dim=dim)[rank].contiguous()
        cut.append(p)
    _shard_lyco_buffers(lyco, kind, rank, world)
    return cut


def _shard_lyco_buffers(lyco: nn.Module, kind: str, rank: int, world: int) -> None:
    """Cut the adapter's own buffers with the axis they follow (`lora_up.scale`, when a
    build carries one, follows the output axis)."""
    if kind != COLUMN:
        return
    for name, b in list(lyco.named_buffers()):
        if name == "lora_up.scale" and b.numel() and b.numel() % world == 0:
            setattr_buf(lyco, name, b.chunk(world, 0)[rank].contiguous())


def setattr_buf(root: nn.Module, path: str, value: Tensor) -> None:
    """Replace buffer `path` (`a.b.c`) of `root`."""
    *parents, leaf = path.split(".")
    module = root
    for part in parents:
        module = module._modules[part]
    module._buffers[leaf] = value


class TPPlan:
    """Every sharded linear of one backbone and the replicated parameters whose
    gradients are summed once per step (`replicated`)."""

    def __init__(self, backbone: nn.Module, world: int, rank: int):
        self.backbone = backbone
        self.world = world
        self.rank = rank
        self.sharded: list[tuple[nn.Module, str]] = []
        self.replicated: list[nn.Parameter] = []
        # every parameter tensor id that was cut (base weights / biases, adapter
        # matrices): the grad norm sums their squared norms over the TP ranks
        self.sharded_ids: set[int] = set()

    def add(self, linear: nn.Module, kind: str) -> None:
        """Record `linear`'s split and shard its weight (bias too, on the same axis)."""
        self.sharded.append((linear, kind))

    def check(self) -> list[str]:
        """Widths that do not divide over the TP ranks, checked before anything is cut
        (a zero-width shard would run an empty matmul)."""
        out = []
        for lin, kind in self.sharded:
            checks = [("weight", lin.weight)]
            bias = getattr(lin, "bias", None)
            if bias is not None and kind == COLUMN:
                checks.append(("bias", bias))
            for name, w in checks:
                if w is None or w.numel() == 0:
                    continue
                if name == "weight" and getattr(lin, "_tp_gate_up", False):
                    width = w.shape[0] // 2
                    what = "fused up/gate half"
                else:
                    dim = 0 if (kind == "column" or w.dim() < 2) else 1
                    width = w.shape[dim]
                    what = f"dim {dim}"
                if not valid_width(width, self.world):
                    out.append(
                        f"TP={self.world}: {_name_of(self.backbone, lin)} {kind} "
                        f"{name} {what} {width} does not divide by {self.world}"
                    )
        return out

    def apply(self) -> None:
        """Cut every recorded linear (widths were checked) and bind the TP forwards. A
        column-parallel bias shards with the output width; a row-parallel one is
        replicated (its output is the all-reduced full width). The fused up/gate
        projection cuts its two halves separately. A LyCORIS adapter's matrices are cut
        with the axis they read / write."""
        for lin, kind in self.sharded:
            if getattr(lin, "_tp_gate_up", False):
                shard_gate_up_parameter(lin.weight, self.rank, self.world)
            else:
                shard_parameter(lin.weight, kind, self.rank, self.world)
            if getattr(lin, "bias", None) is not None and kind == COLUMN:
                shard_parameter(lin.bias, kind, self.rank, self.world)
            # the base weight is sharded (the fused up/gate split included)
            self.sharded_ids.add(id(lin.weight))
            if getattr(lin, "bias", None) is not None and kind == COLUMN:
                self.sharded_ids.add(id(lin.bias))
            self.sharded_ids.update(
                id(p) for p in _shard_adapter(lin, kind, self.rank, self.world)
            )

    def wrap(self) -> None:
        """Install the TP forward on every sharded linear (the module's own
        parameters); the parameters that stay identical on every rank (norms, the
        embedders, the conditioning linears, the heads) are replicated. An adapter
        matrix is *never* replicated: even an uncut one (`lora_up` under a row-parallel
        base) takes a different slice of the output gradient on each rank, so its
        gradients SUM."""
        for lin, kind in self.sharded:
            _install_tp_forward(lin, kind)
        adapter_ids = {
            id(param)
            for name, param in self.backbone.named_parameters()
            if ".lyco." in name
        }
        self.replicated = [
            p
            for p in self.backbone.parameters()
            if id(p) not in self.sharded_ids and id(p) not in adapter_ids
        ]

    def note_sharded(self, params) -> None:
        """Record parameters that were cut (the adapter matrices) for the grad norm."""
        self.sharded_ids.update(id(p) for p in params)

    def sharded_params(self) -> list[nn.Parameter]:
        """Every parameter whose value is a slice (the sharded base weights / biases and
        the adapter matrices cut with them)."""
        return [
            p
            for p in self.backbone.parameters()
            if id(p) in self.sharded_ids
        ]


def _name_of(root: nn.Module, module: nn.Module) -> str:
    for name, m in root.named_modules():
        if m is module:
            return name or root.__class__.__name__
    return module.__class__.__name__


def _install_tp_forward(linear: nn.Module, kind: str) -> None:
    """Bind `MossLinear.forward` (and the adapters' bound forward, if any) to the TP
    communication around the module's own call. A module with a `forward` in its
    instance dict (an adapter bound it) keeps that callable as the inner one."""
    inner: Callable[[Tensor], Tensor] = vars(linear).get("forward") or linear.forward

    def forward(x: Tensor) -> Tensor:
        match kind:
            case "column":
                return inner(copy_to_parallel(x))
            case "row":
                return reduce_from_parallel(inner(x))
            case _:
                return inner(x)

    linear.forward = forward
    linear._tp_kind = kind


def plan_of(backbone: nn.Module) -> TPPlan | None:
    """The plan a previous `shard` installed, None off TP."""
    return getattr(backbone, "_tp_plan", None)


def shard(backbone: nn.Module, world: int | None = None, rank: int | None = None):
    """Shard `backbone` over the TP ranks (in place) and return its `TPPlan`; every
    rank calls it with the same plan. `world` / `rank` default to the process group's.

    The joint blocks and the text-refiner blocks get their projections cut; norms,
    embedders, conditioning linears and heads are replicated. A width that does not
    divide over the ranks raises before anything is cut. Each block's head count
    becomes the local one (the attention is split by head), and the ops are re-bound:
    the fused linear ops take a whole weight, so a sharded layer runs the composed
    path.
    """
    world = tp_size() if world is None else world
    rank = tp_rank() if rank is None else rank
    if world <= 1:
        return TPPlan(backbone, 1, 0)
    plan = TPPlan(backbone, world, rank)
    for block in getattr(backbone, "layers", ()):
        for name, kind in PROJ_KIND.items():
            plan.add(block.proj[name], kind)
        block.proj["ffn_up_gate"]._tp_gate_up = True
    text = getattr(getattr(backbone, "text_embedder", None), "blocks", ())
    for block in text:
        for name, kind in TEXT_PROJ_KIND.items():
            if kind != REPLICATE:
                plan.add(getattr(block, name), kind)
    problems = plan.check()
    if problems:
        raise ValueError("; ".join(problems))
    plan.apply()
    backbone._tp_plan = plan
    plan.wrap()
    _local_heads(backbone, world)
    if getattr(backbone, "ops", None) is not None:
        backbone.bind_ops()
    log.info(
        f"tensor parallel over {world} ranks: {len(plan.sharded)} linears sharded, "
        f"{len(plan.replicated)} parameters replicated"
    )
    return plan


def _local_heads(backbone: nn.Module, world: int) -> None:
    """Every joint block keeps `heads / world` heads (its q/k/v are split by head, the
    head dim unchanged). The text refiner is replicated whole, so its head count
    stays."""
    for block in getattr(backbone, "layers", ()):
        heads = getattr(block, "heads", None)
        if isinstance(heads, int):
            block.heads = max(heads // world, 1)


def split_points(backbone: nn.Module) -> dict[str, int]:
    """Per projection name, the TP kind it was cut with (a record for the run log)."""
    return {name: kind for name, kind in PROJ_KIND.items()}
