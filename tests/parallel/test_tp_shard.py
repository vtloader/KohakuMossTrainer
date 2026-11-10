"""Tensor parallel: the sharding plan (which weight axis each projection cuts), the
shard arithmetic on a fused up/gate weight and on a 1-D bias, and the gather that
undoes it (model-free, one gloo group of one rank plus a two-rank world simulated by
plain tensor math)."""

import pytest
import torch

from kohakumoss.parallel.tp import (
    COLUMN,
    ROW,
    shard_slice,
    split_dim,
    valid_width,
)
from kohakumoss.parallel.tp_shard import (
    PROJ_KIND,
    TEXT_PROJ_KIND,
    shard_gate_up,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_split_dims():
    assert split_dim(COLUMN) == 0
    assert split_dim(ROW) == 1
    assert split_dim("replicate") is None


def test_projection_kinds():
    """q/k/v and the fused up/gate output are column parallel; o and ffn_down sum
    their partial outputs (row parallel)."""
    assert PROJ_KIND == {
        "q": COLUMN,
        "k": COLUMN,
        "v": COLUMN,
        "o": ROW,
        "ffn_up_gate": COLUMN,
        "ffn_down": ROW,
    }


def test_text_refiner_is_replicated():
    """The text refiner's qkv is one fused weight: cutting it along the output axis
    would split q, k and v at different points, so it stays whole."""
    assert set(TEXT_PROJ_KIND.values()) == {"replicate"}


def test_widths_must_divide():
    assert valid_width(128, 2) and valid_width(128, 1)
    assert not valid_width(127, 2)
    with pytest.raises(ValueError):
        shard_slice(torch.zeros(6, 5), COLUMN, 0, 4)


def test_column_shard_is_the_output_slice():
    w = torch.arange(16.0).reshape(8, 2)
    first, second = (shard_slice(w, COLUMN, r, 2) for r in range(2))
    assert torch.equal(torch.cat((first, second)), w), "the two shards reassemble w"
    assert first.shape == (4, 2)


def test_row_shard_is_the_input_slice():
    w = torch.arange(8.0).reshape(2, 4)
    first, second = (shard_slice(w, ROW, r, 2) for r in range(2))
    assert torch.equal(torch.cat((first, second), dim=1), w), "reassembled on dim 1"
    assert first.shape == (2, 2)


def test_bias_shards_on_dim_zero():
    b = torch.arange(4.0)
    first, second = (shard_slice(b, COLUMN, r, 2) for r in range(2))
    assert torch.equal(torch.cat((first, second)), b)


def test_fused_gate_up_keeps_both_halves():
    """A plain chunk of a [up | gate] weight would give one rank all of up and the
    other all of gate; `situ_glu`'s chunk(2) could not read it. Each rank gets half of
    up and half of gate."""
    up = torch.arange(8.0).reshape(8, 1)
    gate = up + 100
    fused = torch.cat((up, gate), dim=0)
    for rank in range(2):
        shard = shard_gate_up(fused, rank, 2)
        assert shard.shape == (8, 1)
        assert torch.equal(shard[:4], up[rank * 4 : (rank + 1) * 4]), "its up half"
        assert torch.equal(shard[4:], gate[rank * 4 : (rank + 1) * 4]), "its gate half"
    first, second = (shard_gate_up(fused, r, 2) for r in range(2))
    assert torch.equal(
        torch.cat((first[:4], second[:4])), up
    ), "the up halves reassemble up"
    assert torch.equal(
        torch.cat((first[4:], second[4:])), gate
    ), "the gate halves reassemble gate"


def test_gate_up_odd_half_fails():
    with pytest.raises(ValueError):
        shard_gate_up(torch.zeros(6, 2), 0, 2)


def check_wrong_axis(w: torch.Tensor) -> None:
    """A row shard cuts the input axis and a column shard the output axis: on a
    square weight they are different slices."""
    row, col = shard_slice(w, ROW, 0, 2), shard_slice(w, COLUMN, 0, 2)
    assert row.shape == col.shape
    must_fail(lambda: _same(row, col), "the row and column shards are the same slice")


def _same(a, b) -> None:
    assert torch.equal(a, b)


def test_row_and_column_shards_differ():
    must_fail(lambda: check_wrong_axis(torch.arange(16.0).reshape(4, 4)), "same cut")
