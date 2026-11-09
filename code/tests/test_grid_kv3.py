"""The three-axis grid KV sharing (AttentionConfig.grid_depth / depth_dim on top of kv_sharing "grid"; ced/model/attention.py "Grid KV sharing").

 1. Off is main: with the two new keys at their defaults every existing model (MQA, GQA-2, GQA-4, the 2-axis grids) is what it was, bit for bit, in its
    parameters, state-dict names and shapes, saved config, random-number use and outputs (checksums taken from main at 46e034675, BEFORE the change; and the
    two-axis assembly against a verbatim copy of main's code).
 2. The key, its validation by name, the saved config.
 3. The layout: head h of an R x C x D x P grid sits in the cell c = h // P = (i, j, k) = (c // (C D), (c // D) % C, c % D) and reads
    [axis-1 component i ; axis-2 component j ; axis-3 component k]; assemble_grid against a hand-built reference.
 4. The keys and values against a reference written from the definition (norm over the whole head with ONE k_norm, then the leading rotary dimensions).
 5. The layer's output and the whole model against the multi-head layer with block-assembled weights (logits and gradients) on the reference, sdpa and flex
    backends, with the cap, the NoPE mix, ssmax, R1-entry and TST's token-scale rotary.
 6. Causality, cached generation equal to the full forward (every chunking), the cache holds the raw components (256 numbers in the arm), batch-axis
    concatenation, checkpoint resume.
 7. The parameter counts (the 2 x 2 x 2 attention has the 2 x 2's and GQA-2's), and the consumers that read a grid (the decode engine refuses it by name,
    cost_model, the attention table).
(The arm's real 200M shapes, the runs file and the release lines are tests/test_gridkv_runs.py; the diagnostics on 3-axis models are tests/test_attn_diag_grid.py.)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from ced.config import AttentionConfig, TrainConfig, from_dict, tiny_v2_control_config, to_dict
from ced.model.attention import GatedAttention, GridSpec, assemble_grid, grid_spec
from ced.model.control import build_model
from ced.model.rope import rotary_dims
from ced.model.sequence import sequence_context
from ced.train.trainer import SAVED_ATTENTION_DEFAULTS, Trainer, saved_config_dict

from helpers import make_model, perturb_parameters, set_config
from test_fast_attention import flex_cpu_backward  # noqa: F401  (a pytest fixture)
from test_grid_kv import (ATT, CAPPED_FIVE, FIVE, GQA2_4Q, GQA2_8Q, GQA4_8Q, GRID22, GRID22P2, GRID24, GRIDU, HPC, MIXCAP, NOQK, R1E_KEY, TOL, _cat_leaf, cache_numbers_a_token,
                          grid_layer, grid_train_config, key_sets, rms, rot)
from pos_arms_helpers import chunked, fast, long_doc_rows
from test_resume import assert_same_state

L = 48
D3, DD3 = ATT + "grid_depth", ATT + "depth_dim"
# THE ARM IN SMALL: 8 query heads, a 2 x 2 x 2 grid of single-head cells, axis widths 8 / 4 / 4 = the tiny model's head dimension 16 (the arm: 64 / 32 / 32 = 128; the
# rotary part rope_fraction 0.25 is the leading 4 of 16 dimensions here, 32 of 128 there, inside axis 1's first 8 / 64)
GRID222 = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 2, D3: 2, ATT + "row_dim": 8, ATT + "col_dim": 4, DD3: 4, ATT + "n_q_heads": 8}
# a general grid with unequal counts and widths and a narrower head (not an arm): 2 x 3 x 2 = 12 heads, widths 6 / 4 / 2 = 12
GRID232 = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 3, D3: 2, ATT + "row_dim": 6, ATT + "col_dim": 4, DD3: 2, ATT + "n_q_heads": 12, ATT + "head_dim": 12}
# two heads to each cell of the 2 x 2 x 2 grid (not an arm): 16 query heads
GRID222P2 = {**GRID222, ATT + "n_q_heads": 16, HPC: 2}
SHAPES3 = {"2x2x2": GRID222, "2x3x2_unequal": GRID232, "2x2x2_pairs": GRID222P2}
THIRD_AXIS_KEYS = ("grid_depth", "depth_dim")


# ------------------------------------------------------------------------------------------
# 1. off is main: the numbers of the existing models, taken from main BEFORE the change

def checksum(t) -> float:
    """sum(w * t) over the flattened tensor with the fixed weights w_n = cos(0.37 n + 1) (NaN and inf as 0): sensitive to every element and its place."""
    t = torch.as_tensor(t).detach().double().flatten()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.cos(torch.arange(t.numel(), dtype=torch.float64) * 0.37 + 1.0)
    return float((t * w).sum())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def flat(tensors) -> torch.Tensor:
    return torch.cat([t.detach().double().flatten() for t in tensors])


def attention_sha(model_cfg) -> str:
    saved = saved_config_dict(TrainConfig(model=model_cfg))["model"]
    return sha(json.dumps([saved["encoder"]["attention"], saved["decoder"]["attention"]], sort_keys=True))


DOCS36 = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])


def layer_pin(over: dict) -> dict:
    """What one attention layer is, in numbers: its state-dict names and shapes, its parameters at the seed (and so the random numbers its construction drew),
    the generator state after construction, its output and cache on a fixed input, the gradients, the output read in chunks through the cache, and the saved
    attention config. Float64."""
    model_cfg = set_config(tiny_v2_control_config(), over)
    torch.manual_seed(0)
    layer = GatedAttention(32, model_cfg.encoder.attention, 1e-6).double()
    rng = torch.rand(4, dtype=torch.float64)                      # the global generator after construction: the layer's random-number use
    perturb_parameters(layer, 0.3, seed=1)
    sd = layer.state_dict()
    x = torch.randn(2, 36, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
    ctx, _ = sequence_context(DOCS36, None)
    out, keys = layer(x, ctx)
    out.square().mean().backward()
    pieces, cache, state = [], None, None
    with torch.no_grad():
        for a, b in [(0, 10), (10, 11), (11, 30), (30, 36)]:
            c, state = sequence_context(DOCS36[:, a:b], state)
            o, cache = layer(x.detach()[:, a:b], c, cache)
            pieces.append(o)
    return {
        "names": sha("\n".join(f"{k}{tuple(v.shape)}" for k, v in sd.items())), "n_params": sum(p.numel() for p in layer.parameters()),
        "params": checksum(flat(sd.values())), "rng": checksum(rng), "out": checksum(out), "cache_k": checksum(keys.k), "cache_v": checksum(keys.v),
        "cache_pos": -1.0 if keys.pos is None else checksum(keys.pos), "grads": checksum(flat([p.grad for p in layer.parameters()] + [x.grad])),
        "chunked": checksum(torch.cat(pieces, dim=1)), "attention_cfg": attention_sha(model_cfg),
    }


def model_pin(over: dict) -> dict:
    """The same for a whole tiny model: names and shapes, parameters, generator state, logits, gradients, cached generation, the cache, the saved attention config."""
    model_cfg = set_config(tiny_v2_control_config(), over)
    torch.manual_seed(0)
    model = build_model(model_cfg).double().eval()
    rng = torch.rand(4, dtype=torch.float64)
    perturb_parameters(model, seed=0)
    sd = model.state_dict()
    tokens, docs, dec = long_doc_rows(2, L, model_cfg.vocab_size, seed=11)
    logits = model(tokens, docs).logits
    logits.square().mean().backward()
    sets = key_sets(model.prefill(tokens, docs, dec).cache.encoder)
    return {
        "names": sha("\n".join(f"{k}{tuple(v.shape)}" for k, v in sd.items())), "n_params": sum(p.numel() for p in model.parameters()),
        "params": checksum(flat(sd.values())), "rng": checksum(rng), "logits": checksum(logits),
        "grads": checksum(flat([p.grad for p in model.parameters() if p.grad is not None])),
        "cached": checksum(chunked(model, tokens, docs, dec, [5, 17, 30])), "cache": checksum(flat([t for s in sets for t in (s.k, s.v)])),
        "attention_cfg": attention_sha(model_cfg),
    }


CAP7 = {ATT + "rope_max_distance": 7, ATT + "temperature": "ssmax", ATT + "temperature_ref_len": 32}
LAYER_PINS = {
    "mqa": {}, "mqa_capped_ssmax": CAP7, "mqa_nope": {ATT + "rope_fraction": 0.0},
    "gqa2_8q": GQA2_8Q, "gqa2_8q_capped_ssmax": {**GQA2_8Q, **CAP7}, "gqa4_8q": GQA4_8Q, "gqa2_4q": GQA2_4Q,
    "grid22": GRID22, "grid22_capped_ssmax": {**GRID22, **CAP7}, "grid22_pairs": GRID22P2, "grid22_pairs_capped_ssmax": {**GRID22P2, **CAP7},
    "grid24": GRID24, "grid24_capped": {**GRID24, ATT + "rope_max_distance": 7}, "grid24_unequal": GRIDU, "grid22_rotary_into_columns": {**GRID22, ATT + "rope_fraction": 1.0},
}
MODEL_PINS = {
    "mqa_r1e_capped_mix": {**MIXCAP, R1E_KEY: "block_entry"}, "gqa2_8q_r1e_capped_mix": {**GQA2_8Q, **MIXCAP, R1E_KEY: "block_entry"},
    "grid22_pairs_r1e_capped_mix": {**GRID22P2, **MIXCAP, R1E_KEY: "block_entry"}, "grid24_capped_mix": {**GRID24, **MIXCAP},
}


def all_pins() -> dict:
    """The pins of every existing model; the golden file is this function's output on the tree BEFORE the three-axis keys existed."""
    return {"layer": {n: layer_pin(o) for n, o in LAYER_PINS.items()}, "model": {n: model_pin({**o, **FIVE}) for n, o in MODEL_PINS.items()}}


# The numbers of main at 46e034675 (this tree before grid_depth / depth_dim existed): tests/golden/grid3_pins_46e034675.json, written by running all_pins() on that tree
# (the control VM, Python 3.12, torch 2.14.1+cpu, float64). The tolerance is 1e-9 relative, which float64 on another build holds by orders of magnitude; the strings and
# integers (state-dict names and shapes, the parameter count, the saved attention config) are exact.
GOLDEN = json.loads((Path(__file__).resolve().parent / "golden" / "grid3_pins_46e034675.json").read_text())


def assert_pin(got: dict, want: dict, label: str) -> None:
    assert set(got) == set(want), label
    for key, w in want.items():
        if isinstance(w, (str, int)):
            assert got[key] == w, (label, key)
        else:
            assert got[key] == pytest.approx(w, rel=1e-9, abs=1e-9), (label, key)


@pytest.mark.parametrize("name", list(LAYER_PINS))
def test_an_attention_layer_without_the_third_axis_is_main_bit_for_bit(name):
    """MQA, GQA-2, GQA-4 and the two-axis grids (single heads, pairs, 2 x 4, unequal, rotary into the columns), plain and capped: the same state-dict names and shapes, the same
    parameter count and values at the seed (so the same random numbers drawn), the same generator state afterwards, the same output, cache, gradients and chunked output, and the
    saved attention config without the new keys, as main at 46e034675 had them."""
    assert_pin(layer_pin(LAYER_PINS[name]), GOLDEN["layer"][name], name)


@pytest.mark.parametrize("name", list(MODEL_PINS))
def test_a_whole_model_without_the_third_axis_is_main_bit_for_bit(name):
    assert_pin(model_pin({**MODEL_PINS[name], **FIVE}), GOLDEN["model"][name], name)


def old_assemble_grid(comp: torch.Tensor, g: GridSpec) -> torch.Tensor:
    """ced/model/attention.py assemble_grid of main at 46e034675, verbatim (the two-axis code before the third axis existed)."""
    B, S, one, W = comp.shape
    assert one == 1 and W == g.width, (tuple(comp.shape), g)
    P = g.heads_per_cell
    rows = comp[..., : g.row_width].reshape(B, S, g.rows, 1, 1, g.row_dim).expand(B, S, g.rows, g.cols, P, g.row_dim)
    cols = comp[..., g.row_width :].reshape(B, S, 1, g.cols, 1, g.col_dim).expand(B, S, g.rows, g.cols, P, g.col_dim)
    return torch.cat([rows, cols], dim=-1).reshape(B, S, g.n_heads, g.head_dim)


@pytest.mark.parametrize("spec", [GridSpec(2, 2, 8, 8), GridSpec(2, 2, 8, 8, 2), GridSpec(2, 4, 8, 8), GridSpec(2, 3, 5, 4, 3), GridSpec(1, 1, 3, 2), GridSpec(2, 2, 64, 64, 2)])
def test_the_two_axis_assembly_is_mains_code_bit_for_bit(spec):
    comp = torch.randn(2, 5, 1, spec.width, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    assert spec.depth == 0 and spec.depth_dim == 0 and spec.n_axes == 2 and spec.depth_width == 0
    got, want = assemble_grid(comp, spec), old_assemble_grid(comp, spec)
    assert got.shape == want.shape and got.dtype == want.dtype and torch.equal(got, want)
    # the two-axis properties are the old ones, and cell_of still returns (i, j)
    assert spec.n_cells == spec.rows * spec.cols and spec.n_heads == spec.n_cells * spec.heads_per_cell and spec.head_dim == spec.row_dim + spec.col_dim
    assert spec.width == spec.row_width + spec.col_width
    for h in range(spec.n_heads):
        c = h // spec.heads_per_cell
        assert spec.cell_of(h) == (c // spec.cols, c % spec.cols)


# ------------------------------------------------------------------------------------------
# 2. the keys, their validation by name, the saved config


def test_the_third_axis_is_off_by_default_and_saves_nothing():
    a = AttentionConfig()
    assert (a.grid_depth, a.depth_dim) == (0, 0) and grid_spec(a) is None
    assert set(THIRD_AXIS_KEYS) <= set(SAVED_ATTENTION_DEFAULTS) and SAVED_ATTENTION_DEFAULTS["grid_depth"] == 0 and SAVED_ATTENTION_DEFAULTS["depth_dim"] == 0
    saved = saved_config_dict(TrainConfig())
    for section in ("encoder", "decoder"):
        for key in THIRD_AXIS_KEYS:
            assert key not in saved["model"][section]["attention"], (section, key)
    assert to_dict(from_dict(TrainConfig, saved)) == to_dict(TrainConfig())
    # a two-axis grid saves exactly the keys it saved before the third axis existed
    base = set(saved["model"]["encoder"]["attention"])
    two = saved_config_dict(TrainConfig(model=set_config(tiny_v2_control_config(), GRID22)))["model"]["encoder"]["attention"]
    assert set(two) - base == {"kv_sharing", "grid_rows", "grid_cols", "row_dim", "col_dim"}


def test_the_third_axis_keys_are_saved_and_round_trip():
    for over in (GRID222, GRID232, GRID222P2):
        cfg = TrainConfig(model=set_config(tiny_v2_control_config(), over))
        att = saved_config_dict(cfg)["model"]["encoder"]["attention"]
        assert att["grid_depth"] == over[D3] and att["depth_dim"] == over[DD3] and att["kv_sharing"] == "grid"
        assert "grid_depth" not in saved_config_dict(cfg)["model"]["decoder"]["attention"]
        assert from_dict(TrainConfig, saved_config_dict(cfg)).model.encoder.attention == cfg.model.encoder.attention


@pytest.mark.parametrize("over,match", [
    ({D3: 2}, "gqa"),                                                                          # a third-axis key under gqa would be ignored
    ({DD3: 4}, "gqa"),
    ({"decoder.attention.grid_depth": 2}, "gqa"),
    ({**GRID22, D3: 2}, "set together"),                                                       # a count without a width, and the other way
    ({**GRID22, DD3: 4}, "set together"),
    ({**GRID222, D3: -1}, "set together"),
    ({**GRID222, D3: 2.0}, "set together"),                                                    # integers only
    ({**GRID222, D3: 3}, "n_q_heads"),                                                         # 2 x 2 x 3 != 8 heads
    ({**GRID222, DD3: 2}, "head_dim"),                                                         # 8 + 4 + 2 != 16
    ({**GRID222, ATT + "col_dim": 8}, "head_dim"),                                             # 8 + 8 + 4 != 16
    ({**GRID222P2, HPC: 1}, "n_q_heads"),                                                      # 2 x 2 x 2 x 1 != 16 heads
    ({**GRID222, ATT + "n_kv_heads": 2}, "n_kv_heads"),
    ({**GRID222, ATT + "grid_cols": 0}, "integers"),
    ({**GRID222, "decoder.attention.kv_sharing": "grid"}, "decoder"),
    ({**GRID222, ATT + "backend": "fa4"}, "backend"),
    ({**GRID222, ATT + "normalizer": "asentmax"}, "normalizer"),
    ({**GRID222, ATT + "row_term": "pdp"}, "row_term"),
    ({**GRID222, "loop.pass2_gate": "uniform"}, "pass2_gate|kv_sharing"),
    ({**GRID222, "cp_head.enabled": True}, "cp_head|kv_sharing"),
    ({**GRID222, "mtp.enabled": True}, "mtp|kv_sharing"),
])
def test_the_third_axis_refusals_are_by_name(over, match):
    with pytest.raises((ValueError, AssertionError), match=match):
        build_model(set_config(tiny_v2_control_config(), over))


def test_another_arch_refuses_the_three_axis_grid():
    with pytest.raises(ValueError, match="arch"):
        build_model(set_config(tiny_v2_control_config(), {**GRID222, "arch": "ced_v2"}))


def test_the_decode_engine_refuses_the_three_axis_grid_by_name():
    from ced.infer.engine import DecodeEngine

    with pytest.raises(AssertionError, match="kv_sharing 'grid'"):
        DecodeEngine(make_model("v2_control", {**GRID222, **FIVE}), max_len=8, attn_backend="matmul")


def test_the_grid_spec_of_the_arm_is_the_designed_one():
    """The arm at 200M (8 query heads of 128): axes 2 x 2 x 2 of widths 64 / 32 / 32, one head to a cell, 256 numbers a token for the keys and for the values."""
    cfg = AttentionConfig(n_q_heads=8, n_kv_heads=1, head_dim=128, kv_sharing="grid", grid_rows=2, grid_cols=2, grid_depth=2, row_dim=64, col_dim=32, depth_dim=32)
    g = grid_spec(cfg)
    assert g == GridSpec(2, 2, 64, 32, 1, 2, 32)
    assert g.axes == ((2, 64), (2, 32), (2, 32)) and g.n_axes == 3 and g.n_cells == g.n_heads == 8 and g.head_dim == 128
    assert (g.row_width, g.col_width, g.depth_width, g.width) == (128, 64, 64, 256)
    assert g.width == GridSpec(2, 2, 64, 64, 2).width == 2 * 128                  # = the 2 x 2 pairs grid's cache = GQA-2's (2 KV heads of 128)
    assert [g.cell_of(h) for h in range(8)] == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    # the rotary part (rope_fraction 0.25: the leading 32 of 128 dimensions) lies inside axis 1 (64 wide), as in the 2 x 2's row
    assert rotary_dims(128, 0.25) == 32 <= g.row_dim


# ------------------------------------------------------------------------------------------
# 3. the layout


def cell_by_hand(g: GridSpec, h: int) -> tuple:
    """(i, j, k) of query head h, written out: c = h // heads_per_cell, row-major over (i, j, k) with the third axis the fastest."""
    c = h // g.heads_per_cell
    return c // (g.cols * g.depth), (c // g.depth) % g.cols, c % g.depth


def head_parts(comp: torch.Tensor, g: GridSpec, i: int, j: int, k: int) -> torch.Tensor:
    """[axis-1 component i ; axis-2 component j ; axis-3 component k] of comp (..., width), from the layout: axis 1's components, then axis 2's, then axis 3's."""
    a2, a3 = g.rows * g.row_dim, g.rows * g.row_dim + g.cols * g.col_dim
    return torch.cat([comp[..., i * g.row_dim:(i + 1) * g.row_dim], comp[..., a2 + j * g.col_dim:a2 + (j + 1) * g.col_dim],
                      comp[..., a3 + k * g.depth_dim:a3 + (k + 1) * g.depth_dim]], dim=-1)


@pytest.mark.parametrize("hpc", [1, 2, 3])
def test_assemble_grid_layout_with_three_axes(hpc):
    g = GridSpec(rows=2, cols=3, row_dim=5, col_dim=4, heads_per_cell=hpc, depth=2, depth_dim=3)
    comp = torch.randn(2, 7, 1, g.width, dtype=torch.float64)
    heads = assemble_grid(comp, g)
    assert g.width == 2 * 5 + 3 * 4 + 2 * 3 and g.head_dim == 12 and g.n_cells == 12 and g.n_heads == 12 * hpc
    assert heads.shape == (2, 7, 12 * hpc, 12)
    for h in range(g.n_heads):
        i, j, k = cell_by_hand(g, h)
        assert g.cell_of(h) == (i, j, k)
        assert (i, j, k) == ((h // hpc) // 6, ((h // hpc) // 2) % 3, (h // hpc) % 2)           # the documented mapping: row-major, the third axis the fastest
        assert torch.equal(heads[:, :, h], head_parts(comp[..., 0, :], g, i, j, k)), (h, i, j, k)
    # a pure re-indexing: heads of a cell are the same numbers; heads that share an axis index share that axis' part and no other
    for a in range(g.n_heads):
        for b in range(g.n_heads):
            ca, cb = g.cell_of(a), g.cell_of(b)
            for axis, part in enumerate((slice(0, 5), slice(5, 9), slice(9, 12))):
                assert torch.equal(heads[:, :, a, part], heads[:, :, b, part]) == (ca[axis] == cb[axis]), (a, b, axis)
            assert torch.equal(heads[:, :, a], heads[:, :, b]) == (ca == cb)


def test_the_arm_refines_the_two_by_two_pairs_grid():
    """SET A's 2 x 2 grid over head pairs puts heads 2c and 2c + 1 in cell c = (i, j); the 2 x 2 x 2 arm puts them in (i, j, 0) and (i, j, 1): the same row and column of every
    head, the pair told apart by the third axis."""
    pairs, arm = GridSpec(2, 2, 64, 64, 2), GridSpec(2, 2, 64, 32, 1, 2, 32)
    assert pairs.n_heads == arm.n_heads == 8
    for h in range(8):
        assert arm.cell_of(h)[:2] == pairs.cell_of(h) and arm.cell_of(h)[2] == h % 2
    # and the same columns it shares: heads 0-3 read row 0's component, 4-7 row 1's (GQA-2's two KV groups, too)
    assert [arm.cell_of(h)[0] for h in range(8)] == [h // 4 for h in range(8)]


def test_the_grid_gradient_is_the_assembly_transpose():
    """assemble_grid is expand + cat: the gradient of a loss on the heads w.r.t. a component is the sum over the heads that read it."""
    g = GridSpec(2, 2, 3, 2, 1, 2, 2)
    comp = torch.randn(1, 4, 1, g.width, dtype=torch.float64, requires_grad=True)
    heads = assemble_grid(comp, g)
    w = torch.randn_like(heads)
    (heads * w).sum().backward()
    want = torch.zeros(g.width, dtype=torch.float64)
    for h in range(g.n_heads):
        i, j, k = cell_by_hand(g, h)
        wh = w[0, :, h].sum(0)
        want[i * 3:(i + 1) * 3] += wh[:3]
        want[6 + j * 2:6 + (j + 1) * 2] += wh[3:5]
        want[10 + k * 2:10 + (k + 1) * 2] += wh[5:7]
    torch.testing.assert_close(comp.grad[0, :, 0].sum(0), want, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------
# 4. the keys and values against the definition


@pytest.mark.parametrize("shape", list(SHAPES3))
@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_keys_and_values_are_axis_concatenations(shape, fraction):
    """The head's key is [axis-1 component i ; axis-2 component j ; axis-3 component k] normed over the WHOLE head by the one k_norm, then the leading rotary dimensions rotated
    (rope_fraction 0.25 inside axis 1, 0.5 exactly axis 1 in the arm, 1.0 reaching into axes 2 and 3); the value is the same concatenation, not normed."""
    layer = grid_layer({**SHAPES3[shape], ATT + "rope_fraction": fraction})
    g, c = layer.grid, layer.cfg
    assert g.n_axes == 3
    B, T = 2, 11
    x = torch.randn(B, T, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    pos = torch.stack([torch.arange(T), torch.arange(T) % 6])
    k, v = layer.keys_values(x, pos)
    assert k.shape == v.shape == (B, T, g.n_heads, g.head_dim)
    r = rotary_dims(g.head_dim, fraction)
    assert (r > g.row_dim) == (fraction == 1.0)
    wk, wv = layer.k_proj.weight.double(), layer.v_proj.weight.double()
    assert layer.k_proj.weight.shape[0] == layer.v_proj.weight.shape[0] == g.width
    assert layer.k_norm.weight.shape == (g.head_dim,) and not hasattr(layer, "k_row_norm") and not hasattr(layer, "k_col_norm")
    for b in range(B):
        kc, vc = x[b] @ wk.T, x[b] @ wv.T
        for h in range(g.n_heads):
            i, j, k_ = cell_by_hand(g, h)
            raw = head_parts(kc, g, i, j, k_)
            key = rot(rms(raw, layer.k_norm.weight), pos[b], r, c.rope_theta)
            torch.testing.assert_close(k[b, :, h], key, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(v[b, :, h], head_parts(vc, g, i, j, k_), rtol=1e-12, atol=1e-12)
    # the same keys through the cache form: the RAW components, assembled, normed, rotated at the read
    raw_k, raw_v = layer._grid_raw(x)
    assert raw_k.shape == (B, T, 1, g.width) and raw_v.shape == (B, T, 1, g.width)
    torch.testing.assert_close(raw_k[:, :, 0], x @ wk.T, rtol=1e-12, atol=1e-12)
    read_k, none_far = layer._grid_keys(raw_k, pos)
    assert none_far is None and torch.equal(read_k, k) and torch.equal(assemble_grid(raw_v, g), v)


@pytest.mark.parametrize("shape,distinct", [("2x2x2", 8), ("2x3x2_unequal", 12), ("2x2x2_pairs", 8)])
def test_the_axis_keys_are_distinct_for_random_weights(shape, distinct):
    """R C D keys from R + C + D components: 8 distinct keys from the arm's 8 heads (6 components of 3 kinds), the heads of one cell identical; two heads share an axis' part of
    the VALUE exactly iff they share that axis' index, and of the KEY up to the one scalar a token of the whole-head norm (proportional parts)."""
    layer = grid_layer({**SHAPES3[shape], ATT + "rope_fraction": 0.5})
    g = layer.grid
    assert g.n_cells == distinct
    x = torch.randn(1, 9, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    k, v = layer.keys_values(x, torch.arange(9)[None])
    heads = [k[0, :, h] for h in range(g.n_heads)]
    cuts = (0, g.row_dim, g.row_dim + g.col_dim, g.head_dim)
    n_distinct = 0
    for a in range(g.n_heads):
        n_distinct += not any(torch.equal(heads[a], heads[b]) for b in range(a))
        for b in range(a + 1, g.n_heads):
            ca, cb = g.cell_of(a), g.cell_of(b)
            if ca == cb:
                assert torch.equal(heads[a], heads[b]) and torch.equal(v[0, :, a], v[0, :, b]), (a, b)
                continue
            assert (heads[a] - heads[b]).abs().max() > 1e-3, (a, b)                       # two cells never read the same key
            for axis in range(3):
                part = slice(cuts[axis], cuts[axis + 1])
                pa, pb = heads[a][:, part], heads[b][:, part]
                proportional = torch.allclose(pa * pb[:, :1], pb * pa[:, :1], rtol=1e-9, atol=1e-12)
                assert proportional == (ca[axis] == cb[axis]), (a, b, axis)
                assert torch.equal(v[0, :, a, part], v[0, :, b, part]) == (ca[axis] == cb[axis]), (a, b, axis)
    assert n_distinct == distinct == g.rows * g.cols * g.depth


def test_the_grid_key_is_computed_by_the_gqa_layers_statements_with_three_axes():
    """The three-axis layer's key is the gqa layer's key of the same assembled head (n_kv_heads = n_q_heads, k_proj rows = the assembled rows, the same k_norm gains):
    keys_values and capped_keys_values agree with the twin's to rounding."""
    for cap in (None, 7):
        over = {ATT + "rope_fraction": 0.5, **GRID222P2, **({} if cap is None else {ATT + "rope_max_distance": cap})}
        layer = grid_layer(over)
        g = layer.grid
        gqa_cfg = dataclasses.replace(layer.cfg, kv_sharing="gqa", grid_rows=0, grid_cols=0, row_dim=0, col_dim=0, heads_per_cell=1, grid_depth=0, depth_dim=0, n_kv_heads=g.n_heads)
        twin = GatedAttention(32, gqa_cfg, 1e-6).double()
        with torch.no_grad():
            twin.k_proj.weight.copy_(assembled_rows3(layer.k_proj.weight, g))
            twin.v_proj.weight.copy_(assembled_rows3(layer.v_proj.weight, g))
            twin.k_norm.weight.copy_(layer.k_norm.weight)
        x = torch.randn(2, 13, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        pos = torch.stack([torch.arange(13), torch.arange(13) % 5])
        if cap is None:
            got, want = layer.keys_values(x, pos), twin.keys_values(x, pos)
        else:
            got, want = layer.capped_keys_values(x, pos), twin.capped_keys_values(x, pos)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------
# 5. the layer and the model against the explicit per-head reference and the multi-head twin


def reference_output3(layer: GatedAttention, x: torch.Tensor, docs: torch.Tensor, cap: int | None = None, s: int = 1) -> torch.Tensor:
    """The layer written head by head from the definition (fixed temperature): head (i, j, k) attends over the keys [axis-1 i ; axis-2 j ; axis-3 k] and the values of the
    same concatenation, of its document at or before the query, the logit q . k / sqrt(head_dim); under the cap a key more than cap tokens back (s * distance > cap under
    token-scale rotary) is scored by the query rotated by the cap against the unrotated key."""
    g, c = layer.grid, layer.cfg
    B, T, _ = x.shape
    H, D = g.n_heads, g.head_dim
    r = rotary_dims(D, c.rope_fraction)
    wq, wk, wv, wo = (m.weight.double() for m in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj))
    out = torch.zeros(B, T, x.shape[-1], dtype=torch.float64)
    for b in range(B):
        doc = docs[b]
        pos = torch.zeros(T, dtype=torch.long)
        for t in range(1, T):
            pos[t] = pos[t - 1] + 1 if doc[t] == doc[t - 1] else 0
        qg = (x[b] @ wq.T).view(T, H, 2 * D)
        q_all, gate = qg[..., :D], qg[..., D:]
        kc, vc = x[b] @ wk.T, x[b] @ wv.T
        heads = []
        for h in range(H):
            i, j, k = cell_by_hand(g, h)
            qn = rms(q_all[:, h], layer.q_norm.weight)
            kn = rms(head_parts(kc, g, i, j, k), layer.k_norm.weight)         # the head's key normed over the whole head, before the rotary part
            v_h = head_parts(vc, g, i, j, k)
            q_near, k_near = rot(qn, pos * s, r, c.rope_theta), rot(kn, pos * s, r, c.rope_theta)
            near = q_near @ k_near.T
            if cap is not None:
                q_far = rot(qn, torch.full((T,), cap), r, c.rope_theta)
                far = q_far @ kn.T
                dist = s * (torch.arange(T)[:, None] - torch.arange(T)[None, :])
                scores = torch.where(dist > cap, far, near)
            else:
                scores = near
            scores = scores * D ** -0.5
            allowed = (torch.arange(T)[None, :] <= torch.arange(T)[:, None]) & (doc[None, :] == doc[:, None])
            probs = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            heads.append((probs @ v_h) * torch.sigmoid(gate[:, h]))
        out[b] = torch.cat(heads, dim=-1) @ wo.T
    return out


@pytest.mark.parametrize("backend", ["reference", "sdpa", "flex"])
@pytest.mark.parametrize("shape,cap,s", [("2x2x2", None, 1), ("2x2x2_pairs", None, 1), ("2x3x2_unequal", None, 1), ("2x2x2", 7, 1), ("2x3x2_unequal", 7, 1),
                                         ("2x2x2_pairs", 12, 2), ("2x2x2", 12, 3)])
def test_layer_output_equals_the_per_head_reference_with_three_axes(backend, shape, cap, s, flex_cpu_backward):  # noqa: F811
    over = {ATT + "rope_fraction": 0.5, ATT + "backend": backend, **SHAPES3[shape], **({} if cap is None else {ATT + "rope_max_distance": cap})}
    layer = grid_layer(over)
    T = 36
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    x = torch.randn(2, T, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    ctx, _ = sequence_context(docs, None)
    if s != 1:
        assert cap is not None
        ctx.rope_scale = s
    out, keys = layer(x, ctx)
    torch.testing.assert_close(out, reference_output3(layer, x, docs, cap, s), rtol=1e-11, atol=1e-11)
    assert keys.k.shape == (2, T, 1, layer.grid.width)       # the cache is the components, RAW: no norm, no rotary, no k_far
    torch.testing.assert_close(keys.k[:, :, 0], x @ layer.k_proj.weight.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(keys.v[:, :, 0], x @ layer.v_proj.weight.T, rtol=1e-12, atol=1e-12)
    assert keys.k_far is None and torch.equal(keys.pos, ctx.pos * s)


def assembled_rows3(w: torch.Tensor, g: GridSpec) -> torch.Tensor:
    """(width, d) component weights to the (n_heads * head_dim, d) rows of the multi-head layer: head h = [axis-1 i ; axis-2 j ; axis-3 k] of its cell."""
    a2, a3 = g.row_width, g.row_width + g.col_width
    blocks = []
    for h in range(g.n_heads):
        i, j, k = cell_by_hand(g, h)
        blocks += [w[i * g.row_dim:(i + 1) * g.row_dim], w[a2 + j * g.col_dim:a2 + (j + 1) * g.col_dim], w[a3 + k * g.depth_dim:a3 + (k + 1) * g.depth_dim]]
    return torch.cat(blocks, dim=0)


def mha_twin3(grid_model, over: dict):
    """The gqa model with n_kv_heads = n_q_heads whose k_proj / v_proj are the grid model's weights assembled, every other tensor copied (the reference backend: the ground truth)."""
    a = grid_model.cfg.encoder.attention
    drop = (ATT + "kv_sharing", ATT + "grid_rows", ATT + "grid_cols", ATT + "row_dim", ATT + "col_dim", HPC, D3, DD3)
    twin_over = {k: v for k, v in over.items() if k not in drop and not k.endswith(".backend")}
    twin = make_model("v2_control", {**twin_over, ATT + "n_kv_heads": a.n_q_heads})
    sd, state = grid_model.state_dict(), twin.state_dict()
    g = next(layer.mixer for layer in grid_model.encoder.layers if layer.kind == "attention").grid
    for name in state:
        state[name] = assembled_rows3(sd[name], g) if name.endswith(("mixer.k_proj.weight", "mixer.v_proj.weight")) else sd[name]
    twin.load_state_dict(state)
    return twin


CASES3 = {
    "plain": {**FIVE},
    "capped_mix_ssmax": {**MIXCAP},
    "capped_mix_ssmax_unequal": {**MIXCAP, **GRID232},
    "capped_mix_ssmax_pairs": {**MIXCAP, **GRID222P2},
    "r1e_capped_mix_ssmax": {**MIXCAP, R1E_KEY: "block_entry"},                              # the arm's whole stack in small: R1-entry, the cap, the NoPE mix, ssmax
    "plain_rotary_into_axes_2_and_3": {**FIVE, ATT + "rope_fraction": 1.0},                  # the rotary part reaches past axis 1 (8 of 16) into axes 2 and 3
    "capped_mix_ssmax_rotary_into_axes_2_and_3": {**MIXCAP, ATT + "rope_fraction": 1.0},
}
QK = {"qk_on": {}, "qk_off": NOQK}


@pytest.mark.parametrize("qk", list(QK))
@pytest.mark.parametrize("case", list(CASES3))
@pytest.mark.parametrize("backend", ["reference", "sdpa", "flex"])
def test_three_axis_grid_equals_the_multi_head_layer_with_assembled_weights(case, backend, qk, flex_cpu_backward):  # noqa: F811
    over = {**GRID222, **QK[qk], **CASES3[case], **(fast(backend) if backend != "reference" else {})}
    grid = make_model("v2_control", over)
    assert grid.cfg.encoder.attention.qk_norm == (qk == "qk_on")
    twin = mha_twin3(grid, over)
    tokens, docs, dec = long_doc_rows(2, L, grid.cfg.vocab_size, seed=11)
    torch.testing.assert_close(grid(tokens, docs).logits, twin(tokens, docs).logits, **TOL)


@pytest.mark.parametrize("qk", list(QK))
@pytest.mark.parametrize("case", ["plain", "capped_mix_ssmax", "capped_mix_ssmax_unequal", "capped_mix_ssmax_pairs", "capped_mix_ssmax_rotary_into_axes_2_and_3"])
def test_three_axis_gradients_are_the_assembled_weights_gradients_summed(case, qk):
    over = {**GRID222, **QK[qk], **CASES3[case]}
    grid = make_model("v2_control", over)
    twin = mha_twin3(grid, over)
    tokens, docs, _ = long_doc_rows(2, L, grid.cfg.vocab_size, seed=12)
    for m in (grid, twin):
        m.zero_grad()
        m(tokens, docs).logits.square().mean().backward()
    gp, tp = dict(grid.named_parameters()), dict(twin.named_parameters())
    g = next(layer.mixer for layer in grid.encoder.layers if layer.kind == "attention").grid
    a2, a3 = g.row_width, g.row_width + g.col_width
    for name, p in gp.items():
        if name.endswith(("mixer.k_proj.weight", "mixer.v_proj.weight")):
            want = torch.zeros_like(p.grad)
            tg = tp[name].grad
            for h in range(g.n_heads):         # the twin's gradient on the rows of head (i, j, k) flows back to the three components it reads
                i, j, k = cell_by_hand(g, h)
                blk = tg[h * g.head_dim:(h + 1) * g.head_dim]
                want[i * g.row_dim:(i + 1) * g.row_dim] += blk[:g.row_dim]
                want[a2 + j * g.col_dim:a2 + (j + 1) * g.col_dim] += blk[g.row_dim:g.row_dim + g.col_dim]
                want[a3 + k * g.depth_dim:a3 + (k + 1) * g.depth_dim] += blk[g.row_dim + g.col_dim:]
            torch.testing.assert_close(p.grad, want, rtol=1e-9, atol=1e-12)
        else:
            torch.testing.assert_close(p.grad, tp[name].grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("qk", list(QK))
def test_token_scale_rotary_through_the_three_axis_model(qk, flex_cpu_backward):  # noqa: F811
    """TST's bags (rope_scale 3): the three-axis model equals the multi-head twin, on every backend, QK-norm on and off."""
    tokens = torch.randint(0, 256, (2, 40, 3), generator=torch.Generator().manual_seed(2))
    docs = torch.tensor([[0] * 25 + [1] * 15, [0] * 40])
    outs = {}
    for backend in ("reference", "sdpa", "flex"):
        over = {**GRID222, **QK[qk], **MIXCAP, ATT + "backend": backend}
        grid = make_model("v2_control", over)
        outs[backend] = grid(tokens, docs, rope_scale=3).logits
        if backend == "reference":
            twin = mha_twin3(grid, over)
            torch.testing.assert_close(outs[backend], twin(tokens, docs, rope_scale=3).logits, **TOL)
    torch.testing.assert_close(outs["sdpa"], outs["reference"], **TOL)
    torch.testing.assert_close(outs["flex"], outs["reference"], **TOL)
    assert not torch.allclose(outs["flex"], grid(tokens, docs).logits)


def test_the_three_axis_layer_traces_without_graph_breaks():
    """torch.compile (dynamo, the eager backend) traces the capped three-axis layer with the encoder's shared rotation tables as one graph and gives the eager values: the
    assembly (a loop over the three axes, static) is fixed-shape code with no boolean-mask indexing."""
    from ced.model.attention import CAPPED_ROTATION_KEY, capped_rotation

    layer = grid_layer({**GRID222, ATT + "rope_fraction": 0.5, ATT + "rope_max_distance": 7})
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    x = torch.randn(2, 36, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    ctx, _ = sequence_context(docs, None)
    ctx.memo[CAPPED_ROTATION_KEY] = capped_rotation(ctx.pos, layer.cfg, torch.float64)
    ref, ref_keys = layer(x, ctx)
    torch._dynamo.reset()
    compiled = torch.compile(lambda a: layer(a, ctx), backend="eager", fullgraph=True)
    out, keys = compiled(x)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    assert torch.equal(keys.k, ref_keys.k) and keys.k_far is None and torch.equal(keys.v, ref_keys.v) and torch.equal(keys.pos, ref_keys.pos)




# ------------------------------------------------------------------------------------------
# 6. causality, cached generation, the cache, resume


GRID_MODELS3 = {
    "2x2x2_plain": {**GRID222, **FIVE},
    "2x2x2_capped_mix_ssmax": {**GRID222, **MIXCAP},
    "2x3x2_unequal_capped_mix_ssmax": {**GRID232, **MIXCAP},
    "2x2x2_pairs_capped_mix_ssmax": {**GRID222P2, **MIXCAP},
    "2x2x2_capped_mix_qknorm_off": {**GRID222, **MIXCAP, **NOQK},
    "2x2x2_r1e_capped_mix_ssmax": {**GRID222, **MIXCAP, R1E_KEY: "block_entry"},
}
# the equal-cache GQA-2 control in small, for the cache and the parameter comparisons, and the 2 x 2 grid over head pairs (SET A's)
GQA2 = {**GQA2_8Q, **MIXCAP}
PAIRS = {**GRID22P2, **MIXCAP}
GRID222_MIX = GRID_MODELS3["2x2x2_capped_mix_ssmax"]


@pytest.mark.parametrize("name", list(GRID_MODELS3))
def test_future_tokens_do_not_change_past_outputs_with_three_axes(name):
    """Perturb the tokens after t, require the outputs at or before t unchanged (on the reference path)."""
    model = make_model("v2_control", GRID_MODELS3[name])
    tokens, docs, _ = long_doc_rows(2, L, model.cfg.vocab_size, seed=1)
    base = model(tokens, docs).logits
    for j in [1, 7, 20, 33, L - 1]:
        changed = tokens.clone()
        changed[:, j] = (changed[:, j] + 1 + j) % model.cfg.vocab_size
        out = model(changed, docs).logits
        assert torch.equal(out[:, :j], base[:, :j]), f"outputs before {j} changed"
        assert not torch.equal(out[:, j:], base[:, j:]), "the change had no effect (the test is vacuous)"


@pytest.mark.parametrize("name", ["2x2x2_capped_mix_ssmax", "2x3x2_unequal_capped_mix_ssmax"])
def test_gradients_do_not_flow_from_future_inputs_with_three_axes(name):
    model = make_model("v2_control", GRID_MODELS3[name])
    tokens, docs, _ = long_doc_rows(1, L, model.cfg.vocab_size, seed=3)
    captured = {}

    def keep(module, inputs, output):
        output.retain_grad()
        captured["h"] = output

    hook = model.embed.register_forward_hook(keep)
    try:
        for t in [0, 9, 24, 40]:
            model.zero_grad(set_to_none=True)
            model(tokens, docs).logits[:, t].sum().backward()
            grad = captured["h"].grad
            assert torch.count_nonzero(grad[:, t + 1:]) == 0, f"future inputs influence position {t}"
            assert torch.count_nonzero(grad[:, : t + 1]) > 0
    finally:
        hook.remove()


@pytest.mark.parametrize("name", list(GRID_MODELS3))
def test_teacher_forced_equals_token_by_token_cached_generation_with_three_axes(name):
    model = make_model("v2_control", GRID_MODELS3[name])
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=10)
    teacher = model(tokens, docs).logits
    stepwise = chunked(model, tokens, docs, dec, list(range(1, L)))
    torch.testing.assert_close(stepwise, teacher, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["2x2x2_capped_mix_ssmax", "2x3x2_unequal_capped_mix_ssmax", "2x2x2_r1e_capped_mix_ssmax"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_teacher_forced_equals_random_chunked_prefill_with_three_axes(name, seed):
    model = make_model("v2_control", GRID_MODELS3[name])
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=20 + seed)
    teacher = model(tokens, docs).logits
    g = torch.Generator().manual_seed(seed)
    boundaries = sorted(torch.randperm(L - 1, generator=g)[:5].add(1).tolist())
    torch.testing.assert_close(chunked(model, tokens, docs, dec, boundaries), teacher, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["2x2x2_capped_mix_ssmax", "2x3x2_unequal_capped_mix_ssmax"])
@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_cached_generation_on_the_fast_backends_with_three_axes(name, backend, flex_cpu_backward):  # noqa: F811
    """The fast paths against the reference path: sdpa and flex, cached in chunks, equal the reference backend's full forward (same weights)."""
    model = make_model("v2_control", {**GRID_MODELS3[name], **fast(backend)})
    ref = make_model("v2_control", GRID_MODELS3[name])
    ref.load_state_dict(model.state_dict())
    tokens, docs, dec = long_doc_rows(1, 40, model.cfg.vocab_size, seed=30)
    teacher = ref(tokens, docs).logits
    torch.testing.assert_close(chunked(model, tokens, docs, dec, [13, 14, 30]), teacher, **TOL)
    torch.testing.assert_close(model(tokens, docs).logits, teacher, **TOL)


def test_the_cache_numbers_a_token_of_the_three_axis_arm_and_its_equals():
    """The tiny model's head_dim 16 stands for 128: the MQA baseline stores 16 numbers for K and for V; the 2 x 2 x 2 arm (axes 8 + 4 + 4, two components each) 2 x 8 + 2 x 4 +
    2 x 4 = 32 = 2 heads' worth, as the 2 x 2 pairs grid and GQA-2 over 8 heads do. In the arm's units (x 8): 256 against MQA's 128."""
    got = {n: cache_numbers_a_token(o) for n, o in (("mqa", MIXCAP), ("arm", GRID222_MIX), ("unequal", GRID_MODELS3["2x3x2_unequal_capped_mix_ssmax"]), ("pairs3", GRID_MODELS3["2x2x2_pairs_capped_mix_ssmax"]),
                                                  ("pairs2", PAIRS), ("gqa2_8q", GQA2), ("gqa4_8q", {**GQA4_8Q, **MIXCAP}))}
    assert got["arm"] == got["pairs3"] == got["pairs2"] == got["gqa2_8q"] == 32 == 2 * got["mqa"] and got["mqa"] == 16
    assert got["unequal"] == 2 * 6 + 3 * 4 + 2 * 2 == 28
    assert got["gqa4_8q"] == 64


def test_the_cache_holds_the_raw_components_and_their_positions_with_three_axes():
    model = make_model("v2_control", GRID222_MIX)
    g = next(layer.mixer for layer in model.encoder.layers if layer.kind == "attention").grid
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=13)
    cache = model.prefill(tokens, docs, dec).cache
    sets = key_sets(cache.encoder)
    assert len(sets) == 5
    per_token = g.rows * g.row_dim + g.cols * g.col_dim + g.depth * g.depth_dim
    assert per_token == g.width == 8 * 2 + 4 * 2 + 4 * 2 == 32
    for ks in sets:
        assert ks.k.shape == ks.v.shape == (2, L, 1, per_token) and ks.k.shape[2] != g.n_heads        # one stored "head" of components, never the 8 assembled heads
        assert ks.k_far is None and torch.equal(ks.pos, sequence_context(docs, None)[0].pos)
    out = model.extend(tokens[:, :3], docs[:, -3:], dec[:, :3], cache)
    assert all(ks.k.shape == (2, L + 3, 1, per_token) for ks in key_sets(out.cache.encoder))


@pytest.mark.parametrize("cap,with_tables", [(None, False), (7, False), (7, True)], ids=["plain", "capped", "capped_shared_tables"])
@pytest.mark.parametrize("shape", ["2x2x2", "2x3x2_unequal", "2x2x2_pairs"])
def test_the_cache_stores_raw_components_and_a_cached_read_matches_the_full_read_with_three_axes(cap, with_tables, shape):
    """The layer's cache after every chunk is the RAW projections with each entry's rotary position; the read assembles, norms and rotates the heads at the entries' own
    positions (also across a document boundary inside a chunk), so the chunked outputs equal the full forward's."""
    from ced.model.attention import CAPPED_ROTATION_KEY, capped_rotation

    layer = grid_layer({ATT + "rope_fraction": 0.5, **SHAPES3[shape], **({} if cap is None else {ATT + "rope_max_distance": cap})})
    T = 36
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    x = torch.randn(2, T, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(17))

    def run(a, b, cache, state):
        ctx, state = sequence_context(docs[:, a:b], state)
        if with_tables:
            ctx.memo[CAPPED_ROTATION_KEY] = capped_rotation(ctx.pos, layer.cfg, torch.float64)
        out, keys = layer(x[:, a:b], ctx, cache)
        return out, keys, state

    full_out, full_keys, _ = run(0, T, None, None)
    cache, state, pieces = None, None, []
    for a, b in [(0, 10), (10, 11), (11, 30), (30, T)]:
        out, cache, state = run(a, b, cache, state)
        pieces.append(out)
        torch.testing.assert_close(cache.k[:, :, 0], x[:, :b] @ layer.k_proj.weight.T, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cache.v[:, :, 0], x[:, :b] @ layer.v_proj.weight.T, rtol=1e-12, atol=1e-12)
        assert cache.k_far is None and cache.pos.shape == (2, b)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full_out, rtol=1e-12, atol=1e-12)
    assert torch.equal(cache.pos, full_keys.pos)
    torch.testing.assert_close(cache.k, full_keys.k, rtol=1e-12, atol=1e-12)




@pytest.mark.parametrize("name", ["2x2x2_capped_mix_ssmax", "2x2x2_pairs_capped_mix_ssmax"])
def test_caches_concatenate_on_the_batch_axis_with_three_axes(name):
    """The cache is one KeySet head of components, the shape of any gqa cache: a batch-axis concatenation (the batched RULER generator's cat_caches) needs nothing
    grid-specific; the rows of the concatenated cache decode exactly as in their own caches."""
    model = make_model("v2_control", GRID_MODELS3[name])
    tokens, docs, dec = long_doc_rows(3, 30, model.cfg.vocab_size, seed=14)
    caches = [model.prefill(tokens[r:r + 1], docs[r:r + 1], dec[r:r + 1]).cache for r in range(3)]
    joined = _cat_leaf(caches)
    nxt = torch.randint(0, model.cfg.vocab_size, (3, 1), generator=torch.Generator().manual_seed(15))
    batched = model.extend(nxt, docs[:, -1:], dec[:, -1:], joined).logits
    for r in range(3):
        alone = model.extend(nxt[r:r + 1], docs[r:r + 1, -1:], dec[r:r + 1, -1:], caches[r]).logits
        torch.testing.assert_close(batched[r:r + 1], alone, rtol=0, atol=1e-10)


def test_packed_documents_are_isolated_with_three_axes():
    model = make_model("v2_control", GRID222_MIX)
    tokens = torch.randint(0, model.cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(4))
    split = 19
    docs = torch.zeros(1, L, dtype=torch.long)
    docs[:, split:] = 1
    packed = model(tokens, docs).logits
    alone = model(tokens[:, split:], torch.zeros(1, L - split, dtype=torch.long)).logits
    torch.testing.assert_close(packed[:, split:], alone, rtol=0, atol=1e-12)


@pytest.mark.parametrize("over", [None, CAPPED_FIVE], ids=["plain", "capped_nope"])
def test_checkpoint_resume_matches_uninterrupted_with_three_axes(over, tmp_path):
    cfg = grid_train_config(8, over, GRID222)
    straight = Trainer(cfg, tmp_path / "straight")
    straight_losses = [straight.train_step()["loss"] for _ in range(8)]
    first = Trainer(cfg, tmp_path / "resumed")
    losses = [first.train_step()["loss"] for _ in range(4)]
    path = first.save_checkpoint()
    del first
    torch.manual_seed(999)
    second = Trainer.from_checkpoint(path)
    att = second.cfg.model.encoder.attention
    assert att == cfg.model.encoder.attention and att.kv_sharing == "grid" and (att.grid_depth, att.depth_dim) == (2, 4)
    losses += [second.train_step()["loss"] for _ in range(4)]
    assert losses == straight_losses
    assert_same_state(second, straight)


# ------------------------------------------------------------------------------------------
# 7. the parameter counts, the consumers


def test_the_three_axis_layer_has_the_two_by_two_layers_and_gqa2s_parameters():
    """The 2 x 2 x 2 attention layer (axes 8 + 4 + 4, 8 heads) has the 2 x 2 pairs grid's and GQA-2's parameter count: k_proj and v_proj are (2 x head_dim, d) in all three
    (256 outputs in the arm), q_proj / gate / o_proj and the one k_norm of head_dim numbers are the same, and the MQA baseline has half the K/V projection."""
    def layer(over: dict) -> GatedAttention:
        return GatedAttention(64, set_config(tiny_v2_control_config(), over).encoder.attention, 1e-6)

    def n(m) -> int:
        return sum(p.numel() for p in m.parameters())

    arm, pairs, gqa2, mqa = layer(GRID222), layer(GRID22P2), layer(GQA2_8Q), layer({ATT + "n_q_heads": 8})
    for m in (arm, pairs, gqa2):
        assert m.k_proj.weight.shape == m.v_proj.weight.shape == (32, 64) and m.k_norm.weight.shape == (16,) and m.q_proj.weight.shape == (8 * 16 * 2, 64)
    assert mqa.k_proj.weight.shape == (16, 64)
    assert n(arm) == n(pairs) == n(gqa2) == n(mqa) + 2 * 16 * 64
    assert sorted((k, tuple(v.shape)) for k, v in arm.state_dict().items()) == sorted((k, tuple(v.shape)) for k, v in gqa2.state_dict().items())


def test_whole_three_axis_models_have_the_two_by_two_models_and_gqa2s_parameters():
    from ced.utils.params import build_on_meta

    def total(over: dict) -> int:
        return sum(p.numel() for p in build_on_meta(set_config(tiny_v2_control_config(), over)).parameters())

    assert total({**GRID222, **FIVE}) == total({**GRID22P2, **FIVE}) == total({**GQA2_8Q, **FIVE}) == total({ATT + "n_q_heads": 8, **FIVE}) + 5 * 2 * 16 * 64
    assert total({**GRID222P2, **FIVE}) > total({**GRID222, **FIVE})              # 16 heads: more q / gate / o, not the arm


@pytest.mark.parametrize("grid_over,gqa_over", [(GRID222, GQA2_8Q), ({**GRID222, **MIXCAP, R1E_KEY: "block_entry"}, {**GQA2_8Q, **MIXCAP, R1E_KEY: "block_entry"})],
                         ids=["200m-shape", "200m-shape-whole-stack"])
def test_the_three_axis_grid_and_its_gqa2_control_start_from_identical_parameters(grid_over, gqa_over):
    """With the same seed the 2 x 2 x 2 grid model and its equal-cache GQA-2 control are the same in EVERY parameter at initialisation (the same names, shapes and values, the
    one k_norm of head_dim numbers included), so they differ in the sharing alone."""
    def model_state(over: dict):
        torch.manual_seed(3)
        model = build_model(set_config(tiny_v2_control_config(), {**over, **FIVE}))
        return model, model.state_dict()

    (mg, g), (mq, q) = model_state(grid_over), model_state(gqa_over)
    assert g.keys() == q.keys() and len(g) > 20
    for k in g:
        assert g[k].shape == q[k].shape and torch.equal(g[k], q[k]), k
    assert sum(p.numel() for p in mg.parameters()) == sum(p.numel() for p in mq.parameters())




