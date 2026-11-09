"""Grid KV sharing for the global attention layers (AttentionConfig.kv_sharing "grid"; ced/model/attention.py "Grid KV sharing").

 1. The key and its refusals: defaults (gqa, nothing saved), the grid's values checked by name, control.check_kv_sharing.
 2. The layout: the query heads sit on a rows x cols grid of cells with heads_per_cell (1 or 2) consecutive heads to a cell (head h is in the cell
    c = h // heads_per_cell = (i, j) = (c // cols, c % cols)); the heads of cell (i, j) read [row i ; column j] as their key and as their value; the
    r * c keys are distinct for random weights (4 distinct keys from 8 heads on the pairs grid); the keys and values of a layer equal an explicit
    per-head reference written from the definition (float64): [row i ; column j] normed over the whole head, then rotated.
 3. The layer's output equals an explicit per-head reference: plain rotary, the rotary distance cap, TST's token-scale rotary.
 4. The grid is the multi-head layer with block-assembled weights: with QK-norm ON and OFF (the grid norms the assembled per-head key over
    the whole head dimension with ONE k_norm, as the gqa layer does; answering the review's finding that the per-component norm separated a grid
    from its GQA control), a grid model equals a gqa model with n_kv_heads = n_q_heads whose k_proj / v_proj rows are [row i; column j] (logits
    and gradients), on the reference, sdpa and flex backends, with the cap, the by-layer NoPE mix, Scalable-Softmax and TST's token-scale
    rotary; also with the rotary part reaching into the column component.
 5. Causality, teacher forcing against cached generation (every chunking; the cache holds RAW components and the read norms and rotates at
    each entry's own position), the cache holds components (numbers a token) and their positions, a batch-axis concatenation of caches,
    checkpoint resume.
 6. gqa is untouched: the same parameters and state-dict names, the saved config without the keys; and a grid and its GQA control have the
    same state-dict names, shapes and, at initialisation, the same value in every parameter.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import pytest
import torch

from ced.config import AttentionConfig, DataConfig, TrainConfig, from_dict, tiny_control_config, tiny_v2_control_config, to_dict
from ced.model.attention import (GatedAttention, GridSpec, KeySet, assemble_grid, grid_spec)
from ced.model.control import build_model
from ced.model.rope import rotary_dims
from ced.model.sequence import sequence_context
from ced.train.trainer import SAVED_ATTENTION_DEFAULTS, Trainer, saved_config_dict

from helpers import make_model, perturb_parameters, random_rows, set_config
from test_fast_attention import flex_cpu_backward  # noqa: F401  (a pytest fixture)
from pos_arms_helpers import CAP, SS_ENC, chunked, fast, long_doc_rows
from test_resume import assert_same_state

L = 48
ATT = "encoder.attention."
HPC = ATT + "heads_per_cell"
GRID22 = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 2, ATT + "row_dim": 8, ATT + "col_dim": 8}  # the tiny model's head_dim 16: the 60M grid in small
# the 200M grid over head PAIRS in small: 8 query heads, a 2 x 2 grid of cells, 2 heads to a cell, components 8 + 8 = the head dimension 16
GRID22P2 = {**GRID22, ATT + "n_q_heads": 8, HPC: 2}
# the 200M 2 x 4 grid over single heads in small: 8 query heads, components 8 + 8
GRID24 = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 4, ATT + "row_dim": 8, ATT + "col_dim": 8, ATT + "n_q_heads": 8}
# a general grid with unequal components and a narrower head (not an arm): 2 x 4, row 8 + col 4 = head_dim 12
GRIDU = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 4, ATT + "row_dim": 8, ATT + "col_dim": 4, ATT + "n_q_heads": 8, ATT + "head_dim": 12}
GQA2_4Q = {ATT + "n_kv_heads": 2}                          # the 60M control in small: 4 query heads, 2 KV heads
GQA2_8Q = {ATT + "n_q_heads": 8, ATT + "n_kv_heads": 2}    # the 200M control in small
GQA4_8Q = {ATT + "n_q_heads": 8, ATT + "n_kv_heads": 4}    # the 200M 4-KV-head control in small
FIVE = {"encoder.full_attn_interval": 2}  # five global layers in the tiny control's ten
MIXCAP = {**SS_ENC, **FIVE, ATT + "rope_max_distance": CAP, "encoder.nope_global_layers": (3, 4)}  # capped rotary below, NoPE above, ssmax
NOQK = {ATT + "qk_norm": False}
R1E_KEY = "encoder.attn_res"        # R1-entry (the arm's trunk): depth mixing at each block's entry
TOL = dict(rtol=1e-10, atol=1e-10)


def gspec(over: dict) -> GridSpec:
    return grid_spec(set_config(tiny_v2_control_config(), over).encoder.attention)


def mixers(model) -> list:
    return [layer.mixer for layer in model.encoder.layers if layer.kind == "attention"]


def key_sets(obj) -> list:
    """Every KeySet in a cache object (a list per layer, tuples and dataclasses nested)."""
    if isinstance(obj, KeySet):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [k for o in obj for k in key_sets(o)]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [k for f in dataclasses.fields(obj) for k in key_sets(getattr(obj, f.name))]
    return []


# ------------------------------------------------------------------------------------------
# 1. the key


def test_default_is_gqa_and_nothing_is_saved():
    a = AttentionConfig()
    assert (a.kv_sharing, a.grid_rows, a.grid_cols, a.row_dim, a.col_dim, a.heads_per_cell) == ("gqa", 0, 0, 0, 0, 1)
    assert grid_spec(a) is None
    cfg = TrainConfig()
    saved = saved_config_dict(cfg)
    assert {"kv_sharing", "grid_rows", "grid_cols", "row_dim", "col_dim", "heads_per_cell"} <= set(SAVED_ATTENTION_DEFAULTS)
    for section in ("encoder", "decoder"):
        for key in ("kv_sharing", "grid_rows", "grid_cols", "row_dim", "col_dim", "heads_per_cell"):
            assert key not in saved["model"][section]["attention"], (section, key)
    assert to_dict(from_dict(TrainConfig, saved)) == to_dict(cfg)


def test_grid_keys_are_saved_and_round_trip():
    cfg = TrainConfig(model=set_config(tiny_v2_control_config(), GRID22))
    saved = saved_config_dict(cfg)
    att = saved["model"]["encoder"]["attention"]
    assert (att["kv_sharing"], att["grid_rows"], att["grid_cols"], att["row_dim"], att["col_dim"]) == ("grid", 2, 2, 8, 8)
    assert "heads_per_cell" not in att                       # 1 is the default: a single-head grid saves the keys it had
    assert "kv_sharing" not in saved["model"]["decoder"]["attention"]
    assert from_dict(TrainConfig, saved).model.encoder.attention == cfg.model.encoder.attention
    pairs = saved_config_dict(TrainConfig(model=set_config(tiny_v2_control_config(), GRID22P2)))["model"]["encoder"]["attention"]
    assert pairs["heads_per_cell"] == 2 and pairs["n_q_heads"] == 8
    pair_cfg = TrainConfig(model=set_config(tiny_v2_control_config(), GRID22P2))
    assert from_dict(TrainConfig, saved_config_dict(pair_cfg)).model.encoder.attention == pair_cfg.model.encoder.attention


@pytest.mark.parametrize("over,match", [
    ({ATT + "kv_sharing": "mqa"}, "kv_sharing"),
    ({ATT + "grid_rows": 2}, "gqa"),                                                         # a grid key under gqa would be ignored
    ({HPC: 2}, "gqa"),                                                                       # so would heads_per_cell
    ({**GRID22, ATT + "grid_rows": 3}, "n_q_heads"),                                         # 3 x 2 != 4 heads
    ({**GRID22, HPC: 2}, "n_q_heads"),                                                       # 2 x 2 x 2 != 4 heads
    ({**GRID22P2, HPC: 1}, "n_q_heads"),                                                     # 2 x 2 x 1 != 8 heads
    ({**GRID22P2, HPC: 0}, "integers"),
    ({**GRID22, ATT + "col_dim": 4}, "head_dim"),                                            # 8 + 4 != 16
    ({**GRID22, ATT + "n_kv_heads": 2}, "n_kv_heads"),
    ({**GRID22, ATT + "row_dim": 0, ATT + "col_dim": 16}, "integers"),
    ({**GRID22, "decoder.attention.kv_sharing": "grid"}, "decoder"),
    ({**GRID22, ATT + "backend": "fa4"}, "backend"),
    ({**GRID22, ATT + "normalizer": "asentmax"}, "normalizer"),
    ({**GRID22, ATT + "row_term": "pdp"}, "row_term"),
    ({**GRID22, "loop.pass2_gate": "uniform"}, "pass2_gate|kv_sharing"),
    ({**GRID22, "cp_head.enabled": True}, "cp_head|kv_sharing"),
    ({**GRID22, "mtp.enabled": True}, "mtp|kv_sharing"),
])
def test_refusals(over, match):
    with pytest.raises((ValueError, AssertionError), match=match):
        build_model(set_config(tiny_v2_control_config(), over))


def test_another_arch_is_refused():
    with pytest.raises(ValueError, match="arch"):
        build_model(set_config(tiny_v2_control_config(), {**GRID22, "arch": "ced_v2"}))


def test_a_slot_read_is_refused():
    layer = GatedAttention(64, set_config(tiny_v2_control_config(), GRID22).encoder.attention, 1e-6)
    with pytest.raises(ValueError, match="slot read"):
        layer.read(torch.zeros(1, 2, 64), torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long), None)


# ------------------------------------------------------------------------------------------
# 2. the layout and the keys


def comp_slices(comp: torch.Tensor, g: GridSpec, i: int, j: int) -> torch.Tensor:
    """[row i ; column j] of the cache's components (..., width), written from the definition (rows first, then columns)."""
    row = comp[..., i * g.row_dim:(i + 1) * g.row_dim]
    col = comp[..., g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]
    return torch.cat([row, col], dim=-1)


@pytest.mark.parametrize("hpc", [1, 2, 3])
def test_assemble_grid_layout(hpc):
    g = GridSpec(rows=2, cols=3, row_dim=5, col_dim=4, heads_per_cell=hpc)
    comp = torch.randn(2, 7, 1, g.width, dtype=torch.float64)
    heads = assemble_grid(comp, g)
    assert heads.shape == (2, 7, 6 * hpc, 9) and g.n_heads == 6 * hpc and g.n_cells == 6
    for h in range(g.n_heads):
        i, j = g.cell_of(h)
        # the documented mapping: head h is in the cell h // hpc, row-major over (i, j)
        assert (i, j) == ((h // hpc) // 3, (h // hpc) % 3)
        assert torch.equal(heads[:, :, h], comp_slices(comp[..., 0, :], g, i, j)), (h, i, j)
    # a pure re-indexing: the heads of a cell are the same numbers, a row's heads share its row component, a column's its column component
    if hpc > 1:
        assert torch.equal(heads[:, :, 0], heads[:, :, hpc - 1]) and not torch.equal(heads[:, :, 0], heads[:, :, hpc])
    assert torch.equal(heads[:, :, 0, :5], heads[:, :, 1 * hpc, :5]) and torch.equal(heads[:, :, 0, 5:], heads[:, :, 3 * hpc, 5:])


def test_the_two_by_two_pairs_mapping_is_the_documented_one():
    """200M's grid: 8 query heads, a 2 x 2 grid of cells, 2 heads to a cell: heads 0-1 are cell (0,0), 2-3 (0,1), 4-5 (1,0), 6-7 (1,1)."""
    g = GridSpec(rows=2, cols=2, row_dim=8, col_dim=8, heads_per_cell=2)
    assert [g.cell_of(h) for h in range(8)] == [(0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0), (1, 1), (1, 1)]
    # and 60M's: 4 query heads, one to a cell: 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1); the 2 x 4 over 8 single heads: head h = i * 4 + j
    assert [GridSpec(2, 2, 8, 8).cell_of(h) for h in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [GridSpec(2, 4, 8, 8).cell_of(h) for h in range(8)] == [(h // 4, h % 4) for h in range(8)]


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)


def rot(x, pos, r, theta):
    """NeoX partial rotary written from the definition: x (T, w), pos (T,), the first r dimensions rotate."""
    if r == 0:
        return x
    inv = theta ** (-torch.arange(0, r, 2, dtype=torch.float64) / r)
    ang = pos[:, None].double() * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[:, : r // 2], x[:, r // 2 : r]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[:, r:]], dim=-1)


def grid_layer(over: dict | None = None, d: int = 32, seed: int = 0) -> GatedAttention:
    cfg = set_config(tiny_v2_control_config(), {**GRID22, **(over or {})})
    torch.manual_seed(seed)
    layer = GatedAttention(d, cfg.encoder.attention, 1e-6, True).double()
    perturb_parameters(layer, 0.3, seed=seed + 1)
    return layer


SHAPES = {"2x2": GRID22, "2x2_pairs": GRID22P2, "2x4": GRID24, "2x4_unequal": GRIDU}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_keys_and_values_are_row_column_concatenations(shape, fraction):
    """The head's key is [row i ; column j] normed over the WHOLE head by the one k_norm, then rotated in the head's leading rotary dimensions (which, at
    rope_fraction 1.0, reach into the column component: the rotation is at the read, on the assembled head)."""
    over = {**SHAPES[shape], ATT + "rope_fraction": fraction}
    layer = grid_layer(over)
    g, c = layer.grid, layer.cfg
    B, T = 2, 11
    x = torch.randn(B, T, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    pos = torch.stack([torch.arange(T), torch.arange(T) % 6])
    k, v = layer.keys_values(x, pos)
    assert k.shape == v.shape == (B, T, g.n_heads, g.head_dim)
    r = rotary_dims(g.head_dim, fraction)
    assert (r > g.row_dim) == (fraction == 1.0)
    wk, wv = layer.k_proj.weight.double(), layer.v_proj.weight.double()
    assert layer.k_norm.weight.shape == (g.head_dim,) and not hasattr(layer, "k_row_norm") and not hasattr(layer, "k_col_norm")
    for b in range(B):
        kc, vc = x[b] @ wk.T, x[b] @ wv.T  # (T, width): the raw components
        for h in range(g.n_heads):
            i, j = (h // g.heads_per_cell) // g.cols, (h // g.heads_per_cell) % g.cols      # the documented mapping, written out
            assert g.cell_of(h) == (i, j)
            k_row, k_col = kc[:, i * g.row_dim:(i + 1) * g.row_dim], kc[:, g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]
            key = rot(rms(torch.cat([k_row, k_col], dim=-1), layer.k_norm.weight), pos[b], r, c.rope_theta)
            torch.testing.assert_close(k[b, :, h], key, rtol=1e-12, atol=1e-12)
            v_row = vc[:, i * g.row_dim:(i + 1) * g.row_dim]
            v_col = vc[:, g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]
            torch.testing.assert_close(v[b, :, h], torch.cat([v_row, v_col], dim=-1), rtol=1e-12, atol=1e-12)
    # the same keys through the cache form: the RAW components the cache stores (no norm, no rotary), assembled, normed, rotated at the read
    raw_k, raw_v = layer._grid_raw(x)
    assert raw_k.shape == (B, T, 1, g.width) and raw_v.shape == (B, T, 1, g.width)
    torch.testing.assert_close(raw_k[:, :, 0], x @ wk.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(raw_v[:, :, 0], x @ wv.T, rtol=1e-12, atol=1e-12)
    read_k, none_far = layer._grid_keys(raw_k, pos)
    assert none_far is None and torch.equal(read_k, k) and torch.equal(assemble_grid(raw_v, g), v)


@pytest.mark.parametrize("shape,distinct", [("2x2", 4), ("2x2_pairs", 4), ("2x4", 8), ("2x4_unequal", 8)])
def test_the_row_times_column_keys_are_distinct_for_random_weights(shape, distinct):
    """The distinct-key count: r x c keys from r + c components; the heads of a cell share their key exactly (4 distinct keys from the pairs grid's 8 heads,
    from 2 + 2 components), the cells of a row share the row component and the cells of a column the column component (the values exactly; the keys up to
    the one scalar a token of the whole-head norm: proportional parts)."""
    layer = grid_layer({ATT + "rope_fraction": 0.5, **SHAPES[shape]})
    g = layer.grid
    assert g.n_cells == distinct
    x = torch.randn(1, 9, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    k, v = layer.keys_values(x, torch.arange(9)[None])
    heads = [k[0, :, h] for h in range(g.n_heads)]
    n_distinct = 0
    for a in range(g.n_heads):
        if not any(torch.equal(heads[a], heads[b]) for b in range(a)):
            n_distinct += 1
        for b in range(a + 1, g.n_heads):
            (ia, ja), (ib, jb) = g.cell_of(a), g.cell_of(b)
            if (ia, ja) == (ib, jb):
                assert torch.equal(heads[a], heads[b]) and torch.equal(v[0, :, a], v[0, :, b]), (a, b)        # the heads of one cell read one key
                continue
            assert (heads[a] - heads[b]).abs().max() > 1e-3, (a, b)                                          # two cells never read the same key
            # The key is normed over the whole head ([row ; column] together), so the cells of a row share the row part of the key up to ONE scalar a token (the cell's
            # 1 / rms): proportional, not equal; likewise the column part for the cells of a column. The values are not normed: they share the part exactly.
            for part in (slice(0, g.row_dim), slice(g.row_dim, g.head_dim)):
                pa, pb = heads[a][:, part], heads[b][:, part]
                proportional = torch.allclose(pa * pb[:, :1], pb * pa[:, :1], rtol=1e-9, atol=1e-12)
                assert proportional == ((ia == ib) if part.start == 0 else (ja == jb)), (a, b, part)
            assert torch.equal(v[0, :, a, : g.row_dim], v[0, :, b, : g.row_dim]) == (ia == ib)               # and the values: the same sharing, exactly
            assert torch.equal(v[0, :, a, g.row_dim:], v[0, :, b, g.row_dim:]) == (ja == jb)
    assert n_distinct == distinct == g.rows * g.cols


def test_the_grid_key_is_computed_by_the_gqa_layers_statements():
    """The grid layer's key is the gqa layer's key of the same assembled head: a gqa layer (n_kv_heads = n_q_heads) whose k_proj rows are the assembled
    rows, sharing the grid's k_norm gains, gives the grid layer's keys_values (and capped_keys_values): the same statements (norm over the head, then
    apply_rope / _rotate) on the same numbers."""
    for cap in (None, 7):
        over = {ATT + "rope_fraction": 0.5, **GRID22P2, **({} if cap is None else {ATT + "rope_max_distance": cap})}
        layer = grid_layer(over)
        g = layer.grid
        gqa_cfg = dataclasses.replace(layer.cfg, kv_sharing="gqa", grid_rows=0, grid_cols=0, row_dim=0, col_dim=0, heads_per_cell=1, n_kv_heads=g.n_heads)
        twin = GatedAttention(32, gqa_cfg, 1e-6).double()
        with torch.no_grad():
            twin.k_proj.weight.copy_(assembled_rows(layer.k_proj.weight, g))
            twin.v_proj.weight.copy_(assembled_rows(layer.v_proj.weight, g))
            twin.k_norm.weight.copy_(layer.k_norm.weight)
        x = torch.randn(2, 13, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        pos = torch.stack([torch.arange(13), torch.arange(13) % 5])
        if cap is None:
            got, want = layer.keys_values(x, pos), twin.keys_values(x, pos)
        else:
            got, want = layer.capped_keys_values(x, pos), twin.capped_keys_values(x, pos)
        for a, b in zip(got, want):  # (equal up to the GEMM's rounding: the two projections have different widths)
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------
# 3. the layer's output against an explicit per-head reference


def reference_output(layer: GatedAttention, x: torch.Tensor, docs: torch.Tensor, cap: int | None = None, s: int = 1) -> torch.Tensor:
    """The layer written head by head from the definition (fixed temperature): head (i, j) attends over the keys [row i ; column j] and the
    values [row i ; column j] of its document at or before the query, the logit q . k / sqrt(head_dim); under the cap a key more than
    cap tokens back (s * distance > cap under token-scale rotary) is scored by the query rotated by the cap against the unrotated key."""
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
            i, j = (h // g.heads_per_cell) // g.cols, (h // g.heads_per_cell) % g.cols      # head h is in the cell h // heads_per_cell, row-major
            k_row = kc[:, i * g.row_dim:(i + 1) * g.row_dim]
            k_col = kc[:, g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]
            qn = rms(q_all[:, h], layer.q_norm.weight)
            kn = rms(torch.cat([k_row, k_col], dim=-1), layer.k_norm.weight)  # the head's key [row i ; column j] normed over the whole head, before the rotary part
            v_h = torch.cat([vc[:, i * g.row_dim:(i + 1) * g.row_dim], vc[:, g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]], dim=-1)
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
@pytest.mark.parametrize("shape,cap,s", [("2x2", None, 1), ("2x2_pairs", None, 1), ("2x4", None, 1), ("2x2", 7, 1), ("2x2_pairs", 7, 1), ("2x4", 7, 1),
                                         ("2x2_pairs", 12, 2), ("2x4", 12, 3), ("2x4_unequal", 7, 1)])
def test_layer_output_equals_the_per_head_reference(backend, shape, cap, s, flex_cpu_backward):  # noqa: F811
    over = {ATT + "rope_fraction": 0.5, ATT + "backend": backend, **SHAPES[shape], **({} if cap is None else {ATT + "rope_max_distance": cap})}
    layer = grid_layer(over)
    T = 36
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    x = torch.randn(2, T, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    ctx, _ = sequence_context(docs, None)
    if s != 1:
        assert cap is not None
        ctx.rope_scale = s
    out, keys = layer(x, ctx)
    torch.testing.assert_close(out, reference_output(layer, x, docs, cap, s), rtol=1e-11, atol=1e-11)
    assert keys.k.shape == (2, T, 1, layer.grid.width)  # the cache is the components, RAW: no norm, no rotary, and no k_far (derived at the read)
    torch.testing.assert_close(keys.k[:, :, 0], x @ layer.k_proj.weight.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(keys.v[:, :, 0], x @ layer.v_proj.weight.T, rtol=1e-12, atol=1e-12)
    assert keys.k_far is None and torch.equal(keys.pos, ctx.pos * s)       # and the rotary position of each entry (TST: s x the document position)


def test_the_reference_has_pairs_beyond_the_cap():
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    ctx, _ = sequence_context(docs, None)
    assert int(ctx.pos.max()) > 7 and int((ctx.pos > 7).sum()) > 20  # documents longer than the caps used above: far pairs exist




def test_the_grid_layer_traces_without_graph_breaks():
    """torch.compile (dynamo, the eager backend: no C++ compiler needed) traces the capped grid layer with the encoder's shared rotation tables as one graph and gives
    the eager values: the assembly, the head norm and the rotation at the read are static code."""
    from ced.model.attention import CAPPED_ROTATION_KEY, capped_rotation

    layer = grid_layer({ATT + "rope_fraction": 0.5, ATT + "rope_max_distance": 7})
    docs = torch.tensor([[0] * 25 + [1] * 11, [0] * 36])
    x = torch.randn(2, 36, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    ctx, _ = sequence_context(docs, None)
    ctx.memo[CAPPED_ROTATION_KEY] = capped_rotation(ctx.pos, layer.cfg, torch.float64)
    ref, ref_keys = layer(x, ctx)
    torch._dynamo.reset()
    compiled = torch.compile(lambda a: layer(a, ctx), backend="eager", fullgraph=True)
    out, keys = compiled(x)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    assert torch.equal(keys.k, ref_keys.k) and keys.k_far is None and ref_keys.k_far is None and torch.equal(keys.v, ref_keys.v)
    assert torch.equal(keys.pos, ref_keys.pos)
    # and the shared tables are what the layer used: the same values as the tables-free path
    ctx.memo.pop(CAPPED_ROTATION_KEY)
    torch.testing.assert_close(layer(x, ctx)[0], ref, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------------------
# 4. the grid is the multi-head layer with block-assembled weights


def assembled_rows(w: torch.Tensor, g: GridSpec) -> torch.Tensor:
    """(width, d) component weights to the (n_heads * head_dim, d) rows of the multi-head layer: head h = [row i ; column j] of its cell (i, j)."""
    blocks = []
    for h in range(g.n_heads):
        i, j = g.cell_of(h)
        blocks += [w[i * g.row_dim:(i + 1) * g.row_dim], w[g.row_width + j * g.col_dim:g.row_width + (j + 1) * g.col_dim]]
    return torch.cat(blocks, dim=0)


def mha_twin(grid_model, over: dict):
    """The gqa model with n_kv_heads = n_q_heads whose k_proj / v_proj are the grid model's weights assembled, every other tensor copied. over:
    the grid model's overrides; the twin takes them without the grid keys and on the reference backend (the ground truth)."""
    a = grid_model.cfg.encoder.attention
    drop = (ATT + "kv_sharing", ATT + "grid_rows", ATT + "grid_cols", ATT + "row_dim", ATT + "col_dim", HPC)
    twin_over = {k: v for k, v in over.items() if k not in drop and not k.endswith(".backend")}
    twin = make_model("v2_control", {**twin_over, ATT + "n_kv_heads": a.n_q_heads})
    sd, state = grid_model.state_dict(), twin.state_dict()
    g = mixers(grid_model)[0].grid
    for name in state:
        if name.endswith(("mixer.k_proj.weight", "mixer.v_proj.weight")):
            state[name] = assembled_rows(sd[name], g)
        else:
            state[name] = sd[name]
    twin.load_state_dict(state)
    return twin


CASES = {
    "plain": {**FIVE},
    "capped_mix_ssmax": {**MIXCAP},
    "capped_mix_ssmax_2x4": {**MIXCAP, **GRID24},
    "capped_mix_ssmax_pairs": {**MIXCAP, **GRID22P2},
    "r1e_capped_mix_ssmax": {**MIXCAP, R1E_KEY: "block_entry"},         # the arm's whole stack in small: R1-entry, the cap, the NoPE mix, ssmax
    "r1e_capped_mix_ssmax_pairs": {**MIXCAP, **GRID22P2, R1E_KEY: "block_entry"},
    # the rotary part (rope_fraction 1.0: all 16 dimensions of a head) reaches into the column component (row_dim 8): the rotation is on the assembled head
    "plain_rotary_into_columns": {**FIVE, ATT + "rope_fraction": 1.0},
    "capped_mix_ssmax_rotary_into_columns": {**MIXCAP, ATT + "rope_fraction": 1.0},
}
# QK-norm ON (the arm's setting: the grid norms the assembled per-head key over the whole head with one k_norm, the multi-head layer's) and OFF
QK = {"qk_on": {}, "qk_off": NOQK}


@pytest.mark.parametrize("qk", list(QK))
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("backend", ["reference", "sdpa", "flex"])
def test_grid_equals_the_multi_head_layer_with_assembled_weights(case, backend, qk, flex_cpu_backward):  # noqa: F811
    over = {**GRID22, **QK[qk], **CASES[case], **(fast(backend) if backend != "reference" else {})}
    grid = make_model("v2_control", over)
    assert grid.cfg.encoder.attention.qk_norm == (qk == "qk_on")
    twin = mha_twin(grid, over)
    tokens, docs, dec = long_doc_rows(2, L, grid.cfg.vocab_size, seed=11)
    torch.testing.assert_close(grid(tokens, docs).logits, twin(tokens, docs).logits, **TOL)


@pytest.mark.parametrize("qk", list(QK))
@pytest.mark.parametrize("case", ["plain", "capped_mix_ssmax", "capped_mix_ssmax_pairs", "capped_mix_ssmax_2x4", "capped_mix_ssmax_rotary_into_columns"])
def test_grid_gradients_are_the_assembled_weights_gradients_summed(case, qk):
    over = {**GRID22, **QK[qk], **CASES[case]}
    grid = make_model("v2_control", over)
    twin = mha_twin(grid, over)
    tokens, docs, _ = long_doc_rows(2, L, grid.cfg.vocab_size, seed=12)
    for m in (grid, twin):
        m.zero_grad()
        m(tokens, docs).logits.square().mean().backward()
    gp, tp = dict(grid.named_parameters()), dict(twin.named_parameters())
    g = mixers(grid)[0].grid
    for name, p in gp.items():
        if name.endswith(("mixer.k_proj.weight", "mixer.v_proj.weight")):
            # the twin's gradient on the rows of the heads of cell (i, j) flows back to row i's and column j's components: summed over the cell's heads,
            # over the columns of a row and over the rows of a column
            full = tp[name].grad.view(g.rows, g.cols, g.heads_per_cell, g.head_dim, -1).sum(2)
            want = torch.cat([full[..., : g.row_dim, :].sum(1).reshape(g.row_width, -1), full[..., g.row_dim:, :].sum(0).reshape(g.col_width, -1)], dim=0)
            torch.testing.assert_close(p.grad, want, rtol=1e-9, atol=1e-12)
        else:
            torch.testing.assert_close(p.grad, tp[name].grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("qk", list(QK))
def test_token_scale_rotary_through_the_model(qk, flex_cpu_backward):  # noqa: F811
    """TST's bags (rope_scale 3): the grid model equals the multi-head twin, on every backend, QK-norm on and off."""
    tokens = torch.randint(0, 256, (2, 40, 3), generator=torch.Generator().manual_seed(2))
    docs = torch.tensor([[0] * 25 + [1] * 15, [0] * 40])
    outs = {}
    for backend in ("reference", "sdpa", "flex"):
        over = {**GRID22, **QK[qk], **MIXCAP, ATT + "backend": backend}
        grid = make_model("v2_control", over)
        outs[backend] = grid(tokens, docs, rope_scale=3).logits
        if backend == "reference":
            twin = mha_twin(grid, over)
            torch.testing.assert_close(outs[backend], twin(tokens, docs, rope_scale=3).logits, **TOL)
    torch.testing.assert_close(outs["sdpa"], outs["reference"], **TOL)
    torch.testing.assert_close(outs["flex"], outs["reference"], **TOL)
    assert not torch.allclose(outs["flex"], grid(tokens, docs).logits)  # the scale changes the function


# ------------------------------------------------------------------------------------------
# 5. causality, cached generation, the cache, resume


GRID_MODELS = {
    "2x2_plain": {**GRID22, **FIVE},
    "2x2_capped_mix_ssmax": {**GRID22, **MIXCAP},
    "2x4_capped_mix_ssmax": {**GRID24, **MIXCAP},
    "2x2_pairs_capped_mix_ssmax": {**GRID22P2, **MIXCAP},
    "2x4_unequal_capped_mix_ssmax": {**GRIDU, **MIXCAP},
    "2x2_capped_mix_qknorm_off": {**GRID22, **MIXCAP, **NOQK},
    "2x2_r1e_capped_mix_ssmax": {**GRID22, **MIXCAP, R1E_KEY: "block_entry"},
    "2x2_pairs_r1e_capped_mix_ssmax": {**GRID22P2, **MIXCAP, R1E_KEY: "block_entry"},
}
# the controls on the existing n_kv_heads path (the arms' stack in small, ssmax / cap / NoPE mix): GQA with 2 KV heads over 4 and over 8 query heads, GQA with 4 over 8
GQA_MODELS = {
    "gqa2_4q": {**GQA2_4Q, **MIXCAP},
    "gqa2_8q": {**GQA2_8Q, **MIXCAP},
    "gqa4_8q": {**GQA4_8Q, **MIXCAP},
    "gqa2_8q_r1e": {**GQA2_8Q, **MIXCAP, R1E_KEY: "block_entry"},
}
ALL_MODELS = {**GRID_MODELS, **GQA_MODELS}


@pytest.mark.parametrize("name", list(ALL_MODELS))
def test_future_tokens_do_not_change_past_outputs(name):
    """Perturb the tokens after t, require the outputs at or before t unchanged (on the reference path), for every grid arm and for the GQA-2 /
    GQA-4 controls."""
    model = make_model("v2_control", ALL_MODELS[name])
    tokens, docs, _ = long_doc_rows(2, L, model.cfg.vocab_size, seed=1)
    base = model(tokens, docs).logits
    for j in [1, 7, 20, 33, L - 1]:
        changed = tokens.clone()
        changed[:, j] = (changed[:, j] + 1 + j) % model.cfg.vocab_size
        out = model(changed, docs).logits
        assert torch.equal(out[:, :j], base[:, :j]), f"outputs before {j} changed"
        assert not torch.equal(out[:, j:], base[:, j:]), "the change had no effect (the test is vacuous)"


@pytest.mark.parametrize("name", ["2x2_capped_mix_ssmax", "2x2_pairs_capped_mix_ssmax", "gqa2_8q"])
def test_gradients_do_not_flow_from_future_inputs(name):
    model = make_model("v2_control", ALL_MODELS[name])
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


@pytest.mark.parametrize("name", list(ALL_MODELS))
def test_teacher_forced_equals_token_by_token_cached_generation(name):
    model = make_model("v2_control", ALL_MODELS[name])
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=10)
    teacher = model(tokens, docs).logits
    stepwise = chunked(model, tokens, docs, dec, list(range(1, L)))
    torch.testing.assert_close(stepwise, teacher, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["2x2_capped_mix_ssmax", "2x2_pairs_capped_mix_ssmax", "gqa4_8q"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_teacher_forced_equals_random_chunked_prefill(name, seed):
    model = make_model("v2_control", ALL_MODELS[name])
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=20 + seed)
    teacher = model(tokens, docs).logits
    g = torch.Generator().manual_seed(seed)
    boundaries = sorted(torch.randperm(L - 1, generator=g)[:5].add(1).tolist())
    torch.testing.assert_close(chunked(model, tokens, docs, dec, boundaries), teacher, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["2x2_capped_mix_ssmax", "2x2_pairs_capped_mix_ssmax", "gqa2_8q"])
@pytest.mark.parametrize("backend", ["sdpa", "flex"])
def test_cached_generation_on_the_fast_backends(name, backend, flex_cpu_backward):  # noqa: F811
    model = make_model("v2_control", {**ALL_MODELS[name], **fast(backend)})
    ref = make_model("v2_control", ALL_MODELS[name])
    ref.load_state_dict(model.state_dict())
    tokens, docs, dec = long_doc_rows(1, 40, model.cfg.vocab_size, seed=30)
    teacher = ref(tokens, docs).logits
    torch.testing.assert_close(chunked(model, tokens, docs, dec, [13, 14, 30]), teacher, **TOL)


def cache_numbers_a_token(over: dict) -> int:
    """The numbers one token stores in a global layer's key cache (the value cache holds as many), read from the cache a prefill builds."""
    model = make_model("v2_control", over)
    tokens, docs, dec = long_doc_rows(1, L, model.cfg.vocab_size, seed=13)
    sets = key_sets(model.prefill(tokens, docs, dec).cache.encoder)
    assert len(sets) == 5 and len({ks.k[0, 0].numel() for ks in sets}) == 1 and sets[0].k[0, 0].numel() == sets[0].v[0, 0].numel()
    return sets[0].k[0, 0].numel()


def test_the_cache_numbers_a_token_of_the_arms_and_their_controls():
    """The tiny model's head_dim 16 stands for 128: the MQA baseline (1 KV head) stores 16 numbers for K and for V; the 2 x 2 grid (components of 8 = the
    arm's 64) stores 2 x 8 + 2 x 8 = 32 = 2 heads' worth, as GQA-2 does over 4 heads and over 8 heads and as the pairs grid does; the 2 x 4 grid stores
    2 x 8 + 4 x 8 = 48 = 3 heads' worth; GQA-4 64 = 4 heads'. In the arms' units (x 8): 128, 256, 256, 256, 384, 512."""
    got = {n: cache_numbers_a_token(o) for n, o in (("mqa", MIXCAP), ("grid22", GRID_MODELS["2x2_capped_mix_ssmax"]), ("pairs", GRID_MODELS["2x2_pairs_capped_mix_ssmax"]),
                                                  ("grid24", GRID_MODELS["2x4_capped_mix_ssmax"]), ("gqa2_4q", GQA_MODELS["gqa2_4q"]), ("gqa2_8q", GQA_MODELS["gqa2_8q"]),
                                                  ("gqa4_8q", GQA_MODELS["gqa4_8q"]))}
    assert got == {"mqa": 16, "grid22": 32, "pairs": 32, "grid24": 48, "gqa2_4q": 32, "gqa2_8q": 32, "gqa4_8q": 64}
    assert got["grid22"] == got["pairs"] == got["gqa2_4q"] == got["gqa2_8q"] == 2 * got["mqa"]        # the equal-cache pairs: grid against GQA-2
    assert got["grid24"] == 3 * got["mqa"] and got["gqa4_8q"] == 4 * got["mqa"]


@pytest.mark.parametrize("name,over", [("2x2", GRID_MODELS["2x2_capped_mix_ssmax"]), ("2x2_pairs", GRID_MODELS["2x2_pairs_capped_mix_ssmax"]),
                                       ("2x4", GRID_MODELS["2x4_capped_mix_ssmax"])])
def test_the_cache_holds_components_and_their_numbers_a_token(name, over):
    model = make_model("v2_control", over)
    g = mixers(model)[0].grid
    tokens, docs, dec = long_doc_rows(2, L, model.cfg.vocab_size, seed=13)
    cache = model.prefill(tokens, docs, dec).cache
    sets = key_sets(cache.encoder)
    assert len(sets) == 5  # the five global layers
    per_token = g.rows * g.row_dim + g.cols * g.col_dim
    for ks in sets:
        # one stored "head" of the components, never the n_q_heads assembled ones
        assert ks.k.shape == ks.v.shape == (2, L, 1, per_token)
        assert ks.k[0, 0].numel() == ks.v[0, 0].numel() == per_token
        assert ks.k.shape[2] != g.n_heads
        # RAW components: no k_far (the unrotated normed key is derived at the read), and the rotary position of every entry beside idx and doc
        assert ks.k_far is None and torch.equal(ks.pos, sequence_context(docs, None)[0].pos)
    # what is stored is what the layer projects, raw: the read assembles, norms and rotates it (test_the_cache_stores_raw_components_...)
    layer = mixers(model)[0]
    assert layer.k_proj.weight.shape[0] == layer.v_proj.weight.shape[0] == per_token
    # and a later chunk extends it by the same components
    out = model.extend(tokens[:, :3], docs[:, -3:], dec[:, :3], cache)
    assert all(ks.k.shape == (2, L + 3, 1, per_token) for ks in key_sets(out.cache.encoder))


@pytest.mark.parametrize("cap,with_tables", [(None, False), (7, False), (7, True)], ids=["plain", "capped", "capped_shared_tables"])
@pytest.mark.parametrize("shape", ["2x2_pairs", "2x4"])
def test_the_cache_stores_raw_components_and_a_cached_read_matches_the_full_read(cap, with_tables, shape):
    """The layer's cache after every chunk is the RAW projections (no norm, no rotary; no k_far under the cap) with each entry's rotary position; the
    read norms and rotates the assembled heads at the entries' own positions (also across a document boundary inside a chunk), so the chunked outputs equal the
    full forward's, with the encoder's shared rotation tables in the context (the chunk's: the cached entries rotate with tables of their own positions) or without."""
    from ced.model.attention import CAPPED_ROTATION_KEY, capped_rotation

    layer = grid_layer({ATT + "rope_fraction": 0.5, **SHAPES[shape], **({} if cap is None else {ATT + "rope_max_distance": cap})})
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
    assert torch.equal(cache.pos, full_keys.pos) and torch.equal(cache.pos, sequence_context(docs, None)[0].pos)
    torch.testing.assert_close(cache.k, full_keys.k, rtol=1e-12, atol=1e-12)






def _cat_leaf(objs: list):
    """The batch-dimension concatenation of one cache field across caches (ced/eval/recall_ruler_gen.py: cat_caches)."""
    first = objs[0]
    if first is None:
        assert all(o is None for o in objs)
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(objs, 0)
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        return type(first)(**{f.name: _cat_leaf([getattr(o, f.name) for o in objs]) for f in dataclasses.fields(first)})
    if isinstance(first, (list, tuple)):
        return type(first)(_cat_leaf(list(z)) for z in zip(*objs))
    assert all(o == first for o in objs)
    return first


@pytest.mark.parametrize("name", ["2x2_capped_mix_ssmax", "2x2_pairs_capped_mix_ssmax", "2x4_capped_mix_ssmax", "gqa2_8q"])
def test_caches_concatenate_on_the_batch_axis(name):
    """The batched generator's cat_caches (unmerged ced/eval/recall_ruler_gen.py: a cat of every cache tensor along dim 0, and _cat_leaf here is a copy
    of it) works on the grid's components: the cache is one KeySet "head" of components, the shape of any gqa cache, so a batch-axis concatenation
    needs nothing grid-specific; the rows of the concatenated cache decode exactly as in their own caches."""
    model = make_model("v2_control", ALL_MODELS[name])
    tokens, docs, dec = long_doc_rows(3, 30, model.cfg.vocab_size, seed=14)
    caches = [model.prefill(tokens[r:r + 1], docs[r:r + 1], dec[r:r + 1]).cache for r in range(3)]
    joined = _cat_leaf(caches)
    nxt = torch.randint(0, model.cfg.vocab_size, (3, 1), generator=torch.Generator().manual_seed(15))
    batched = model.extend(nxt, docs[:, -1:], dec[:, -1:], joined).logits
    for r in range(3):
        alone = model.extend(nxt[r:r + 1], docs[r:r + 1, -1:], dec[r:r + 1, -1:], caches[r]).logits
        torch.testing.assert_close(batched[r:r + 1], alone, rtol=0, atol=1e-10)


def test_packed_documents_are_isolated():
    model = make_model("v2_control", GRID_MODELS["2x2_capped_mix_ssmax"])
    tokens, _, _ = random_rows(1, L, model.cfg.vocab_size, seed=4)
    split = 19
    docs = torch.zeros(1, L, dtype=torch.long)
    docs[:, split:] = 1
    packed = model(tokens, docs).logits
    alone = model(tokens[:, split:], torch.zeros(1, L - split, dtype=torch.long)).logits
    torch.testing.assert_close(packed[:, split:], alone, rtol=0, atol=1e-12)


def grid_train_config(steps: int = 8, over: dict | None = None, base: dict | None = None) -> TrainConfig:
    """base: the arm's keys (default the 60M grid in small); {} gives a gqa model (the control), with n_kv_heads from `over`."""
    model = set_config(tiny_control_config(), {ATT + "n_kv_heads": 1, **(GRID22 if base is None else base), **(over or {})})
    return TrainConfig(model=model, data=DataConfig(seq_len=32, batch_size=2), steps=steps, log_every=1000, ckpt_every_steps=1000, ckpt_every_minutes=1e9)


CAPPED_FIVE = {**FIVE, ATT + "rope_max_distance": 8, "encoder.nope_global_layers": (1,)}


@pytest.mark.parametrize("base,over,kind", [(None, None, "grid"), (None, CAPPED_FIVE, "grid"), (GRID22P2, CAPPED_FIVE, "grid"), (GRID24, CAPPED_FIVE, "grid"),
                                            ({}, {**CAPPED_FIVE, **GQA2_4Q}, "gqa"), ({}, {**CAPPED_FIVE, **GQA2_8Q}, "gqa")])
def test_checkpoint_resume_matches_uninterrupted(base, over, kind, tmp_path):
    """Resume equals uninterrupted: the single-head grid (plain and capped), the pairs grid, the 2 x 4 grid and the GQA-2 controls on the existing path."""
    cfg = grid_train_config(8, over, base)
    straight = Trainer(cfg, tmp_path / "straight")
    straight_losses = [straight.train_step()["loss"] for _ in range(8)]
    first = Trainer(cfg, tmp_path / "resumed")
    losses = [first.train_step()["loss"] for _ in range(4)]
    path = first.save_checkpoint()
    del first
    torch.manual_seed(999)  # disturb the global RNG: the checkpoint must restore everything needed
    second = Trainer.from_checkpoint(path)
    att = second.cfg.model.encoder.attention
    assert att == cfg.model.encoder.attention and att.kv_sharing == kind
    if kind == "grid":
        assert att.row_dim == 8 and att.heads_per_cell == cfg.model.encoder.attention.heads_per_cell
    losses += [second.train_step()["loss"] for _ in range(4)]
    assert losses == straight_losses
    assert_same_state(second, straight)


# ------------------------------------------------------------------------------------------
# 6. gqa is untouched


def test_gqa_layer_is_the_layer_it_was():
    cfg = set_config(tiny_v2_control_config(), {})
    layer = GatedAttention(64, cfg.encoder.attention, 1e-6)
    assert layer.grid is None
    assert sorted(layer.state_dict()) == ["k_norm.weight", "k_proj.weight", "o_proj.weight", "q_norm.weight", "q_proj.weight", "v_proj.weight"]
    assert layer.k_proj.weight.shape == (16, 64) and layer.q_proj.weight.shape == (128, 64)
    grid = GatedAttention(64, set_config(tiny_v2_control_config(), GRID22).encoder.attention, 1e-6)
    # the grid layer has the gqa layer's parameter NAMES (one k_norm of head_dim numbers; no per-component norms)
    assert sorted(grid.state_dict()) == sorted(layer.state_dict())
    assert grid.k_proj.weight.shape == grid.v_proj.weight.shape == (2 * 8 + 2 * 8, 64) and grid.k_norm.weight.shape == layer.k_norm.weight.shape == (16,)


def test_a_gqa_model_has_the_same_parameters_as_before():
    """A model without the key: no k_row_norm / k_col_norm anywhere (the per-component norms of the first grid build are gone from the code too), the same
    names and shapes (the existing suites compare its outputs)."""
    model = make_model("v2_control", {})
    names = [n for n, _ in model.named_parameters()]
    assert not any("k_row_norm" in n or "k_col_norm" in n for n in names)
    assert all(m.grid is None for m in mixers(model))
    grid_names = [n for n, _ in make_model("v2_control", GRID22).named_parameters()]
    assert not any("k_row_norm" in n or "k_col_norm" in n for n in grid_names) and any(n.endswith("mixer.k_norm.weight") for n in grid_names)


def test_the_grid_and_its_gqa2_control_have_equal_kv_projection_sizes_and_parameters():
    """The grid 2 x 2 (components of 8, the arm's 64) against GQA with 2 KV heads at the same head dimension: k_proj and v_proj are (2 x head_dim, d)
    in both, the QK-norms are the same module with head_dim numbers in both (the grid's one k_norm over the assembled head, GQA's k_norm over a KV head),
    q_proj / gate / o_proj are untouched, so a layer has the same parameters; the same for the pairs grid over 8 heads against GQA-2 over 8. The MQA
    baseline has half the K/V projection."""
    def layer(over: dict) -> GatedAttention:
        return GatedAttention(64, set_config(tiny_v2_control_config(), over).encoder.attention, 1e-6)

    def n(m) -> int:
        return sum(p.numel() for p in m.parameters())

    mqa, grid, gqa2 = layer({}), layer(GRID22), layer(GQA2_4Q)
    assert grid.k_proj.weight.shape == grid.v_proj.weight.shape == gqa2.k_proj.weight.shape == gqa2.v_proj.weight.shape == (32, 64)
    assert mqa.k_proj.weight.shape == (16, 64)
    assert grid.q_proj.weight.shape == gqa2.q_proj.weight.shape == mqa.q_proj.weight.shape and grid.o_proj.weight.shape == gqa2.o_proj.weight.shape
    assert grid.k_norm.weight.shape == gqa2.k_norm.weight.shape == mqa.k_norm.weight.shape == (16,) and type(grid.k_norm) is type(gqa2.k_norm)
    assert sorted((k, tuple(v.shape)) for k, v in grid.state_dict().items()) == sorted((k, tuple(v.shape)) for k, v in gqa2.state_dict().items())
    assert n(grid) == n(gqa2) == n(mqa) + 2 * 16 * 64
    pairs, gqa2_8 = layer(GRID22P2), layer(GQA2_8Q)
    assert n(pairs) == n(gqa2_8) and pairs.q_proj.weight.shape == gqa2_8.q_proj.weight.shape == (8 * 16 * 2, 64)
    # the 2 x 4 grid (6 components of 8 = 3 heads' worth) and GQA-4 (4 heads' worth) differ by one head's K/V projections and norm
    g24, gqa4 = layer(GRID24), layer(GQA4_8Q)
    assert g24.k_proj.weight.shape[0] == 48 and gqa4.k_proj.weight.shape[0] == 64
    assert n(gqa4) - n(g24) == 2 * 16 * 64            # the K and V projections of one more head; both norms hold 16 numbers


def test_models_with_grids_and_their_controls_have_equal_parameters_but_for_the_kv_width():
    """Whole models in small: the 60M grid and GQA-2 have equal totals (and equal non-embedding totals); both exceed the MQA baseline by the K/V width alone."""
    from ced.utils.params import build_on_meta

    def total(over: dict) -> int:
        return sum(p.numel() for p in build_on_meta(set_config(tiny_v2_control_config(), over)).parameters())

    base = total(FIVE)
    n_glob, d = 5, 64
    assert total({**GRID22, **FIVE}) == total({**GQA2_4Q, **FIVE}) == base + n_glob * 2 * 16 * d
    assert total({**GRID22P2, **FIVE}) == total({**GQA2_8Q, **FIVE})
    assert total({**GQA4_8Q, **FIVE}) - total({**GRID24, **FIVE}) == n_glob * 2 * 16 * d


@pytest.mark.parametrize("grid_over,gqa_over", [(GRID22, GQA2_4Q), (GRID22P2, GQA2_8Q), ({**GRID22P2, **MIXCAP, R1E_KEY: "block_entry"}, {**GQA2_8Q, **MIXCAP, R1E_KEY: "block_entry"})],
                         ids=["60m-shape", "200m-shape", "200m-shape-whole-stack"])
def test_the_grid_and_its_gqa2_control_start_from_identical_parameters(grid_over, gqa_over):
    """The reading rule's statement: with the same seed a grid model and its equal-cache GQA-2 control are the same in EVERY parameter at initialisation
    (embeddings, q / k / v / o projections, gates, temperatures, FFNs, the QK-norms including the key norm), the same names and shapes, the same
    total: since the grid norms the assembled per-head key with ONE k_norm of head_dim numbers (the gqa layer's module), the arms differ in the
    sharing alone (the review found the per-component norm of the first build a second difference; wanted a difference pinned on the grid)."""
    def model_state(over: dict):
        torch.manual_seed(3)
        model = build_model(set_config(tiny_v2_control_config(), {**over, **FIVE}))
        return model, model.state_dict()

    (mg, g), (mq, q) = model_state(grid_over), model_state(gqa_over)
    assert g.keys() == q.keys() and len(g) > 20
    for k in g:
        assert g[k].shape == q[k].shape and torch.equal(g[k], q[k]), k
    key_norms = [k for k in g if k.endswith("mixer.k_norm.weight")]
    assert len(key_norms) == 5 and not any("k_row_norm" in k or "k_col_norm" in k for k in g)                  # the five global layers, one key norm each, as GQA's
    assert all(not g[k].any() and g[k].numel() == 16 for k in key_norms)                                      # zero gains (zero-centred), 16 numbers a layer in both
    assert sum(p.numel() for p in mg.parameters()) == sum(p.numel() for p in mq.parameters())
