"""The components of the adopted architecture, each on the plain pure-PyTorch reference path (CPU, float64), written for this code export.

For every component and for their composition, on a tiny control model:
  1. causality: changing a token at position j leaves every output before j bit-identical, and changes the outputs from j on;
  2. teacher-forced logits equal the logits of cached generation (prefill, then one token at a time);
and for the whole stack on the trainer:
  3. a run killed before, at and after the TST switch and resumed is the uninterrupted run, bit for bit (model and optimizer state).

Components: R1-entry (block-entry depth mixing), the GDN write mix `wsl` and its variants qmix / korig / vonly / kqmix, Scalable-Softmax with
capped RoPE and NoPE global layers, tied embeddings, the grid KV sharing (two axes, and the third axis), and the stack of them.
"""

from __future__ import annotations

import pytest
import torch

from helpers import make_model, random_rows
from pos_arms_helpers import assert_same_state, chunked
from test_tst import ENTRY, N, T, cfg_for, run_through_prepare, shards, steps_of  # noqa: F401  (shards: a pytest fixture)
from ced.train.trainer import Trainer

L = 40
ATT = "encoder.attention."
R1E = {"encoder.n_layers": 8, "encoder.attn_res": "block_entry"}  # two blocks of the tiny model's four-layer groups
WSL = {**R1E, "encoder.gdn_write_mix": "layer"}
POSITIONS = {ATT + "temperature": "ssmax", ATT + "temperature_ref_len": 32, ATT + "rope_max_distance": 11, "encoder.nope_global_layers": (1,)}
GRID22 = {ATT + "kv_sharing": "grid", ATT + "grid_rows": 2, ATT + "grid_cols": 2, ATT + "row_dim": 8, ATT + "col_dim": 8, ATT + "n_kv_heads": 1}
GRID24 = {**GRID22, ATT + "grid_cols": 4, ATT + "n_q_heads": 8}
GRID222 = {**GRID22, ATT + "grid_depth": 2, ATT + "col_dim": 4, ATT + "depth_dim": 4, ATT + "n_q_heads": 8}

COMPONENTS = {
    "r1_entry": R1E,
    "wsl": WSL,
    "wsl_qmix": {**WSL, "encoder.gdn_write_variant": "qmix"},
    "wsl_korig": {**WSL, "encoder.gdn_write_variant": "korig"},
    "wsl_vonly": {**WSL, "encoder.gdn_write_variant": "vonly"},
    "wsl_kqmix": {**WSL, "encoder.gdn_write_variant": "kqmix"},
    "ssmax_capped_rope_nope": POSITIONS,
    "tied_embeddings": {"tie_embeddings": True},
    "grid_2x2": GRID22,
    "grid_2x4": GRID24,
    "grid_2x2x2": GRID222,
    "stack": {**WSL, "encoder.gdn_write_variant": "qmix", **POSITIONS, "tie_embeddings": True},
}


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_causality(name):
    model = make_model("control", COMPONENTS[name])
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=1)
    base = model(tokens, docs, dec).logits
    for j in (1, 9, 25):
        changed = tokens.clone()
        changed[:, j] = (changed[:, j] + 1 + j) % model.cfg.vocab_size
        out = model(changed, docs, dec).logits
        assert torch.equal(out[:, :j], base[:, :j]), f"outputs before {j} changed"
        assert not torch.equal(out[:, j:], base[:, j:]), "the change had no effect (the test is vacuous)"


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_teacher_forced_equals_cached_generation(name):
    model = make_model("control", COMPONENTS[name])
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=2)
    teacher = model(tokens, docs, dec).logits
    stepwise = chunked(model, tokens, docs, dec, list(range(1, L)))
    torch.testing.assert_close(stepwise, teacher, rtol=0, atol=1e-10)


def test_the_mixes_start_as_the_plain_model():
    """The mixes and R1-entry are zero-initialised gates: at initialisation (no perturbation) the model with wsl + qmix computes the
    same logits as the model with R1-entry alone, because every mix weight is zero."""
    plain = make_model("control", R1E)
    mixed = make_model("control", {**WSL, "encoder.gdn_write_variant": "qmix"})
    # same seed, so the shared parameters are equal; the mix parameters are the extra ones. Zero them and compare.
    shared = dict(plain.named_parameters())
    with torch.no_grad():
        for n, p in mixed.named_parameters():
            if n in shared:
                p.copy_(shared[n])
            else:
                p.zero_()
    tokens, docs, dec = random_rows(2, L, plain.cfg.vocab_size, seed=3)
    torch.testing.assert_close(mixed(tokens, docs, dec).logits, plain(tokens, docs, dec).logits, rtol=0, atol=1e-10)


STACK_OVER = {**ENTRY, "encoder.gdn_write_mix": "layer", "encoder.gdn_write_variant": "qmix", "tie_embeddings": True}


@pytest.mark.parametrize("stop_offset", [-1, 0, 1])
def test_stack_resume_around_the_tst_switch(stop_offset, shards, tmp_path):
    """The adopted stack (R1-entry + wsl + qmix + tied + TST) killed and resumed around the TST switch equals the uninterrupted run."""
    schedule = [[0, 4], [5, 1]]
    cfg = cfg_for(shards, schedule, STACK_OVER)
    cfg.optim.zero_gate_decay = "none"
    cfg.optim.mix_query_decay = "none_qmix"
    whole = steps_of(run_through_prepare(cfg, tmp_path / "whole"))
    assert sorted(whole) == list(range(1, N + 1))
    ref = Trainer.from_checkpoint(Trainer(cfg, tmp_path / "whole").latest_checkpoint())
    stop = schedule[-1][0] + stop_offset
    d = tmp_path / "killed"
    run_through_prepare(cfg, d, until=stop)
    recs = run_through_prepare(cfg, d)
    assert steps_of(recs) == whole
    assert sum(r.get("event") == T.SWITCH_EVENT for r in recs) == 1  # the switch happens once
    assert_same_state(Trainer.from_checkpoint(Trainer(cfg, d).latest_checkpoint()), ref)
