"""Helper definitions copied from the repository's tests/test_pos_arms.py (CAP, SS_ENC, ENC, L, chunked, fast, long_doc_rows,
assert_same_state): the shared small-model settings and row builders of the position and grid tests."""

from __future__ import annotations

import torch
from ced.model.sequence import sequence_context
from ced.train.trainer import Trainer
from helpers import random_rows

CAP = 11  # the tests' cap: rows of 40-96 tokens whose documents are longer than it

L = 48

ENC = {"encoder.attention.rope_max_distance": CAP}

SS_ENC = {"encoder.attention.temperature": "ssmax", "encoder.attention.temperature_ref_len": 32}

def fast(backend: str) -> dict:
    return {"encoder.attention.backend": backend, "decoder.attention.backend": backend}

@torch.no_grad()
def chunked(model, tokens, docs, dec, boundaries):
    """Teacher forcing in cached chunks: prefill, then extend (the cache carries every layer's key set)."""
    edges = [0, *boundaries, tokens.shape[1]]
    out, pieces = None, []
    for a, b in zip(edges[:-1], edges[1:]):
        args = (tokens[:, a:b], docs[:, a:b], dec[:, a:b])
        out = model.prefill(*args) if out is None else model.extend(*args, out.cache)
        pieces.append(out.logits)
    return torch.cat(pieces, dim=1)

def long_doc_rows(batch: int, length: int, vocab: int, seed: int, n_docs: int = 2, n_dec: int | None = None):
    """random_rows whose documents are long enough to hold pairs beyond the cap."""
    tokens, docs, dec = random_rows(batch, length, vocab, seed=seed, n_docs=n_docs, n_dec=n_dec)
    ctx, _ = sequence_context(docs, None)
    assert int(ctx.pos.max()) > CAP + 3, "the rows hold no pair beyond the cap"
    return tokens, docs, dec

def assert_same_state(a: Trainer, b: Trainer):
    assert a.step == b.step
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert list(sa) == list(sb)
    for name in sa:
        assert torch.equal(sa[name], sb[name]), f"parameter {name} differs"
    oa, ob = a.optimizer.state_dict(), b.optimizer.state_dict()
    assert oa["param_groups"] == ob["param_groups"]
    for i in oa["state"]:
        for key in oa["state"][i]:
            assert torch.equal(oa["state"][i][key], ob["state"][i][key]), f"optimizer state {i}/{key} differs"
