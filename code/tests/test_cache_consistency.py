"""Train/inference consistency: teacher-forced logits equal the logits of cached generation.

The teacher-forced side is the training forward (fixed decoder capacity, gathered slots, whole
rows). The cached side processes the same rows token by token (and in random chunks) with the
caches: GDN and conv state, encoder global KV, memory KV, and per-(loop, layer) own KV. Everything
runs in float64 so the tolerance can be tight.
"""

import pytest
import torch

from helpers import LOOPED_VARIANTS, make_model, random_rows

L = 48
ATOL = 1e-10


def run_in_chunks(model, tokens, docs, dec, T, boundaries):
    """Process the rows in chunks [0, b1), [b1, b2), ... with caches; return the concatenated logits."""
    edges = [0, *boundaries, tokens.shape[1]]
    out, pieces = None, []
    for a, b in zip(edges[:-1], edges[1:]):
        if out is None:
            out = model.prefill(tokens[:, a:b], docs[:, a:b], dec[:, a:b], T=T)
        else:
            out = model.extend(tokens[:, a:b], docs[:, a:b], dec[:, a:b], out.cache)
        pieces.append(out.logits)
    return torch.cat(pieces, dim=1)


@pytest.mark.parametrize("arch,variant", [("looped_ced", v) for v in LOOPED_VARIANTS] + [("control", "default")])
def test_teacher_forced_equals_token_by_token(arch, variant):
    model = make_model(arch, LOOPED_VARIANTS[variant] if arch == "looped_ced" else None)
    T = model.cfg.loop.t_eval
    n_dec = model.capacity(L) if arch == "looped_ced" else L // 4
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=10, n_dec=n_dec)

    teacher = model(tokens, docs, dec, T=T)  # the training path, default capacity
    assert int(teacher.n_dropped.sum()) == 0
    if arch == "looped_ced":
        assert torch.equal(teacher.kept, dec)

    stepwise = run_in_chunks(model, tokens, docs, dec, T, boundaries=list(range(1, L)))
    torch.testing.assert_close(stepwise, teacher.logits, rtol=0, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_teacher_forced_equals_random_chunked_prefill(seed):
    """Prefill of a random prefix (often splitting an assistant span) followed by random-size chunks."""
    model = make_model("looped_ced")
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=20 + seed, n_dec=model.capacity(L))
    teacher = model(tokens, docs, dec).logits
    g = torch.Generator().manual_seed(seed)
    boundaries = sorted(torch.randperm(L - 1, generator=g)[:5].add(1).tolist())
    chunked = run_in_chunks(model, tokens, docs, dec, model.cfg.loop.t_eval, boundaries)
    torch.testing.assert_close(chunked, teacher, rtol=0, atol=ATOL)


@pytest.mark.parametrize("T", [1, 2, 3])
def test_greedy_generation_matches_teacher_forced_rescoring(T):
    """Generate with the cache, then score prompt + generation in one teacher-forced pass.

    The prompt holds an earlier assistant turn (decoder positions), a user turn (context) and the
    header of the current turn, so later turns must read the earlier turn's own per-loop KVs.
    """
    model = make_model("looped_ced")
    V = model.cfg.vocab_size
    g = torch.Generator().manual_seed(T)
    prompt = torch.randint(0, V, (2, 30), generator=g)
    prompt_docs = torch.zeros(2, 30, dtype=torch.long)
    prompt_dec = torch.zeros(2, 30, dtype=torch.bool)
    prompt_dec[:, 6:15] = True  # earlier assistant turn
    prompt_dec[:, 29] = True  # header of the turn being generated
    n_new = 12

    new_tokens, gen_logits = model.generate(prompt, prompt_docs, prompt_dec, n_new, T=T)

    full = torch.cat([prompt, new_tokens], dim=1)
    full_docs = torch.zeros_like(full)
    full_dec = torch.cat([prompt_dec, torch.ones(2, n_new, dtype=torch.bool)], dim=1)
    teacher = model(full, full_docs, full_dec, T=T, capacity=full.shape[1]).logits
    rescored = teacher[:, 29 : 29 + n_new]
    torch.testing.assert_close(rescored, gen_logits, rtol=0, atol=ATOL)
    assert torch.equal(rescored.argmax(-1), new_tokens)


def test_sampled_generation_matches_teacher_forced_rescoring():
    model = make_model("looped_ced")
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 20), generator=g)
    dec = torch.zeros(1, 20, dtype=torch.bool)
    dec[:, -1] = True
    new_tokens, gen_logits = model.generate(prompt, torch.zeros_like(prompt), dec, 10, greedy=False, generator=g)
    full = torch.cat([prompt, new_tokens], dim=1)
    full_dec = torch.cat([dec, torch.ones(1, 10, dtype=torch.bool)], dim=1)
    teacher = model(full, torch.zeros_like(full), full_dec, capacity=full.shape[1]).logits
    torch.testing.assert_close(teacher[:, 19:29], gen_logits, rtol=0, atol=ATOL)


def test_cache_rejects_a_different_loop_count():
    model = make_model("looped_ced")
    tokens, docs, dec = random_rows(1, 10, model.cfg.vocab_size, seed=3, n_dec=3)
    out = model.prefill(tokens, docs, dec, T=2)
    with pytest.raises(AssertionError):
        model.forward(tokens[:, :1], docs[:, :1], dec[:, :1], T=3, capacity=1, cache=out.cache)


def test_float32_smoke():
    """The same equivalence holds in float32 up to rounding."""
    model = make_model("looped_ced", dtype=torch.float32)
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=11, n_dec=model.capacity(L))
    teacher = model(tokens, docs, dec).logits
    stepwise = run_in_chunks(model, tokens, docs, dec, model.cfg.loop.t_eval, boundaries=list(range(1, L)))
    torch.testing.assert_close(stepwise, teacher, rtol=1e-4, atol=1e-4)
