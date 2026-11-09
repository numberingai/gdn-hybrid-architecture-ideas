"""Text generators of the retrieval-inside-the-training-length evals (single and multi-key needle-in-a-haystack, variable
tracking, up to 8k; the extrapolation cells stay separate). RULER-style definitions, the project tokenizer, the haystack from held-out text
(the documents of the web and general_web eval sets, never in a training shard: the same pool as ced/eval/probe_gen.py's,
tools/build_probes.py), fixed seeds. The extrapolation probes (probes-v1 above 8k,
probes-lc-128k) are not regenerated and not touched: these items have their own seed (BASE_SEED) and their own probe names (PROBES), so a
reader of probes-v1 never sees them. Builder: tools/build_report_pcr.py; the items are scored by scripts/eval_probes.py as every probe.

THE THREE PROBES (every item: a context string and one answer string; the answer starts with a space after a letter or a colon, so tokenizing
the context and the answer apart gives the tokens of tokenizing them together: the builder checks it per item under the tokenizer):

- pcr_niah1 (single-key needle in a haystack): the held-out haystack, one sentence "The access code for <name> is <uuid4>." at the sentence
  start nearest to a depth fraction (0.1, 0.3, 0.5, 0.7, 0.9), then the query "What is the access code for <name>? The access code for
  <name> is"; the answer " <uuid>". Cell: length; group: depth.
- pcr_niahk (multi-key needle): the same with N in {4, 16} such sentences (distinct keys and values) at random depths, one key asked. This
  is RULER's multikey NIAH, the cells of probes-v1's multikey at the training lengths. Cell: length, n_keys; group: the asked needle's depth bin.
- pcr_vt (variable tracking, RULER's VT): two chains of VT_HOPS + 1 variables, assigned at random places of the haystack, the statements of
  one chain in order: "VAR <A> = <number>." then "VAR <B> = VAR <A>." and so on down the chain; the two chains hold different 5-digit
  numbers and the variables are random 5-letter upper-case names that occur nowhere in the haystack. The query "Question: Which variables are
  assigned the value <number>? Answer: The variables assigned the value <number> are:" asks for one chain's variables; the answer lists them in
  the chain's order, " <A> <B> <C> <D>". Copying every VAR name of the text in order gives eight names, not four: the distractor chain is what
  makes it a tracking task. Cell: length.

LENGTH. An item's total tokens (context plus answer) under the tokenizer lie in [(1 - LENGTH_TOL) T, T] for T in LENGTHS (2,048, 4,096, 8,192).
The haystack is the pool's whole documents joined by a blank line in a random order, cut at a whitespace character so that the finished item
fits (the cut falls in the last document; probe_gen.fit_haystack, which never cuts a document, cannot fit a 2k item from long documents).

NO CUT, AND THE REDRAW. The
window [(1 - LENGTH_TOL) T, T] is not widened. A draw whose haystack has no whitespace cut inside the window (fit_cut raises NoCutError: the count jumps over
the window at one whitespace step, which a whitespace-free run longer than the window does) is not an error of the item: the item takes the next draw of its own
seeded sequence, the haystack's document order from stream REDRAW_STREAM + k for k = 1, 2, ... (draw 0 is the stream 1 of before, so every item that built
before is the same item). The content (names' pool, values, depths, numbers, the asked needle) comes from the content stream and is the same in every draw. A
redrawn item carries meta["redraws"] = {"count": k, "failed_draws": [the NoCutError texts]}; an item with no redraw has no such key (its JSON is what it was).
The builder counts the redraws per probe in its log and in manifest["probes"][probe]["redraws"]. After REDRAW_MAX redraws the item raises (RuntimeError).

SCORES (ced/eval/probes.py, as every probe): exact = every answer token is the argmax under teacher forcing, first_acc = the first answer token is,
nats = the gold answer's total negative log-probability (nats_per_token per token); the builder's manifest has the floor of each probe (a guess
at random: ln of the number of candidates for the key's value is not defined, so only the cells' sizes are listed).

Every generator is a pure function of (BASE_SEED, probe, cell, item index) and the pool, except through the cut: the content (names, values,
depths, numbers) comes from one random stream and the haystack order from another, so the content does not change with the length measure.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np

from ced.eval import probe_gen as g

BASE_SEED = 20261029
LENGTHS = (2048, 4096, 8192)
NIAH_DEPTHS = g.NEEDLE_DEPTHS  # 0.1 .. 0.9
MULTI_N = g.MULTIKEY_N  # 4, 16
VT_HOPS = 3  # a chain of VT_HOPS + 1 variables
VT_CHAINS = 2  # the asked chain and one distractor chain
LENGTH_TOL = g.LENGTH_TOL
SEP = g.SEP
PROBES = ("pcr_niah1", "pcr_niahk", "pcr_vt")
PROBE_CODE = {p: i for i, p in enumerate(PROBES)}
PER_DEPTH = 40  # pcr_niah1: items per (length, depth): 200 per length
PER_CELL = 200  # pcr_niahk: items per (length, n_keys); pcr_vt: items per length

REDRAW_MAX = 20  # draws after the first one; an item that needs more raises
REDRAW_STREAM = 1000  # the haystack-order stream of draw k >= 1 is REDRAW_STREAM + k (stream 1 is draw 0; stream 0 is the content)

Count = Callable[[str, str], int]  # (context, answer) -> total tokens


class NoCutError(RuntimeError):
    """fit_cut: no whitespace cut of this haystack gives an item with a token count in the window. A property of the draw, cured by the next draw."""


def rng_for(probe: str, cell: tuple, index: int, stream: int) -> np.random.Generator:
    """An independent stream per (probe, cell, item, purpose). stream 0: content; 1: the haystack's document order."""
    words = [BASE_SEED, PROBE_CODE[probe], *[int(round(x * 1000)) if isinstance(x, float) else int(x) for x in cell], int(index), int(stream)]
    return np.random.default_rng(np.random.SeedSequence(words))


def hay_rng_for(probe: str, cell: tuple, index: int, draw: int = 0) -> np.random.Generator:
    """The haystack-order stream of draw `draw` (0: the stream every item had before the redraw rule)."""
    return rng_for(probe, cell, index, 1 if draw == 0 else REDRAW_STREAM + draw)


def draw_until_cut(make_item: Callable[[np.random.Generator], g.TextItem], probe: str, cell: tuple, index: int) -> g.TextItem:
    """make_item(haystack_rng) with the stream of draw 0, then of draw 1, ... until a draw has a cut in the window (NoCutError is the only error
    that moves to the next draw). The failed draws are recorded in the item's meta["redraws"]; none, no key."""
    failed: list[str] = []
    for draw in range(REDRAW_MAX + 1):
        try:
            item = make_item(hay_rng_for(probe, cell, index, draw))
        except NoCutError as e:
            failed.append(str(e))
            continue
        if failed:
            item.meta["redraws"] = {"count": len(failed), "failed_draws": failed}
        return item
    raise RuntimeError(f"{probe} {cell} item {index}: no cut in the window in {REDRAW_MAX + 1} draws: {failed}")


# --------------------------------------------------------------------------------------------
# the haystack


def gather(pool: g.HaystackPool, rng: np.random.Generator, tokens: int, slack: int = 96) -> tuple[str, list[int]]:
    """Whole documents in random order, joined by the separator, until their (estimated) tokens reach `tokens` + slack. Returns the text
    and the chosen documents' indices. RuntimeError if the whole pool is shorter."""
    order = rng.permutation(len(pool.texts))
    chosen, total = [], 0
    for j in order:
        chosen.append(int(j))
        total += int(pool.counts[j]) + 1
        if total >= tokens + slack:
            return pool.join(chosen), chosen
    raise RuntimeError(f"the haystack pool holds {total} tokens, fewer than the {tokens + slack} one item needs")


def fit_cut(text: str, make: Callable[[str], tuple[str, str]], count: Count, total: int, tol: float = LENGTH_TOL) -> tuple[str, str, int, int]:
    """Cut `text` at a whitespace character so that the finished item (context, answer) = make(cut) has count(context, answer) in
    [ceil((1 - tol) total), total]. Returns (context, answer, n_tokens, characters kept). The count grows with the cut (one more word
    adds about one token), so the search keeps a cut that fits and the next one that does not, and ends on neighbours; RuntimeError if
    the whole text does not exceed `total` (gather more); NoCutError (a RuntimeError) if no cut lands in the window (draw_until_cut takes the next draw)."""
    cuts = [m.start() for m in re.finditer(r"\s", text)]
    lo_t = math.ceil((1 - tol) * total)

    def n_at(i: int) -> tuple[int, str, str]:
        c, a = make(text[: cuts[i]].rstrip())
        return count(c, a), c, a

    if not cuts or n_at(len(cuts) - 1)[0] <= total:
        raise RuntimeError(f"the text ({len(text)} characters) does not exceed {total} tokens as an item")
    lo, hi = 0, len(cuts) - 1  # n(lo) <= total < n(hi) is the invariant once lo holds
    if n_at(lo)[0] > total:
        raise RuntimeError("the needles and the query alone exceed the item length")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_at(mid)[0] <= total:
            lo = mid
        else:
            hi = mid
    n, c, a = n_at(lo)
    if n < lo_t:
        raise NoCutError(f"no cut in [{lo_t}, {total}] tokens: the best fits {n}")
    return c, a, n, cuts[lo]


# --------------------------------------------------------------------------------------------
# single-key and multi-key needles


def _needle_item(pool, count: Count, probe: str, length: int, depths: list[float], values: list[str], asked: int,
                 names_pool: list[str], hay_rng: np.random.Generator):
    """Shared by pcr_niah1 and pcr_niahk: gather a haystack, take the first len(values) names of names_pool absent from it, insert one
    "access code" sentence per (depth, value) and cut to length. Returns (context, answer, chosen docs, n tokens, names, char offsets)."""
    n_keys = len(values)
    text, chosen = gather(pool, hay_rng, length)
    names = g.names_absent_from(names_pool, text, n_keys)
    state: dict = {}

    def make(haystack: str) -> tuple[str, str]:
        pairs = [(d, g.needle_sentence(nm, v)) for d, nm, v in zip(depths, names, values)]
        body, offsets = g.insert_sentences(haystack, pairs)
        state["offsets"] = offsets
        return body + g.needle_query(names[asked]), f" {values[asked]}"

    context, answer, n, kept = fit_cut(text, make, count, length)
    make(text[:kept].rstrip())  # the offsets of the kept text
    return context, answer, chosen, n, names, state["offsets"], kept


def niah1_item(pool, count: Count, length: int, depth: float, index: int) -> g.TextItem:
    cell = (length, depth)
    content = rng_for("pcr_niah1", cell, index, 0)
    value = g.random_uuid(content)
    names_pool = g.candidate_names(content, 40)

    def draw(hay_rng: np.random.Generator) -> g.TextItem:
        context, answer, chosen, n, names, offsets, kept = _needle_item(pool, count, "pcr_niah1", length, [depth], [value], 0, names_pool, hay_rng)
        return g.TextItem(
            id=f"pcr_niah1/len={length}/depth={depth}/{index:04d}", probe="pcr_niah1", cell={"length": length}, group={"depth": depth},
            context=context, answer=answer,
            meta={"name": names[0], "value": value, "char_offset": offsets[0], "haystack_chars": kept,
                  "haystack_docs": [pool.doc_ids[j] for j in chosen], "n_tokens_length_measure": n})

    return draw_until_cut(draw, "pcr_niah1", cell, index)


def niahk_item(pool, count: Count, length: int, n_keys: int, index: int) -> g.TextItem:
    cell = (length, n_keys)
    content = rng_for("pcr_niahk", cell, index, 0)
    values: list[str] = []
    while len(values) < n_keys:  # distinct values, also in their first group
        v = g.random_uuid(content)
        if all(v[:8] != w[:8] for w in values):
            values.append(v)
    depths = sorted(float(x) for x in content.uniform(0.02, 0.98, size=n_keys))
    asked = int(content.integers(n_keys))
    names_pool = g.candidate_names(content, min(120, 3 * n_keys + 40))

    def draw(hay_rng: np.random.Generator) -> g.TextItem:
        context, answer, chosen, n, names, offsets, kept = _needle_item(pool, count, "pcr_niahk", length, depths, values, asked, names_pool, hay_rng)
        return g.TextItem(
            id=f"pcr_niahk/len={length}/n={n_keys}/{index:04d}", probe="pcr_niahk", cell={"length": length, "n_keys": n_keys},
            group={"asked_depth": g.depth_bin(depths[asked])}, context=context, answer=answer,
            meta={"names": names, "values": values, "depths": depths, "asked": asked, "char_offsets": offsets, "haystack_chars": kept,
                  "haystack_docs": [pool.doc_ids[j] for j in chosen], "n_tokens_length_measure": n})

    return draw_until_cut(draw, "pcr_niahk", cell, index)


# --------------------------------------------------------------------------------------------
# variable tracking


def _vt_names(content: np.random.Generator, n: int, avoid: str) -> list[str]:
    """n distinct random 5-letter upper-case names, none of which occurs in `avoid` (the haystack)."""
    out: list[str] = []
    while len(out) < n:
        name = "".join(chr(65 + int(x)) for x in content.integers(0, 26, size=5))
        if name not in out and name not in avoid and "VAR" not in name:
            out.append(name)
    return out


def _vt_numbers(content: np.random.Generator, n: int, avoid: str) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        v = str(int(content.integers(10000, 100000)))
        if v not in out and v not in avoid:
            out.append(v)
    return out


def vt_question(number: str) -> str:
    return f"{SEP}Question: Which variables are assigned the value {number}? Answer: The variables assigned the value {number} are:"


def vt_item(pool, count: Count, length: int, index: int, hops: int = VT_HOPS, chains: int = VT_CHAINS) -> g.TextItem:
    cell = (length,)

    def draw(hay_rng: np.random.Generator) -> g.TextItem:
        content = rng_for("pcr_vt", cell, index, 0)  # the same content stream in every draw (it reads the haystack only to avoid its words)
        text, chosen = gather(pool, hay_rng, length)
        text = re.sub(r"\bVAR\b", "Var", text)  # the statements' keyword occurs only in the planted statements
        size = hops + 1
        numbers = _vt_numbers(content, chains, text)
        names = _vt_names(content, chains * size, text)
        chain_names = [names[c * size:(c + 1) * size] for c in range(chains)]
        labels = [int(x) for x in content.permutation([c for c in range(chains) for _ in range(size)])]  # the chain of each statement, in text order
        depths = sorted(float(x) for x in content.uniform(0.02, 0.98, size=len(labels)))
        asked = int(content.integers(chains))
        seen = [0] * chains
        sentences, order = [], []
        for lab in labels:
            h = seen[lab]
            seen[lab] += 1
            nm = chain_names[lab][h]
            sentences.append(f"VAR {nm} = {numbers[lab]}." if h == 0 else f"VAR {nm} = VAR {chain_names[lab][h - 1]}.")
            order.append((lab, h))
        state: dict = {}

        def make(haystack: str) -> tuple[str, str]:
            body, offsets = g.insert_sentences(haystack, list(zip(depths, sentences)))
            state["offsets"] = offsets
            return body + vt_question(numbers[asked]), " " + " ".join(chain_names[asked])

        context, answer, n, kept = fit_cut(text, make, count, length)
        make(text[:kept].rstrip())
        return g.TextItem(
            id=f"pcr_vt/len={length}/{index:04d}", probe="pcr_vt", cell={"length": length}, group={}, context=context, answer=answer,
            meta={"chains": chain_names, "numbers": numbers, "asked": asked, "statements": sentences, "statement_chain_hop": order,
                  "depths": depths, "char_offsets": state["offsets"], "haystack_chars": kept,
                  "haystack_docs": [pool.doc_ids[j] for j in chosen], "n_tokens_length_measure": n})

    return draw_until_cut(draw, "pcr_vt", cell, index)


# --------------------------------------------------------------------------------------------
# structural checks (raise AssertionError)


def check_item(item: g.TextItem) -> None:
    """The structural rules every item must satisfy; the tokenizer-dependent ones (boundary, the answer's tokens in the context) are the
    builder's."""
    ctx, ans = item.context, item.answer
    assert ans.startswith(" ") and ans.strip(), item.id
    if item.probe in ("pcr_niah1", "pcr_niahk"):
        value = ans.strip()
        assert g.UUID_RE.fullmatch(value), item.id
        q = ctx.rindex(SEP + "What is the access code")
        query = ctx[q:]
        assert ctx.count(value) == 1 and value not in query, f"{item.id}: the answer must occur exactly once, in the haystack"
        n_keys = 1 if item.probe == "pcr_niah1" else item.cell["n_keys"]
        assert len(g.UUID_RE.findall(ctx)) == n_keys, item.id
        names = [item.meta["name"]] if item.probe == "pcr_niah1" else item.meta["names"]
        for nm in names:
            assert ctx[:q].count(nm) == 1, f"{item.id}: key {nm!r} not present exactly once"
        asked = names[0] if item.probe == "pcr_niah1" else names[item.meta["asked"]]
        assert ctx[:q].count(g.needle_sentence(asked, value)) == 1 and query.endswith(f"The access code for {asked} is"), item.id
        if item.probe == "pcr_niahk":
            assert len(set(item.meta["values"])) == len(item.meta["values"]) == n_keys, item.id
    elif item.probe == "pcr_vt":
        chains, numbers, asked = item.meta["chains"], item.meta["numbers"], item.meta["asked"]
        q = ctx.rindex(SEP + "Question: Which variables")
        body, query = ctx[:q], ctx[q:]
        assert query == vt_question(numbers[asked]), item.id
        assert ans == " " + " ".join(chains[asked]), item.id
        for c, number in enumerate(numbers):
            assert body.count(number) == 1, f"{item.id}: the number {number} must occur exactly once before the question"
        assert len(set(numbers)) == len(numbers) and len({n for ch in chains for n in ch}) == sum(len(ch) for ch in chains), item.id
        stmts = re.findall(r"VAR ([A-Z]{5}) = (VAR [A-Z]{5}|\d{5})\.", body)
        n_stmt = sum(len(ch) for ch in chains)
        assert len(stmts) == n_stmt and body.count("VAR") == 2 * n_stmt - len(chains), f"{item.id}: a VAR statement of another form"
        # follow the assignments: the asked chain's variables hold the asked number, in the answer's order, the other chain's do not
        env: dict[str, str] = {}
        for lhs, rhs in stmts:
            assert lhs not in env, f"{item.id}: {lhs} assigned twice"
            if rhs.startswith("VAR "):
                assert rhs[4:] in env, f"{item.id}: {rhs[4:]} read before it is assigned"
                env[lhs] = env[rhs[4:]]
            else:
                env[lhs] = rhs
        got = [lhs for lhs, rhs in stmts if env[lhs] == numbers[asked]]
        assert got == chains[asked], f"{item.id}: the variables holding {numbers[asked]} are {got}, the answer lists {chains[asked]}"
        assert query.count("VAR") == 0 and not any(nm in query for ch in chains for nm in ch), f"{item.id}: a name leaks into the question"
    else:
        raise ValueError(item.probe)
