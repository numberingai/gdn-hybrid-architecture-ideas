"""The multi-fact task (work): text-level generators, the fact bookkeeping and the metrics. Built by
tools/build_probes_multifact.py, scored by ced/eval/probes.score_sets (scripts/eval_long.py).

Every item is a held-out haystack of total length T (8k / 16k / 32k / 64k tokens, the uuid needle's haystacks:
ced/eval/probe_gen.py) with k = 2 / 4 / 8 fact sentences inserted at controlled depths, a query, and an answer that
needs ALL k facts. Two probes, each fact one sentence of one template:

- mf_chain, a chain of k pointers: "The supervisor of N0 is N1.", "The supervisor of N1 is N2.", ..., "The supervisor
  of N(k-1) is Nk." (k + 1 names absent from the haystack). Query: "Who is the supervisor of N0? The supervisor of N0
  is"; answer: " N1. The supervisor of N1 is N2. ... The supervisor of N(k-1) is Nk" (the chain written out: each
  step's key is the value the previous step retrieved, and the last value is the one reached by following all k
  pointers).
- mf_concat, k values in a given order: "The access code for Ni is Ci." (k names, k codes of 8 hex characters with
  distinct first characters). Query: "What are the access codes for N0, N1, ... and N(k-1), in that order? The access
  code for N0 is"; answer: " C0. The access code for N1 is C1. ... The access code for N(k-1) is C(k-1)" (the asked
  order is the order of the list in the question, which is not the order of the facts in the context).

The answer alternates FACT pieces (the retrieved values: " N1", " C0") and SCAFFOLD pieces (". The supervisor of N1
is"), recorded in the item's meta ("pieces": [text, is_fact]). Scoring is teacher-forced over the whole answer
(probes.score_sets); the metrics (`multifact_metrics`) read the fact tokens from the per-token record:

    exact        every token of every fact is the argmax given the gold prefix: all k items retrieved (the headline)
    fact_acc     the share of the k facts whose tokens are all right
    nats         -log p per FACT token (the scaffold is not counted)
    first_nats, first_ok     the first token of each fact (the one that decides which value is read; distinct across the
                 facts of an mf_concat item), mean over facts
    rest_nats    tokens 2..n of each fact, per token
    exact_all    every answer token, scaffold included, is the argmax: greedy decoding writes the whole answer
    hop_ok       per fact, in answer order: 1.0 if that fact is right (first fact against later ones; by depth)

Controlled depths (`layout_depths`), the item's group: "spread" (the k facts at the centres of k equal parts of the
whole haystack), "front" (all inside the first 5-30% of it, the far end from the query) and "back" (all inside
70-95%); layouts cycle with the item index, and which fact sits at which depth is a random permutation, so the order of
the chain (or of the question's list) is not the order of appearance.

Every generator is a pure function of (probe, length, k, index), the document pool and the length measure, with one
random stream for the content and one for the haystack (probe_gen.rng_for's scheme, another base seed).

Boundary convention (the split pattern of our byte-level BPE tokenizers, tools/data/train_tokenizer.py PATTERN): a
fact piece starts with a space after a letter and a scaffold piece with "." after a letter or a digit, so tokenizing
the pieces separately gives the tokens of the whole answer; the build tool checks it per item and tokenizer
(`fact_spans` raises otherwise).
"""

from __future__ import annotations

import numpy as np

from ced.eval import probe_gen as g
from ced.eval.probe_gen import TextItem

BASE_SEED = 20261010
PROBES = ("mf_chain", "mf_concat")
PROBE_CODE = {p: i for i, p in enumerate(PROBES)}
LENGTHS = g.LENGTHS  # 8k, 16k, 32k, 64k
KS = (2, 4, 8)
LAYOUTS = ("spread", "front", "back")
LAYOUT_RANGE = {"spread": (0.0, 1.0), "front": (0.05, 0.30), "back": (0.70, 0.95)}
HEX = "0123456789abcdef"
CODE_LEN = 8
CHAIN_PHRASE = "supervisor of"
CONCAT_PHRASE = "access code"


def rng_for(probe: str, length: int, k: int, index: int, stream: int) -> np.random.Generator:
    """Independent stream per (probe, length, k, item, purpose). stream 0: content; 1: haystack."""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, PROBE_CODE[probe], int(length), int(k), int(index), stream]))


def layout_of(index: int) -> str:
    return LAYOUTS[index % len(LAYOUTS)]


def layout_depths(layout: str, k: int) -> list[float]:
    """The k depth fractions of a layout, increasing: the centres of k equal parts of the layout's range of the
    haystack (spread: all of it; front: 5-30%; back: 70-95%)."""
    lo, hi = LAYOUT_RANGE[layout]
    return [lo + (hi - lo) * (j + 0.5) / k for j in range(k)]


def usable_haystack_doc(text: str) -> bool:
    """probe_gen's rule (no UUID-shaped string, no "access code") plus no "supervisor of": a haystack document must not
    hold a sentence of either template."""
    return g.usable_haystack_doc(text) and CHAIN_PHRASE not in text.lower()


def build_pool(docs: list[tuple[str, str]], count: g.Count) -> g.HaystackPool:
    """The haystack pool of (id, text) documents under this module's rule."""
    keep = [(i, t.strip()) for i, t in docs if t.strip() and usable_haystack_doc(t)]
    return g.HaystackPool(texts=[t for _, t in keep], doc_ids=[i for i, _ in keep],
                          counts=np.array([count(t) for _, t in keep], dtype=np.int64))


def chain_sentence(key: str, value: str) -> str:
    return f"The supervisor of {key} is {value}."


def chain_query(key: str) -> str:
    return f"{g.SEP}Who is the supervisor of {key}? The supervisor of {key} is"


def concat_sentence(key: str, value: str) -> str:
    return f"The access code for {key} is {value}."


def concat_query(names: list[str]) -> str:
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"{g.SEP}What are the access codes for {listed}, in that order? The access code for {names[0]} is"


def random_codes(rng: np.random.Generator, k: int) -> list[str]:
    """k codes of CODE_LEN hex characters with distinct first characters (k <= 16), so the first character of a code
    already tells which fact it is."""
    assert k <= len(HEX)
    firsts = rng.choice(len(HEX), size=k, replace=False)
    return [HEX[int(f)] + "".join(HEX[int(c)] for c in rng.integers(0, len(HEX), size=CODE_LEN - 1)) for f in firsts]


def answer_pieces(probe: str, keys: list[str], values: list[str]) -> list[list]:
    """[[text, is_fact], ...]: the answer as alternating fact and scaffold pieces (facts in answer order). The scaffold
    after fact j names the next step's key: the value just retrieved (mf_chain) or the next listed name (mf_concat)."""
    k = len(values)
    pieces = []
    for j in range(k):
        pieces.append([f" {values[j]}", True])
        if j < k - 1:
            pieces.append([f". The supervisor of {values[j]} is" if probe == "mf_chain"
                           else f". The access code for {keys[j + 1]} is", False])
    return pieces


def _item(probe: str, pool: g.HaystackPool, count_item, length: int, k: int, index: int) -> TextItem:
    assert probe in PROBES and k >= 2
    content = rng_for(probe, length, k, index, 0)
    hay_rng = rng_for(probe, length, k, index, 1)
    layout = layout_of(index)
    slots = layout_depths(layout, k)
    order = [int(x) for x in content.permutation(k)]  # fact j sits at depth slots[order[j]]
    depths = [slots[order[j]] for j in range(k)]
    by_depth = sorted(range(k), key=lambda j: depths[j])
    n_names = k + 1 if probe == "mf_chain" else k
    names_pool = g.candidate_names(content, min(120, 3 * n_names + 40))
    codes = random_codes(content, k) if probe == "mf_concat" else None
    state = {"names": names_pool[:n_names]}

    def parts(names):
        if probe == "mf_chain":
            keys, values = names[:k], names[1 : k + 1]
            return keys, values, [chain_sentence(a, b) for a, b in zip(keys, values)], chain_query(names[0])
        return names, codes, [concat_sentence(a, b) for a, b in zip(names, codes)], concat_query(names)

    def make(haystack: str) -> tuple[str, str]:
        keys, values, sentences, query = parts(state["names"])
        # inserted in depth order, so two facts that land on the same sentence start still appear in depth order
        text, offs = g.insert_sentences(haystack, [(depths[j], sentences[j]) for j in by_depth])
        state["offsets"] = [offs[by_depth.index(j)] for j in range(k)]
        return text + query, "".join(t for t, _ in answer_pieces(probe, keys, values))

    lo_t = (1 - g.LENGTH_TOL) * length
    for _ in range(20):
        _, _, chosen, _ = g.fit_haystack(pool, hay_rng, length, make, count_item)
        haystack = pool.join(chosen)
        if codes is not None and any(c in haystack for c in codes):
            codes = random_codes(content, k)  # a code that happens to occur in the haystack: draw again
            continue
        state["names"] = g.names_absent_from(names_pool, haystack, n_names)
        context, answer = make(haystack)
        n = count_item(context, answer)
        if lo_t <= n <= length:
            keys, values, sentences, _ = parts(state["names"])
            return TextItem(
                id=f"{probe}/len={length}/k={k}/{index:04d}", probe=probe, cell={"length": length, "k": k},
                group={"layout": layout}, context=context, answer=answer,
                meta={"keys": list(keys), "values": list(values), "names": list(state["names"]), "depths": depths,
                      "depth_order": order, "layout": layout, "pieces": answer_pieces(probe, keys, values),
                      "sentences": sentences, "char_offsets": state["offsets"], "haystack_chars": len(haystack),
                      "haystack_docs": [pool.doc_ids[j] for j in chosen], "n_tokens_length_measure": n})
    raise RuntimeError("could not fit a multi-fact item")


def chain_item(pool: g.HaystackPool, count_item, length: int, k: int, index: int) -> TextItem:
    return _item("mf_chain", pool, count_item, length, k, index)


def concat_item(pool: g.HaystackPool, count_item, length: int, k: int, index: int) -> TextItem:
    return _item("mf_concat", pool, count_item, length, k, index)


def make_item(probe: str, pool: g.HaystackPool, count_item, length: int, k: int, index: int) -> TextItem:
    return _item(probe, pool, count_item, length, k, index)


QUESTION_OPENING = {"mf_chain": "Who is the supervisor of", "mf_concat": "What are the access codes for"}


def query_start(item: TextItem) -> int:
    """Character offset in the context where the query begins."""
    return item.context.rindex(g.SEP + QUESTION_OPENING[item.probe])


def question_text(probe: str, context: str) -> str:
    """The question of an item, as the model reads it right before the answer: the context from the question's first
    word ("Who is the supervisor of ..." / "What are the access codes for ...") to its end, the answer's lead-in
    included ("... The supervisor of N0 is"). The separator before it (probe_gen.SEP) is not part of it. This is the
    item's "last turn" for the replay window of the long-context suite (scripts/eval_long.py --replay-window turn)."""
    return context[context.rindex(g.SEP + QUESTION_OPENING[probe]) + len(g.SEP):]


def question_token_count(context_ids, question_ids, item_id: str = "") -> int:
    """The number of context tokens the question covers: len(question_ids), the question tokenized on its own, after
    checking that those ARE the context's last tokens (so the count is exact for the stored item). They are under
    our tokenizers' split pattern (tools/data/train_tokenizer.py PATTERN): the question follows a blank line, which
    ends a pre-token, and starts with a letter. AssertionError otherwise."""
    q = [int(x) for x in question_ids]
    assert q and [int(x) for x in context_ids[len(context_ids) - len(q):]] == q, \
        f"{item_id}: the question tokenized alone is not the context's last {len(q)} tokens"
    return len(q)


def attach_question_tokens(es, questions: dict, encode_ids) -> int:
    """Give every item record of an EncodedSet its "question_tokens" (for a set built before the builder stored it).
    questions: {item id: question_text}; encode_ids: str -> token ids under the set's tokenizer. Every count is
    checked against the item's stored tokens (`question_token_count`). Returns the number of items."""
    for i, rec in enumerate(es.items):
        ids = es.item_ids(i).tolist()
        ctx = ids[: len(ids) - int(es.n_answer[i])]
        rec["question_tokens"] = question_token_count(ctx, encode_ids(questions[rec["id"]]), rec["id"])
    return len(es.items)


def check_item(item: TextItem) -> None:
    """Structural rules every item must satisfy (raises AssertionError)."""
    ctx, ans, meta = item.context, item.answer, item.meta
    k = item.cell["k"]
    keys, values, sentences = meta["keys"], meta["values"], meta["sentences"]
    q = query_start(item)
    body, query = ctx[:q], ctx[q:]
    assert len(keys) == len(values) == len(sentences) == len(meta["depths"]) == k, item.id
    assert ans == "".join(t for t, _ in meta["pieces"]), f"{item.id}: the answer is not its pieces"
    facts = [t for t, f in meta["pieces"] if f]
    assert facts == [f" {v}" for v in values] and len(meta["pieces"]) == 2 * k - 1, item.id
    assert [f for _, f in meta["pieces"]] == [j % 2 == 0 for j in range(2 * k - 1)], f"{item.id}: pieces do not alternate"
    for s in sentences:
        assert body.count(s) == 1, f"{item.id}: fact sentence not present exactly once"
        assert s not in query, item.id
    assert len(set(values)) == k and len(set(keys)) == k, item.id
    assert sorted(meta["depth_order"]) == list(range(k)), item.id
    assert meta["depths"] == [layout_depths(meta["layout"], k)[o] for o in meta["depth_order"]], item.id
    # the facts are in the context in the order of their depths
    offs = meta["char_offsets"]
    assert [body.find(s) for s in sentences] == offs, f"{item.id}: recorded offsets"
    by_depth = sorted(range(k), key=lambda j: meta["depths"][j])
    assert all(offs[a] < offs[b] for a, b in zip(by_depth, by_depth[1:])), f"{item.id}: facts out of depth order"
    if item.probe == "mf_chain":
        names = meta["names"]
        assert keys == names[:k] and values == names[1 : k + 1] and len(set(names)) == k + 1, item.id
        assert body.count(names[0]) == 1 and body.count(names[k]) == 1, item.id
        for nm in names[1:k]:
            assert body.count(nm) == 2, f"{item.id}: an inner name of the chain must occur as a value and as a key"
        assert query.endswith(f"The supervisor of {names[0]} is") and query.count(names[0]) == 2, item.id
        for nm in names[1:]:
            assert nm not in query, f"{item.id}: a value leaks into the query"
        for j in range(k - 1):
            assert meta["pieces"][2 * j + 1][0] == f". The supervisor of {values[j]} is", item.id
    else:
        assert keys == meta["names"], item.id
        for nm, c in zip(keys, values):
            assert body.count(nm) == 1 and ctx.count(c) == 1, f"{item.id}: key or code not present exactly once"
            assert len(c) == CODE_LEN and set(c) <= set(HEX), item.id
        assert len({c[0] for c in values}) == k, f"{item.id}: codes share a first character"
        assert query.endswith(f"The access code for {keys[0]} is"), item.id
        at = [query.index(nm) for nm in keys]
        assert at == sorted(at), f"{item.id}: the question lists the names in another order"
        for j in range(k - 1):
            assert meta["pieces"][2 * j + 1][0] == f". The access code for {keys[j + 1]} is", item.id


# --------------------------------------------------------------------------------------------
# tokens: which answer tokens are fact tokens
# --------------------------------------------------------------------------------------------


def fact_spans(item: TextItem, encode) -> dict:
    """The fact tokens of an item's answer under `encode` (str -> TokenizedDoc: ids, byte_lens): {"fact_spans": [[a, b)
    per fact, in answer tokens], "n_answer"}. The pieces are tokenized one by one; their concatenation must be the
    tokens of the whole answer (the boundary convention), else AssertionError. A fact's span starts at its first token
    that covers a non-whitespace character: the space before a value is a token of its own when the value starts with
    a digit (digits are always split alone), and that token does not say which value follows, so it counts as
    scaffold."""
    from ced.eval.probes import _first_informative

    whole = list(encode(item.answer).ids)
    spans, ids = [], []
    for text, is_fact in item.meta["pieces"]:
        tok = encode(text)
        t = list(tok.ids)
        if is_fact:
            spans.append([len(ids) + _first_informative(text, tok), len(ids) + len(t)])
            assert spans[-1][0] < spans[-1][1], f"{item.id}: an empty fact"
        ids += t
    assert ids == whole, f"{item.id}: the answer's pieces do not tokenize like the whole answer"
    return {"fact_spans": spans, "n_answer": len(whole)}


def encode_items(items: list[TextItem], encode, probe: str | None = None):
    """probes.encode_items plus, per item, what the metrics need: fact_spans, the facts' depths and the layout
    (EncodedSet.items[i]: "fact_spans", "fact_depths", "layout", "k"), and "question_tokens", the number of context
    tokens the question covers (`question_text`, `question_token_count`)."""
    from ced.eval import probes

    es = probes.encode_items(items, encode, probe=probe)
    for i, it in enumerate(items):
        rec = fact_spans(it, encode)
        assert rec["n_answer"] == int(es.n_answer[i]), it.id
        ids = es.item_ids(i).tolist()
        q = question_token_count(ids[: len(ids) - rec["n_answer"]], encode(question_text(it.probe, it.context)).ids, it.id)
        es.items[i].update(fact_spans=rec["fact_spans"], fact_depths=[round(float(d), 4) for d in it.meta["depths"]],
                           layout=it.meta["layout"], k=it.cell["k"], question_tokens=q)
    return es


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------

METRICS = ("exact", "fact_acc", "nats", "first_nats", "first_ok", "rest_nats", "exact_all", "nats_all")


def multifact_metrics(result: dict, meta: dict) -> dict:
    """The item's metrics (module docstring) from a score_sets result (tok_logp, tok_correct over the whole answer) and
    the set's item record (fact_spans). hop_ok: per fact in answer order."""
    lp, ok = result["tok_logp"], result["tok_correct"]
    spans = meta["fact_spans"]
    assert len(lp) == len(ok) == result["n_answer"] and spans and spans[-1][1] <= len(lp), result.get("id")
    hop_ok = [float("0" not in ok[a:b]) for a, b in spans]
    n_fact = sum(b - a for a, b in spans)
    first = [-lp[a] for a, _ in spans]
    rest = [-x for a, b in spans for x in lp[a + 1 : b]]
    out = {"exact": float(all(hop_ok)), "fact_acc": sum(hop_ok) / len(spans),
           "nats": -sum(x for a, b in spans for x in lp[a:b]) / n_fact,
           "first_nats": sum(first) / len(first), "first_ok": sum(ok[a] == "1" for a, _ in spans) / len(spans),
           "exact_all": float("0" not in ok), "nats_all": -sum(lp) / len(lp), "hop_ok": hop_ok}
    if rest:
        out["rest_nats"] = sum(rest) / len(rest)
    return out


def summarize(results: list[dict], metas: list[dict]) -> dict:
    """{probe: {"k=<k>": {"<length>": cell}}}; cell: n, each metric's item mean and item-level SE (<metric>_se),
    "layouts": {layout: the same}, "hops": the mean of hop_ok per answer position (fact 1, 2, ...), "depth": the mean of
    hop_ok per depth fifth of the haystack (0.1, 0.3, ..., 0.9; None where no fact sits there)."""
    from collections import defaultdict

    from ced.eval.probes import mean_se

    tree: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r, m in zip(results, metas):
        assert r["id"] == m["id"], "results and item records are not aligned"
        tree[r["probe"]][f"k={r['cell']['k']}"][str(r["cell"]["length"])].append((multifact_metrics(r, m), m))

    def agg(pairs):
        out = {"n": len(pairs)}
        for key in METRICS:
            out[key], out[key + "_se"] = mean_se(x[key] for x, _ in pairs if key in x)
        return out

    def cell(pairs):
        out = agg(pairs)
        by_layout = defaultdict(list)
        for x, m in pairs:
            by_layout[m.get("layout", "")].append((x, m))
        out["layouts"] = {name: agg(v) for name, v in sorted(by_layout.items())}
        k = max(len(x["hop_ok"]) for x, _ in pairs)
        out["hops"] = [float(np.mean([x["hop_ok"][j] for x, _ in pairs if len(x["hop_ok"]) > j])) for j in range(k)]
        bins = defaultdict(list)
        for x, m in pairs:
            for hit, d in zip(x["hop_ok"], m.get("fact_depths") or []):
                bins[g.depth_bin(d)].append(hit)
        out["depth"] = {str(d): (float(np.mean(bins[d])) if bins[d] else None) for d in g.NEEDLE_DEPTHS}
        return out

    return {probe: {kk: {L: cell(pairs) for L, pairs in sorted(cells.items(), key=lambda kv: int(kv[0]))}
                    for kk, cells in sorted(ks.items(), key=lambda kv: int(kv[0].split("=")[1]))}
            for probe, ks in tree.items()}
