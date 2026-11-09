"""Latent two-hop probe: text-level generator, item checks, the per-item scores and the
paired summary. Scored by ced/eval/probes.score_sets (scripts/eval_probes.py).

What it measures (architecture spec v5, section 5 "Why pass 2": "Latent two-hop lookup within one token: B can only be
located after A has been read"; section 12 probes: "Latent two-hop without chain of thought"): the answer is found only
by reading one fact and using what it says as the key of a second fact, and the model must give the answer's FIRST
token at once. No token is written between the question and the answer, so the first hop is never written out (the
multi-fact chain, probe_gen_multifact.py, writes it out and is another task).

The item. A held-out haystack (the uuid needle's: whole web and general_web eval documents) with D + 1 chains planted
in it, each two sentences:

    fact A, the link:   "The supervisor of <head> is <bridge>."
    fact B, the value:  "The access code for <bridge> is <code>."

Chain 0 is asked; the other D are distractor chains of the same form at other places. From one base item (one
haystack, one set of chains, one layout) come five scored items, the `variant`:

    twohop    "What is the access code for the supervisor of <head>? The access code for the supervisor of <head> is"
              answer " <code>": both facts are needed; the bridge is never named in the question or the answer.
    onehop    "What is the access code for <bridge>? The access code for <bridge> is", answer " <code>": the same B asked
              directly by its key. The control: what the same answer costs when no hop is needed.
    link      "Who is the supervisor of <head>? The supervisor of <head> is", answer " <bridge>": fact A alone (reported;
              it says whether the first hop can be read at all).
    shuffled  the two-hop question on a context whose asked link fact names a stranger ("The supervisor of <head> is
              <stranger>.", a name with no access code): the link points nowhere. Scored on the same " <code>". No
              strategy can find the code through the link there, so this is the ceiling of guessing: position, salience,
              and (with adjacent facts) the code that stands next to the head. NOT matched in one respect: the asked
              bridge is then named once and every other bridge twice, so a rule that never reads the asked head ("a
              code whose key some link names") scores chance on twohop and 0 here.
    swapped   the two-hop question on a context whose links are moved one chain on: the link of chain c names the
              bridge of chain c + 1 (the last one names chain 0's). Scored on the same " <code>". Every fact stands
              where it stood, every head has one link, every bridge is still named twice, and no chain differs from
              the others in form: whatever does not read the asked link scores here exactly as on twohop, and a model
              that follows the link gives another chain's code. So twohop - swapped is the share of answers that
              follow the link. It is the matched control.

Difficulty settings (the cell): length (2,048 or 8,192 tokens by default: inside the 8k training row; longer lengths
are a build option); gap between the two facts of a chain: "adjacent" (consecutive sentences), "near" (about
NEAR_TOKENS tokens apart) or "far" (half the haystack apart); distractors D (1 or 3: a guess among the codes present is
right 1 / (D + 1) of the time); value: "word" (the code is one common word, ONE token: the hop has to happen inside
that token) or "uuid" (a uuid4 whose first character is a letter: its first token already says which code it is, the
other 20 to 35 tokens are a copy). The group is the order of the two facts in the text: "AB" (the link first) or "BA".
Every chain of an item has the item's gap and order, so the chains differ only in their names and codes. The order and
the asked chain's rank among the D + 1 chains (in text order) cycle with the item index, so "the first code" or "the
last code" is right exactly 1 / (D + 1) of the time in a cell whose size is a multiple of 2 (D + 1).

The first answer token decides: the D + 1 codes of an item start with different tokens (different words; uuids with
different first letters, and a letter-first uuid starts with a space-plus-letters token), and it is answer token 0 (no
separate space token before it, which would give the model one more position to work at). The build tool checks both
per item and tokenizer.

Scores per scored item (`item_scores`, from the scorer's per-token record): first_ok (the first answer token is the
argmax), first_nats (its -log p), exact (every answer token is the argmax), nats (per answer token), rest_nats (tokens
2..n, uuid values). Summary per setting (`summarize`): each variant's level with item SEs, the chance level, and two
differences paired per base item with their SE: hop_cost = twohop - onehop, link_use = twohop - shuffled, link_follow
= twohop - swapped.

Every generator is a pure function of (length, gap, distractors, value, index), the document pool and the length
measure, with one random stream for the content and one for the haystack (probe_gen.rng_for's scheme, another base
seed). Boundary convention as the other probes: the answer starts with a space after a letter.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ced.eval import probe_gen as g
from ced.eval.probe_gen import TextItem

BASE_SEED = 20261011
PROBE = "twohop"
LENGTHS = (2048, 8192)  # the default cells: inside the 8k training row
GAPS = ("adjacent", "near", "far")
NEAR_TOKENS = 256  # "near": the second fact about this many tokens after the first
FAR_FRACTION = 0.5  # "far": the second fact this share of the haystack after the first
ORDERS = ("AB", "BA")  # A = the link fact, B = the value fact; which comes first in the text
DISTRACTORS = (1, 3)
VALUES = ("word", "uuid")
VARIANTS = ("twohop", "onehop", "link", "shuffled", "swapped")
CODE_VARIANTS = ("twohop", "onehop", "shuffled", "swapped")  # the variants whose answer is the code
HOP_VARIANTS = ("twohop", "shuffled", "swapped")  # the variants that ask the two-hop question
SLOT_RANGE = (0.05, 0.95)  # the facts sit inside this part of the haystack (fractions of its characters)
FAR_FIRST = (0.05, 0.45)  # "far": the first facts' part; the second facts follow FAR_FRACTION later
SHORT_LENGTH = 4096  # below this an item may be up to SHORT_TOL short of its nominal length (whole documents only)
SHORT_TOL = 0.10
LINK_PHRASE = "supervisor of"
VALUE_PHRASE = "access code"
HEX_LETTERS = "abcdef"
# Code words: each " <word>" is ONE token under both tokenizers the sets are built for (t64k, t128k; checked
# against the tokenizer files and again by the build for every item), five letters or more.
CODE_WORDS = """amber anchor bamboo banner barley basalt beacon blossom brass breeze bronze butter cabin cactus camel
candle carbon castle cedar cherry citrus comet copper coral cotton crane crater cricket crystal feather fossil ginger
glacier granite grape gravel harbor honey horizon lemon linen magnet maple marble mercury mirror monsoon nectar nickel
olive oyster pearl pebble pepper planet pollen prairie quartz rabbit ribbon rocket saddle salmon shadow silver summit
sunset thunder tiger timber turtle walnut zebra""".split()
METRICS = ("first_ok", "first_nats", "exact", "nats", "rest_nats")
PAIRS = {"hop_cost": ("twohop", "onehop"), "link_use": ("twohop", "shuffled"), "link_follow": ("twohop", "swapped")}


class NoRoom(Exception):
    """The haystack has too few sentence starts where the layout needs them; other documents are drawn."""


def rng_for(length: int, gap: str, distractors: int, value: str, index: int, stream: int) -> np.random.Generator:
    """Independent stream per (cell, item, purpose). stream 0: content; 1: haystack."""
    words = [BASE_SEED, int(length), GAPS.index(gap), int(distractors), VALUES.index(value), int(index), stream]
    return np.random.default_rng(np.random.SeedSequence(words))


def length_tol(length: int) -> float:
    """An item's total tokens lie in [(1 - tol) * length, length]: probe_gen's 3%, or SHORT_TOL below SHORT_LENGTH
    (a short haystack of whole documents rarely lands inside 3%)."""
    return g.LENGTH_TOL if length >= SHORT_LENGTH else SHORT_TOL


def order_of(index: int) -> str:
    return ORDERS[index % len(ORDERS)]


def rank_of(index: int, distractors: int) -> int:
    """The asked chain's place among the D + 1 chains in text order; cycles with the item index, inside each order."""
    return (index // len(ORDERS)) % (distractors + 1)


def usable_haystack_doc(text: str) -> bool:
    """probe_gen's rule (no UUID-shaped string, no "access code") and no "supervisor of": a haystack document must
    not hold a sentence of either template."""
    return g.usable_haystack_doc(text) and LINK_PHRASE not in text.lower()


def build_pool(docs: list[tuple[str, str]], count: g.Count) -> g.HaystackPool:
    """The haystack pool of (id, text) documents under this module's rule."""
    keep = [(i, t.strip()) for i, t in docs if t.strip() and usable_haystack_doc(t)]
    return g.HaystackPool(texts=[t for _, t in keep], doc_ids=[i for i, _ in keep],
                          counts=np.array([count(t) for _, t in keep], dtype=np.int64))


def link_sentence(head: str, bridge: str) -> str:
    return f"The supervisor of {head} is {bridge}."


def value_sentence(bridge: str, code: str) -> str:
    return f"The access code for {bridge} is {code}."


def swapped_links(heads: list[str], bridges: list[str]) -> list[str]:
    """The link sentences of the swapped variant, by chain: chain c's head with the bridge of chain c + 1 (cyclic)."""
    n = len(heads)
    return [link_sentence(heads[c], bridges[(c + 1) % n]) for c in range(n)]


def twohop_query(head: str) -> str:
    return f"{g.SEP}What is the access code for the supervisor of {head}? The access code for the supervisor of {head} is"


def onehop_query(bridge: str) -> str:
    return f"{g.SEP}What is the access code for {bridge}? The access code for {bridge} is"


def link_query(head: str) -> str:
    return f"{g.SEP}Who is the supervisor of {head}? The supervisor of {head} is"


def word_absent(word: str, text: str) -> bool:
    """`word` occurs nowhere in `text`, in any case and not inside another word either ("amber" is not absent from a
    text with "chamber" or "Amber"). Stricter than a token needs, so that a code is in its item exactly once as plain
    text too, and " <word>" is never a token of the haystack."""
    return word.lower() not in text.lower()


def letter_uuid(rng: np.random.Generator, letter: str) -> str:
    """A uuid4 whose first character is `letter` (a-f): under the two tokenizers its first token is a space plus letters,
    so the answer's first token is the one that tells the codes apart."""
    assert letter in HEX_LETTERS
    return letter + g.random_uuid(rng)[1:]


def _layout(haystack: str, gap: str, n_chains: int, gap_chars: float) -> list[tuple[int, int]]:
    """(first fact, second fact) insertion offsets of every chain slot, slots in text order. Offsets are sentence
    starts of the haystack (or its start). Different chains never share an offset, so a sentence of haystack text
    separates their facts; the two facts of one chain share theirs only when the gap is "adjacent". Raises NoRoom when
    the haystack has too few sentence starts."""
    cand = np.unique(np.concatenate([[0], g.sentence_starts(haystack)])).astype(np.int64)
    n = len(haystack)
    lo, hi = FAR_FIRST if gap == "far" else (SLOT_RANGE[0], SLOT_RANGE[1] - gap_chars / max(1, n))
    if hi <= lo:
        raise NoRoom("the gap does not fit the haystack")
    used: set[int] = set()
    firsts, last = [], -1
    for j in range(n_chains):  # first facts: the centres of n_chains equal parts of [lo, hi]
        i = max(int(np.argmin(np.abs(cand - (lo + (hi - lo) * (j + 0.5) / n_chains) * n))), last + 1)
        if i >= len(cand):
            raise NoRoom("too few sentence starts for the first facts")
        used.add(i)
        firsts.append(i)
        last = i
    out = []
    for i in firsts:
        if gap == "adjacent":
            k = i
        else:
            k = max(int(np.argmin(np.abs(cand - (cand[i] + gap_chars)))), i + 1)
            while k in used:
                k += 1
            if k >= len(cand):
                raise NoRoom("too few sentence starts for the second facts")
            used.add(k)
        out.append((int(cand[i]), int(cand[k])))
    return out


def _assemble(haystack: str, slots: list[tuple[int, int]], chain_at: list[int], order: str, links: list[str],
              values: list[str]) -> tuple[str, list[int], list[int]]:
    """The haystack with every chain's two facts inserted: (text, link offsets, value offsets), offsets by chain.
    chain_at[j]: the chain in slot j. Order "AB": a chain's link is its first fact; "BA": its value fact is."""
    points, sentences = [], []
    for (p_first, p_second), c in zip(slots, chain_at):
        a, b = (links[c], values[c]) if order == "AB" else (values[c], links[c])
        points += [p_first, p_second]
        sentences += [a, b]  # equal offsets keep the input order: the first fact, then the second
    text, offs = g.insert_at(haystack, points, sentences)
    link_off, value_off = [0] * len(chain_at), [0] * len(chain_at)
    for j, c in enumerate(chain_at):
        first, second = offs[2 * j], offs[2 * j + 1]
        link_off[c], value_off[c] = (first, second) if order == "AB" else (second, first)
    return text, link_off, value_off


def make_items(pool: g.HaystackPool, count_item, length: int, gap: str, distractors: int, value: str, index: int,
               variants=VARIANTS, tries: int = 60) -> list[TextItem]:
    """The scored items of base item `index` of a cell, one per variant (module docstring), in the order of
    `variants`. count_item(context, answer): the length measure. Every variant has at most `length` tokens and the
    longest at least (1 - length_tol(length)) * length."""
    assert gap in GAPS and value in VALUES and distractors >= 1 and all(v in VARIANTS for v in variants), (gap, value, variants)
    n_chains = distractors + 1
    content, hay_rng = rng_for(length, gap, distractors, value, index, 0), rng_for(length, gap, distractors, value, index, 1)
    order, rank = order_of(index), rank_of(index, distractors)
    n_names = 2 * n_chains + 1  # a head and a bridge per chain, and the stranger of the shuffled variant
    names_pool = g.candidate_names(content, min(120, 3 * n_names + 40))
    if value == "word":
        assert n_chains <= len(CODE_WORDS)
        code_pool = [CODE_WORDS[i] for i in content.permutation(len(CODE_WORDS))]
    else:
        assert n_chains <= len(HEX_LETTERS), "uuid values: at most 6 chains (distinct first letters)"
        code_pool = [letter_uuid(content, HEX_LETTERS[int(i)]) for i in content.choice(len(HEX_LETTERS), size=n_chains, replace=False)]
    # which chain sits in which slot (slots in text order): the asked chain (0) in slot `rank`, the others at random
    others = [int(c) + 1 for c in content.permutation(distractors)]
    chain_at = others[:rank] + [0] + others[rank:]
    base = f"{PROBE}/len={length}/gap={gap}/d={distractors}/value={value}/{index:04d}"

    def build(chosen: list[int]) -> list[TextItem]:
        haystack = pool.join(chosen)
        names = g.names_absent_from(names_pool, haystack, n_names)
        heads, bridges, stranger = names[:n_chains], names[n_chains : 2 * n_chains], names[-1]
        # word codes: the first of the item's shuffled list that the haystack does not hold; uuid codes: as drawn
        codes = [c for c in code_pool if word_absent(c, haystack)][:n_chains]
        if len(codes) < n_chains:
            raise NoRoom("the codes occur in the haystack")
        hay_tokens = max(1, count_item(haystack, ""))
        request = {"adjacent": 0, "near": NEAR_TOKENS, "far": int(round(FAR_FRACTION * hay_tokens))}[gap]
        slots = _layout(haystack, gap, n_chains, FAR_FRACTION * len(haystack) if gap == "far" else request * len(haystack) / hay_tokens)
        links = [link_sentence(h, b) for h, b in zip(heads, bridges)]
        values = [value_sentence(b, c) for b, c in zip(bridges, codes)]
        body, link_off, value_off = _assemble(haystack, slots, chain_at, order, links, values)
        shuffled_links = [link_sentence(heads[0], stranger)] + links[1:]
        s_body, s_link_off, s_value_off = _assemble(haystack, slots, chain_at, order, shuffled_links, values)
        moved_links = swapped_links(heads, bridges)
        w_body, w_link_off, w_value_off = _assemble(haystack, slots, chain_at, order, moved_links, values)
        common = {"base": base, "order": order, "rank": rank, "gap": gap, "distractors": distractors, "value": value,
                  "heads": heads, "bridges": bridges, "codes": codes, "stranger": stranger, "chain_at": chain_at,
                  "value_sentences": values, "gap_request_tokens": request, "haystack_chars": len(haystack),
                  "haystack_docs": [pool.doc_ids[j] for j in chosen]}
        made = {"twohop": (body, twohop_query(heads[0]), f" {codes[0]}", links, link_off, value_off),
                "onehop": (body, onehop_query(bridges[0]), f" {codes[0]}", links, link_off, value_off),
                "link": (body, link_query(heads[0]), f" {bridges[0]}", links, link_off, value_off),
                "shuffled": (s_body, twohop_query(heads[0]), f" {codes[0]}", shuffled_links, s_link_off, s_value_off),
                "swapped": (w_body, twohop_query(heads[0]), f" {codes[0]}", moved_links, w_link_off, w_value_off)}
        items = []
        for v in variants:
            text, query, answer, link_sentences, lo, vo = made[v]
            items.append(TextItem(
                id=f"{base}/{v}", probe=PROBE,
                cell={"variant": v, "length": length, "gap": gap, "distractors": distractors, "value": value},
                group={"order": order}, context=text + query, answer=answer,
                meta={**common, "variant": v, "link_sentences": link_sentences, "link_offsets": lo, "value_offsets": vo,
                      "n_tokens_length_measure": count_item(text + query, answer)}))
        return items

    # Choose whole documents so that the finished items fit: fit_haystack's search (probe_gen), on the longest variant
    tol = length_tol(length)
    lo_t, hi_t = int(np.ceil((1 - tol) * length)), length
    chain_text = " ".join(link_sentence(a, b) + " " + value_sentence(b, c) for a, b, c in
                          zip(names_pool[:n_chains], names_pool[n_chains:], code_pool[:n_chains]))
    extra = count_item(chain_text + twohop_query(names_pool[0]), f" {code_pool[0]}") + 2
    margin = max(4, int(0.004 * length))
    lo, hi = lo_t - extra + margin, hi_t - extra - margin
    for _ in range(tries):
        chosen = pool.pick(hay_rng, lo, hi)
        if chosen is None:
            lo, hi = lo - margin, hi + margin
            continue
        try:
            items = build(chosen)
        except (NoRoom, RuntimeError):  # too few sentence starts, codes or names present: other documents
            continue
        n = max(it.meta["n_tokens_length_measure"] for it in items)
        if lo_t <= n <= hi_t:
            return items
        if n > hi_t:
            hi -= n - hi_t + margin
            lo = min(lo, hi - margin)
        else:
            lo += lo_t - n + margin
            hi = max(hi, lo + margin)
    raise RuntimeError(f"{base}: could not fit a haystack into [{lo_t}, {hi_t}] tokens")


def query_start(item: TextItem) -> int:
    """Character offset in the context where the question begins."""
    opening = "Who is the supervisor of" if item.meta["variant"] == "link" else "What is the access code for"
    return item.context.rindex(g.SEP + opening)


def check_item(item: TextItem) -> None:
    """Structural rules every item must satisfy (raises AssertionError)."""
    m, ctx, ans = item.meta, item.context, item.answer
    v, order, D = m["variant"], m["order"], m["distractors"]
    heads, bridges, codes = m["heads"], m["bridges"], m["codes"]
    n = D + 1
    assert item.probe == PROBE and item.id == f"{m['base']}/{v}" and v in VARIANTS, item.id
    assert item.cell == {"variant": v, "length": item.cell["length"], "gap": m["gap"], "distractors": D, "value": m["value"]}, item.id
    assert item.group == {"order": order} and order in ORDERS and m["gap"] in GAPS, item.id
    assert len(heads) == len(bridges) == len(codes) == n and sorted(m["chain_at"]) == list(range(n)), item.id
    names = heads + bridges + [m["stranger"]]
    assert len({x.split()[0] for x in names}) == len(names) == len({x.split()[1] for x in names}), f"{item.id}: names share a part"
    assert len(set(codes)) == n, f"{item.id}: codes repeat"
    if m["value"] == "word":
        assert all(c in CODE_WORDS for c in codes), item.id
    else:
        assert all(g.UUID_RE.fullmatch(c) for c in codes) and len({c[0] for c in codes}) == n, f"{item.id}: uuid first characters"
        assert all(c[0] in HEX_LETTERS for c in codes), f"{item.id}: a uuid code starts with a digit"
    q = query_start(item)
    body, query = ctx[:q], ctx[q:]
    links, values = m["link_sentences"], m["value_sentences"]
    assert values == [value_sentence(b, c) for b, c in zip(bridges, codes)], item.id
    if v == "swapped":  # every link names the next chain's bridge: none its own, each bridge by exactly one link
        assert links == swapped_links(heads, bridges) and all(bridges[c] not in links[c] for c in range(n)), item.id
    else:
        told = m["stranger"] if v == "shuffled" else bridges[0]  # whom the asked link names
        assert links == [link_sentence(heads[0], told)] + [link_sentence(h, b) for h, b in zip(heads[1:], bridges[1:])], item.id
    for s in links + values:
        assert body.count(s) == 1 and s not in query, f"{item.id}: a fact not exactly once in the context"
    assert [body.find(s) for s in links] == m["link_offsets"] and [body.find(s) for s in values] == m["value_offsets"], f"{item.id}: offsets"
    assert body.count(LINK_PHRASE) == n == body.count(VALUE_PHRASE), f"{item.id}: another sentence of a template"
    for c in codes:
        assert ctx.count(c) == 1 and c not in query, f"{item.id}: a code not exactly once, or in the question"
    for h in heads:
        assert body.count(h) == 1, f"{item.id}: a head not exactly once"
    for c, b in enumerate(bridges):
        assert body.count(b) == (1 if (v == "shuffled" and c == 0) else 2), f"{item.id}: a bridge is a link's value and a code's key"
    assert body.count(m["stranger"]) == (1 if v == "shuffled" else 0), f"{item.id}: the stranger"
    assert f"{VALUE_PHRASE} for {m['stranger']}" not in ctx, f"{item.id}: the stranger has an access code"
    # the layout: each chain's facts in the item's order, at the item's gap; the chains in slot order
    first = [m["link_offsets"][c] if order == "AB" else m["value_offsets"][c] for c in range(n)]
    second = [m["value_offsets"][c] if order == "AB" else m["link_offsets"][c] for c in range(n)]
    first_s = [links[c] if order == "AB" else values[c] for c in range(n)]
    second_s = [values[c] if order == "AB" else links[c] for c in range(n)]
    assert all(a < b for a, b in zip(first, second)), f"{item.id}: a chain's facts out of order"
    assert [m["chain_at"][j] for j in np.argsort([first[c] for c in m["chain_at"]])] == m["chain_at"], f"{item.id}: slots out of order"
    assert m["chain_at"][m["rank"]] == 0, f"{item.id}: the asked chain is not in its slot"
    for c in range(n):
        between = body[first[c] + len(first_s[c]) : second[c]]
        if m["gap"] == "adjacent":
            assert between in (" ", "\n"), f"{item.id}: adjacent facts with text between them"
        else:
            assert len(between.strip()) > 0, f"{item.id}: no text between a chain's facts"
    # facts of different chains never stand together: only the two facts of one chain do, and only when adjacent
    placed = sorted([(o, s, c) for c, (o, s) in enumerate(zip(m["link_offsets"], links))]
                    + [(o, s, c) for c, (o, s) in enumerate(zip(m["value_offsets"], values))])
    for (o1, s1, c1), (o2, _, c2) in zip(placed, placed[1:]):
        if body[o1 + len(s1) : o2] in (" ", "\n"):
            assert m["gap"] == "adjacent" and c1 == c2, f"{item.id}: two facts stand together"
    # the question and the answer
    if v in HOP_VARIANTS:
        assert query == twohop_query(heads[0]) and ans == f" {codes[0]}", item.id
        assert all(b not in query for b in bridges) and m["stranger"] not in query, f"{item.id}: the bridge is named in the question"
    elif v == "onehop":
        assert query == onehop_query(bridges[0]) and ans == f" {codes[0]}", item.id
        assert all(h not in query for h in heads), f"{item.id}: a head is named in the one-hop question"
    else:
        assert query == link_query(heads[0]) and ans == f" {bridges[0]}", item.id
    assert g.check_boundary_free(ctx, ans), item.id


# --------------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------------


def _token_index(byte_lens, text: str, char_offset: int) -> int:
    """The number of tokens of `text` (its tokens' byte lengths: byte_lens) that end at or before character
    char_offset: the index of the token that holds that character."""
    ends = np.cumsum(np.asarray(byte_lens, dtype=np.int64))
    return int(np.searchsorted(ends, len(text[:char_offset].encode("utf-8")), side="right"))


def token_record(item: TextItem, encode) -> dict:
    """What the build checks and the analyses read for one item under `encode` (str -> TokenizedDoc: ids, byte_lens):

    gap_tokens        tokens from the start of the asked chain's first fact to the start of its second fact
    answer_distance   tokens from the start of the asked value fact to the context's last token
    link_distance     the same for the asked link fact
    first_index       the answer's first token that covers a non-whitespace character (0: the first token decides)
    answer_count      occurrences of the answer's tokens in the context's tokens (1; 2 for the link variant, whose
                      answer is the bridge: a link's value and a code's key)
    codes_first_distinct, codes_single   over the item's D + 1 codes in answer form (" <code>"): their first tokens
                      differ; each is one token (word values)."""
    from ced.eval.probes import _first_informative

    m = item.meta
    ctx, ans = encode(item.context), encode(item.answer)
    n_ctx = len(ctx.ids)
    link_at = _token_index(ctx.byte_lens, item.context, m["link_offsets"][0])
    value_at = _token_index(ctx.byte_lens, item.context, m["value_offsets"][0])
    code_ids = [list(encode(f" {c}").ids) for c in m["codes"]]
    c_ids, a_ids = list(ctx.ids), list(ans.ids)
    count = sum(1 for a in range(len(c_ids) - len(a_ids) + 1) if c_ids[a] == a_ids[0] and c_ids[a : a + len(a_ids)] == a_ids)
    return {"gap_tokens": abs(value_at - link_at), "answer_distance": n_ctx - 1 - value_at, "link_distance": n_ctx - 1 - link_at,
            "first_index": _first_informative(item.answer, ans), "answer_count": count,
            "codes_first_distinct": len({ids[0] for ids in code_ids}) == len(code_ids),
            "codes_single": all(len(ids) == 1 for ids in code_ids)}


def item_record(item: TextItem, tokens: dict) -> dict:
    """The "twohop" entry of an EncodedSet item record (and, through the scorer, of its result): what the summary
    groups and pairs by, and where the facts sit in tokens."""
    m = item.meta
    return {"base": m["base"], "variant": m["variant"], "length": item.cell["length"], "gap": m["gap"],
            "distractors": m["distractors"], "value": m["value"], "order": m["order"], "rank": m["rank"],
            "gap_tokens": tokens["gap_tokens"], "answer_distance": tokens["answer_distance"], "link_distance": tokens["link_distance"]}


def encode_items(items: list[TextItem], encode):
    """probes.encode_items plus the "twohop" record of every item (`item_record`)."""
    from ced.eval import probes

    es = probes.encode_items(items, encode, probe=PROBE)
    for rec, it in zip(es.items, items):
        rec["twohop"] = item_record(it, token_record(it, encode))
    return es


# --------------------------------------------------------------------------------------------
# scores and the summary
# --------------------------------------------------------------------------------------------


def item_scores(r: dict) -> dict:
    """The item's scores from a score_sets result (module docstring). The first token is the answer's first token that
    is not whitespace (`first_index`; 0 in a built set)."""
    lp, ok, k = r["tok_logp"], r["tok_correct"], r["first_index"]
    assert len(lp) == len(ok) == r["n_answer"] and 0 <= k < len(lp), r.get("id")
    out = {"first_ok": float(ok[k] == "1"), "first_nats": -float(lp[k]), "exact": float("0" not in ok), "nats": -float(sum(lp)) / len(lp)}
    if len(lp) > k + 1:
        out["rest_nats"] = -float(sum(lp[k + 1 :])) / (len(lp) - k - 1)
    return out


def settings_of(t: dict) -> list[str]:
    """The settings a base item is reported under: all, each design level, and its cell."""
    return ["all", f"length={t['length']}", f"gap={t['gap']}", f"order={t['order']}", f"distractors={t['distractors']}",
            f"value={t['value']}", cell_name(t)]


def cell_name(t: dict) -> str:
    return f"length={t['length']},gap={t['gap']},distractors={t['distractors']},value={t['value']}"


def is_cell(setting: str) -> bool:
    return "," in setting


def by_base(results: list[dict]) -> dict:
    """{base id: {"t": the base's settings record, "fp": {variant: fingerprint}, variant: scores}} of the probe's
    results (other probes' results are passed over)."""
    out: dict = {}
    for r in results:
        if r.get("probe") != PROBE:
            continue
        t = r["twohop"]
        rec = out.setdefault(t["base"], {"t": t, "fp": {}})
        assert t["variant"] not in rec["fp"], f"{r['id']}: scored twice"
        rec[t["variant"]] = item_scores(r)
        rec["fp"][t["variant"]] = r["fingerprint"]
    return out


def _agg(rows: list[dict]) -> dict:
    from ced.eval.probes import mean_se

    out = {"n": len(rows)}
    for k in METRICS:
        vals = [x[k] for x in rows if k in x]
        if vals:
            out[k], out[k + "_se"] = mean_se(vals)
    return out


def _sort_settings(names) -> list[str]:
    def key(s: str):
        if s == "all":
            return (0, 0, [])
        parts = []
        for kv in s.split(","):
            k, _, v = kv.partition("=")
            try:
                parts.append((k, 0, float(v), ""))
            except ValueError:
                parts.append((k, 1, 0.0, v))
        order = ["length", "gap", "order", "distractors", "value"]
        return (2 if is_cell(s) else 1, order.index(parts[0][0]) if parts[0][0] in order else 9, parts)
    return sorted(names, key=key)


def summarize(results: list[dict]) -> dict:
    """{setting: {"n": base items, "chance": the mean of 1 / (D + 1), "chance_nats": the mean of ln(D + 1),
    "variants": {variant: {"n", metric: item mean, metric_se}},
    "paired": {"hop_cost": twohop - onehop, "link_use": twohop - shuffled, "link_follow": twohop - swapped; each {"n",
    metric: mean of the per-base differences, metric_se: their SE}}}}. Settings: all, every design level and every
    cell (`settings_of`)."""
    bases = by_base(results)
    tree: dict = defaultdict(list)
    for rec in bases.values():
        for s in settings_of(rec["t"]):
            tree[s].append(rec)
    out = {}
    for s in _sort_settings(tree):
        recs = tree[s]
        cell = {"n": len(recs), "chance": float(np.mean([1.0 / (r["t"]["distractors"] + 1) for r in recs])),
                "chance_nats": float(np.mean([math.log(r["t"]["distractors"] + 1) for r in recs])),
                "variants": {v: _agg([r[v] for r in recs if v in r]) for v in VARIANTS if any(v in r for r in recs)},
                "paired": {}}
        for name, (a, b) in PAIRS.items():
            both = [r for r in recs if a in r and b in r]
            if both:
                cell["paired"][name] = _agg([{k: r[a][k] - r[b][k] for k in METRICS if k in r[a] and k in r[b]} for r in both])
        out[s] = cell
    return out


def compare(a: list[dict], b: list[dict]) -> dict:
    """Two scorings of the same set (two checkpoints, or one checkpoint at --T 2 and --T 1), a - b, paired per base
    item: {setting: {"n", <variant>: that variant's scores (twohop, onehop, link, shuffled, swapped), "hop_cost":
    (twohop - onehop of a) - (twohop - onehop of b), "link_use": the same with shuffled, "link_follow": the same with
    swapped; each {"n", metric: mean difference, metric_se}}}. The items must be the same tokens (fingerprints). A
    gain on twohop that the controls share is not a gain in using the link: read link_follow and link_use."""
    A, B = by_base(a), by_base(b)
    tree: dict = defaultdict(list)
    for base, x in A.items():
        y = B.get(base)
        if y is None:
            continue
        for v in x["fp"]:
            assert v not in y["fp"] or x["fp"][v] == y["fp"][v], f"{base}/{v}: the two scorings read different items"
        for s in settings_of(x["t"]):
            tree[s].append((x, y))
    out = {}
    for s in _sort_settings(tree):
        pairs = tree[s]
        cell = {"n": len(pairs)}
        for v in VARIANTS:
            rows = [{k: x[v][k] - y[v][k] for k in METRICS if k in x[v] and k in y[v]} for x, y in pairs if v in x and v in y]
            if rows:
                cell[v] = _agg(rows)
        for name, (p, q) in PAIRS.items():
            rows = [{k: (x[p][k] - x[q][k]) - (y[p][k] - y[q][k]) for k in METRICS if all(k in z[w] for z in (x, y) for w in (p, q))}
                    for x, y in pairs if all(w in z for z in (x, y) for w in (p, q))]
            if rows:
                cell[name] = _agg(rows)
        out[s] = cell
    return out
