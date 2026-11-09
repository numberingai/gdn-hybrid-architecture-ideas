"""Fixed-distance needles: uuid_needle and multikey items whose needle sits a requested number of tokens before the
question (build option tools/build_probes.py --needle-distances).

probes-v1 places a needle by depth, a fraction of the haystack's characters, so its token distance from the question
is known only afterwards and takes five values per length. Here the distance is the request. With the same distance
inside two item lengths (2,048 ... 30,720 tokens inside 32,768 and inside 65,536), a score that follows the distance
is told apart from one that follows the length or the depth.

An item is a probes-v1 item in every other respect: the same sentences (probe_gen.needle_sentence / needle_query),
the same held-out haystack pool, whole documents joined by a blank line, the needle at a sentence start, the same
checks (probe_gen.check_item) and the same EncodedSet files, so the probe scorer and the length diagnostics read it
unchanged. cell: {"length"} (uuid_needle) or {"length", "n_keys"} (multikey); group: {"depth" or "asked_depth":
1 - distance / length, "distance": the request}.

Distance (as attn_diag.NeedleSpans.distance): the tokens from the needle's first value token to the first answer
query, the context's last token, under the placement tokenizer (the build's first length tokenizer).

How the request is met without cutting a document or a sentence:
- Documents. One fit per (probe, length, n_keys, item index), independent of the distance: the documents, the key
  names and the values of item i are the same at every distance of a length (the haystack stream does not see the
  distance). So the scores of one item index across distances are paired: what differs is where the needle sits
  and the order of the documents.
- Order and place. A sentence inserted at a sentence start c of a document is followed by the rest of that document
  and by every later document, so its distance is known from per-document token offsets alone (DocTokens: one
  tokenization per document, cached). The documents' order is redrawn (a stream of its own per distance) until some
  sentence start gives the requested distance exactly by that count, or `order_tries` orders have been tried (the
  nearest is then taken: a needle close to the question lies in the last document, which leaves few places); the
  item is then built and its distance measured by tokenizing the context. A difference (tokens that merge across a
  boundary in a way the count missed) is fed back as a bias and the search repeated; the item is accepted when the
  measured distance is within `tol` tokens of the request.
- Fallback. When no order of the item's documents gets there (documents with few sentence starts near the wanted
  place), other documents are fitted for this one distance (meta "refits" counts them: 0 for nearly every item; such
  an item is not paired with the item index's other distances), and after `refits` of those the request is refused
  (RuntimeError).
- multikey: the other keys keep random depths (fractions of the haystack's characters, as in probes-v1; drawn once
  per item index); those that land after the asked key add their sentences' tokens to its distance, which the count
  includes. A request that falls where another key's sentence stands cannot be met in any order: the other keys'
  depths are then drawn again (from the distance's stream; the item's meta has the depths used).
- The count is exact where tokens do not merge across the inserted text's edges in a way the per-document offsets
  miss; the joiner after a sentence (a space, or a newline at a line start) is counted by TokenIndex.joiner_tokens.
- Length: the documents are fitted to `length` - DIST_SLACK tokens, so the item stays at most `length` tokens after
  the reordering (a boundary token more or less); its lower bound is (1 - LENGTH_TOL) * length - 2 * DIST_SLACK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ced.eval import probe_gen as g

DIST_TAG = 2_000_003  # a word of the random streams' cells that no probes-v1 cell has (depths x 1000 and key counts are small)
DIST_SLACK = 16  # tokens kept free when the documents are fitted (the order and the needle's place are set afterwards)
DIST_TOL = 4  # accepted |measured distance - request|, tokens
MIN_HEAD = 512  # a distance fits a length when distance + MIN_HEAD <= (1 - LENGTH_TOL) * length (room before the needle)

Starts = Callable[[str], np.ndarray]  # text -> the character offset where each of its tokens starts (ascending)


def fits(length: int, distance: int, min_head: int = MIN_HEAD) -> bool:
    """A needle `distance` tokens before the question has at least min_head tokens before it in the shortest item of
    the length."""
    return distance + min_head <= (1 - g.LENGTH_TOL) * length


def distances_for(length: int, distances, min_head: int = MIN_HEAD) -> list[int]:
    return [int(d) for d in distances if fits(length, int(d), min_head)]


def depth_of(length: int, distance: int) -> float:
    return round(1.0 - distance / length, 4)


@dataclass
class DocTokens:
    """One haystack document followed by the separator, under the placement tokenizer.

    n:    tokens of text + SEP (a full stop and the blank line after it can be one token, so the separator is counted
          with the document it follows)
    cand: character offsets a sentence may be inserted at: 0 and the document's sentence starts (probe_gen.SENTENCE_START)
    tail: per candidate, the tokens of text + SEP that end after it: what follows a sentence inserted there (a token
          that straddles the offset, " Next" for a sentence start after a space, follows it)
    line: per candidate, whether it is a line start (the inserted sentence is then followed by a newline, else by a
          space: probe_gen.insert_at)"""

    n: int
    cand: np.ndarray
    tail: np.ndarray
    line: np.ndarray


class TokenIndex:
    """DocTokens per pool document, computed on first use (one tokenization each) and kept."""

    def __init__(self, pool: g.HaystackPool, starts: Starts):
        self.pool, self.starts = pool, starts
        self._docs: dict[int, DocTokens] = {}
        self.fits: dict = {}  # (probe, length, n_keys, index) -> the fitted documents and names (distance_item)
        self._joiner: dict = {}

    def count(self, text: str) -> int:
        return int(len(self.starts(text)))

    def joiner_tokens(self, joiner: str) -> int:
        """Tokens the joiner after an inserted sentence adds between the sentence's full stop and the text that
        follows: 0 where it merges into a neighbour (a BPE whose spaces go with the next word, a newline with the
        full stop), 1 under a byte tokenizer. Measured on a fixed example."""
        if joiner not in self._joiner:
            left, right = " 0a.", "The quick"
            self._joiner[joiner] = self.count(left + joiner + right) - self.count(left) - self.count(right)
        return self._joiner[joiner]

    def doc(self, j: int) -> DocTokens:
        d = self._docs.get(j)
        if d is None:
            text = self.pool.texts[j]
            unit = text + g.SEP
            st = np.asarray(self.starts(unit), dtype=np.int64)
            ends = np.append(st[1:], len(unit))
            cand = np.unique(np.concatenate([[0], g.sentence_starts(text)])).astype(np.int64)
            tail = len(st) - np.searchsorted(ends, cand, side="right")
            line = np.array([c == 0 or text[c - 1] == "\n" for c in cand.tolist()], dtype=bool)
            d = self._docs[j] = DocTokens(n=int(len(st)), cand=cand, tail=tail.astype(np.int64), line=line)
        return d


def measured_distance(context: str, answer: str, starts: Starts) -> int:
    """The item's needle distance in tokens: from the token that holds the first character of the answer's one
    occurrence in the context (the needle's value, with its leading space) to the context's last token."""
    assert context.count(answer) == 1, "the answer must occur exactly once in the context"
    v = context.index(answer)
    st = np.asarray(starts(context), dtype=np.int64)
    ends = np.append(st[1:], len(context))
    first = int(np.searchsorted(ends, v, side="right"))  # tokens that end at or before the value's first character
    return len(st) - 1 - first


def _content(probe: str, cell: tuple, index: int, n_keys: int):
    """Values, the other keys' depths, the asked key and the candidate names of item `index`: one stream per (probe,
    length, n_keys, index), the same at every distance."""
    rng = g.rng_for(probe, cell, index, 0)
    if n_keys == 1:
        return [g.random_uuid(rng)], [0.5], 0, g.candidate_names(rng, 40)
    values: list[str] = []
    while len(values) < n_keys:  # distinct values, also in their first group (as probe_gen.multikey_item)
        v = g.random_uuid(rng)
        if all(v[:8] != w[:8] for w in values):
            values.append(v)
    fracs = [float(x) for x in rng.uniform(0.02, 0.98, size=n_keys)]
    asked = int(rng.integers(n_keys))
    return values, fracs, asked, g.candidate_names(rng, min(120, 3 * n_keys + 40))


def distance_item(pool: g.HaystackPool, count_item, tokens: TokenIndex, probe: str, length: int, distance: int, index: int,
                  n_keys: int = 1, tol: int = DIST_TOL, order_tries: int = 300, build_tries: int = 8, refits: int = 4) -> g.TextItem:
    """A uuid_needle (n_keys 1) or multikey item of nominal length `length` whose asked needle is `distance` tokens
    before the question, to within `tol` (module docstring). count_item(context, answer): the build's length measure;
    tokens: the placement tokenizer's TokenIndex over `pool`. Raises RuntimeError when neither the item's documents
    nor `refits` other fits of documents can put the needle there (a pool with very few sentence starts)."""
    assert probe in ("uuid_needle", "multikey") and (n_keys == 1) == (probe == "uuid_needle"), (probe, n_keys)
    cell = (length, DIST_TAG, n_keys)
    values, fracs, asked, names_pool = _content(probe, cell, index, n_keys)
    key = (probe, length, n_keys, index)
    if key not in tokens.fits:
        if len(tokens.fits) >= 8:  # jobs come grouped by item index: a few entries are enough
            tokens.fits.pop(next(iter(tokens.fits)))
        fit = g._haystack_item(pool, count_item, length - DIST_SLACK, fracs, values, asked, names_pool, g.rng_for(probe, cell, index, 1))
        tokens.fits[key] = (list(fit[2]), list(fit[4]))
    chosen, names = tokens.fits[key]
    tried = []
    for refit in range(refits + 1):
        if refit:  # other documents, for this distance only
            fit = g._haystack_item(pool, count_item, length - DIST_SLACK, fracs, values, asked, names_pool,
                                   g.rng_for(probe, (length, DIST_TAG, n_keys, distance), index, 2 + refit))
            chosen, names = list(fit[2]), list(fit[4])
        item, note = _place(pool, count_item, tokens, probe, length, distance, index, n_keys, tol, order_tries, build_tries,
                            values, fracs, asked, chosen, names)
        if item is not None:
            item.meta["refits"] = refit
            return item
        tried.append(note)
    raise RuntimeError(f"{probe} length {length} distance {distance} item {index}: no order of the documents puts the needle "
                       f"within {tol} tokens of the request ({'; '.join(tried)})")


def _place(pool, count_item, tokens: TokenIndex, probe: str, length: int, distance: int, index: int, n_keys: int, tol: int,
           order_tries: int, build_tries: int, values: list, fracs: list, asked: int, chosen: list, names: list):
    """distance_item for one set of documents: (the item, None), or (None, what was tried)."""
    sentences = [g.needle_sentence(nm, v) for nm, v in zip(names, values)]
    query, answer = g.needle_query(names[asked]), f" {values[asked]}"
    # from the value's first token to the end of its sentence, plus the question after the last separator, less the
    # last token (the first answer query)
    const = tokens.count(answer + ".") + tokens.count(query[len(g.SEP):]) - 1
    other_tok = [tokens.count(s) for s in sentences]
    others = [k for k in range(n_keys) if k != asked]
    info = [tokens.doc(j) for j in chosen]
    chars = np.array([len(pool.texts[j]) + len(g.SEP) for j in chosen], dtype=np.int64)
    n_tok = np.array([d.n for d in info], dtype=np.int64)
    # per candidate, the tokens its joiner adds (a newline at a line start, else a space)
    j_line, j_space = tokens.joiner_tokens("\n"), tokens.joiner_tokens(" ")
    join = [np.where(d.line, j_line, j_space) for d in info]
    order_rng = g.rng_for(probe, (length, DIST_TAG, n_keys, distance, len(chosen), int(n_tok.sum())), index, 2)
    lo_t = int(np.ceil((1 - g.LENGTH_TOL) * length)) - 2 * DIST_SLACK
    fr = list(fracs)  # the other keys' depths (the asked key's entry is not read)
    bias, builds, orders, rounds = 0, 0, 0, 0
    while builds < build_tries and rounds < 2 * build_tries:
        rounds += 1
        # redraw the documents' order until a sentence start gives the requested distance by the token count
        found = None  # (|predicted error|, order, candidate offset, the other keys' offsets, prediction)
        for _ in range(order_tries):
            orders += 1
            perm = order_rng.permutation(len(chosen))
            off = np.concatenate([[0], np.cumsum(chars[perm])])  # character offset of each document, and the total
            after = np.concatenate([np.cumsum(n_tok[perm][::-1])[::-1][1:], [0]])  # tokens of the later documents
            cand = np.concatenate([off[i] + info[p].cand for i, p in enumerate(perm)])
            pred = const + bias + np.concatenate([info[p].tail + join[p] + after[i] for i, p in enumerate(perm)])
            hay_chars = int(off[-1]) - len(g.SEP)
            lines = None  # the candidates' line-start flags in this order (multikey only)
            pts = {}
            for k in others:  # another key: at the candidate nearest to its depth; where its sentence follows the asked
                at = int(np.argmin(np.abs(cand - fr[k] * hay_chars)))  # one (ties: the lower key index goes first), its
                pts[k] = int(cand[at])                                 # tokens and its joiner's add to the distance
                if lines is None:
                    lines = np.concatenate([info[p].line for p in perm])
                adds = other_tok[k] + (j_line if lines[at] else j_space)
                pred = pred + adds * ((cand < pts[k]) | ((cand == pts[k]) & (asked < k)))
            j = int(np.argmin(np.abs(pred - distance)))
            err = abs(int(pred[j]) - distance)
            if found is None or err < found[0]:
                found = (err, perm, int(cand[j]), pts, int(pred[j]))
            if err == 0:
                break
        if found is None or found[0] > tol:
            if not others:
                break
            # multikey: the request falls where another key's sentence stands in every order (their depths are
            # fractions of the haystack): draw the other keys' depths again, from the distance's stream
            for k in others:
                fr[k] = float(order_rng.uniform(0.02, 0.98))
            continue
        _, perm, c, pts, pred_c = found
        builds += 1
        docs = [chosen[p] for p in perm]
        haystack = pool.join(docs)
        points = [c if k == asked else pts[k] for k in range(n_keys)]
        text, offsets = g.insert_at(haystack, points, sentences)
        context = text + query
        real = measured_distance(context, answer, tokens.starts)
        n = count_item(context, answer)
        if abs(real - distance) <= tol and lo_t <= n <= length:
            depth = depth_of(length, distance)
            meta = {"distance": distance, "distance_tokens": real, "haystack_chars": len(haystack),
                    "haystack_docs": [pool.doc_ids[j] for j in docs], "n_tokens_length_measure": n,
                    "orders_tried": orders, "builds": builds}
            if probe == "uuid_needle":
                return g.TextItem(
                    id=f"uuid_needle/len={length}/dist={distance}/{index:04d}", probe=probe, cell={"length": length},
                    group={"depth": depth, "distance": distance}, context=context, answer=answer,
                    meta={"name": names[0], "value": values[0], "char_offset": offsets[0], **meta}), None
            depths = [depth if k == asked else fr[k] for k in range(n_keys)]
            return g.TextItem(
                id=f"multikey/len={length}/n={n_keys}/dist={distance}/{index:04d}", probe=probe,
                cell={"length": length, "n_keys": n_keys}, group={"asked_depth": depth, "distance": distance},
                context=context, answer=answer,
                meta={"names": names, "values": values, "depths": depths, "asked": asked, "char_offsets": offsets, **meta}), None
        bias += real - pred_c  # what the count missed (boundary tokens); the next search starts from it
    return None, f"{len(chosen)} documents: {orders} orders, {builds} builds"
