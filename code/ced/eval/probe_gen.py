"""Text-level generators for the v2 ladder probes.

Four probes, every item a context string plus one answer string (or, for multiple choice, a list
of candidate continuations and the index of the gold one). Nothing here depends on a tokenizer
except the length of the long-context items, which is measured by a `count` callable (the build
tool passes the largest token count over the tokenizers the set is built for; the tests pass a
byte count):

- uuid_needle: held-out haystack of total length T (1x, 2x, 4x, 8x the 8,192-token training rows),
  one sentence "The access code for <name> is <uuid4>." inserted at a sentence start nearest to a
  depth fraction, then the query "What is the access code for <name>? The access code for <name>
  is" with " <uuid>" as the answer.
- multikey: the same with N in {4, 16} such sentences at random depths (distinct keys and values);
  one key is asked.
- path_copy: a code-like index of K "class <Name> -> <path>" lines whose paths share a long prefix
  and differ in one directory (K x 1 x 1 grids; that directory is one token of its own under each
  tokenizer, GROUP_WORDS) or are near-duplicates of each other (one segment or one digit apart:
  2 x 3 x 4); the query repeats the form, "class <Name> ->", with " <path>" as the answer (format
  "arrow"; the "sentence" query "The file that defines class <Name> is" is kept as a variant: 60M
  models never start a path there).
- depth: Python assignments forming C chains of h + 1 lines each (the asked chain and C - 1
  distractor chains) and filler statements, interleaved at random in the body of `def main():`,
  then "    assert <var> == " and the digit (format "body"), scored as 10-way multiple choice over
  "0".."9" and as the choice among the C assigned digits. The asked literal's rank among the C
  literal lines and the number of lines after the asked variable cycle with the item index, so no
  position or chain-length heuristic beats 1 / C (layout "equal"; the calibration round's "free"
  layout and top-level formats are kept as variants).

Haystacks are whole held-out documents (the web and general_web eval sets, never training shards)
joined by a blank line, in random order. Every generator is a pure function of (seed, probe, cell,
item index) and, for haystack items, the document pool and the `count` function. Values and depths
come from one random stream and the haystack from another; keys are the first names of a fixed
candidate list that do not occur in the haystack. So an item's content changes with the length
measure only when a different haystack happens to contain one of the first candidate names.

Boundary convention: the answer starts with a space after a letter or ">" (" <uuid>", " <path>"),
or is a single digit (depth), so tokenizing context and answer separately gives the same tokens as
tokenizing them together under the split pattern of our byte-level BPE tokenizers
(tools/data/train_tokenizer.py PATTERN). The build tool checks this per item and tokenizer.
"""

from __future__ import annotations

import keyword
import re
import uuid as uuidlib
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

BASE_SEED = 20261007
TRAIN_LEN = 8192
LENGTHS = tuple(TRAIN_LEN * m for m in (1, 2, 4, 8))  # 8k, 16k, 32k, 64k
NEEDLE_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
MULTIKEY_N = (4, 16)
PATH_PREFIX = {"shallow": 2, "deep": 6}  # shared directory segments after the root
# path_copy cells (K, grid G x S x D), set by the calibration on the v1 60M models: K x 1 x 1, one
# choice (the directory after the shared prefix) decides the path; 2 x 3 x 4, three choices
# (directory, stem, digit) among near-duplicates one segment or one digit apart.
PATH_CELLS = ((4, (4, 1, 1)), (8, (8, 1, 1)), (16, (16, 1, 1)), (12, (2, 3, 4)))
LEGACY_PATH_K = (12, 24, 40)  # the first build's cells (grids PATH_GRID): about twice the floor at 60M
HOPS = tuple(range(0, 9))  # hops = 0: the asked variable holds the digit itself (one lookup, no composition)
DEPTH_CHAINS = 3  # the queried chain + 2 distractor chains (calibration: 4 chains sat at the floor at every h)
# Depth layouts. "equal" (default since the review of 2026-10-07): every chain is h + 1 lines and
# filler statements keep the item at DEPTH_STATEMENTS lines, so no chain is shorter than the asked
# one; the chains' first h lines are exchangeable (a uniform region) and their last lines are in a
# uniformly ordered final segment, so the asked variable's line is uniform over it at every h. "free" (the calibration round of 2026-10-07,
# 3 chains in 16 lines): the distractor chains fill the item, so the asked chain is the shortest at
# small h and the longest at large h (a chain-length cue).
DEPTH_LAYOUTS = ("equal", "free")
DEPTH_LAYOUT = "equal"
DEPTH_STATEMENTS = 32  # equal layout: 3 chains of h + 1 lines (27 at h = 8) plus filler, whatever h (the region is 24 lines)
FREE_DEPTH_STATEMENTS = 16  # the free layout's calibrated size
DEPTH_FINAL = 8  # the final segment: every chain's last line and filler, in random order (equal layout)
FILLER_ATOMS = ("None", "True", "False", "[]", "{}", '""')  # filler right-hand sides: never a digit
LEGACY_DEPTH = (4, 24)  # (chains, statements) of the first build: its items keep their random streams
LENGTH_TOL = 0.03  # a haystack item's total tokens lie in [(1 - tol) T, T]
SEP = "\n\n"  # between haystack documents
PROBES = ("uuid_needle", "multikey", "path_copy", "depth")
PROBE_CODE = {p: i for i, p in enumerate(PROBES)}

Count = Callable[[str], int]


@dataclass
class TextItem:
    id: str
    probe: str
    cell: dict  # the aggregation cell (uuid_needle: length; multikey: n_keys, length; path_copy: k, prefix; depth: hops)
    context: str
    answer: str  # gold continuation (for multiple choice: candidates[gold])
    candidates: list[str] | None = None
    gold: int | None = None
    group: dict = field(default_factory=dict)  # finer grouping inside a cell (uuid_needle: depth)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_json(cls, d: dict) -> "TextItem":
        return cls(**d)


def cell_key(item_or_cell) -> str:
    cell = item_or_cell.cell if isinstance(item_or_cell, TextItem) else item_or_cell
    return ",".join(f"{k}={cell[k]}" for k in sorted(cell))


def rng_for(probe: str, cell: tuple, index: int, stream: int) -> np.random.Generator:
    """Independent stream per (probe, cell, item, purpose). stream 0: content; 1: haystack."""
    words = [BASE_SEED, PROBE_CODE[probe], *[int(round(x * 1000)) if isinstance(x, float) else int(x) for x in cell], index, stream]
    return np.random.default_rng(np.random.SeedSequence(words))


# --------------------------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------------------------

FIRST_NAMES = """Aaron Abigail Adele Adrian Agnes Alba Alden Alma Alvin Amara Anders Anika Ansel Aria Arlo Astrid
Aurelio Basil Beatrix Bennett Bianca Boris Bram Brielle Callum Camille Casimir Cecily Cedric Celeste Cyrus Dalia
Damian Daphne Darius Delphine Desmond Dorian Edda Edmund Elio Elodie Emeric Enzo Esme Evander Fabian Faye Felix
Fiona Florian Freya Gareth Gemma Gideon Greta Gunnar Hadley Halvard Hazel Hector Helena Hugo Ida Ignatius Ilse
Imogen Ingrid Isidore Ivo Jasper Juno Kaia Kasimir Keira Lazlo Leander Lenora Leopold Linnea Lorcan Lucian Lydia
Magnus Malin Marcel Matilda Maxim Mireille Nadia Nikolai Noemi Octavia Odile Olin Orla Osric Otto Pascal Petra
Philippa Quentin Quinn Rafael Rhea Rolf Rosalind Rufus Sabine Saoirse Severin Signe Silas Solveig Stellan Sylvie
Tamsin Teodor Thea Tobias Ulla Ulric Valentin Vera Viggo Wilhelmina Wren Xavier Yara Yusuf Zara Zeno""".split()

LAST_NAMES = """Abernathy Achterberg Aldana Ambrose Arkwright Ashdown Bachelard Balfour Barclay Beaumont Bellweather
Birkeland Blackwood Bramley Brennan Calloway Castellan Chadwick Coldwell Corrigan Crowhurst Dalgaard Davenport
Delacroix Drummond Duquesne Eastwick Edevane Ellery Elmsworth Fairbanks Falkner Fenwick Fitzgerald Fontaine
Galloway Garnier Gilchrist Granger Gravesend Halloran Hargreaves Hawthorne Heathcote Holloway Ingersoll Inglewood
Jardine Kavanagh Kettering Kingsley Lachance Langford Larkspur Lindqvist Lockhart Lowell MacAllister Marchetti
Montague Moreau Mortensen Nightingale Northcott Oakhurst Ogilvie Okonkwo Ormsby Pemberton Penhaligon Quenneville
Radcliffe Ravensworth Redgrave Rosenthal Rutherford Saltonstall Sandoval Sinclair Southwell Stanhope Sterling
Strickland Sutherland Tallis Templeton Thackeray Thornbury Treadwell Underhill Vasquez Vandermeer Wainwright
Wakefield Westbrook Whitlock Wilberforce Winterbourne Wolcott Yardley Zimmerman""".split()

# path_copy: directory segments, file stems, class-name words (disjoint lists)
ROOTS = ["/srv", "/opt", "/home/deploy", "/var/lib", "/usr/local/share", "/workspace"]
DIR_WORDS = """app apps services service core internal handlers handler api apis v1 v2 v3 backend server worker workers
jobs tasks pipelines plugins modules lib libs pkg src common shared utils adapters domain infra storage gateway
billing ledger payments accounts auth identity search catalog orders shipping inventory reports analytics metrics
alerts notify scheduler queue cache sync export import admin portal console tenants regions""".split()
# The directory that decides the path in the one-choice grids (K x 1 x 1). Every "/<word>" here is
# ONE token under both tokenizers the sets are built for (t128k v0, t64k v1-64k; the build checks
# it per item), so the choice token is the whole directory name: no two listed paths share it,
# none is a bare "/", and a class-blind pick scores exactly 1/K under either tokenizer. (DIR_WORDS
# failed this: under t64k 17 of its words start with a bare "/" and app/apps, lib/libs, v1/v2/v3,
# ... share a first token; review of 2026-10-07.) Plural twins (services, apis) are left out.
GROUP_WORDS = """app service core internal api server jobs plugins modules lib pkg src common shared utils storage auth
search cache import admin config models schemas templates static assets scripts tools tests docs events helpers graph
loader parser network users build release vendor stable web cli media images files uploads comments rest registry
oauth stream""".split()
FILE_STEMS = """invoice_sync ledger_merge payment_batch refund_queue account_audit token_refresh session_store
order_router stock_check price_rules tax_table rate_limit retry_policy event_bus job_runner cron_plan report_daily
export_csv import_feed schema_diff cache_warm index_build search_rank alert_rule notify_mail user_merge tenant_seed
region_map quota_guard feature_flag audit_trail health_probe""".split()
CLASS_ADJ = """Amber Arctic Azure Bold Brisk Cedar Cobalt Copper Crimson Dusky Ember Fallow Gilded Granite Hollow
Indigo Ivory Jade Lunar Maple Misty Nimble Obsidian Pale Quiet Rapid Russet Silent Silver Solar Stormy Tawny
Velvet Verdant Wild Woven""".split()
CLASS_NOUN = """Anchor Badger Beacon Bramble Canyon Cinder Comet Condor Falcon Fennel Glacier Harbor Heron Juniper
Kestrel Lantern Meadow Nebula Orchard Osprey Pebble Quarry Raven Ridge Sparrow Summit Thicket Thistle Tundra Willow
Wren Yarrow""".split()

_VAR_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_BAD_VARS = {"as", "if", "in", "is", "or", "id", "io", "os", "re"}
VAR_NAMES = [a + b for a in _VAR_LETTERS for b in _VAR_LETTERS if a + b not in _BAD_VARS and not keyword.iskeyword(a + b)]

DIGITS = [str(d) for d in range(10)]

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
# A sentence start: after end punctuation (optionally a closing quote or bracket) and spaces, before
# an upper-case letter or an opening quote; or at a line start. The start is the match's end.
SENTENCE_START = re.compile(r"[.!?][\"')\]]? +(?=[A-Z\"'(])|\n(?=\S)")


def random_uuid(rng: np.random.Generator) -> str:
    return str(uuidlib.UUID(bytes=rng.bytes(16), version=4))


def candidate_names(rng: np.random.Generator, n: int) -> list[str]:
    """n full names in a fixed random order with distinct first names and distinct last names
    (n <= 120). Items take the first names of this list that do not occur in their haystack."""
    firsts = rng.choice(len(FIRST_NAMES), size=n, replace=False)
    lasts = rng.choice(len(LAST_NAMES), size=n, replace=False)
    return [f"{FIRST_NAMES[a]} {LAST_NAMES[b]}" for a, b in zip(firsts, lasts)]


def names_absent_from(candidates: list[str], text: str, n: int) -> list[str]:
    """The first n candidates whose first and last name do not occur as words in text (so no key is
    even half-present in the haystack)."""
    words = set(re.findall(r"[A-Za-z]+", text))
    out = [c for c in candidates if c.split()[0] not in words and c.split()[1] not in words and c not in text]
    if len(out) < n:
        raise RuntimeError("not enough names absent from the haystack")
    return out[:n]


# --------------------------------------------------------------------------------------------
# haystack
# --------------------------------------------------------------------------------------------


def usable_haystack_doc(text: str) -> bool:
    """A document may serve as haystack unless it already contains what a probe plants: a
    UUID-shaped string or the phrase "access code" (either would be an unplanned distractor)."""
    return UUID_RE.search(text) is None and "access code" not in text.lower()


@dataclass
class HaystackPool:
    texts: list[str]  # whole documents, stripped
    doc_ids: list[str]
    counts: np.ndarray  # tokens per document under the length measure

    @classmethod
    def build(cls, docs: list[tuple[str, str]], count: Count) -> "HaystackPool":
        keep = [(i, t.strip()) for i, t in docs if t.strip() and usable_haystack_doc(t)]
        return cls(texts=[t for _, t in keep], doc_ids=[i for i, _ in keep],
                   counts=np.array([count(t) for _, t in keep], dtype=np.int64))

    def pick(self, rng: np.random.Generator, lo: int, hi: int) -> list[int] | None:
        """Documents in random order whose estimated joined length lies in [lo, hi]: walk a random
        permutation and take every document that still fits. None if this walk cannot reach lo."""
        order = rng.permutation(len(self.texts))
        chosen, total = [], 0
        for j in order:
            c = int(self.counts[j]) + 1  # + the separator
            if total + c <= hi:
                chosen.append(int(j))
                total += c
                if total >= lo:
                    return chosen
        return None

    def join(self, chosen: list[int]) -> str:
        return SEP.join(self.texts[j] for j in chosen)


def sentence_starts(text: str) -> np.ndarray:
    return np.array([m.end() for m in SENTENCE_START.finditer(text)], dtype=np.int64)


def insertion_point(text: str, starts: np.ndarray, fraction: float) -> int:
    """The sentence start nearest to fraction * len(text) (characters); a document boundary or the
    text start if the text has no sentence start."""
    target = fraction * len(text)
    if len(starts) == 0:
        return 0
    return int(starts[np.argmin(np.abs(starts - target))])


def insert_sentences(text: str, sentences: list[tuple[float, str]]) -> tuple[str, list[int]]:
    """Insert each (depth fraction, sentence) at the sentence start nearest to its depth. Returns the
    new text and the character offset of each inserted sentence in it (in input order). A sentence
    is followed by a space where the next sentence follows on the same line, by a newline at a line
    start."""
    starts = sentence_starts(text)
    points = [insertion_point(text, starts, f) for f, _ in sentences]
    return insert_at(text, points, [s for _, s in sentences])


def insert_at(text: str, points: list[int], sentences: list[str]) -> tuple[str, list[int]]:
    """Insert sentences[i] at character offset points[i] of text (a sentence start; equal offsets: in input
    order). Returns the new text and the character offset of each inserted sentence in it (in input order). A
    sentence is followed by a space where the next sentence follows on the same line, by a newline at a line start.
    insert_sentences chooses the offsets by depth; the fixed-distance items (probe_gen_distance.py) by token count."""
    order = sorted(range(len(sentences)), key=lambda i: (points[i], i))
    out, prev, offsets = [], 0, [0] * len(sentences)
    length = 0
    for i in order:
        p = points[i]
        out.append(text[prev:p])
        length += p - prev
        prev = p
        joiner = "\n" if p == 0 or text[p - 1] == "\n" else " "
        piece = sentences[i] + joiner
        offsets[i] = length
        out.append(piece)
        length += len(piece)
    out.append(text[prev:])
    return "".join(out), offsets


def fit_haystack(pool: HaystackPool, rng: np.random.Generator, total: int, make: Callable[[str], tuple[str, str]],
                 count_item: Callable[[str, str], int], tries: int = 40) -> tuple[str, str, list[int], int]:
    """Choose haystack documents so that the finished item (context, answer) = make(haystack) has
    count_item(context, answer) in [(1 - LENGTH_TOL) * total, total]. Starts from an estimate of the
    extra tokens and corrects it from the exact counts."""
    lo_t, hi_t = int(np.ceil((1 - LENGTH_TOL) * total)), total
    probe_ctx, probe_ans = make("")
    extra = count_item(probe_ctx, probe_ans) + 2
    margin = max(4, int(0.004 * total))
    lo, hi = lo_t - extra + margin, hi_t - extra - margin
    for _ in range(tries):
        chosen = pool.pick(rng, lo, hi)
        if chosen is None:
            lo, hi = lo - margin, hi + margin
            continue
        context, answer = make(pool.join(chosen))
        n = count_item(context, answer)
        if lo_t <= n <= hi_t:
            return context, answer, chosen, n
        if n > hi_t:
            hi -= n - hi_t + margin
            lo = min(lo, hi - margin)
        else:
            lo += lo_t - n + margin
            hi = max(hi, lo + margin)
    raise RuntimeError(f"could not fit a haystack item into [{lo_t}, {hi_t}] tokens")


# --------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------


def needle_sentence(name: str, value: str) -> str:
    return f"The access code for {name} is {value}."


def needle_query(name: str) -> str:
    return f"{SEP}What is the access code for {name}? The access code for {name} is"


def _haystack_item(pool: HaystackPool, count_item, length: int, depths: list[float], values: list[str],
                   asked: int, names_pool: list[str], hay_rng: np.random.Generator):
    """Shared by uuid_needle and multikey: fit a haystack with one "access code" sentence per
    (depth, value), keys = the first len(values) names of names_pool absent from the haystack."""
    n_keys = len(values)
    state = {"names": names_pool[:n_keys]}

    def make(haystack: str) -> tuple[str, str]:
        names = state["names"]
        pairs = [(d, needle_sentence(nm, v)) for d, nm, v in zip(depths, names, values)]
        text, offsets = insert_sentences(haystack, pairs)
        state["offsets"] = offsets
        return text + needle_query(names[asked]), f" {values[asked]}"

    lo_t = (1 - LENGTH_TOL) * length
    for _ in range(20):
        _, _, chosen, _ = fit_haystack(pool, hay_rng, length, make, count_item)
        haystack = pool.join(chosen)
        state["names"] = names_absent_from(names_pool, haystack, n_keys)
        context, answer = make(haystack)
        n = count_item(context, answer)
        if lo_t <= n <= length:
            return context, answer, chosen, n, state["names"], state["offsets"], len(haystack)
    raise RuntimeError("could not fit a haystack item")


def uuid_needle_item(pool: HaystackPool, count_item, length: int, depth: float, index: int) -> TextItem:
    cell = (length, depth)
    content = rng_for("uuid_needle", cell, index, 0)
    value = random_uuid(content)
    names_pool = candidate_names(content, 40)
    context, answer, chosen, n, names, offsets, hay_chars = _haystack_item(
        pool, count_item, length, [depth], [value], 0, names_pool, rng_for("uuid_needle", cell, index, 1))
    return TextItem(
        id=f"uuid_needle/len={length}/depth={depth}/{index:04d}", probe="uuid_needle",
        cell={"length": length}, group={"depth": depth}, context=context, answer=answer,
        meta={"name": names[0], "value": value, "char_offset": offsets[0], "haystack_chars": hay_chars,
              "haystack_docs": [pool.doc_ids[j] for j in chosen], "n_tokens_length_measure": n},
    )


def depth_bin(fraction: float) -> float:
    """The nearest of NEEDLE_DEPTHS (bins of width 0.2)."""
    return NEEDLE_DEPTHS[min(len(NEEDLE_DEPTHS) - 1, int(fraction * len(NEEDLE_DEPTHS)))]


def multikey_item(pool: HaystackPool, count_item, length: int, n_keys: int, index: int) -> TextItem:
    cell = (length, n_keys)
    content = rng_for("multikey", cell, index, 0)
    values: list[str] = []
    while len(values) < n_keys:  # distinct values, also in their first group
        v = random_uuid(content)
        if all(v[:8] != w[:8] for w in values):
            values.append(v)
    depths = sorted(float(x) for x in content.uniform(0.02, 0.98, size=n_keys))
    asked = int(content.integers(n_keys))
    names_pool = candidate_names(content, min(120, 3 * n_keys + 40))
    context, answer, chosen, n, names, offsets, hay_chars = _haystack_item(
        pool, count_item, length, depths, values, asked, names_pool, rng_for("multikey", cell, index, 1))
    return TextItem(
        id=f"multikey/len={length}/n={n_keys}/{index:04d}", probe="multikey",
        cell={"length": length, "n_keys": n_keys}, group={"asked_depth": depth_bin(depths[asked])},
        context=context, answer=answer,
        meta={"names": names, "values": values, "depths": depths, "asked": asked, "char_offsets": offsets,
              "haystack_chars": hay_chars, "haystack_docs": [pool.doc_ids[j] for j in chosen],
              "n_tokens_length_measure": n},
    )


PATH_GRID = {12: (2, 3, 4), 24: (3, 3, 4), 40: (3, 4, 5)}  # K -> (groups, stems, digits)


PATH_FORMATS = ("arrow", "sentence")
# "body" (default): the statements are the indented body of `def main():` and the query is
# `    assert <x> == `, so every occurrence of a variable (left-hand side, right-hand side, query) is
# the same pre-token " <x>" under our split pattern and tokenizes identically (the build checks it).
# "comment" / "repl" / "assert" are top-level forms, where a variable is "<x>" at a line start,
# " <x>" on a right-hand side and "(<x>" in print(<x>): the query token often differs from the
# definition's (review of 2026-10-07).
DEPTH_FORMATS = ("body", "comment", "repl", "assert")
DEPTH_FORMAT = "body"


def parse_grid(grid: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """"GxSxD" (groups x stems x digits) or a tuple."""
    if isinstance(grid, str):
        G, S, D = (int(x) for x in grid.split("x"))
        return G, S, D
    return tuple(int(x) for x in grid)  # type: ignore[return-value]


def path_copy_item(k: int, prefix: str, index: int, fmt: str = "arrow", grid: str | tuple | None = None) -> TextItem:
    """K class -> path lines. Paths = root / shared segments / <group> / <stem>_<digit>.py, drawn
    as K distinct points of a small (group, stem, digit) grid G x S x D (default PATH_GRID[k]), so
    most paths have neighbours one segment or one digit away. The asked path has at least one
    neighbour differing only in the digit (if D > 1) and one differing only in one directory or
    stem segment (if G > 1 or S > 1). A K x 1 x 1 grid gives K paths that differ in one directory
    only: one choice, right after the shared prefix, decides the path.

    The group directories come from GROUP_WORDS (one token each, no shared first token), except in
    the first build's grids (PATH_GRID), which keep their DIR_WORDS draws and random streams."""
    G, S, D = parse_grid(grid) if grid is not None else PATH_GRID[k]
    assert G * S * D >= k, "path_copy: grid smaller than K"
    legacy = PATH_GRID.get(k) == (G, S, D)
    n_shared = PATH_PREFIX[prefix]
    rng = rng_for("path_copy", (k, n_shared) if legacy else (k, n_shared, G, S, D), index, 0)
    for _ in range(1000):
        if legacy:
            words = list(rng.choice(len(DIR_WORDS), size=n_shared + G, replace=False))
            shared = [DIR_WORDS[w] for w in words[:n_shared]]
            groups = [DIR_WORDS[w] for w in words[n_shared:]]
        else:
            groups = [GROUP_WORDS[w] for w in rng.choice(len(GROUP_WORDS), size=G, replace=False)]
            rest = [w for w in DIR_WORDS if w not in groups]
            shared = [rest[w] for w in rng.choice(len(rest), size=n_shared, replace=False)]
        stems = [FILE_STEMS[s] for s in rng.choice(len(FILE_STEMS), size=S, replace=False)]
        digits = sorted(int(d) for d in rng.choice(10, size=D, replace=False))
        root = ROOTS[int(rng.integers(len(ROOTS)))]
        grid = [(g, s, d) for g in range(G) for s in range(S) for d in range(D)]
        picks = [grid[i] for i in rng.choice(len(grid), size=k, replace=False)]
        pick_set = set(picks)

        def has_digit_nb(p):
            return any((p[0], p[1], d) in pick_set for d in range(D) if d != p[2])

        def has_segment_nb(p):
            return (any((g, p[1], p[2]) in pick_set for g in range(G) if g != p[0])
                    or any((p[0], s, p[2]) in pick_set for s in range(S) if s != p[1]))

        askable = [i for i, p in enumerate(picks) if (D == 1 or has_digit_nb(p)) and (G * S == 1 or has_segment_nb(p))]
        if askable:
            break
    else:
        raise RuntimeError("path_copy: no askable path")
    paths = [f"{root}/{'/'.join(shared)}/{groups[g]}/{stems[s]}_{digits[d]}.py" for g, s, d in picks]
    classes = []
    while len(classes) < k:
        c = CLASS_ADJ[int(rng.integers(len(CLASS_ADJ)))] + CLASS_NOUN[int(rng.integers(len(CLASS_NOUN)))]
        if c not in classes:
            classes.append(c)
    asked = askable[int(rng.integers(len(askable)))]
    lines = sorted(zip(paths, classes))  # a sorted listing puts near-duplicates next to each other
    body = "".join(f"class {c} -> {p}\n" for p, c in lines)
    # "arrow": the query repeats the listing's form, so the answer is a path for sure and the test is
    # which path is copied; "sentence": "The file that defines class X is" (a small model often
    # continues the sentence with a word rather than a path, so its first token is mostly wrong).
    query = f"class {classes[asked]} ->" if fmt == "arrow" else f"# The file that defines class {classes[asked]} is"
    context = (f"# Index of class definitions in {root}/{'/'.join(shared)}.\n"
               "# Generated by tools/index_classes.py; do not edit by hand.\n"
               "# Each line maps a class to the file that defines it.\n\n"
               f"{body}\n# Lookup\n{query}")
    shape = f"{G}x{S}x{D}"
    tag = "" if legacy else f"grid={shape}/"
    return TextItem(
        id=f"path_copy/k={k}/{tag}prefix={prefix}/{index:04d}", probe="path_copy",
        cell={"k": k, "prefix": prefix, "grid": shape}, context=context, answer=f" {paths[asked]}",
        meta={"asked_class": classes[asked], "paths": [p for p, _ in lines], "classes": [c for _, c in lines],
              "shared_segments": n_shared, "format": fmt, "choice_floor": path_choice_floor([p for p, _ in lines], paths[asked])},
    )


def path_choice_floor(paths: list[str], target: str) -> dict:
    """The copy-without-binding baseline of a path_copy item: a model that copies a listed path
    perfectly but picks uniformly among the listed paths consistent with the teacher-forced prefix
    at each of the three choice points (group, stem, digit). Returns its exact-match probability
    and its answer nats (log of the number of options, summed over the choice points)."""
    def parts(path):
        head, tail = path.rsplit("/", 1)
        stem, digit = tail[:-3].rsplit("_", 1)
        return head, stem, digit

    t, P = parts(target), [parts(x) for x in paths]
    nats = 0.0
    for level in range(3):
        n = len({q[level] for q in P if q[:level] == t[:level]})
        nats += float(np.log(n))
    return {"exact": float(np.exp(-nats)), "nats": round(nats, 6)}


def _chain_sizes(rng: np.random.Generator, total: int, n: int, lo: int = 2, hi: int = 9) -> list[int]:
    for _ in range(10000):
        sizes = [int(x) for x in rng.integers(lo, hi + 1, size=n)]
        if sum(sizes) == total:
            return sizes
    raise RuntimeError("chain sizes")


def _composition(rng: np.random.Generator, total: int, n: int) -> list[int]:
    """n positive sizes summing to total, uniform over the compositions (n - 1 distinct cut points)."""
    if n == 0:
        assert total == 0
        return []
    assert total >= n, "depth: too few statements for the chains"
    cuts = sorted(int(x) + 1 for x in rng.choice(total - 1, size=n - 1, replace=False)) if n > 1 else []
    bounds = [0, *cuts, total]
    return [b - a for a, b in zip(bounds[:-1], bounds[1:])]


def _chains(names: list[str], sizes: list[int], literals: list[int]) -> list[tuple[list[str], list[str]]]:
    """(variable names, statements) of each chain: `v0 = <digit>`, `v1 = v0`, ..."""
    out, at = [], 0
    for c, size in enumerate(sizes):
        chain_names = names[at : at + size]
        at += size
        stmts = [f"{chain_names[0]} = {literals[c]}"] + [f"{chain_names[i]} = {chain_names[i - 1]}" for i in range(1, size)]
        out.append((chain_names, stmts))
    return out


def _equal_layout(rng: np.random.Generator, hops: int, chains: int, statements: int, rank: int) -> np.ndarray:
    """Line labels of an "equal" item (0 = the asked chain, 1.. = distractor chains, -1 = filler).

    The item is a region followed by a final segment of DEPTH_FINAL lines, each in a uniformly
    random order: the region holds the first `hops` lines of every chain (its literal and all copies
    but the last) and filler; the final segment holds every chain's last line and filler. So the
    chains are exchangeable in both parts: a statistic of the layout (which literal comes last,
    which literal is copied first, whose last line comes last) singles out the asked chain only by
    chance; every literal precedes the asked variable for hops >= 1; the asked variable's line is
    uniform over the final segment at every h; and the only chain lines after it are other chains'
    last lines (ruling a chain out from there takes as many lookups as the task). The asked literal
    is the rank-th literal line (rejection); with the rank cycled, every strategy that ignores the
    asked name scores 1 / chains (exactly for position-of-literal strategies, in expectation for
    the others)."""
    n = hops + 1
    final = DEPTH_FINAL
    region_fill = statements - chains * n - (final - chains)
    assert region_fill >= 0, f"depth: {statements} statements cannot hold {chains} chains of {n} lines"
    segment = np.array(list(range(chains)) + [-1] * (final - chains), dtype=np.int64)
    region = np.array([c for c in range(chains) for _ in range(n - 1)] + [-1] * region_fill, dtype=np.int64)
    for _ in range(100000):
        labels = np.concatenate([rng.permutation(region), rng.permutation(segment)])
        firsts = [int(np.argmax(labels == c)) for c in range(chains)]
        if sorted(firsts).index(firsts[0]) == rank:
            return labels


def depth_item(hops: int, index: int, fmt: str = DEPTH_FORMAT, chains: int = DEPTH_CHAINS,
               statements: int | None = None, layout: str = DEPTH_LAYOUT) -> TextItem:
    """`chains` chains of assignments `v0 = <digit>`, `v1 = v0`, ..., interleaved at random (each
    chain in order). The asked chain has `hops` copies after its literal (hops = 0: the asked
    variable is assigned the digit itself). Literal digits are distinct across chains, so a
    distractor's value is always a wrong answer, and the asked literal is the (index mod chains)-th
    literal line of the item, so position heuristics ("the first / last digit assigned") score
    1 / chains in every hop cell.

    layout "equal" (default; `statements` defaults to DEPTH_STATEMENTS): every chain has hops + 1
    lines and filler statements (`v = None`, `v = []`, ..., or a copy of an earlier filler variable;
    never a digit) fill the item to `statements` lines, so no chain is shorter or longer than the
    asked one. The first h lines of every chain are interleaved uniformly with filler (a region in
    which the chains are exchangeable); every chain's last line is in the final DEPTH_FINAL lines,
    also in uniform order, so the asked variable's line is uniform over them at every h
    (`_equal_layout`). So "the literal nearest above the asked variable" is the last literal for
    h >= 1 (1 / chains by the rank balance), and any strategy that does not follow the asked
    variable's chain scores 1 / chains in expectation.

    layout "free" (the calibration round; `statements` defaults to FREE_DEPTH_STATEMENTS): the
    distractor chains fill the item (the first build, 4 chains in 24 lines: 2..9 lines each;
    otherwise any positive sizes, uniform over the compositions). Kept to rebuild those sets."""
    assert layout in DEPTH_LAYOUTS and fmt in DEPTH_FORMATS, (layout, fmt)
    if statements is None:
        statements = DEPTH_STATEMENTS if layout == "equal" else FREE_DEPTH_STATEMENTS
    legacy = (chains, statements) == LEGACY_DEPTH
    assert not (legacy and layout == "equal"), "the first build's shape is a free layout"
    rank = index % chains
    meta: dict = {}
    if layout == "free":
        rng = rng_for("depth", (hops,) if legacy else (hops, chains, statements), index, 0)
        rest = statements - hops - 1
        sizes = [hops + 1] + (_chain_sizes(rng, rest, chains - 1) if legacy else _composition(rng, rest, chains - 1))
        literals = [int(x) for x in rng.choice(10, size=chains, replace=False)]
        names = [VAR_NAMES[i] for i in rng.choice(len(VAR_NAMES), size=statements, replace=False)]
        chain_list = _chains(names, sizes, literals)
        labels = np.repeat(np.arange(chains), sizes)
        for _ in range(10000):
            rng.shuffle(labels)
            firsts = [int(np.argmax(labels == c)) for c in range(chains)]
            if sorted(firsts).index(firsts[0]) == rank:
                break
        else:
            raise RuntimeError("depth: could not place the asked literal")
        filler_names: list[str] = []
        tag = "" if legacy else f"c={chains}/s={statements}/"
    else:
        rng = rng_for("depth", (hops, chains, statements, 1), index, 0)
        sizes = [hops + 1] * chains
        literals = [int(x) for x in rng.choice(10, size=chains, replace=False)]
        names = [VAR_NAMES[i] for i in rng.choice(len(VAR_NAMES), size=statements, replace=False)]
        chain_list = _chains(names, sizes, literals)
        filler_names = names[sum(sizes):]
        labels = _equal_layout(rng, hops, chains, statements, rank)
        tail = len(labels) - 1 - int(np.flatnonzero(labels == 0)[-1])
        meta = {"layout": "equal", "tail": tail, "fillers": len(filler_names), "final": DEPTH_FINAL}
        tag = f"eq/c={chains}/s={statements}/"
    pos = [0] * chains
    lines, fillers = [], []
    for c in labels:
        if c < 0:  # a filler: an atom, or (half the time) a copy of an earlier filler variable
            v = filler_names[len(fillers)]
            if fillers and rng.random() < 0.5:
                rhs = fillers[int(rng.integers(len(fillers)))]
            else:
                rhs = FILLER_ATOMS[int(rng.integers(len(FILLER_ATOMS)))]
            fillers.append(v)
            lines.append(f"{v} = {rhs}")
        else:
            lines.append(chain_list[c][1][pos[c]])
            pos[c] += 1
    asked_var = chain_list[0][0][-1]
    # "body": `def main():`, the statements indented, `    assert x == ` then the digit; "comment": a
    # script, `print(x)  # prints ` then the digit; "repl": an interactive session, `>>> print(x)`
    # then the digit on the next line; "assert": `assert x == ` then the digit.
    if fmt == "body":
        context = "def main():\n" + "".join(f"    {s}\n" for s in lines) + f"    assert {asked_var} == "
    elif fmt == "repl":
        context = "".join(f">>> {s}\n" for s in lines) + f">>> print({asked_var})\n"
    elif fmt == "assert":
        context = "".join(f"{s}\n" for s in lines) + f"assert {asked_var} == "
    else:
        context = "".join(f"{s}\n" for s in lines) + f"print({asked_var})  # prints "
    gold = literals[0]
    return TextItem(
        id=f"depth/{tag}hops={hops}/{index:04d}", probe="depth", cell={"hops": hops, "chains": chains},
        group={"literal_rank": rank}, context=context, answer=DIGITS[gold], candidates=list(DIGITS), gold=gold,
        meta={"asked_var": asked_var, "literals": literals, "chain_sizes": sizes, "distractor_digits": literals[1:],
              "format": fmt, "chains": chains, "statements": statements, **meta},
    )


STATEMENT_RE = re.compile(r"^(?:>>> |    )?(\w+) = (\S+)$", flags=re.M)


def depth_statements(context: str) -> list[tuple[str, str]]:
    """(left-hand side, right-hand side) of every assignment line of a depth item, in order."""
    return STATEMENT_RE.findall(context)


def depth_heuristics(item: TextItem) -> dict:
    """What simple non-binding strategies answer on a depth item (the digit each one picks), and
    where the asked variable sits. Used to check that every cue scores 1 / chains in every hop cell:
    - first_literal / last_literal: the first / last digit assigned;
    - shortest_chain: the literal of the shortest chain (ties: the first such literal line);
    - nearest_above: the last literal line strictly above the asked variable's own line (h >= 1);
    - asked_line_from_end: statements after the asked variable's line (0 = the last before the query);
    - chain_lines_after: chain statements after the asked variable's line (in the equal layout only
      other chains' last lines: ruling a chain out from there takes as many lookups as the task);
    - first_read_last: the literal whose variable is first copied last (with the asked chain's
      first copy forced after every literal, as in an earlier draft, this scored 0.57-0.66 at h = 2-3);
    - last_read_last: the literal of the chain whose last line comes last (needs h lookups)."""
    stmts = depth_statements(item.context)
    lhs = [a for a, _ in stmts]
    lit_lines = [i for i, (_, b) in enumerate(stmts) if b.isdigit()]
    lit_vals = [int(stmts[i][1]) for i in lit_lines]
    sizes = item.meta["chain_sizes"]
    lits = item.meta["literals"]
    m = min(sizes)
    tied = {lits[c] for c in range(len(sizes)) if sizes[c] == m}
    at = lhs.index(item.meta["asked_var"])
    above = [int(stmts[i][1]) for i in lit_lines if i < at]
    chain_vars = set(lhs[i] for i in lit_lines)
    for a, b in stmts:  # every variable that holds a digit (chain variables)
        if b in chain_vars:
            chain_vars.add(a)
    first_read = {}
    for i, (_, b) in enumerate(stmts):
        if b in lit_vars_of(stmts) and b not in first_read:
            first_read[b] = i
    lit_of = {a: int(b) for a, b in stmts if b.isdigit()}
    return {"first_literal": lit_vals[0], "last_literal": lit_vals[-1],
            "shortest_chain": next(v for v in lit_vals if v in tied),
            "nearest_above": above[-1] if above else None, "asked_line_from_end": len(stmts) - 1 - at,
            "chain_lines_after": sum(1 for a in lhs[at + 1 :] if a in chain_vars),
            "first_read_last": lit_of[max(first_read, key=first_read.get)] if first_read else None,
            "last_read_last": _resolve(stmts, [a for a in lhs if a in chain_vars and a not in {b for _, b in stmts}][-1])}


def _resolve(stmts: list[tuple[str, str]], var: str):
    """The value a variable holds after the statements (a digit, or None for filler)."""
    rhs = dict(stmts)
    while var in rhs and not rhs[var].isdigit():
        var = rhs[var]
    return int(rhs[var]) if var in rhs and rhs[var].isdigit() else None


def lit_vars_of(stmts: list[tuple[str, str]]) -> set[str]:
    return {a for a, b in stmts if b.isdigit()}


# --------------------------------------------------------------------------------------------
# item checks (also run by the build tool on every item)
# --------------------------------------------------------------------------------------------


def query_start(item: TextItem) -> int:
    """Character offset in the context where the query begins."""
    if item.probe in ("uuid_needle", "multikey"):
        return item.context.rindex(SEP + "What is the access code")
    if item.probe == "path_copy":
        return item.context.rindex("\n# Lookup\n")
    return max(item.context.rfind(">>> print("), item.context.rfind("\nprint("), item.context.rfind("\nassert "),
               item.context.rfind("\n    assert "))


NEEDLE_QUESTION_OPENING = "What is the access code"  # needle_query's first words, after the separator


def needle_question_text(context: str) -> str:
    """The question of a uuid_needle or multikey item, as the model reads it right before the answer: the context from
    the question's first word to its end, the answer's lead-in included ("What is the access code for N? The access
    code for N is"). The separator before it (SEP) is not part of it. This is the item's "last turn" for the replay
    window (ced/eval/replay_window.py; ced/eval/probe_turn.py), as probe_gen_multifact.question_text is for a
    multi-fact item."""
    return context[context.rindex(SEP + NEEDLE_QUESTION_OPENING) + len(SEP):]


def check_item(item: TextItem) -> None:
    """Structural rules every item must satisfy (raises AssertionError)."""
    ctx, ans = item.context, item.answer
    q = query_start(item)
    query = ctx[q:]
    assert ans, item.id
    if item.probe in ("uuid_needle", "multikey"):
        value = ans.strip()
        assert UUID_RE.fullmatch(value), item.id
        assert ctx.count(value) == 1, f"{item.id}: answer not present exactly once"
        assert value not in query, f"{item.id}: answer leaks into the query"
        assert len(UUID_RE.findall(ctx)) == (1 if item.probe == "uuid_needle" else item.cell["n_keys"]), item.id
        names = [item.meta["name"]] if item.probe == "uuid_needle" else item.meta["names"]
        for nm in names:
            assert ctx[:q].count(nm) == 1, f"{item.id}: key {nm!r} not present exactly once"
        asked = item.meta["name"] if item.probe == "uuid_needle" else names[item.meta["asked"]]
        assert ctx[:q].count(needle_sentence(asked, value)) == 1, item.id
        assert query.endswith(f"The access code for {asked} is"), item.id
        if item.probe == "multikey":
            assert len(set(item.meta["values"])) == len(item.meta["values"]), item.id
            assert len({n.split()[0] for n in names}) == len(names) and len({n.split()[1] for n in names}) == len(names)
    elif item.probe == "path_copy":
        path = ans.strip()
        assert ctx.count(path) == 1, f"{item.id}: path not present exactly once"
        assert path not in query, item.id
        cls = item.meta["asked_class"]
        assert ctx[:q].count(f"class {cls} -> ") == 1, item.id
        assert query.endswith(f"class {cls} ->" if item.meta["format"] == "arrow" else f"defines class {cls} is"), item.id
        assert len(set(item.meta["paths"])) == item.cell["k"] == len(set(item.meta["classes"])), item.id
    elif item.probe == "depth":
        assert item.candidates == DIGITS and item.answer == DIGITS[item.gold], item.id
        assert not any(ch.isdigit() for ch in query), f"{item.id}: digit in the query"
        lits = item.meta["literals"]
        assert len(set(lits)) == len(lits) and lits[0] == item.gold, item.id
        stmts = depth_statements(ctx[:q])
        assert len(re.findall(r" = ", ctx[:q])) == len(stmts), f"{item.id}: an assignment line of another form"
        assigned = [a for a, _ in stmts]
        assert len(assigned) == len(set(assigned)) == item.meta["statements"], item.id
        assert len(lits) == item.meta["chains"] == sum(1 for _, b in stmts if b.isdigit()), item.id
        # resolve every variable by following the assignments: the asked one must hold the gold digit
        # after exactly `hops` copies; filler variables never hold a digit
        env, src = {}, {}
        for lhs, rhs in stmts:
            if rhs.isdigit():
                env[lhs], src[lhs] = int(rhs), 0
            elif rhs in FILLER_ATOMS:
                env[lhs], src[lhs] = None, None
            else:
                assert rhs in env, f"{item.id}: {rhs} used before assignment"
                env[lhs], src[lhs] = env[rhs], None if src[rhs] is None else src[rhs] + 1
        var = item.meta["asked_var"]
        assert env[var] == item.gold and src[var] == item.cell["hops"], item.id
        assert sum(1 for v in env.values() if v is not None) == sum(item.meta["chain_sizes"]), item.id
        assert query.rstrip().endswith((f"print({var})  # prints", f"print({var})", f"assert {var} ==")), item.id
        if item.meta.get("layout") == "equal":
            assert len(set(item.meta["chain_sizes"])) == 1, f"{item.id}: chains of different lengths"
            h = item.cell["hops"]
            assert sum(item.meta["chain_sizes"]) + item.meta["fillers"] == len(stmts), item.id
            at = assigned.index(var)
            assert len(stmts) - 1 - at == item.meta["tail"], f"{item.id}: tail"
            # every chain's last line (a chain variable nobody reads) is in the final segment, and
            # every other chain line before it
            read = {b for _, b in stmts}
            final = len(stmts) - item.meta["final"]
            for i, (a, _) in enumerate(stmts):
                if env[a] is not None:
                    assert (a not in read) == (i >= final), f"{item.id}: chain line {a} on the wrong side of the final segment"
    else:
        raise ValueError(item.probe)


def check_boundary_free(context: str, answer: str) -> bool:
    """The context ends and the answer starts on characters our split pattern never joins: the
    answer starts with a space after a letter or a ">" (a letter run or punctuation run ends there),
    or the context ends with a newline, or the answer is a digit (digits are always split alone)."""
    return ((answer.startswith(" ") and (context[-1:].isalpha() or context.endswith(">"))) or context.endswith("\n")
            or (answer.isdigit() and len(answer) == 1))
