"""A RULER subset within the training length, with an open haystack (RULER, Hsieh et al. 2024, https://github.com/NVIDIA/RULER, Apache-2.0):
multi-key, multi-value and multi-query needle-in-a-haystack and variable tracking, at 2,048 / 4,096 / 8,192 tokens of the project tokenizer, scored by RULER's
own string-match recall on greedy generations.

ATTRIBUTION. This file adapts the task templates, the needle and noise haystack generators, the variable-tracking chains and the string-match scoring of RULER
(https://github.com/NVIDIA/RULER at commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a; Copyright (c) 2024, NVIDIA CORPORATION; licensed under the Apache License, Version 2.0,
https://www.apache.org/licenses/LICENSE-2.0) and was modified: it is a re-implementation with numpy and this project's helpers (no file of RULER is copied), and its
changes from RULER are listed under DEVIATIONS below. The Apache License 2.0 text is in LICENSES/Apache-2.0.txt; see also THIRD_PARTY_NOTICES.md.

This module is pure text and arithmetic (numpy, no torch); the model-facing code is scripts/recall_ruler_score.py and ced/eval/recall_ruler_gen.py, the builder is
tools/build_recall_ruler.py.

THE NAME OF THE VARIANT. "RULER subset, open haystack". It is NOT the essay version: RULER's essay haystack is Paul Graham's essays (scraped from
paulgraham.com and the gkamradt/LLMTest_NeedleInAHaystack repository; no open licence), and this project uses open data only. The three tasks whose haystack
is "essay" in RULER's scripts/synthetic.yaml (niah_multikey_1, niah_multivalue, niah_multiquery) take instead whole held-out documents
(the web and general_web eval sets, the pool of the pcr evals, ced/eval/report_pcr_gen.py): never in a training shard, deduplicated against training at build.
The tasks with RULER's "needle" haystack (niah_multikey_2, niah_multikey_3) and "noise" haystack (vt) are generated exactly as RULER's scripts do. The numbers are
therefore not comparable with published RULER numbers of the essay version; a task's recall at a length is comparable between models and seeds.

THE SIX TASKS (RULER scripts/synthetic.yaml at commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a):
    task              haystack (RULER)  needle keys  values   keys  values/key  queried keys  tokens to generate
    niah_multikey_1   essay -> open     words        numbers   4        1            1             128
    niah_multikey_2   needle            words        numbers   1        1            1             128
    niah_multikey_3   needle            uuids        uuids     1        1            1             128
    niah_multivalue   essay -> open     words        numbers   1        4            1             128
    niah_multiquery   essay -> open     words        numbers   4        1            4             128   (RULER: num_needle_k = max(num_needle_k, num_needle_q))
    vt                noise             (variables)  numbers   1 chain of 4 hops (5 variables)                                30

PROMPT. RULER's "base" model template is the task template followed by the answer prefix (scripts/data/template.py, scripts/data/synthetic/constants.py); the
prompt is the whole string and the model continues after the answer prefix. For a single answer (queried keys x values per key == 1) niah.py edits the template
text ("Some" -> "A", "are all" -> "is", "are" -> "is", the value type loses its "s"); the needle sentence keeps the plural type. DEVIATIONS from the scripts, each
a design choice of this build: (1) the open haystack; (2) the word pool of the keys is this module's own adjective and noun lists (RULER's
is the wonderwords package's), keys are "adjective-noun", and every key in an item is distinct from every other (RULER draws at random and does not check);
(3) the variable-tracking answer prefix has no trailing space (a byte-level BPE attaches the space to the next word: the answer then starts with " ABCDE", as it does
after the in-context example); (4) the needle positions are drawn as fractions and mapped to the unit count of the item being fitted (RULER draws indices of the
final unit count), and the open haystack inserts a needle at the sentence start nearest to its depth (probe_gen.insert_sentences) where RULER cuts the essay
into nltk sentences; the depths are RULER's grid (40 values from 0 to 100) sampled without replacement; (5) the length of an item is the token count of the
prompt plus the tokens to generate, the largest that is at most T: within LENGTH_TOL of T for the open haystack (cut at a whitespace) and within UNIT_TOL for the
needle and noise haystacks (a whole sentence is the step, as RULER's binary search over the haystack size ends within one unit of the maximum).

SCORING (RULER scripts/eval/synthetic/constants.py): string_match_all = the share of the reference strings that occur, lower-cased, in the generated text after
postprocess_pred (strip, control characters to newlines, strip). niah and vt both use it. The recall of an item is the share found: in {0, 1/4, ..., 1} for the four-reference
tasks (multivalue, multiquery), {0, 1/5, ..., 1} for vt (five variables) and {0, 1} for the others; a cell's value is the mean over its items. "all_found" (every reference present) is printed beside for the multi-reference tasks.

IDS. An item is a pure function of (BASE_SEED, task, length, index) and the pool; its id is "<task>/len=<T>/<index:04d>"; the string a flagging list uses is
"<task>:len=<T>/<index:04d>" (item_key).

VALUES FILE (the common output of the three recall builds): <out>/<run>.values.json = {"run", "family": "ruler", "code_sha", "cells", "n_items", "excluded"}; cells
"ruler|<task>|<length>|recall" (the headline cells) and the same number as "ruler|<task>|<length>|string_match_all" (RULER's name for it: TWIN_METRIC),
"ruler|<task>|<length>|all_found" (multi-reference tasks), "ruler|<task>|all|recall" (the three lengths pooled by item) and "ruler|mean|<length>|recall" (the six
tasks' cell values averaged), all in [0, 1]. "excluded" lists every flagged id given, also those of items this run did not score.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
import uuid as uuidlib
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ced.eval import probe_gen as g
from ced.eval import report_pcr_gen as pg

BASE_SEED = 20261102
LENGTHS = (2048, 4096, 8192)
TASKS = ("niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt")
TASK_CODE = {t: i for i, t in enumerate(TASKS)}
PER_CELL = 500  # RULER's default (NUM_SAMPLES=500 in its task configuration)
LENGTH_TOL = 0.05  # the prompt plus the tokens to generate lie in [(1 - tol) T, T] (the open haystack, cut at a whitespace: a word's resolution)
UNIT_TOL = 0.10  # the same for a haystack of units (the needle sentences, the noise sentences): RULER's resolution is one unit, so the item ends within one
# unit of T whatever the unit's size (a uuid needle sentence is 50 to 100 tokens); the 10 % is a sanity bound, the realised lengths are in the build's manifest
NIAH_NEW = 128  # RULER constants.py 'niah' tokens_to_generate
VT_NEW = 30  # RULER constants.py 'variable_tracking' tokens_to_generate
ICL_TOKENS = 500  # RULER's variable-tracking in-context example: max_seq_length=500 (variable_tracking.py main)
REDRAW_MAX = 20
REDRAW_STREAM = 1000
STREAM_CONTENT, STREAM_HAY, STREAM_ICL = 0, 1, 2
# The headline metric is written under two names: "recall", and "string_match_all", RULER's own name for the same number
# (scripts/eval/synthetic/constants.py). The twin has the same value and n_items.
TWIN_METRIC = "string_match_all"
DEPTH_GRID = tuple(int(x) for x in np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))  # RULER: DEPTHS

# --------------------------------------------------------------------------------------------
# RULER's texts (scripts/data/synthetic/constants.py TASKS['niah'], ['variable_tracking']; niah.py needle; variable_tracking.py haystack)

NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
NIAH_TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the "
                 "{type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?")
NIAH_ANSWER_PREFIX = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
VT_TEMPLATE = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are "
               "assigned the value {query} in the text above.")
VT_ANSWER_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, "
                    "they are:")  # RULER's ends with ": " (a trailing space); deviation (3)
VT_NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

# the six tasks of scripts/synthetic.yaml (the "essay" haystack is replaced by "open": held-out documents)
SPEC = {
    "niah_multikey_1": dict(kind="niah", ruler_haystack="essay", haystack="open", type_k="words", type_v="numbers", n_k=4, n_v=1, n_q=1),
    "niah_multikey_2": dict(kind="niah", ruler_haystack="needle", haystack="needle", type_k="words", type_v="numbers", n_k=1, n_v=1, n_q=1),
    "niah_multikey_3": dict(kind="niah", ruler_haystack="needle", haystack="needle", type_k="uuids", type_v="uuids", n_k=1, n_v=1, n_q=1),
    "niah_multivalue": dict(kind="niah", ruler_haystack="essay", haystack="open", type_k="words", type_v="numbers", n_k=1, n_v=4, n_q=1),
    "niah_multiquery": dict(kind="niah", ruler_haystack="essay", haystack="open", type_k="words", type_v="numbers", n_k=1, n_v=1, n_q=4),
    "vt": dict(kind="vt", ruler_haystack="noise", haystack="noise", n_chains=1, n_hops=4),
}
MULTI_REF = tuple(t for t in TASKS if t in ("niah_multivalue", "niah_multiquery", "vt"))  # tasks with more than one reference string
OPEN_TASKS = tuple(t for t in TASKS if SPEC[t].get("haystack") == "open")

# the key words of the 'words' needle type: "adjective-noun" (RULER: wonderwords' lists; here this module's own, about 60,000 pairs)
ADJECTIVES = sorted(set("""able abrupt absent active adept airy alert amber ample ancient angular antique arctic ardent artful ashen astute atomic avid azure balmy
bare basic bitter blunt bold brass breezy brief bright brisk broad bronze brown bulky burly busy calm candid casual cheap chilly civic clean clear clever close cloudy
coarse cold common compact cool coral cosmic cozy crisp cruel curious curly dainty damp dapper dark decent deep dense dim distant dizzy dry dull dusky eager early
earthy easy elastic eerie elegant empty endless even exact faint fair false famous fancy fast fierce firm fit flat fluffy fond formal fragile frank fresh frosty
gentle giant gleaming glossy golden grand gray green grim hardy harsh hasty heavy hollow honest humble icy idle jolly keen kind large late lazy lean light lively
lofty lone loud lucky lunar mellow merry mild minor misty modest narrow neat noble noisy odd pale plain polite proud pure quick quiet rapid rare raw rich rigid ripe
rough round royal rustic sandy sharp shiny short shy silent silky silver simple sleek slim slow small smooth snowy soft solar solid sour spare steady steep stiff
stormy sturdy sunny swift tame tidy tiny tough vague vast vivid warm weary wide wild windy wise woolen young zesty""".split()))
NOUNS = sorted(set("""abbey acorn alley amulet anchor apple apron arrow atlas attic avenue badge bagpipe bakery balloon bamboo banjo barn barrel basket beacon beaker
bench biscuit blanket bonnet bottle bouquet boulder bowl bracket branch bridge brook brush bucket button cabin cable cactus camel candle canal canoe canyon cape
carpet carrot castle cedar cellar chalk chapel chariot cherry chest chimney cliff cloak clock clover cloud cobweb comet compass copper cottage cradle crane crown
crystal cushion dagger daisy delta desert dinner dolphin dragon drum dune dusk eagle eclipse elm ember engine fabric falcon feather fence ferry fiddle finch fjord
flag flame flask flint flute foam forest fossil fountain fox frost gable garden garlic gate gem glacier glove goat goblet granite grove guitar hammer harbor harp
haven hawk hedge helmet heron hickory hill hinge hive horizon island ivory jacket jar jewel jungle kayak kettle kitchen kite knot ladder lagoon lake lamp lantern lark
lava leaf ledger lemon lighthouse lilac linen lizard locket loom lute magnet mallet mantle maple marble mast meadow melon mill mirror mitten moon mosaic moss moth
mountain mushroom needle nest nickel oak oasis ocean olive onion orbit orchard otter owl paddle paddock palace panther parcel peak pearl pebble pencil pepper pier
pillar pillow pine plank planet plateau plow plume pocket pond puzzle pylon quarry quilt quiver rabbit radar rafter rain raven reef ribbon ridge river robin rocket
root rudder saddle sail salmon sand sandal satchel scroll shadow shelf shell shore shovel signal silo sketch spade sparrow spider spindle spire spring spruce
squirrel stable star statue stone storm stream summit sunrise swallow swan tablet tapestry teapot thimble thistle thorn thunder tide timber tower trail trellis
trumpet tulip tunnel turnip umbrella valley vault velvet vessel vine violet violin wagon walnut wave well wharf wheat wheel whistle willow window winter wizard wolf
yacht yarn zephyr""".split()))
N_KEY_PAIRS = len(ADJECTIVES) * len(NOUNS)

Count = Callable[[str], int]  # text -> tokens under the tokenizer (the length measure)


class NoFitError(RuntimeError):
    """No haystack size gives an item with a length in the window: the draw is redrawn from the item's own seeded sequence (as pcr's NoCutError)."""


# --------------------------------------------------------------------------------------------
# items


@dataclass
class RulerItem:
    id: str  # "<task>/len=<T>/<index:04d>"
    task: str
    length: int  # the nominal length T (prompt tokens + tokens to generate lie in [(1 - tol) T, T])
    index: int
    prompt: str  # the whole prompt: ends with the answer prefix; the model continues from there
    refs: list  # the reference strings (RULER "outputs")
    max_new: int  # tokens to generate
    n_prompt_tokens: int  # the length measure of the prompt (the builder's tokenizer)
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return item_key(self.task, self.length, self.index)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "RulerItem":
        return cls(**d)


def item_id(task: str, length: int, index: int) -> str:
    return f"{task}/len={length}/{index:04d}"


def item_key(task: str, length: int, index: int) -> str:
    """The "task:id" string of an exclusion list: the task and the id inside the task."""
    return f"{task}:len={length}/{index:04d}"


def text_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def rng_for(task: str, length: int, index: int, stream: int) -> np.random.Generator:
    """An independent stream per (task, length, item, purpose). 0: content; 1: the haystack's document order; 2: the in-context example."""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, TASK_CODE[task], int(length), int(index), int(stream)]))


def hay_rng_for(task: str, length: int, index: int, draw: int = 0) -> np.random.Generator:
    return rng_for(task, length, index, STREAM_HAY if draw == 0 else REDRAW_STREAM + draw)


# --------------------------------------------------------------------------------------------
# random needle material


def random_number(rng: np.random.Generator, digits: int = 7) -> str:
    """RULER generate_random_number: a number with exactly `digits` digits."""
    return str(int(rng.integers(10 ** (digits - 1), 10 ** digits)))


def random_uuid(rng: np.random.Generator) -> str:
    return str(uuidlib.UUID(bytes=rng.bytes(16), version=4))


def key_word(adj: int, noun: int) -> str:
    return f"{ADJECTIVES[adj]}-{NOUNS[noun]}"


def draw_keys(rng: np.random.Generator, kind: str, n: int, avoid_text: str = "", exclude: set | None = None) -> list[str]:
    """n distinct keys of the type ('words': adjective-noun; 'uuids'; 'numbers') none of which occurs in avoid_text (lower-cased comparison)
    or in `exclude`."""
    out: list[str] = []
    seen = set(exclude or ())
    low = avoid_text.lower()
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 10000 + 100 * n:
            raise RuntimeError("cannot draw distinct keys")
        if kind == "words":
            k = key_word(int(rng.integers(len(ADJECTIVES))), int(rng.integers(len(NOUNS))))
        elif kind == "uuids":
            k = random_uuid(rng)
        elif kind == "numbers":
            k = random_number(rng)
        else:
            raise ValueError(kind)
        if k in seen or (low and k.lower() in low):
            continue
        seen.add(k)
        out.append(k)
    return out


def draw_values(rng: np.random.Generator, kind: str, n: int, avoid_text: str = "", exclude: set | None = None) -> list[str]:
    """n distinct values of the type that occur nowhere in avoid_text (a number of the haystack's would make the reference ambiguous)."""
    return draw_keys(rng, kind, n, avoid_text, exclude)


# --------------------------------------------------------------------------------------------
# prompts


def render_niah(task: str, context: str, query: str) -> str:
    """RULER niah.py: template + answer prefix with the single-answer edits; the needle sentence is in the context already."""
    s = SPEC[task]
    template = NIAH_TEMPLATE + NIAH_ANSWER_PREFIX
    type_v = s["type_v"]
    if s["n_q"] * s["n_v"] == 1:
        template = template.replace("Some", "A")
        template = template.replace("are all", "is")
        template = template.replace("are", "is")
        template = template.replace("answers", "answer")
        type_v = type_v[:-1]
    return template.format(type_needle_v=type_v, context=context, query=query)


def render_vt(context: str, value: str, num_v: int) -> str:
    return (VT_TEMPLATE + VT_ANSWER_PREFIX).format(context=context, query=value, num_v=num_v)


def query_string(keys: list[str]) -> str:
    """RULER: ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]."""
    return ", ".join(keys[:-1]) + ", and " + keys[-1] if len(keys) > 1 else keys[0]


# --------------------------------------------------------------------------------------------
# fitting an item to its length


def fit_units(make: Callable[[int], str], count: Count, total: int, n_lo: int, n_hi: int, tol: float = UNIT_TOL) -> tuple[str, int, int]:
    """The largest unit count n in [n_lo, n_hi] whose text has count(make(n)) <= total. Returns (text, count, n). The count grows with n, so the search
    ends on neighbours (n fits, n + 1 does not). RuntimeError if n_hi is too small or n_lo alone does not fit; NoFitError if the fit is below
    (1 - tol) total."""
    if count(make(n_hi)) <= total:
        raise RuntimeError(f"n_hi = {n_hi} units still fit in {total} tokens")
    if count(make(n_lo)) > total:
        raise RuntimeError(f"{n_lo} units (the needles, the template) alone exceed {total} tokens")
    lo, hi = n_lo, n_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(make(mid)) <= total:
            lo = mid
        else:
            hi = mid
    text = make(lo)
    n = count(text)
    if n < math.ceil((1 - tol) * total):
        raise NoFitError(f"no unit count in [{math.ceil((1 - tol) * total)}, {total}] tokens: the best fits {n}")
    return text, n, lo


def content_rng_for(task: str, length: int, index: int, draw: int = 0) -> np.random.Generator:
    """The content stream of draw `draw`. The open-haystack tasks and vt keep the content of draw 0 in every draw (the redraw moves the haystack); the
    needle-haystack tasks have no haystack to move, so their redraw takes a content stream of its own."""
    return rng_for(task, length, index, STREAM_CONTENT if draw == 0 else 2 * REDRAW_STREAM + draw)


def draw_until_fit(make_item: Callable[[np.random.Generator, int], RulerItem], task: str, length: int, index: int) -> RulerItem:
    """make_item(haystack rng, draw) with the stream of draw 0, then of draw 1, ... until a draw fits (NoFitError, or pcr's NoCutError, is the only error
    that moves on to the next draw). The failed draws are recorded in meta["redraws"]; none, no key."""
    failed: list[str] = []
    for draw in range(REDRAW_MAX + 1):
        try:
            item = make_item(hay_rng_for(task, length, index, draw), draw)
        except (NoFitError, pg.NoCutError) as e:
            failed.append(str(e))
            continue
        if failed:
            item.meta["redraws"] = {"count": len(failed), "failed_draws": failed}
        return item
    raise RuntimeError(f"{task} len={length} item {index}: no fit in {REDRAW_MAX + 1} draws: {failed}")


# --------------------------------------------------------------------------------------------
# needle-in-a-haystack


def _needles(content: np.random.Generator, task: str, avoid_text: str):
    """RULER niah.py generate_input_output, the needle part: keys, values per key, the needle sentences (shuffled), the queried keys and the references."""
    s = SPEC[task]
    n_keys = max(s["n_k"], s["n_q"])
    keys = draw_keys(content, s["type_k"], n_keys, avoid_text)
    used = set(keys)
    values: list[list[str]] = []
    needles: list[str] = []
    for k in keys:
        vs = draw_values(content, s["type_v"], s["n_v"], avoid_text, used)
        used.update(vs)
        values.append(vs)
        needles.extend(NEEDLE.format(type_needle_v=s["type_v"], key=k, value=v) for v in vs)
    order = content.permutation(len(needles))  # RULER shuffles the needles
    needles = [needles[int(i)] for i in order]
    asked = [int(i) for i in content.choice(n_keys, size=s["n_q"], replace=False)]  # RULER: random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in asked]
    refs = [a for i in asked for a in values[i]]
    return keys, values, needles, queries, refs, used


def niah_open_item(task: str, length: int, index: int, count: Count, pool) -> RulerItem:
    """A needle task whose haystack is RULER's essay in the scripts: here whole held-out documents (module docstring), the needles at RULER's depths."""
    s = SPEC[task]

    def draw(hay_rng: np.random.Generator, draw_no: int) -> RulerItem:
        content = content_rng_for(task, length, index)
        text, chosen = pg.gather(pool, hay_rng, length)
        keys, values, needles, queries, refs, _ = _needles(content, task, text)
        depths = sorted(int(d) for d in content.choice(np.array(DEPTH_GRID), size=len(needles), replace=False))  # RULER: sorted(random.sample(DEPTHS, n))
        query = query_string(queries)
        state: dict = {}

        def make(haystack: str) -> tuple[str, str]:
            body, offsets = g.insert_sentences(haystack, [(d / 100.0, n) for d, n in zip(depths, needles)])
            state["offsets"] = offsets
            return render_niah(task, body, query), ""

        prompt, _, n_total, kept = pg.fit_cut(text, make, lambda c, a: count(c) + NIAH_NEW, length, tol=LENGTH_TOL)
        make(text[:kept].rstrip())
        return RulerItem(
            id=item_id(task, length, index), task=task, length=length, index=index, prompt=prompt, refs=refs, max_new=NIAH_NEW,
            n_prompt_tokens=n_total - NIAH_NEW,
            meta={"keys": keys, "values": values, "queries": queries, "depths_percent": depths, "char_offsets": state["offsets"], "haystack_chars": kept,
                  "haystack_docs": [pool.doc_ids[j] for j in chosen], "haystack": "open", "ruler_haystack": s["ruler_haystack"]})

    return draw_until_fit(draw, task, length, index)


def niah_needle_item(task: str, length: int, index: int, count: Count) -> RulerItem:
    """A needle task with RULER's 'needle' haystack: sentences of the needle format with random keys and values, one per line; the real needles are inserted."""
    s = SPEC[task]

    def draw(hay_rng: np.random.Generator, draw_no: int) -> RulerItem:  # no documents to order: a redraw takes the content stream of its draw number
        content = content_rng_for(task, length, index, draw_no)
        keys, values, needles, queries, refs, used = _needles(content, task, "")
        n_max = length // 6 + 8  # a distractor sentence is 20 tokens or more
        dist_keys = draw_keys(content, s["type_k"], n_max, "", used)
        dist_vals = draw_values(content, s["type_v"], n_max, "", used | set(dist_keys))
        master = [NEEDLE.format(type_needle_v=s["type_v"], key=k, value=v) for k, v in zip(dist_keys, dist_vals)]
        frac = [float(x) for x in content.random(size=len(needles))]
        query = query_string(queries)

        def make(n: int) -> str:
            sentences = master[:n]
            places = sorted((min(int(f * (n + 1)), n) for f in frac), reverse=True)  # RULER: sorted indices, reverse, zip with the needles
            for place, needle in zip(places, needles):
                sentences.insert(place, needle)
            return render_niah(task, "\n".join(sentences), query)

        prompt, n_total, n_units = fit_units(make, lambda t: count(t) + NIAH_NEW, length, 0, n_max)
        return RulerItem(
            id=item_id(task, length, index), task=task, length=length, index=index, prompt=prompt, refs=refs, max_new=NIAH_NEW,
            n_prompt_tokens=n_total - NIAH_NEW,
            meta={"keys": keys, "values": values, "queries": queries, "n_distractors": n_units, "needle_fractions": frac, "haystack": "needle",
                  "ruler_haystack": s["ruler_haystack"]})

    return draw_until_fit(draw, task, length, index)


# --------------------------------------------------------------------------------------------
# variable tracking


def _vt_names(rng: np.random.Generator, n: int, k: int = 5) -> list[str]:
    """RULER generate_chains: n distinct random k-letter upper-case names (none that contains 'VAR')."""
    out: list[str] = []
    while len(out) < n:
        name = "".join(string.ascii_uppercase[int(x)] for x in rng.integers(0, 26, size=k))
        if name not in out and "VAR" not in name:  # RULER allows it; a name that contains the keyword would make the statements ambiguous to count (the pcr evals exclude it too)
            out.append(name)
    return out


def _vt_chain(names: list[str], value: str) -> list[str]:
    """RULER generate_chains: ['VAR A = <value>', 'VAR B = VAR A ', ...] (each later statement keeps RULER's trailing space)."""
    return [f"VAR {names[0]} = {value}"] + [f"VAR {names[j + 1]} = VAR {names[j]} " for j in range(len(names) - 1)]


def _vt_context(chain: list[str], frac: list[float], n_noise: int) -> str:
    """RULER's noise haystack: n_noise copies of the noise sentence, the chain's statements inserted in order at sorted random positions, one per line."""
    sentences = [VT_NOISE] * n_noise
    positions = sorted(min(int(f * n_noise), max(n_noise - 1, 0)) for f in frac)
    for j, (pos, stmt) in enumerate(zip(positions, chain)):
        sentences.insert(pos + j, stmt)
    return "\n".join(sentences).replace(". \n", ".\n")


def vt_icl_text(rng: np.random.Generator, count: Count, tokens: int = ICL_TOKENS, n_hops: int = 4) -> tuple[str, str, list[str]]:
    """RULER's in-context example (variable_tracking.py main(): sys_vartrack_w_noise_random(max_seq_length=500, incremental=5) with 3-letter variable names,
    then `icl_example['input'] + ' ' + ' '.join(outputs) + '\\n'` with the names and the value drawn again by randomize_icl): a prompt of the same task whose
    noise is the largest that keeps it within `tokens`, its answer after it, and a newline. Returns (text, its value, its variable names)."""
    names = _vt_names(rng, n_hops + 1, 3)
    value = str(int(rng.integers(10000, 99999)))
    chain = _vt_chain(names, value)
    frac = sorted(float(x) for x in rng.random(size=len(chain)))

    def make(n: int) -> str:
        return render_vt(_vt_context(chain, frac, n), value, n_hops + 1)

    lo, hi = 1, tokens // 6 + 8
    if count(make(hi)) <= tokens:
        raise RuntimeError("the in-context example's noise bound is too small")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(make(mid)) <= tokens:
            lo = mid
        else:
            hi = mid
    return make(lo) + " " + " ".join(names) + "\n", value, names


def vt_item(length: int, index: int, count: Count, n_hops: int = 4) -> RulerItem:
    task = "vt"

    def draw(hay_rng: np.random.Generator, draw_no: int) -> RulerItem:
        content = content_rng_for(task, length, index, draw_no)
        names = _vt_names(content, n_hops + 1, 5)
        value = str(int(content.integers(10000, 99999)))  # np.random.randint(10000, 99999)
        chain = _vt_chain(names, value)
        frac = sorted(float(x) for x in content.random(size=len(chain)))
        for attempt in range(8):  # the example never shares the item's value (its names have 3 letters, the item's 5)
            icl, icl_value, icl_names = vt_icl_text(rng_for(task, length, index, STREAM_ICL + 10 * attempt), count)
            if icl_value != value:
                break
        else:
            raise RuntimeError("no in-context example with another value")
        prefix = icl + "\n"

        def make(n: int) -> str:
            return prefix + render_vt(_vt_context(chain, frac, n), value, n_hops + 1)

        prompt, n_total, n_units = fit_units(make, lambda t: count(t) + VT_NEW, length, 1, length // 6 + 8)
        return RulerItem(
            id=item_id(task, length, index), task=task, length=length, index=index, prompt=prompt, refs=list(names), max_new=VT_NEW,
            n_prompt_tokens=n_total - VT_NEW,
            meta={"chain": names, "value": value, "n_noise": n_units, "statement_fractions": frac, "icl_value": icl_value, "icl_names": icl_names,
                  "haystack": "noise", "ruler_haystack": "noise"})

    return draw_until_fit(draw, task, length, index)


# --------------------------------------------------------------------------------------------
# the items


def make_item(task: str, length: int, index: int, count: Count, pool=None) -> RulerItem:
    """One item. `pool` (a probe_gen.HaystackPool of the held-out documents) is needed by the three open-haystack tasks only."""
    if task not in SPEC:
        raise ValueError(f"task {task!r}: one of {TASKS}")
    if length < 256:
        raise ValueError(f"length {length}: the template and the in-context example alone take about 700 tokens")
    s = SPEC[task]
    if s["kind"] == "vt":
        return vt_item(length, index, count)
    if s["haystack"] == "open":
        if pool is None:
            raise ValueError(f"{task} needs the held-out document pool (the open haystack)")
        return niah_open_item(task, length, index, count, pool)
    return niah_needle_item(task, length, index, count)


def usable_doc(text: str) -> bool:
    """A document may be haystack unless it holds what the tasks plant or ask about: a UUID-shaped string, the phrase 'access code' (the pcr tasks'), or
    'special magic' (this family's needle sentence)."""
    return g.usable_haystack_doc(text) and "special magic" not in text.lower()


def check_item(item: RulerItem) -> None:
    """The structural rules every item must satisfy (AssertionError). The prompt is the string the model reads; the references are what is scored."""
    s = SPEC[item.task]
    p = item.prompt
    assert item.id == item_id(item.task, item.length, item.index) and item.max_new == (VT_NEW if s["kind"] == "vt" else NIAH_NEW), item.id
    assert item.refs and len(set(item.refs)) == len(item.refs), item.id
    assert item.n_prompt_tokens + item.max_new <= item.length, f"{item.id}: {item.n_prompt_tokens} + {item.max_new} tokens exceed {item.length}"
    tol = LENGTH_TOL if s.get("haystack") == "open" else UNIT_TOL
    assert item.n_prompt_tokens + item.max_new >= math.ceil((1 - tol) * item.length), f"{item.id}: below the length window"
    if s["kind"] == "niah":
        assert p.endswith(" mentioned in the provided text " + ("is" if s["n_q"] * s["n_v"] == 1 else "are")), item.id
        assert len(item.refs) == s["n_q"] * s["n_v"] and len(item.meta["queries"]) == s["n_q"], item.id
        body, tail = p.rsplit("\nWhat ", 1)
        for v in item.refs:
            assert body.count(v) == 1 and v not in tail, f"{item.id}: the reference {v!r} must occur exactly once, in a needle"
        needle_n = body.count("One of the special magic ")
        n_needles = max(s["n_k"], s["n_q"]) * s["n_v"]
        assert needle_n == n_needles + (item.meta["n_distractors"] if s["haystack"] == "needle" else 0), f"{item.id}: {needle_n} needle sentences"
        for k in item.meta["queries"]:
            assert tail.count(k) == 2, f"{item.id}: the queried key {k!r} must be in the question and in the answer prefix"
            assert body.count(f" for {k} is: ") == s["n_v"], f"{item.id}: key {k!r} must head {s['n_v']} needle(s)"
        assert len(set(item.meta["keys"])) == len(item.meta["keys"]), item.id
        # the pairing: the references are, in the order of the queried keys, exactly the values of the needle sentences that carry those keys (a reference that occurs
        # once in a needle of ANOTHER key would pass every check above while the question asks for a different key)
        want = []
        for k in item.meta["queries"]:
            vs = item.meta["values"][item.meta["keys"].index(k)]
            for v in vs:
                sentence = NEEDLE.format(type_needle_v=s["type_v"], key=k, value=v)
                assert body.count(sentence) == 1, f"{item.id}: the needle sentence of key {k!r} with its value {v!r} is not in the prompt exactly once"
                want.append(v)
        assert want == item.refs, f"{item.id}: the references are not the values of the queried keys"
    else:
        names, value = item.meta["chain"], item.meta["value"]
        assert p.endswith(f"they are:") and item.refs == names and len(names) == s["n_hops"] + 1, item.id
        q = p.rindex("Memorize and track the chain(s)")
        body, task_text = p[:q], p[q:]
        ctx = task_text.split("\n\n", 1)[1].rsplit("\nQuestion:", 1)[0]
        stmts = re.findall(r"VAR ([A-Z]{5}) = (VAR [A-Z]{5}|\d{5})", ctx)
        assert len(stmts) == len(names) and ctx.count("VAR") == 2 * len(names) - 1, f"{item.id}: a VAR statement of another form"
        env: dict = {}
        for lhs, rhs in stmts:
            assert lhs not in env, f"{item.id}: {lhs} assigned twice"
            if rhs.startswith("VAR "):
                assert rhs[4:] in env, f"{item.id}: {rhs[4:]} read before it is assigned"
                env[lhs] = env[rhs[4:]]
            else:
                env[lhs] = rhs
        assert [l for l, _ in stmts if env[l] == value] == names, f"{item.id}: the variables holding {value} are not the references"
        assert ctx.count(value) == 1 and task_text.count(value) == 3, f"{item.id}: the value {value} must occur once in the text and in the question and prefix"
        assert body.endswith("\n") and body.count("they are:") == 1 and item.meta["icl_value"] not in ctx and not (set(item.meta["icl_names"]) & set(names)), item.id


# --------------------------------------------------------------------------------------------
# scoring (RULER scripts/eval/synthetic/constants.py string_match_all; scripts/eval/evaluate.py postprocess_pred)


def postprocess_pred(text: str) -> str:
    """RULER evaluate.py postprocess_pred: strip, every control character (0x00-0x1f) to a newline, strip."""
    text = text.strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def recall(pred: str, refs: list[str]) -> float:
    """string_match_all of one item: the share of the references that occur, lower-cased, in the prediction."""
    low = postprocess_pred(pred).lower()
    return sum(1.0 if r.lower() in low else 0.0 for r in refs) / len(refs)


def all_found(pred: str, refs: list[str]) -> float:
    return 1.0 if recall(pred, refs) == 1.0 else 0.0


def cell_key(task: str, length, metric: str) -> str:
    return f"ruler|{task}|{length}|{metric}"


def binom_se(p: float, n: int) -> float:
    """The binomial standard error of a share p over n independent items."""
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n > 0 else float("nan")


def mean_se(values: list[float]) -> tuple[float, float]:
    """(mean, standard error of the mean) of item scores (n - 1 divisor)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1) / n)


def values_file(run: str, results: list[dict], code_sha: str = "local", excluded: list[str] | None = None) -> dict:
    """The common values file of one scored checkpoint. `results`: one dict per scored item with "task", "length", "recall" (0..1) and "key" (the item key);
    an item whose key is in `excluded` is left out of every cell. Cells as in the module docstring; n_items has the same keys."""
    drop = set(excluded or ())
    rows = [r for r in results if r["key"] not in drop]
    cells: dict = {}
    n_items: dict = {}
    per: dict = {}
    for r in rows:
        per.setdefault((r["task"], r["length"]), []).append(float(r["recall"]))
    for (task, length), vals in sorted(per.items()):
        k = cell_key(task, length, "recall")
        cells[k], n_items[k] = sum(vals) / len(vals), len(vals)
        k2 = cell_key(task, length, TWIN_METRIC)  # the same number under RULER's own name for the metric (see TWIN_METRIC)
        cells[k2], n_items[k2] = cells[k], n_items[k]
        if task in MULTI_REF:
            k2 = cell_key(task, length, "all_found")
            cells[k2], n_items[k2] = sum(1.0 for v in vals if v == 1.0) / len(vals), len(vals)
    for task in TASKS:
        vals = [v for (t, _), vs in sorted(per.items()) if t == task for v in vs]
        if vals:
            k = cell_key(task, "all", "recall")
            cells[k], n_items[k] = sum(vals) / len(vals), len(vals)
    for length in LENGTHS:
        cs = [cells[cell_key(t, length, "recall")] for t in TASKS if cell_key(t, length, "recall") in cells]
        if len(cs) == len(TASKS):
            k = cell_key("mean", length, "recall")
            cells[k], n_items[k] = sum(cs) / len(cs), min(n_items[cell_key(t, length, "recall")] for t in TASKS)
    return {"run": run, "family": "ruler", "code_sha": code_sha, "cells": cells, "n_items": n_items, "excluded": sorted(drop)}
