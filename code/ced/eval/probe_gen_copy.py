"""Generators for the two copying probes (copy_span and edit_old).

Both probes are conversations in the chat format (tools/data/chat_format.py): an item is the list of
(text, control) pieces of the context, the answer text, and for the edit probe the "tail" (see
below). Pieces are encoded exactly as the training-shard builder does: a control piece is one
special token, every text piece is tokenized on its own.
SYNTHETIC material only (small Python modules and records built from fixed word lists, as in
ced/eval/probe_gen_ext.py): no training data and no held-out document.

copy_span: reproduce a span that appeared earlier, exactly.
  The context is a short conversation: a source message that holds K = 4 labelled snippets of one
  type and length (the target and K - 1 look-alikes), then filler turns (an assistant `view` call and
  a tool result with a synthetic code file, or a `run` call with a log) until the target is the
  wanted distance back, then a user message that names the snippet by its label and asks for it
  verbatim. The answer is the target's text. In the current version the
  reply opens with a fixed, teacher-forced lead-in (LEADIN_COPY: a short sentence and a code fence, the
  place where a chat-format v1 assistant puts raw text) and the answer follows it inside the same text
  piece (a tail item with an empty suffix): the answer no longer opens a fresh assistant reply, whose first
  token a v1-trained model expects to be a tool call, <think> or prose. `leadin=None` builds the v1 item.
  Cells (cell dict): type {code, path, ident, random} x len {8, 32, 128} nominal tokens x dist
  {1024, 4096, 8192, 16384, 32768} x src {tool, assistant} x variant {plain, near_dup}.
    - dist 1024 / 4096: tokens from the end of the target span to the start of the answer, in
      [0.9 D, D] (the length measure of the build tool: the larger of the two tokenizers' counts).
    - dist 8192 / 16384 / 32768: the item (context + answer) is T = D tokens long at most and at
      least 0.96 T (the source sits at the start of the conversation); the distance back is then
      T minus about 0.6k to 1k. dist 8192 is the largest that fits a training row (8,192 tokens),
      16384 and 32768 are beyond the training length. The realised distance is stored per item.
    - src tool: the snippets are the content of a tool result; src assistant: of an earlier assistant
      message (the model's own turn).
    - near_dup (type code only): one look-alike is the target with one token changed in each of 1 / 2
      / 3 lines (len 8 / 32 / 128): the near-miss copy the failure analysis found. The scorer reports
      diff_acc / diff_nats at the differing tokens.
  Rules: a span starts with a non-space character and ends with a letter, digit or underscore (a
  closing bracket would merge with the newline after it in the source but not before <|im_end|>);
  the target occurs exactly once in the context; its label occurs next to it and in the question.
  Lead-in (v2): the lead-in and the answer are one text piece (an assistant message's content is one
  piece in the shard builder), so a token could straddle their boundary; the builder requires that none
  does (boundary_ok) and that the answer's tokens equal its tokens alone (the v1 tokens), and retries
  otherwise. Nothing follows the answer in the piece (suffix ""), so its end tokenizes as in v1.
  Floor (per item and tokenizer, `floor`): a model that copies one of the K snippets at random
  (perfectly, without binding the label): exact 1 / K, token_acc (1 + sum of the other snippets'
  positionwise agreement with the target) / K, nats ln K.

edit_old: produce the old text of a str_replace call for a stated line range.
  A file view as a tool result (a synthetic module of about 40 / 200 lines), then
  `str_replace(path, old_str, new_str)` for "lines A-B" (1 / 3 / 8 lines; unique in the file, first
  and last line not blank). The model is teacher-forced up to the start of the old_str value and
  scored on the tokens of the old text. Six renderings of the SAME content (cell `render`):
    json_num    tool arguments as escaped JSON (chat_format v1), view with line numbers (cat -n)
    json_plain  v1, view without line numbers (chat_format v2's stripped form, with its header)
    dec_num     decoded arguments between reserved control tokens (chat_format v2), view with numbers
    dec_plain   v2, view without numbers (what a v2-trained model sees)
    raw_num     DIAGNOSTIC: the v1 JSON frame, old text unescaped (real newlines, quotes), view with numbers
    raw_plain   DIAGNOSTIC: the same with the view without numbers
  The four main renderings are json_/dec_ x num/plain; raw_* separate escaping from the novelty of
  v2's delimiters. The view message is the same text in v1 and v2 for a given numbering (v2 would
  strip the numbers of a numbered view; here the numbered view is kept in dec_num on purpose).
  Floors: a random window of the same number of lines from the view (W = file lines - span + 1
  windows): exact 1 / W, token_acc estimated on up to 16 other windows, nats ln W.

Every generator is a pure function of (probe, cell, index, attempt) and the `count` callable.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ced.eval import probe_gen_ext as ge
from ced.eval.probe_gen import BASE_SEED

_TOOLS_DATA = str(Path(__file__).resolve().parents[2] / "tools" / "data")
if _TOOLS_DATA not in sys.path:
    sys.path.insert(0, _TOOLS_DATA)
import chat_format as cf  # noqa: E402

COPY_PROBE, EDIT_PROBE = "copy_span", "edit_old"
SPAN_PROBES = (COPY_PROBE, EDIT_PROBE)
PROBE_CODE_COPY = {COPY_PROBE: 103, EDIT_PROBE: 104}

TRAIN_LEN = 8192
COPY_TYPES = ("code", "path", "ident", "random")
COPY_LENGTHS = (8, 32, 128)
COPY_DISTS = (1024, 4096, 8192, 16384, 32768)
WITHIN_DISTS = (1024, 4096, 8192)  # within the training length (8,192 tokens)
BEYOND_DISTS = (16384, 32768)
COPY_SOURCES = ("tool", "assistant")
COPY_VARIANTS = ("plain", "near_dup")
NEAR_DIFFS = {8: 1, 32: 2, 128: 3}
K_SNIPPETS = 4
DIST_LO = 0.90  # dist 1024 / 4096: the distance lies in [DIST_LO D, D]
TOTAL_LO = 0.96  # dist >= 8192: the item length lies in [TOTAL_LO T, T]
SPAN_TOL = 0.5  # the span's token count lies within max(SPAN_TOL x nominal, SPAN_ABS) of the nominal length
SPAN_ABS = 5


def span_length_ok(n_tokens: int, nominal: int) -> bool:
    return abs(n_tokens - nominal) <= max(SPAN_TOL * nominal, SPAN_ABS)

EDIT_LINES = (40, 200)
EDIT_SPANS = (1, 3, 8)
MAIN_RENDERS = ("json_num", "json_plain", "dec_num", "dec_plain")
DIAG_RENDERS = ("raw_num", "raw_plain")
RENDERS = MAIN_RENDERS + DIAG_RENDERS
MARK = "@@OLD@@"

Count = Callable[[str], int]
Piece = tuple[str, bool]  # (text, control)


@dataclass
class ConvItem:
    id: str
    probe: str
    cell: dict
    context_pieces: list  # [[text, control], ...] up to the answer (for a tail item: up to and excluding the tail prefix)
    answer: str
    group: dict = field(default_factory=dict)
    tail: dict | None = None  # {"prefix": str, "suffix": str}: the answer sits between them inside ONE text piece
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ConvItem":
        return cls(**d)

    @property
    def text(self) -> str:
        """The context as text (control tokens left out) followed by the answer."""
        return context_text(self) + self.answer


def context_text(item: ConvItem) -> str:
    out = "".join(t for t, c in item.context_pieces if not c)
    return out + (item.tail["prefix"] if item.tail else "")


def rng_copy(probe: str, cell: tuple, index: int, attempt: int, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, PROBE_CODE_COPY[probe], *[int(x) for x in cell], index, attempt, stream]))


def pieces_len(count: Count, pieces) -> int:
    """Token length of pieces under the shard builder's convention: a control piece is one token, a
    text piece is tokenized alone."""
    return sum(1 if c else count(t) for t, c in pieces)


def _pc(pieces) -> list[Piece]:
    return [(t, bool(c)) for t, _, c in pieces]


def render_messages(messages: list[dict], tools: list | None = None, version: int = 1) -> list[Piece]:
    return _pc(cf.pieces({"messages": messages, "tools": tools or []}, version))


GEN_HEADER = [(cf.IM_START, True), ("assistant\n", False)]  # what the harness adds before the model's reply

# The fixed, teacher-forced start of the reply before a raw answer (probes-copy-v2): a
# short sentence and a code fence opening, where chat-format v1 assistants put raw text. Context, not scored.
LEADIN_COPY = "Here it is:\n```\n"


# --------------------------------------------------------------------------------------------
# copy: spans
# --------------------------------------------------------------------------------------------

ADJ = """amber brisk calm dusty eager fuzzy gentle hollow ivory jolly keen lucid mellow noble opal proud quiet rustic silent tidy
umber vivid warm young zesty bold crisp dim faint grand hazy icy jade lunar misty nimble plain royal swift tawny wild""".split()
ANIMAL = """falcon heron otter badger lynx marten osprey pelican quail raven stoat tern urchin viper walrus yak zebra bison cobra
dingo egret ferret gecko hare ibis jackal koala lemur mole newt ocelot panda robin shrew toad vole wombat finch crane drake""".split()
WORDS = ge.VERBS + ge.NOUNS
ALNUM = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
SPAN_END_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _closest(cands: list[str], count: Count, nominal: int) -> str:
    best = None
    for text in cands:
        if not text or text[0].isspace() or text[-1] not in SPAN_END_OK:
            continue
        gap = abs(count(text) - nominal)
        if best is None or gap < best[0]:
            best = (gap, text)
    if best is None:
        raise RuntimeError("copy: no candidate span")
    return best[1]


def _path(rng: np.random.Generator, depth: int) -> str:
    dirs = [ge._pick(rng, ge.DIRS) for _ in range(depth)]
    return "/repo/" + "/".join(dirs) + f"/{ge._pick(rng, ge.NOUNS)}_{ge._pick(rng, WORDS)}.py"


def gen_span(rng: np.random.Generator, ctype: str, count: Count, nominal: int) -> tuple[str, dict]:
    """(span text, meta) of one snippet whose token count (the `count` measure) is closest to nominal."""
    if ctype == "code":
        consts = [f"{ge._pick(rng, ge.CONST_ADJ)}_{ge._pick(rng, ge.NOUNS).upper()}" for _ in range(3)]
        params = ge.make_params(rng)
        stmts = ge.gen_statements(rng, params, 16, [], consts, simple_only=nominal <= 8, ind="")
        cands = ["\n".join(line for s in stmts[:k] for line in s) for k in range(1, len(stmts) + 1)]
        text = _closest(cands, count, nominal)
        return text, {"lines": text.count("\n") + 1}
    if ctype == "path":
        cands = [_path(rng, d) for d in (1, 2, 3)]
        lines = [_path(rng, int(rng.integers(1, 4))) for _ in range(40)]
        cands += ["\n".join(lines[:m]) for m in range(2, 41)]
        return _closest(cands, count, nominal), {}
    if ctype == "ident":
        def one() -> str:
            return "_".join(ge._pick(rng, WORDS) for _ in range(int(rng.integers(2, 5))))
        items = [one() for _ in range(48)]
        cands = [", ".join(items[:m]) for m in range(1, 49)]
        return _closest(cands, count, nominal), {}
    if ctype == "random":
        s = "".join(ALNUM[rng.integers(len(ALNUM), size=640)])
        lo = max(4, int(0.35 * nominal))
        cands = [s[:n] for n in range(lo, min(len(s), 6 * nominal + 40), max(1, nominal // 16))]
        return _closest(cands, count, nominal), {}
    raise ValueError(ctype)


def mutate_span(rng: np.random.Generator, span: str, n_diff: int):
    """The code span with one token changed in each of n_diff lines: (text, [(start, end)] changed char spans in the NEW
    text's line coordinates converted to the span coordinates). None if too few lines have a site."""
    lines = span.split("\n")
    order = [int(i) for i in rng.permutation(len(lines))]
    picked = []
    for li in order:
        mut = ge.mutate_line(rng, lines[li])
        if mut is not None:
            picked.append((li, mut[0], mut[1]))
        if len(picked) == n_diff:
            break
    if len(picked) < n_diff:
        return None
    new = list(lines)
    for li, text, _ in picked:
        new[li] = text
    starts = np.cumsum([0] + [len(x) + 1 for x in lines])
    diff_chars = sorted((int(starts[li] + a), int(starts[li] + b)) for li, _, (a, b) in picked)
    return "\n".join(new), diff_chars


def _labels(rng: np.random.Generator, k: int) -> list[str]:
    out: list[str] = []
    while len(out) < k:
        lab = f"{ge._pick(rng, ADJ)}-{ge._pick(rng, ANIMAL)}"
        # no two labels share an adjective or an animal word (avoids look-alike labels such as bold-ibis / lucid-ibis)
        if lab not in out and all(lab.split("-")[0] != o.split("-")[0] and lab.split("-")[1] != o.split("-")[1] for o in out):
            out.append(lab)
    return out


def block(label: str, span: str) -> str:
    return f"[{label}]\n{span}\n[/{label}]"


# --------------------------------------------------------------------------------------------
# copy: filler and the item
# --------------------------------------------------------------------------------------------

SYSTEM = "You are a coding assistant working in a repository. Use the tools when they help."


def _view_pair(rng, names, consts, n_funcs: int) -> list[dict]:
    path = ge._path(rng)
    text = ge._plain_file(rng, names, n_funcs, consts, [])
    return [{"role": "assistant", "content": f"Let me look at {path.rsplit('/', 1)[1]}.",
             "tool_calls": [{"name": "view", "arguments": {"path": path}}]},
            {"role": "tool", "content": text}]


def _run_pair(rng, n_lines: int) -> list[dict]:
    cmd = f"pytest -q tests/test_{ge._pick(rng, ge.NOUNS)}.py"
    lines = [f"tests/test_{ge._pick(rng, ge.NOUNS)}.py::test_{ge._pick(rng, ge.VERBS)} PASSED [{int(rng.integers(1, 100))}%]" for _ in range(n_lines)]
    return [{"role": "assistant", "content": "Running the tests.", "tool_calls": [{"name": "run", "arguments": {"cmd": cmd}}]},
            {"role": "tool", "content": "\n".join(lines) if lines else "no tests ran"}]


def _filler_candidates(rng, names, consts, room: int):
    """Candidate filler pairs, largest first (a random order inside a size class); the caller takes the first that fits."""
    for n_funcs in (7, 5, 3, 2, 1):
        yield _view_pair(rng, names, consts, n_funcs)
    for n_lines in (24, 12, 6, 3, 1, 0):
        yield _run_pair(rng, n_lines)


def build_filler(rng, count: Count, names, consts, lo: int, hi: int, used: int) -> list[dict]:
    """Filler messages (assistant call + tool result pairs) that raise `used` into [lo, hi]. Raises
    RuntimeError if no filler lands in the window."""
    out: list[dict] = []
    guard = 0
    while used < lo:
        guard += 1
        if guard > 4000:
            raise RuntimeError("copy: filler did not converge")
        room = hi - used
        # try a random size class first (so the filler is not all one size), then smaller ones
        start = int(rng.integers(0, 4)) if room > 1800 else 0
        pairs = list(_filler_candidates(rng, names, consts, room))
        for pair in pairs[start:] + pairs[:start]:
            n = pieces_len(count, render_messages(pair))
            if used + n <= hi:
                out += pair
                used += n
                break
        else:
            raise RuntimeError("copy: no filler fits the window")
    return out


def copy_item(count: Count, ctype: str, length: int, dist: int, src: str, variant: str, index: int, attempt: int = 0,
              k_snippets: int = K_SNIPPETS, leadin: str | None = LEADIN_COPY) -> ConvItem:
    """One copy_span item. Raises RuntimeError when the layout misses a window (the caller retries with the next attempt).
    `leadin`: the fixed start of the reply before the answer (a tail prefix, counted in the distance back); None gives
    the v1 item, whose answer opens the reply."""
    assert ctype in COPY_TYPES and length in NEAR_DIFFS and src in COPY_SOURCES and variant in COPY_VARIANTS
    assert variant == "plain" or ctype == "code"
    cell = (COPY_TYPES.index(ctype), length, dist, COPY_SOURCES.index(src), COPY_VARIANTS.index(variant))
    rng = rng_copy(COPY_PROBE, cell, index, attempt)
    spans = []
    seen = set()
    near = None
    target, tmeta = gen_span(rng, ctype, count, length)
    n_other = k_snippets - 1
    mutated = None
    if variant == "near_dup":
        n_diff = min(NEAR_DIFFS[length], target.count("\n") + 1)
        mutated = mutate_span(rng, target, n_diff)
        if mutated is None:
            raise RuntimeError("copy: no mutation sites")
        n_other -= 1
    others = []
    for _ in range(n_other):
        t, _ = gen_span(rng, ctype, count, length)
        others.append(t)
    items = [target] + ([mutated[0]] if mutated else []) + others
    for t in items:
        if t in seen or any(t in u or u in t for u in seen):
            raise RuntimeError("copy: duplicate snippets")
        seen.add(t)
    labels = _labels(rng, k_snippets)
    order = [int(i) for i in rng.permutation(k_snippets)]  # position -> index in items
    t_pos = order.index(0)
    blocks = [block(labels[p], items[order[p]]) for p in range(k_snippets)]
    label = labels[t_pos]
    names = ge.Names(rng)
    consts = [f"{ge._pick(rng, ge.CONST_ADJ)}_{ge._pick(rng, ge.NOUNS).upper()}" for _ in range(3)]
    # the source message and the text around the target span
    joined = "\n\n".join(blocks)
    head = "".join(b + "\n\n" for b in blocks[:t_pos])
    pre_span = head + f"[{label}]\n"
    if src == "tool":
        user0 = "First fetch the saved snippet bank, then go through the repository files."
        lead = {"role": "assistant", "content": "I'll fetch the snippet bank.",
                "tool_calls": [{"name": "lookup", "arguments": {"bank": "snippets"}}]}
        source = {"role": "tool", "content": joined}
        msgs_a = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}, lead]
        msgs_b = [source]
        prefix_text = "\n"  # the tool piece is "\n{content}\n"
    else:
        user0 = "Draft some snippets for me, then go through the repository files."
        msgs_a = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}]
        source = {"role": "assistant", "content": "Here are the snippets I drafted:\n\n" + joined + "\n\nNow the repository files.",
                  "tool_calls": [{"name": "view", "arguments": {"path": ge._path(rng)}}]}
        msgs_b = [source]
        prefix_text = "Here are the snippets I drafted:\n\n"
    query = (f"Reproduce the snippet labelled `{label}` exactly as it appeared above, character for character: "
             f"the text between `[{label}]` and `[/{label}]`, without the tags. Output nothing else.")
    qmsg = {"role": "user", "content": query}
    # lengths: before the span, the span, and what follows the span in the source piece
    p_a = render_messages(msgs_a)
    p_src = render_messages(msgs_a + msgs_b)[len(p_a):]
    # locate the piece holding the target: the one whose text contains pre_span + span
    needle = pre_span + target
    at = next(i for i, (t, c) in enumerate(p_src) if not c and needle in t)
    text = p_src[at][0]
    cut = text.index(needle)
    n_span = count(target)
    before = pieces_len(count, p_a) + pieces_len(count, p_src[:at]) + count(text[: cut + len(pre_span)])
    after_src = count(text[cut + len(pre_span) + len(target):]) + pieces_len(count, p_src[at + 1:])
    p_q = render_messages([{"role": "system", "content": SYSTEM}, qmsg])[len(render_messages([{"role": "system", "content": SYSTEM}])):]
    tail = pieces_len(count, p_q) + pieces_len(count, GEN_HEADER) + (count(leadin) if leadin else 0)
    used = after_src + tail
    if dist in (1024, 4096):
        lo, hi = int(math.ceil(DIST_LO * dist)), dist
    else:
        room = dist - before - 2 * n_span  # context + answer = before + span + after + answer
        lo, hi = int(math.ceil(TOTAL_LO * dist)) - before - 2 * n_span, room
    if used > hi:
        raise RuntimeError("copy: the source alone is farther than the window")
    filler = build_filler(rng, count, names, consts, lo, hi, used)
    messages = msgs_a + msgs_b + filler + [qmsg]
    pieces = render_messages(messages) + GEN_HEADER
    dist_measure = used + pieces_len(count, render_messages(filler))
    meta = {"label": label, "target_pos": t_pos, "type": ctype, "span_tokens_measure": n_span, "k": k_snippets,
            "distance_measure": dist_measure, "before_measure": before, "attempt": attempt, "source_label_text": prefix_text,
            "snippets": [items[order[p]] for p in range(k_snippets)], "labels": labels, "leadin": leadin, **tmeta}
    if mutated:
        meta.update(n_diff=len(mutated[1]), diff_chars=mutated[1], decoy_text=mutated[0])
    return ConvItem(
        id=f"{COPY_PROBE}/{ctype}/len={length}/dist={dist}/{src}/{variant}/{index:04d}", probe=COPY_PROBE,
        cell={"type": ctype, "len": length, "dist": dist, "src": src, "variant": variant},
        context_pieces=[[t, c] for t, c in pieces], answer=target, group={},
        tail={"prefix": leadin, "suffix": ""} if leadin else None, meta=meta)


def check_copy_item(item: ConvItem) -> None:
    """Text-level rules (AssertionError): the answer occurs once in the context text (the lead-in included); the label
    sits next to the target in the source and in the question; the question does not hold the answer; the reply starts
    with the harness header and, in v2, the fixed lead-in (a tail prefix with nothing after the answer)."""
    m = item.meta
    text = context_text(item)
    ans = item.answer
    lead = m.get("leadin")
    assert item.tail == ({"prefix": lead, "suffix": ""} if lead else None), f"{item.id}: lead-in frame"
    assert [tuple(p) for p in item.context_pieces[-2:]] == GEN_HEADER, f"{item.id}: the reply header ends the context pieces"
    assert ans and text.count(ans) == 1, f"{item.id}: answer not present exactly once"
    assert ans[0] != " " and ans[-1] in SPAN_END_OK, item.id
    lab = m["label"]
    assert text.count(f"[{lab}]\n{ans}\n[/{lab}]") == 1, item.id
    q = text.rindex("Reproduce the snippet")
    assert ans not in text[q:] and f"`{lab}`" in text[q:] and text[:q].count(f"[{lab}]") == 1 and text[q:].count(f"[{lab}]") == 1, item.id
    for s in m["snippets"]:
        assert text.count(s) == 1, f"{item.id}: a look-alike snippet is not present exactly once"
    if item.cell["variant"] == "near_dup":
        dec = m["decoy_text"]
        assert dec != ans and len(dec.split("\n")) == len(ans.split("\n")), item.id
        assert sum(1 for x, y in zip(ans.split("\n"), dec.split("\n")) if x != y) == m["n_diff"] == len(m["diff_chars"]), item.id


# --------------------------------------------------------------------------------------------
# edit
# --------------------------------------------------------------------------------------------

TOOLS = [
    {"name": "view", "description": "Show a file with its line numbers.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "str_replace", "description": "Replace old_str (which must occur exactly once in the file) with new_str.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
                    "required": ["path", "old_str", "new_str"]}},
]


def numbered_view(path: str, lines: list[str]) -> str:
    """`cat -n` view with the preface of the OpenHands-style views that chat_format.strip_line_numbers knows."""
    return f"Here's the result of running `cat -n` on {path}:\n" + ge.numbered(lines)


def plain_view(path: str, lines: list[str]) -> str:
    """The numbered view as chat_format v2 stores it: without the numbers, with the header that
    names the first line."""
    stripped = cf.strip_line_numbers(numbered_view(path, lines))
    assert stripped is not None
    head, body = stripped
    return head + body


def _edit_task(path: str, a: int, b: int) -> str:
    where = f"line {a}" if a == b else f"lines {a}-{b} (inclusive)"
    return f"In {path}, replace {where} with a TODO comment. View the file first."


def edit_base(n_lines: int, span: int, index: int, attempt: int = 0) -> dict:
    """The content shared by the renderings: file lines, path, the 1-based range (a, b), the old text."""
    rng = rng_copy(EDIT_PROBE, (n_lines, span), index, attempt)
    for _ in range(200):
        try:
            lines, _ = ge.line_file(rng, n_lines)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("edit: could not build a file")
    text = "\n".join(lines)
    starts = [s for s in range(len(lines) - span + 1)
              if lines[s].strip() and lines[s + span - 1].strip() and text.count("\n".join(lines[s : s + span])) == 1]
    if not starts:
        raise RuntimeError("edit: no unique range")
    s = starts[int(rng.integers(len(starts)))]
    return {"lines": lines, "path": ge._path(rng), "a": s + 1, "b": s + span, "old": "\n".join(lines[s : s + span]),
            "n_windows": len(lines) - span + 1, "n_unique_ranges": len(starts), "attempt": attempt}


def edit_item(n_lines: int, span: int, render: str, index: int, attempt: int = 0, base: dict | None = None) -> ConvItem:
    """The edit item of one rendering. All renderings of (n_lines, span, index, attempt) share the content (`base`)."""
    assert render in RENDERS
    b = base or edit_base(n_lines, span, index, attempt)
    version = 2 if render.startswith("dec") else 1
    numbered = render.endswith("num")
    view = numbered_view(b["path"], b["lines"]) if numbered else plain_view(b["path"], b["lines"])
    final_content = f"I'll replace {'line ' + str(b['a']) if b['a'] == b['b'] else 'lines ' + str(b['a']) + '-' + str(b['b'])} of {b['path']}."
    sentinel = MARK
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _edit_task(b["path"], b["a"], b["b"])},
        {"role": "assistant", "content": "I'll view the file first.", "tool_calls": [{"name": "view", "arguments": {"path": b["path"]}}]},
        {"role": "tool", "content": view},
        {"role": "assistant", "content": final_content,
         "tool_calls": [{"name": "str_replace", "arguments": {"path": b["path"], "old_str": sentinel, "new_str": "# TODO"}}]},
    ]
    per_msg = cf.message_pieces({"messages": messages, "tools": TOOLS}, version)
    if version == 2:  # a numbered view must keep its numbers: take the tool message's pieces from v1 (v2 would strip them)
        per_msg[3] = cf.message_pieces({"messages": messages, "tools": TOOLS}, 1)[3]
    pieces = [(t, bool(c)) for _, ps in per_msg[:-1] for t, _, c in ps]
    final = [(t, bool(c)) for t, _, c in per_msg[-1][1]]
    at = next(i for i, (t, c) in enumerate(final) if not c and MARK in t)
    old = b["old"]
    tail = None
    if version == 2:
        assert final[at][0] == MARK
        pieces += final[:at]
        answer = old
    else:
        text = final[at][0]
        i = text.index(MARK)
        assert text[i - 1] == '"' and text[i + len(MARK)] == '"'
        pieces += final[:at]
        answer = old if render.startswith("raw") else json.dumps(old, ensure_ascii=False)[1:-1]
        tail = {"prefix": text[:i], "suffix": text[i + len(MARK):]}
    meta = {"base": f"{EDIT_PROBE}/lines={n_lines}/span={span}/{index:04d}", "a": b["a"], "b": b["b"], "path": b["path"],
            "n_file_lines": len(b["lines"]), "n_windows": b["n_windows"], "old": old, "attempt": b["attempt"], "render": render,
            "n_unique_ranges": b["n_unique_ranges"]}
    return ConvItem(
        id=f"{meta['base']}/{render}", probe=EDIT_PROBE, cell={"lines": n_lines, "span": span, "render": render},
        context_pieces=[[t, c] for t, c in pieces], answer=answer, tail=tail, meta=meta)


def edit_answer_form(old: str, render: str) -> str:
    """The old text as it appears in the rendering (escaped JSON for json_*)."""
    return json.dumps(old, ensure_ascii=False)[1:-1] if render.startswith("json") else old


def check_edit_item(item: ConvItem, base: dict) -> None:
    m = item.meta
    lines = base["lines"]
    text = "\n".join(lines)
    old = "\n".join(lines[m["a"] - 1 : m["b"]])
    assert old == m["old"] and text.count(old) == 1 and m["b"] - m["a"] + 1 == item.cell["span"], item.id
    assert lines[m["a"] - 1].strip() and lines[m["b"] - 1].strip(), item.id
    ctx = "".join(t for t, c in item.context_pieces if not c)
    numbered = item.cell["render"].endswith("num")
    assert (f"{m['a']:6d}\t{lines[m['a'] - 1]}" in ctx) == numbered, item.id
    assert ("[line numbers removed; first line is 1]" in ctx) == (not numbered), item.id
    assert item.answer == edit_answer_form(old, item.cell["render"]), item.id
    assert (item.tail is None) == item.cell["render"].startswith("dec"), item.id
    if item.tail:
        assert item.tail["prefix"].endswith('"old_str": "'), item.id
        assert item.tail["suffix"].startswith('", "new_str": "# TODO"'), item.id
    else:
        assert item.context_pieces[-1] == ["old_str\n", False] and item.context_pieces[-2] == [cf.ARG_STR, True], item.id
    # the stated range is in the task, the final remark and nowhere else as a pair of numbers
    assert (f"lines {m['a']}-{m['b']}" in ctx) == (m["a"] != m["b"]), item.id


# --------------------------------------------------------------------------------------------
# encoding (shard-builder convention)
# --------------------------------------------------------------------------------------------


def encode_conv_item(item: ConvItem, enc: Callable[[str], tuple[list[int], list[tuple[int, int]]]], control_id: Callable[[str], int]):
    """(context ids, answer ids, info). enc(text) -> (ids, char offsets). Control pieces are one token by
    id, text pieces are tokenized alone. A tail item (v1 / raw JSON frame): prefix + answer + suffix are
    tokenized together as the training pipeline tokenizes the whole call, the answer tokens are those whose
    characters overlap the answer; `info["boundary_ok"]` is False when a token straddles the start of the
    answer (a context token merged with it) and `info["end_merge"]` counts the extra characters the last
    answer token covers past the answer (the closing quote, ...)."""
    ctx: list[int] = []
    for text, c in item.context_pieces:
        ctx += [control_id(text)] if c else enc(text)[0]
    info = {"boundary_ok": True, "end_merge": 0}
    if item.tail is None:
        return ctx, enc(item.answer)[0], info
    pre, suf = item.tail["prefix"], item.tail["suffix"]
    ids, offs = enc(pre + item.answer + suf)
    a0, a1 = len(pre), len(pre) + len(item.answer)
    inside = [k for k, (s, e) in enumerate(offs) if s < a1 and e > a0]
    first, last = inside[0], inside[-1]
    info["boundary_ok"] = offs[first][0] >= a0
    info["end_merge"] = max(0, offs[last][1] - a1)
    return ctx + ids[:first], ids[first : last + 1], info
