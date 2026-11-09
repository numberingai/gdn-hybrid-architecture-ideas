"""Text-level generators for the two extension probes of probes-v1 (copy and line-addressed edit).

Both probes use SYNTHETIC code-like files made here (no held-out documents, no
training data): small Python modules built from fixed word lists and statement templates. A model
cannot know their content, so the span it must copy (or the line it must name) is only in the
context.

- copy: tool-result-like file views (`$ cat <path>` blocks), then an instruction to reproduce the
  body of one function (form "repeat") or an `old_str:` field of a str_replace tool call whose old
  text must be copied exactly (form "edit"). The answer is that body, verbatim, with its
  indentation. Cells: (copy length 8 / 32 / 128 tokens) x (source distance 1k / 4k / 8k tokens,
  measured from the end of the source span to the start of the instruction) x (variant "plain" /
  "near_dup"). In "near_dup" the same file view holds a decoy function whose body equals the target's
  except for one token in each of 1 / 2 / 3 lines (a digit, an operator, a comparison or a boolean:
  one token under both tokenizers, checked at build time), right next to the target. A copier that
  loses its place at one of those tokens and continues in the decoy produces the near-miss copy the
  analysis found to be the copy failures that matter.
  Floors, stored per item (`floor`): a model that copies one of the W function bodies of the source
  file at random (perfectly, without binding the name): exact 1 / W, token accuracy (1 + sum of the
  other bodies' token agreement with the target) / W, nats ln W; and, in near_dup, a model that
  copies perfectly but takes the target's or the decoy's token at each differing token with equal
  probability: `diff_acc` 1/2 per differing token (`diff_index`: the answer-token indices of the
  differing tokens).
- line_edit: a file view WITH line numbers (`cat -n` form, as the OpenHands views the analysis
  found), then a question that names one definition (a top-level function, a class or a module
  constant) and asks for its line number. Form "number": `Answer: line` + ` <N>`. Form "sed": a task
  to delete that line, `$ sed -i '` + `<N>d` (the line-range edit command with the number the model
  has to produce). Cells: file length 40 / 160 / 480 lines. Floors (`floor`): a random definition
  line (exact 1 / (definition lines of the file)) and a random line (1 / lines).
  The distance distribution comes from the companion probe `line_edit_foil`: the same contexts with
  other line numbers as the answer (the neighbours +-1, 2, 4, 8 and other definition lines of the
  file). The scorer scores every row as an ordinary item; `ced.eval.probe_ext_analysis.line_choice`
  turns the log-likelihoods of the gold and foil answers of one question into the model's choice
  among them, its accuracy and the histogram of |chosen - gold|.

Every generator is a pure function of (probe, cell, item index, attempt) and, for copy, the `count`
callable (tokens under the length measure: the build tool passes the larger count of the
tokenizers). The build tool retries with the next `attempt` when a per-tokenizer check fails.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np

from ced.eval.probe_gen import BASE_SEED, TextItem

EXT_PROBES = ("copy", "line_edit", "line_edit_foil")
PROBE_CODE_EXT = {"copy": 101, "line_edit": 102}

COPY_LENGTHS = (8, 32, 128)  # nominal tokens of the span to copy
COPY_DISTANCES = (1024, 4096, 8192)  # tokens between the end of the source span and the start of the instruction
COPY_VARIANTS = ("plain", "near_dup")
COPY_FORMS = ("repeat", "edit")
COPY_DIFFS = {8: 1, 32: 2, 128: 3}  # differing tokens (lines) in a near_dup decoy
DIST_LO, DIST_HI = 0.90, 1.00  # the measured distance lies in [DIST_LO D, DIST_HI D]
SPAN_TOL = 0.5  # the span's token count lies within +-50% of the nominal length

LINE_COUNTS = (40, 160, 480)
LINE_FORMS = ("number", "sed")
LINE_KINDS = ("function", "class", "constant")
FOIL_OFFSETS = (-8, -4, -2, -1, 1, 2, 4, 8)
FOIL_RANDOM_DEFS = 4

Count = Callable[[str], int]


def rng_ext(probe: str, cell: tuple, index: int, attempt: int, stream: int = 0) -> np.random.Generator:
    words = [BASE_SEED, PROBE_CODE_EXT[probe], *[int(x) for x in cell], index, attempt, stream]
    return np.random.default_rng(np.random.SeedSequence(words))


# --------------------------------------------------------------------------------------------
# vocabulary and a tiny synthetic Python
# --------------------------------------------------------------------------------------------

VERBS = """load parse merge split build check scan sync fetch store render encode decode filter reduce rank score batch
flush route audit patch probe seal trim unify weigh apply bind clamp dedupe emit fold group index join keep lift mark
open peek pull push queue rotate sample shift sort stamp tally track update verify wrap""".split()
NOUNS = """ledger invoice order token session record config metric cache queue event route schema tenant account alert
report policy quota region rule shard slot stream tag task user vector window bucket cursor digest frame header key
label lease limit mapping offset packet ticket volume""".split()
LOCALS = """acc buf cur idx key val row col tmp res out src dst cnt pos lim ref node item part sum top low gap
mid end beg num cap dep len seq arg opt tab fld""".split()
CONST_ADJ = ["MAX", "MIN", "DEFAULT", "BASE", "LOCAL", "RAW", "SOFT", "HARD", "INITIAL", "FALLBACK"]
IMPORTS = ["import os", "import re", "import json", "import time", "import logging", "from pathlib import Path",
           "from dataclasses import dataclass", "import itertools", "import functools"]
DIRS = """core util model store sync batch parse render audit route cache event queue""".split()

OPS = ("+", "-", "*")
CMPS = ("<", ">", "<=", ">=")
FLAGS = ("True", "False")
BUILTIN_CALLS = ("max", "min")


class Names:
    """Unique function names (verb_noun) for one item, drawn from a shuffled list."""

    def __init__(self, rng: np.random.Generator):
        pairs = [(v, n) for v in VERBS for n in NOUNS]
        self.pairs = [pairs[i] for i in rng.permutation(len(pairs))]
        self.at = 0
        self.used: set[str] = set()

    def next(self) -> str:
        while True:
            v, n = self.pairs[self.at]
            self.at += 1
            if f"{v}_{n}" not in self.used:
                self.used.add(f"{v}_{n}")
                return f"{v}_{n}"

    def sibling(self, name: str, rng: np.random.Generator) -> str:
        """Same noun, another verb, not used yet."""
        noun = name.split("_", 1)[1]
        for v in rng.permutation(VERBS):
            cand = f"{v}_{noun}"
            if cand not in self.used and cand != name:
                self.used.add(cand)
                return cand
        raise RuntimeError("no sibling name")


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _atom(rng: np.random.Generator, names: list[str]) -> str:
    if names and rng.random() < 0.6:
        return _pick(rng, names)
    return str(int(rng.integers(2, 100)))


def _simple(rng: np.random.Generator, ind: str, vars_: list[str], funcs: list[str], consts: list[str]) -> str:
    """One single-line statement. New variables are drawn from LOCALS and appended to vars_."""
    kind = int(rng.integers(8))
    new = _pick(rng, [x for x in LOCALS if x not in vars_] or LOCALS)
    a, b = _atom(rng, vars_), _atom(rng, vars_)
    if kind == 0:
        s = f"{new} = {a} {_pick(rng, OPS)} {b}"
    elif kind == 1:
        s = f"{new} = {_pick(rng, funcs) if funcs else _pick(rng, BUILTIN_CALLS)}({a}, {b})"
    elif kind == 2 and vars_:
        new = _pick(rng, vars_)
        s = f"{new} += {a} * {int(rng.integers(2, 100))}"
    elif kind == 3 and vars_:
        s = f"{_pick(rng, vars_)}.append({a})"
        new = None
    elif kind == 4:
        s = f"{new} = [{a}, {b}, {int(rng.integers(2, 100))}]"
    elif kind == 5:
        s = f"{new} = {_pick(rng, FLAGS)}"
    elif kind == 6 and consts:
        s = f"{new} = {_pick(rng, consts)} {_pick(rng, OPS)} {a}"
    else:
        s = f"{new} = {a} if {a} {_pick(rng, CMPS)} {int(rng.integers(2, 100))} else {b}"
    if new is not None and new not in vars_:
        vars_.append(new)
    return ind + s


def gen_statements(rng: np.random.Generator, params: list[str], n: int, funcs: list[str], consts: list[str],
                   simple_only: bool = False, ind: str = "    ") -> list[list[str]]:
    """n statements, each a list of lines (compound statements are 2-3 lines)."""
    vars_ = list(params)
    out = []
    for _ in range(n):
        if not simple_only and rng.random() < 0.3:
            body = [_simple(rng, ind + "    ", vars_, funcs, consts) for _ in range(int(rng.integers(1, 3)))]
            if rng.random() < 0.5:
                head = f"{ind}if {_atom(rng, vars_)} {_pick(rng, CMPS)} {int(rng.integers(2, 100))}:"
            else:
                i = _pick(rng, [x for x in LOCALS if x not in vars_])
                head = f"{ind}for {i} in range({int(rng.integers(2, 100))}):"
            out.append([head, *body])
        else:
            out.append([_simple(rng, ind, vars_, funcs, consts)])
    return out


def body_lines(rng: np.random.Generator, params: list[str], n: int, funcs: list[str], consts: list[str]) -> list[str]:
    """Exactly n body lines (a compound statement only where it fits)."""
    lines: list[str] = []
    vars_ = list(params)
    while len(lines) < n:
        left = n - len(lines)
        if left >= 2 and rng.random() < 0.3:
            body = [_simple(rng, "        ", vars_, funcs, consts) for _ in range(min(left - 1, int(rng.integers(1, 3))))]
            if rng.random() < 0.5:
                head = f"    if {_atom(rng, vars_)} {_pick(rng, CMPS)} {int(rng.integers(2, 100))}:"
            else:
                i = _pick(rng, [x for x in LOCALS if x not in vars_])
                head = f"    for {i} in range({int(rng.integers(2, 100))}):"
            lines += [head, *body]
        else:
            lines.append(_simple(rng, "    ", vars_, funcs, consts))
    return lines


def make_params(rng: np.random.Generator) -> list[str]:
    return [LOCALS[i] for i in rng.choice(len(LOCALS), size=int(rng.integers(2, 4)), replace=False)]


def def_line(name: str, params: list[str]) -> str:
    return f"def {name}({', '.join(params)}):"


def function_lines(rng: np.random.Generator, name: str, n_body: int, funcs: list[str], consts: list[str]) -> list[str]:
    params = make_params(rng)
    return [def_line(name, params), *body_lines(rng, params, n_body, funcs, consts), f"    return {_pick(rng, params)}"]


# --------------------------------------------------------------------------------------------
# one-token mutations (copy, near_dup)
# --------------------------------------------------------------------------------------------

_SWAP = {"+": "-", "-": "*", "*": "+", "<": ">", ">": "<", "<=": ">=", ">=": "<=", "True": "False", "False": "True"}


def mutation_sites(line: str) -> list[tuple[int, int]]:
    """Character spans of the tokens a mutation may change in a line: a digit of a number literal
    (not of an identifier), a binary operator or comparison between spaces, True / False. Digits
    that follow another digit are found too (the regex is applied to every digit of a literal)."""
    sites = []
    for m in re.finditer(r"\d|(?<= )(?:\+|-|\*|<=|>=|<|>)(?= )|\b(?:True|False)\b", line):
        if m.group().isdigit():
            a = m.start()
            while a > 0 and line[a - 1].isdigit():
                a -= 1
            if a > 0 and (line[a - 1].isalpha() or line[a - 1] == "_"):
                continue  # a digit of an identifier
        sites.append((m.start(), m.end()))
    return sites


def mutate_line(rng: np.random.Generator, line: str) -> tuple[str, tuple[int, int]] | None:
    """The line with exactly one site changed to another value of its class (a different digit, a
    different operator, the opposite comparison or boolean), and the changed span; None if the line
    has no site. A digit never becomes a leading zero."""
    sites = mutation_sites(line)
    if not sites:
        return None
    a, b = sites[int(rng.integers(len(sites)))]
    old = line[a:b]
    if old.isdigit():
        lead = (a == 0 or not line[a - 1].isdigit()) and b < len(line) and line[b].isdigit()
        choices = [d for d in "0123456789" if d != old and not (lead and d == "0")]
        new = _pick(rng, choices)
    else:
        new = _SWAP[old]
    return line[:a] + new + line[b:], (a, b)


# --------------------------------------------------------------------------------------------
# copy
# --------------------------------------------------------------------------------------------


def file_block(path: str, text: str) -> str:
    return f"$ cat {path}\n{text}\n\n"


def _file_header(rng: np.random.Generator, noun: str) -> list[str]:
    imports = [IMPORTS[i] for i in sorted(rng.choice(len(IMPORTS), size=int(rng.integers(1, 3)), replace=False))]
    return [f'"""{noun.title()} helpers."""', "", *imports, "", ""]


def _plain_file(rng: np.random.Generator, names: Names, n_funcs: int, consts: list[str], funcs: list[str]) -> str:
    """A file of n_funcs functions (3-9 body lines each), two blank lines between them."""
    path_noun = _pick(rng, NOUNS)
    parts = ["\n".join(_file_header(rng, path_noun)).rstrip("\n")]
    for _ in range(n_funcs):
        parts.append("\n".join(function_lines(rng, names.next(), int(rng.integers(3, 10)), funcs, consts)))
    return "\n\n\n".join(parts)


def _path(rng: np.random.Generator) -> str:
    return f"/repo/src/{_pick(rng, DIRS)}/{_pick(rng, NOUNS)}_{_pick(rng, ['utils', 'core', 'ops', 'index'])}.py"


def copy_item(count: Count, length: int, distance: int, variant: str, index: int, attempt: int = 0,
              before: tuple[int, int] = (0, 6), after: tuple[int, int] = (0, 4)) -> TextItem:
    """One copy item. `before` / `after`: how many functions the source file holds before and after
    the target (and decoy); tests shrink them. Raises RuntimeError when no layout reaches the
    distance within the tolerance (the caller retries with the next attempt)."""
    assert variant in COPY_VARIANTS and length in COPY_DIFFS
    form = COPY_FORMS[index % 2]
    cell = (length, distance, COPY_VARIANTS.index(variant))
    rng = rng_ext("copy", cell, index, attempt)
    names = Names(rng)
    consts = [f"{_pick(rng, CONST_ADJ)}_{_pick(rng, NOUNS).upper()}" for _ in range(3)]
    funcs: list[str] = []  # calls to other functions are left out of the source file: its functions are self-contained
    target_name = names.next()
    params = make_params(rng)
    # the span: the first k statements of a generated body, k chosen so that its token count is closest to the nominal length
    stmts = gen_statements(rng, params, 16, funcs, consts, simple_only=length <= 8)
    best = None
    for k in range(1, len(stmts) + 1):
        text = "\n".join(line for s in stmts[:k] for line in s)
        gap = abs(count(text) - length)
        if best is None or gap < best[0]:
            best = (gap, k, text)
    span_lines = [line for s in stmts[: best[1]] for line in s]
    answer = best[2]
    n_span = count(answer)
    if abs(n_span - length) > SPAN_TOL * length:
        raise RuntimeError("copy: span length off the nominal")
    target_fn = "\n".join([def_line(target_name, params), *span_lines])
    meta: dict = {"name": target_name, "form": form, "variant": variant, "span_lines": len(span_lines),
                  "span_tokens_measure": n_span, "attempt": attempt}
    pieces_mid = [target_fn]
    decoy_answer = None
    if variant == "near_dup":
        n_diff = min(COPY_DIFFS[length], len(span_lines))
        order = [int(i) for i in rng.permutation(len(span_lines))]
        picked: list[tuple[int, str, tuple[int, int]]] = []
        for li in order:
            mut = mutate_line(rng, span_lines[li])
            if mut is not None:
                picked.append((li, mut[0], mut[1]))
            if len(picked) == n_diff:
                break
        if len(picked) < n_diff:
            raise RuntimeError("copy: no mutation sites for the decoy")
        decoy_lines = list(span_lines)
        for li, new, _ in picked:
            decoy_lines[li] = new
        decoy_name = names.sibling(target_name, rng)
        decoy_answer = "\n".join(decoy_lines)
        starts = np.cumsum([0] + [len(x) + 1 for x in span_lines])
        diff_chars = sorted((int(starts[li] + a), int(starts[li] + b)) for li, _, (a, b) in picked)
        decoy_fn = "\n".join([def_line(decoy_name, params), *decoy_lines])
        pieces_mid = [decoy_fn, target_fn] if rng.random() < 0.5 else [target_fn, decoy_fn]
        meta.update(decoy_name=decoy_name, n_diff=n_diff, diff_chars=diff_chars, decoy_answer=decoy_answer,
                    decoy_first=pieces_mid[0] is decoy_fn)
    n_before = int(rng.integers(before[0], before[1] + 1))
    n_after = int(rng.integers(after[0], after[1] + 1))
    path = _path(rng)
    header = "\n".join(_file_header(rng, _pick(rng, NOUNS))).rstrip("\n")
    fn_before = ["\n".join(function_lines(rng, names.next(), int(rng.integers(3, 10)), funcs, consts)) for _ in range(n_before)]
    fn_after = ["\n".join(function_lines(rng, names.next(), int(rng.integers(3, 10)), funcs, consts)) for _ in range(n_after)]
    head_text = "\n\n\n".join([header, *fn_before])
    # the text from the end of the span to the end of the target file's block
    # (a decoy that follows the target is part of it)
    if pieces_mid[0] is target_fn or variant == "plain":
        pre_target = head_text + "\n\n\n"
        after_pieces = ([pieces_mid[1]] if len(pieces_mid) > 1 else []) + fn_after
        target_first = True
    else:
        pre_target = head_text + "\n\n\n" + pieces_mid[0] + "\n\n\n"
        after_pieces = fn_after
        target_first = False
    rest = ("\n\n\n" + "\n\n\n".join(after_pieces) if after_pieces else "") + "\n\n"
    # the target function's own text up to the span's end
    context_head = f"$ cat {path}\n" + pre_target + "\n".join([def_line(target_name, params)]) + "\n"
    after = rest
    lo, hi = int(math.ceil(DIST_LO * distance)), distance
    n_after_tokens = count(after)
    if n_after_tokens > hi:
        raise RuntimeError("copy: the source file alone is farther than the distance")
    guard = 0
    while n_after_tokens < lo and guard < 200:
        guard += 1
        n_funcs = int(rng.integers(2, 8))
        block = None
        while n_funcs >= 1:
            cand = file_block(_path(rng), _plain_file(rng, names, n_funcs, consts, funcs))
            if n_after_tokens + count(cand) <= hi:
                block = cand
                break
            n_funcs -= 1
        if block is None:  # a file of one short function to fill the last gap
            block = file_block(_path(rng), "\n".join(function_lines(rng, names.next(), 2, funcs, consts)))
            if n_after_tokens + count(block) > hi:
                break
        after += block
        n_after_tokens += count(block)
    # compose: the target's own text (body = the answer), then `after`
    full_prefix = context_head + answer
    q_start_marker = after
    if form == "repeat":
        query = (f"User: Copy the whole body of `{target_name}` in {path} exactly as it appears above, verbatim, "
                 f"every line of it, without the def line.\nAssistant:\n")
    else:
        query = (f"User: Update `{target_name}` in {path}.\nAssistant: I will replace the whole body of `{target_name}`; old_str is that whole body, verbatim.\n"
                 f"<tool_call>\nname: str_replace\npath: {path}\nold_str:\n")
    context = full_prefix + q_start_marker + query
    d_measure = count(q_start_marker)
    if not (lo <= d_measure <= hi):
        raise RuntimeError(f"copy: distance {d_measure} outside [{lo}, {hi}]")
    # the other function bodies of the source file (the windows of the floor), as answer-form text
    windows = []
    for fn in [*fn_before, *fn_after, *( [decoy_fn] if variant == "near_dup" else [] )]:
        windows.append("\n".join(fn.split("\n")[1:]))
    meta.update(path=path, distance_tokens=d_measure, windows=len(windows) + 1, window_bodies=windows,
                before_funcs=n_before, after_funcs=n_after, query_chars=len(query))
    return TextItem(
        id=f"copy/len={length}/dist={distance}/{variant}/{index:04d}", probe="copy",
        cell={"len": length, "dist": distance, "variant": variant}, group={"form": form},
        context=context, answer=answer, meta=meta)


def copy_floor(item: TextItem, tok_ids, encode) -> dict:
    """The per-tokenizer floor record of a copy item. `tok_ids`: the answer's token ids; `encode`:
    str -> list of ids. exact 1/W, token_acc (1 + sum of the other bodies' agreement)/W, nats ln W
    for a copier that takes one of the W function bodies of the source file at random; diff_acc 1/2
    per differing token in near_dup."""
    t = list(tok_ids)
    W = item.meta["windows"]
    agree = 0.0
    for body in item.meta["window_bodies"]:
        w = encode(body)
        agree += sum(1 for i in range(len(t)) if i < len(w) and w[i] == t[i]) / len(t)
    floor = {"exact": 1.0 / W, "token_acc": (1.0 + agree) / W, "nats": math.log(W)}
    if item.meta["variant"] == "near_dup":
        floor["diff_acc"] = 0.5
    return floor


def copy_diff_index(item: TextItem, offsets, decoy_ids, target_ids) -> list[int] | None:
    """Answer-token indices of the differing tokens of a near_dup item: for each changed character
    span, the target tokens overlapping it must be exactly one, and the decoy answer must have as
    many tokens, differing from the target's at those indices only. None if any of this fails.
    offsets: the (start, end) character offsets of the target answer's tokens."""
    out = []
    for a, b in item.meta["diff_chars"]:
        hit = [k for k, (s, e) in enumerate(offsets) if s < b and e > a]
        if len(hit) != 1:
            return None
        out.append(hit[0])
    if len(decoy_ids) != len(target_ids):
        return None
    differ = [k for k in range(len(target_ids)) if decoy_ids[k] != target_ids[k]]
    return out if differ == sorted(out) else None


def check_copy_item(item: TextItem) -> None:
    ctx, ans, m = item.context, item.answer, item.meta
    assert ans and ctx.count(ans) == 1, f"{item.id}: answer not present exactly once"
    q = ctx.rindex("User: ")
    assert ans not in ctx[q:], item.id
    assert ctx.count(f"def {m['name']}(") == 1 and ctx[q:].count(f"`{m['name']}`") >= 1, item.id
    assert ctx.endswith("\n") and ans.startswith("    "), item.id
    assert ctx[:q].endswith("\n\n"), item.id
    if item.cell["variant"] == "near_dup":
        dec = m["decoy_answer"]
        assert dec != ans and len(dec.split("\n")) == len(ans.split("\n")), item.id
        assert sum(1 for x, y in zip(ans.split("\n"), dec.split("\n")) if x != y) == m["n_diff"] == len(m["diff_chars"]), item.id
        assert ctx.count(dec) == 1, f"{item.id}: decoy body not present exactly once"
        assert ctx.count(f"def {m['decoy_name']}(") == 1, item.id
        for a, b in m["diff_chars"]:
            assert ans[a:b] != "" and len(ans[a:b]) <= 5, item.id
    else:
        assert "decoy_answer" not in m, item.id


# --------------------------------------------------------------------------------------------
# line-addressed edit
# --------------------------------------------------------------------------------------------


def numbered(lines: list[str]) -> str:
    """`cat -n` form: the line number right-aligned in 6 columns, a tab, the line."""
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, 1))


def line_file(rng: np.random.Generator, n_lines: int) -> tuple[list[str], list[dict]]:
    """A synthetic module of at most n_lines lines (at least 0.8 n_lines) and its definitions
    [{kind: function | class | constant | method, name, line (1-based)}]. Raises RuntimeError when
    the file lacks a function, a class or a constant."""
    names = Names(rng)
    lines = _file_header(rng, _pick(rng, NOUNS))
    lines = lines[:-1]  # one blank line before the constants
    defs: list[dict] = []
    consts: list[str] = []
    used = set()
    for _ in range(int(rng.integers(3, 6))):
        while True:
            c = f"{_pick(rng, CONST_ADJ)}_{_pick(rng, NOUNS).upper()}"
            if c not in used:
                used.add(c)
                break
        consts.append(c)
        value = _pick(rng, [str(int(rng.integers(2, 1000))), f'"{_pick(rng, NOUNS)}"', f"{int(rng.integers(1, 10))}.{int(rng.integers(0, 10))}",
                            f"[{int(rng.integers(2, 99))}, {int(rng.integers(2, 99))}]"])
        lines.append(f"{c} = {value}")
        defs.append({"kind": "constant", "name": c, "line": len(lines)})
    lines += ["", ""]
    # plan the blocks (name, kind, body size) first, so bodies can call any function of the file
    plan = []
    used_lines = len(lines)
    pending_class = True
    for _ in range(400):
        make_class = pending_class and (rng.random() < 0.3 or used_lines > 0.4 * n_lines)
        if make_class:
            sizes = [int(rng.integers(2, 5)) for _ in range(int(rng.integers(1, 3)) if used_lines < 0.4 * n_lines else 1)]
            cls = _pick(rng, NOUNS).title() + _pick(rng, VERBS).title()
            blk = {"type": "class", "name": cls, "sizes": sizes, "lines": 1 + 4 + sum(s + 3 for s in sizes) + 2}
        else:
            n_body = int(rng.integers(2, 9))
            blk = {"type": "function", "name": names.next(), "n_body": n_body, "lines": n_body + 4}
        if used_lines + blk["lines"] - 2 > n_lines:  # the last block's separator is stripped
            if used_lines - 2 >= 0.85 * n_lines:
                break
            continue
        plan.append(blk)
        used_lines += blk["lines"]
        pending_class = pending_class and blk["type"] != "class"
    fn_names = [b["name"] for b in plan if b["type"] == "function"]
    for blk in plan:
        if blk["type"] == "function":
            params = make_params(rng)
            lines.append(def_line(blk["name"], params))
            defs.append({"kind": "function", "name": blk["name"], "line": len(lines)})
            lines += body_lines(rng, params, blk["n_body"], fn_names, consts)
            lines.append(f"    return {_pick(rng, params)}")
        else:
            lines.append(f"class {blk['name']}:")
            defs.append({"kind": "class", "name": blk["name"], "line": len(lines)})
            a, b = [LOCALS[i] for i in rng.choice(len(LOCALS), size=2, replace=False)]
            lines += [f"    def __init__(self, {a}, {b}):", "        self." + a + " = " + a, "        self." + b + " = " + b, ""]
            defs.append({"kind": "method", "name": "__init__", "line": len(lines) - 3})
            for s in blk["sizes"]:
                mname = names.next()
                params = ["self", *make_params(rng)]
                lines.append(f"    def {mname}({', '.join(params)}):")
                defs.append({"kind": "method", "name": mname, "line": len(lines)})
                body = body_lines(rng, params[1:], s, fn_names, consts)
                lines += ["    " + x for x in body]
                lines.append(f"        return {_pick(rng, params[1:])}")
                lines.append("")
        lines += ["", ""] if blk is not plan[-1] else []
    while lines and lines[-1] == "":
        lines.pop()
    if len(lines) > n_lines or len(lines) < 0.8 * n_lines:
        raise RuntimeError("line_file: size off")
    kinds = {d["kind"] for d in defs}
    if not {"function", "class", "constant"} <= kinds:
        raise RuntimeError("line_file: a kind is missing")
    return lines, defs


def _kind_phrase(kind: str, name: str) -> str:
    return {"function": f"the function `{name}`", "class": f"the class `{name}`", "constant": f"the constant `{name}`"}[kind]


def line_edit_item(n_lines: int, index: int, attempt: int = 0) -> tuple[TextItem, list[TextItem]]:
    """One question (gold item) and its foils. The form alternates with the index, the kind of the
    asked definition cycles with index // 2, the asked definition is uniform among those of its kind."""
    form = LINE_FORMS[index % 2]
    kind = LINE_KINDS[(index // 2) % 3]
    rng = rng_ext("line_edit", (n_lines,), index, attempt)
    for _ in range(200):
        try:
            lines, defs = line_file(rng, n_lines)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("line_edit: could not build a file")
    of_kind = [d for d in defs if d["kind"] == kind]
    target = of_kind[int(rng.integers(len(of_kind)))]
    path = f"/repo/src/{_pick(rng, DIRS)}/{_pick(rng, NOUNS)}_{_pick(rng, ['utils', 'core', 'ops', 'index'])}.py"
    view = f"$ cat -n {path}\n{numbered(lines)}\n\n"
    phrase = _kind_phrase(kind, target["name"])
    if form == "number":
        query = f"Question: on which line of {path} is {phrase} defined?\nAnswer: line"
        fmt = lambda n: f" {n}"  # noqa: E731
    else:
        query = f"Task: delete the line of {path} that defines {phrase}.\n$ sed -i '"
        fmt = lambda n: f"{n}d"  # noqa: E731
    context = view + query
    gold = target["line"]
    plausible = [d["line"] for d in defs]
    mentions = sum(1 for i, ln in enumerate(lines, 1) if i != gold and re.search(rf"\b{re.escape(target['name'])}\b", ln))
    base_id = f"line_edit/lines={n_lines}/{form}/{kind}/{index:04d}"
    cell = {"lines": n_lines}
    group = {"form": form, "kind": kind}
    item = TextItem(
        id=base_id, probe="line_edit", cell=cell, group=group, context=context, answer=fmt(gold),
        meta={"name": target["name"], "kind": kind, "form": form, "line": gold, "n_file_lines": len(lines), "path": path,
              "n_plausible": len(plausible), "plausible": plausible, "mentions": mentions, "role": "gold", "base": base_id,
              "floor": {"exact": 1.0 / len(plausible), "any": 1.0 / len(lines)}, "attempt": attempt})
    cands: list[int] = []
    for off in FOIL_OFFSETS:
        ln = gold + off
        if 1 <= ln <= len(lines) and ln not in cands:
            cands.append(ln)
    others = [p for p in plausible if p != gold and p not in cands]
    rng2 = rng_ext("line_edit", (n_lines,), index, attempt, stream=1)
    for j in rng2.permutation(len(others))[:FOIL_RANDOM_DEFS]:
        cands.append(others[int(j)])
    foils = []
    for ln in cands:
        foils.append(TextItem(
            id=f"{base_id}/foil={ln}", probe="line_edit_foil", cell=cell, group=group, context=context, answer=fmt(ln),
            meta={"role": "foil", "base": base_id, "line": ln, "offset": ln - gold, "form": form, "kind": kind}))
    item.meta["n_candidates"] = 1 + len(foils)
    item.meta["floor"]["choice"] = 1.0 / (1 + len(foils))
    return item, foils


def check_line_item(item: TextItem, foils: list[TextItem]) -> None:
    """Structural rules (raises AssertionError): the view is `cat -n` of a file whose line `line`
    defines the asked name exactly once; the query holds no digit; the answers carry the line number."""
    m = item.meta
    ctx = item.context
    q = ctx.index("Question: ") if m["form"] == "number" else ctx.index("Task: ")
    view, query = ctx[:q], ctx[q:]
    assert not any(ch.isdigit() for ch in query), f"{item.id}: digit in the query"
    body = view.split("\n", 1)[1].rstrip("\n").split("\n")
    assert len(body) == m["n_file_lines"], item.id
    for i, row in enumerate(body, 1):
        assert row.startswith(f"{i:6d}\t"), (item.id, i)
    gold = m["line"]
    text = body[gold - 1].split("\t", 1)[1]
    name = m["name"]
    pat = {"function": rf"^def {name}\(", "class": rf"^class {name}:", "constant": rf"^{name} = "}[m["kind"]]
    assert re.match(pat, text), (item.id, text)
    defining = [i for i, row in enumerate(body, 1) if re.match(pat, row.split("\t", 1)[1])]
    assert defining == [gold], f"{item.id}: not a unique definition"
    expect = f" {gold}" if m["form"] == "number" else f"{gold}d"
    assert item.answer == expect and m["line"] in m["plausible"], item.id
    assert query.endswith("Answer: line" if m["form"] == "number" else "$ sed -i '"), item.id
    seen = set()
    for f in foils:
        ln = f.meta["line"]
        assert f.context == ctx and ln != gold and 1 <= ln <= m["n_file_lines"] and ln not in seen, f.id
        seen.add(ln)
        assert f.answer == (f" {ln}" if m["form"] == "number" else f"{ln}d"), f.id
    assert len(foils) + 1 == m["n_candidates"] and len(foils) >= 5, item.id


def line_heuristics(item: TextItem) -> dict:
    """Name-independent picks of the asked line (what a model that ignores the name could answer)
    and the name-dependent first / last mention: first_def / last_def of the asked kind,
    a random-free `median_def`, and `mentions` (lines other than the definition that hold the name)."""
    m = item.meta
    ctx = item.context
    body = ctx.split("\n", 1)[1].split("\n\n", 1)[0].rstrip("\n").split("\n")
    pat = {"function": r"^def \w+\(", "class": r"^class \w+:", "constant": r"^[A-Z_]+ = "}[m["kind"]]
    same = [i for i, row in enumerate(body, 1) if re.match(pat, row.split("\t", 1)[1])]
    return {"first_def": same[0], "last_def": same[-1], "median_def": same[len(same) // 2], "n_same_kind": len(same),
            "mentions": m["mentions"]}
