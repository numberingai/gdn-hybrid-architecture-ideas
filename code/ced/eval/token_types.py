"""Token types of generated (assistant) tokens in agent trajectories, from the token stream alone (2026-10-11).

Why: the pass-gain analysis (tools/analysis/auto/pass_gain.py) splits the loss of a looped model's first-pass and
second-pass outputs by what kind of token is predicted. No model and no text are involved here: the input is the token ids
of a document, the flags saying which tokens the assistant generates (the shards' `generated` flags, LongDocs.assistant,
or an EvalRows' dec_mask), and a table of the bytes of every vocabulary entry. The labels are a pure function of those,
deterministic, and a changed rule is a re-run of this file over stored ids, never a GPU job.

Format. Both renderings of tools/data/chat_format.py are read, found from the stream: v1 (the default, what the ladder's shards and eval
  sets hold) <tool_call>\\n{"name": "N", "arguments": {"k": v...}}\\n</tool_call> v2 (reserved argument delimiters)
  <tool_call>\\nN\\n<|reserved_0|>k\\nvalue<|reserved_2|>\\n ... </tool_call> An assistant message is [<think>\\nR\\n</think>\\n\\n] content
  [\\n<tool_call>...</tool_call>]* <|im_end|>.

Fine types (FINE) and the roll-up the analysis reports (ROLLUP, TYPES). Every generated token gets exactly one fine type:

  scaffold whitespace between the parts of a message, <think> and </think>, </tool_call>, and the call's syntax:
                   the JSON braces, quotes, colons, commas and argument names (v1), or the delimiters and argument names (v2).
                   Rolls up to "other". call_open the <tool_call> token (the decision to call a tool). "other". turn_end
                   <|im_end|>. "other". reasoning_think tokens inside <think> ... </think>. "reasoning". reasoning_call the value
                   of a thought argument of a call (key in REASONING_KEYS, e.g. the think tool's "thought"). "reasoning". narration
                   message text of a message that also makes a tool call ("Opening the config file first"). "reasoning": it is the
                   model's thinking aloud before the call. Reported apart in the fine table. Not when every call of the message is
                   a finish-type tool (FINISH_TOOLS): that text is final_answer. head_name the tool name. "head" (the first tokens
                   of a tool call). head_path the value of a target-file or path argument: a string argument whose name is in
                   PATH_KEYS or ends in a word of PATH_WORDS, in any case and with any separator (path, file_path, filePath,
                   output_file, repo_dir). "head". arg the value of any other argument that is not code and not a copied span.
                   "args". code_arg the value of an argument that carries code or text to be written (key in CODE_KEYS: a shell
                   command, file text, an edit's old and new string, keystrokes), not a copied span. "code". copied_arg a token of
                   an argument value (arg or code_arg) inside a COPIED SPAN: a window of at least COPY_MIN_TOKENS consecutive
                   tokens of one argument value that occurs verbatim earlier in the document, entirely before the window starts
                   (non-overlapping; any role, tool output or earlier call). "copied". The comparison is on bytes, with two things
                   undone on both sides (document_bytes): the JSON escapes of v1 call arguments, and the line-number prefixes of
                   file views (" 12<TAB>", "12:", "12->" at a line start), which otherwise cut every copied line shorter than
                   COPY_MIN_TOKENS out of a span. Measured on an editor tool's `old_str` (a copy by construction: it must match the
                   file): 84% of its tokens are copied without the second, 99.6% with it (30 CoderForge conversations, N = 16).
                   code_content the body of a fenced code block in a message's content: the opening ``` starts a line and the
                   closing ``` is alone on one, as Markdown reads them (inline ``` is text). "code". (Copy detection is for
                   argument values only, as asked: a copied block in content stays code_content.) final_answer message text of a
                   message that makes no tool call, or only finish-type calls (finish, submit, task_complete: the summary written
                   before handing the work in), outside code fences; and the value of the message argument of a finish-type tool
                   (FINISH_TOOLS). "final". A message that writes its action as text (a reply in another tool format that was not
                   parsed into calls) makes no call, so its text is final_answer too: the labeller cannot tell it from an answer.

Rules in order of precedence for an argument value token: head_path (PATH_KEYS) > reasoning_call (REASONING_KEYS) >
final_answer (FINAL_KEYS of a FINISH_TOOLS call) > copied_arg (window rule) > code_arg (CODE_KEYS) > arg.
A token that holds bytes of more than one part takes the type of its FIRST value byte; a token with no value byte
(only syntax) is scaffold.

COPY_MIN_TOKENS = 16, fixed on 2026-10-11 before any model was scored (shares at 8 / 16 / 32 are in the analysis note).

What the labels may depend on. A token's type depends on the tokens before it and on the rest of its own message:
narration or final_answer is decided by whether the message goes on to make a call, and copied_arg by the 16-token
window the token is in. Nothing after the message's <|im_end|> is read, and a copied span is copied only by what
precedes it (tests/test_token_types.py cuts documents and checks the labels do not move). So the labels are for
scoring, not for a decision at generation time: a schedule that must act before the message ends can use only the
types that the tokens already emitted decide (think, call head, argument name).

A second labelling, independent of the types, is for exactly that: SEGMENTS (reasoning, content, call_head, call_body,
unplaced), a function of the tokens up to a position only (segment_positions, segment_targets); and COPY_OFFSETS, the offset
of a copied_arg token inside its copied span (copy_offsets).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOOLS_DATA = str(Path(__file__).resolve().parents[2] / "tools" / "data")
if _TOOLS_DATA not in sys.path:
    sys.path.insert(0, _TOOLS_DATA)
import chat_format as cf  # noqa: E402  (tools/data/chat_format.py: pure Python)

FINE = ("scaffold", "call_open", "turn_end", "reasoning_think", "reasoning_call", "narration", "head_name", "head_path",
        "arg", "code_arg", "copied_arg", "code_content", "final_answer")
F = {name: i for i, name in enumerate(FINE)}

TYPES = ("reasoning", "head", "args", "copied", "code", "final", "other")
ROLLUP = {"scaffold": "other", "call_open": "other", "turn_end": "other", "reasoning_think": "reasoning",
          "reasoning_call": "reasoning", "narration": "reasoning", "head_name": "head", "head_path": "head", "arg": "args",
          "code_arg": "code", "copied_arg": "copied", "code_content": "code", "final_answer": "final"}
FINE_TO_TYPE = np.array([TYPES.index(ROLLUP[f]) for f in FINE], dtype=np.int8)

COPY_MIN_TOKENS = 16
PATH_KEYS = frozenset({"path", "file_path", "filepath", "file", "filename", "target_file", "notebook_path", "directory",
                       "dir", "old_path", "new_path", "cwd", "root"})
# the last word of an argument name that makes it a path whatever comes before it (source_path, outputFile, repo_dir)
PATH_WORDS = frozenset({"path", "filepath", "file", "filename", "dir", "directory", "folder"})
_KEY_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# A file view's line number at a line start, as tools/data/chat_format.py reads one (spaces, digits, a tab or an arrow),
# and the "<n>:" of a numbered viewer or grep -n. Dropped from the bytes copy detection compares, on both sides.
_LINE_NUMBER = re.compile(rb"(?m)^(?: *\d+(?:\t|\xe2\x86\x92)|\d+:)")
REASONING_KEYS = frozenset({"thought", "thoughts", "thinking", "reasoning"})
FINAL_KEYS = frozenset({"message", "answer", "result", "final_answer"})
FINISH_TOOLS = frozenset({"finish", "task_complete", "submit", "final_answer", "complete", "done", "stop"})
CODE_KEYS = frozenset({"command", "cmd", "keystrokes", "content", "file_text", "new_str", "old_str", "new_string",
                       "old_string", "code", "script", "patch", "diff", "insert_text", "replacement"})

SUBCOMMAND_TOOLS = ("editor", "str_replace")  # tools whose "command" argument is a sub-command, not a shell command
COPY_MIN_BYTES = 8  # a window shorter than this in bytes is never a copy (a run of 16 one-byte tokens is padding)

N_SPECIAL = len(cf.SPECIAL_TOKENS)
ID = {name: cf.SPECIAL_TOKENS.index(name) for name in cf.SPECIAL_TOKENS}
CONTROL_IDS = frozenset(range(N_SPECIAL))


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte to printable-character map (the alphabet of a ByteLevel tokenizer's vocabulary strings)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@dataclass
class TokenTable:
    """The bytes of every vocabulary entry. Special tokens (ids 0 .. 31: chat_format.SPECIAL_TOKENS in order, as the
    tokenizers are trained) hold their literal text; they are told apart by id, never by their bytes."""

    token_bytes: list[bytes]
    is_ws: np.ndarray = field(init=False)  # (V,) bool: a non-special token of whitespace only

    def __post_init__(self):
        self.is_ws = np.array([i >= N_SPECIAL and len(b) > 0 and not b.strip() for i, b in enumerate(self.token_bytes)])

    @classmethod
    def from_vocab(cls, vocab: dict[str, int], specials: dict[str, int] | None = None) -> "TokenTable":
        """vocab: {token string of a ByteLevel tokenizer: id} (tokenizer.json model.vocab plus the added tokens)."""
        inv = {c: b for b, c in bytes_to_unicode().items()}
        out: list[bytes] = [b""] * (max(vocab.values()) + 1)
        for tok, i in vocab.items():
            out[i] = bytes(inv[c] for c in tok) if all(c in inv for c in tok) else tok.encode("utf-8")
        for name in cf.SPECIAL_TOKENS:  # the literal text, whatever the vocabulary file holds
            out[ID[name]] = name.encode("utf-8")
        return cls(out)

    @classmethod
    def from_tokenizer_json(cls, path: str) -> "TokenTable":
        """From a tokenizer.json (a ByteLevel BPE: model.vocab plus added_tokens), read as JSON: the same vocabulary
        as tokenizers.Tokenizer.from_file(path).get_vocab() (checked on the v0 tokenizer, 128,000 entries), without
        needing that package where the analysis runs."""
        import json

        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        vocab = dict(spec["model"]["vocab"])
        vocab.update({a["content"]: a["id"] for a in spec.get("added_tokens") or []})
        for name in cf.SPECIAL_TOKENS:  # the tokenizers are trained with the chat format's special tokens at ids 0 .. 31
            if vocab.get(name) != ID[name]:
                raise ValueError(f"{path}: {name!r} has id {vocab.get(name)}, expected {ID[name]}")
        return cls.from_vocab(vocab)


# -------------------------------------------------------------------------------------------------------------------
# a tool call's interior


def is_path_key(key: str) -> bool:
    """An argument name that carries a target file or path: in PATH_KEYS as written, or, split into words at
    underscores, hyphens and capitals, the words joined are in PATH_KEYS or the last word is in PATH_WORDS
    (filePath, file-path, FILE_PATH, output_file, repoDir). "profile" and "xpath" are one word each and are not."""
    if key in PATH_KEYS:
        return True
    words = [w.lower() for w in _KEY_WORDS.findall(key)]
    return bool(words) and ("_".join(words) in PATH_KEYS or "".join(words) in PATH_KEYS or words[-1] in PATH_WORDS)


def _classify(tool: str, key: str, is_string: bool) -> int:
    """The fine type of an argument value (copy detection comes after)."""
    if is_string and is_path_key(key):
        return F["head_path"]
    if key == "command" and is_string and any(w in tool.lower() for w in SUBCOMMAND_TOOLS):
        return F["head_name"]  # an editor tool's sub-command (view, create, str_replace): part of the head
    if key in REASONING_KEYS and is_string:
        return F["reasoning_call"]
    if tool in FINISH_TOOLS and key in FINAL_KEYS and is_string:
        return F["final_answer"]
    if key in CODE_KEYS:
        return F["code_arg"]
    return F["arg"]


_ESC = {"n": 10, "t": 9, "r": 13, '"': 34, "\\": 92, "/": 47, "b": 8, "f": 12}


class _Symbols:
    """The bytes of a run of tokens as one list of symbols (0..255 a byte, 256 + id a control token) and the token each
    symbol belongs to."""

    def __init__(self, ids, a: int, b: int, table: TokenTable):
        sym, tok = [], []
        for t in range(a, b):
            i = int(ids[t])
            if i < N_SPECIAL:
                sym.append(256 + i)
                tok.append(t)
            else:
                bs = table.token_bytes[i]
                sym.extend(bs)
                tok.extend([t] * len(bs))
        self.sym, self.tok = sym, tok
        self.tool = ""  # the call's tool name, set by the parser
        self.role = [-1] * len(sym)  # fine type of a value symbol, -1 syntax
        self.gid = [-1] * len(sym)  # argument-value group of a code_arg / arg symbol
        # the byte each symbol stands for once a JSON string's escapes are undone (-1: none; the second symbol of an
        # escape, and control tokens): copy detection compares these bytes, not token ids
        self.emit = [x if x < 256 else -1 for x in sym]

    def unescape(self, lo: int, hi: int) -> None:
        """Undo the JSON escapes of the string value in symbols lo..hi-1 (v1 renders arguments with json.dumps)."""
        k = lo
        while k < hi:
            if self.sym[k] == 92 and k + 1 < hi and self.sym[k + 1] < 256 and chr(self.sym[k + 1]) in _ESC:
                self.emit[k], self.emit[k + 1] = _ESC[chr(self.sym[k + 1])], -1
                k += 2
            else:
                k += 1

    def __len__(self):
        return len(self.sym)


def _skip_ws(s: list[int], i: int) -> int:
    while i < len(s) and s[i] in (32, 9, 10, 13):
        i += 1
    return i


def _string_end(s: list[int], i: int) -> int:
    """i at an opening quote: the index of the closing quote (len(s) if unterminated)."""
    j = i + 1
    while j < len(s):
        if s[j] == 92:  # backslash
            j += 2
            continue
        if s[j] == 34:
            return j
        j += 1
    return len(s)


def _value_end(s: list[int], i: int) -> int:
    """i at the first symbol of a JSON value: the index one past it (balanced brackets and strings)."""
    if i >= len(s):
        return i
    if s[i] == 34:
        return min(_string_end(s, i) + 1, len(s))
    if s[i] in (123, 91):
        depth, j = 0, i
        while j < len(s):
            c = s[j]
            if c == 34:
                j = _string_end(s, j)
            elif c in (123, 91):
                depth += 1
            elif c in (125, 93):
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return len(s)
    j = i
    while j < len(s) and s[j] not in (44, 125, 93, 32, 10, 13, 9) and s[j] < 256:
        j += 1
    return max(j, i + 1)


def _text(s: list[int], a: int, b: int) -> str:
    return bytes(x for x in s[a:b] if x < 256).decode("utf-8", errors="replace")


def _parse_v1(S: _Symbols, group0: int) -> tuple[bool, int]:
    """{"name": ..., "arguments": {...}} over the symbols. Returns (well formed, next group id)."""
    s = S.sym
    g = group0
    tool = ""
    i = _skip_ws(s, 0)
    if i >= len(s) or s[i] != 123:
        return False, g
    i += 1
    while True:
        i = _skip_ws(s, i)
        if i >= len(s):
            return False, g
        if s[i] == 125:
            return True, g
        if s[i] == 44:
            i += 1
            continue
        if s[i] != 34:
            return False, g
        e = _string_end(s, i)
        key = _text(s, i + 1, e)
        i = _skip_ws(s, e + 1)
        if i >= len(s) or s[i] != 58:
            return False, g
        i = _skip_ws(s, i + 1)
        if i >= len(s):
            return False, g
        if key == "name" and s[i] == 34:
            e = _string_end(s, i)
            tool = S.tool = _text(s, i + 1, e)
            for k in range(i + 1, e):
                S.role[k] = F["head_name"]
            i = min(e + 1, len(s))
        elif key == "arguments" and s[i] == 123:
            i += 1
            while True:
                i = _skip_ws(s, i)
                if i >= len(s):
                    return False, g
                if s[i] == 125:
                    i += 1
                    break
                if s[i] == 44:
                    i += 1
                    continue
                if s[i] != 34:
                    return False, g
                e = _string_end(s, i)
                akey = _text(s, i + 1, e)
                i = _skip_ws(s, e + 1)
                if i >= len(s) or s[i] != 58:
                    return False, g
                i = _skip_ws(s, i + 1)
                if i >= len(s):
                    return False, g
                is_str = s[i] == 34
                end = _value_end(s, i)
                lo, hi = (i + 1, max(i + 1, end - 1)) if is_str else (i, end)
                if is_str and end <= len(s) and (end - 1 < len(s)) and s[end - 1] != 34:
                    hi = end  # unterminated string
                t = _classify(tool, akey, is_str)
                if is_str:
                    S.unescape(lo, min(hi, len(s)))
                for k in range(lo, min(hi, len(s))):
                    S.role[k] = t
                    if t in (F["arg"], F["code_arg"]):
                        S.gid[k] = g
                g += 1
                i = end
        else:
            i = _value_end(s, i)


def _parse_v2(S: _Symbols, group0: int) -> tuple[bool, int]:
    """\\nNAME\\n then (ARG_STR | ARG_JSON) key\\n value ARG_END \\n ... over the symbols."""
    s = S.sym
    g = group0
    i = _skip_ws(s, 0)
    e = i
    while e < len(s) and s[e] != 10 and s[e] < 256:
        e += 1
    if e >= len(s) or s[e] != 10:
        return False, g
    tool = S.tool = _text(s, i, e)
    for k in range(i, e):
        S.role[k] = F["head_name"]
    i = e + 1
    ctl = {256 + ID[cf.ARG_STR]: True, 256 + ID[cf.ARG_JSON]: False}
    end_ctl = 256 + ID[cf.ARG_END]
    while i < len(s):
        if s[i] in (10, 32):
            i += 1
            continue
        if s[i] not in ctl:
            return False, g
        is_str = ctl[s[i]]
        i += 1
        k0 = i
        while i < len(s) and s[i] != 10 and s[i] < 256:
            i += 1
        if i >= len(s):
            return False, g
        akey = _text(s, k0, i)
        i += 1
        v0 = i
        while i < len(s) and s[i] != end_ctl:
            i += 1
        t = _classify(tool, akey, is_str)
        for k in range(v0, i):
            S.role[k] = t
            if t in (F["arg"], F["code_arg"]):
                S.gid[k] = g
        g += 1
        if i < len(s):
            i += 1
    return True, g


# -------------------------------------------------------------------------------------------------------------------
# a message, a document


_FENCE_OPEN = re.compile(rb"(?m)^[ \t]*```[^`\n]*$")
_FENCE_CLOSE = re.compile(rb"(?m)^[ \t]*```[ \t]*$")


def _fence_regions(raw: bytes) -> list[tuple[int, int, int, int]]:
    """Fenced code blocks in `raw` as (open start, body start, body end, close end) byte offsets, read as Markdown reads
    them: the opening ``` starts a line (after spaces or tabs) and the rest of its line is an info string without a
    backtick; the body runs from the next line to the line that holds a closing ``` alone (an unclosed block runs to
    the end). A ``` inside a line of text (inline code, a fence inside a JSON string whose newlines are escaped) opens
    nothing: read as a fence it would turn the rest of its line into syntax and the text after it into code."""
    out, pos = [], 0
    while True:
        m = _FENCE_OPEN.search(raw, pos)
        if m is None:
            return out
        a = m.start() + (m.group(0).index(b"```"))
        body = min(m.end() + 1, len(raw))
        c = _FENCE_CLOSE.search(raw, body)
        if c is None:
            out.append((a, body, len(raw), len(raw)))
            return out
        out.append((a, body, c.start(), c.end()))
        pos = c.end()


def _label_content(ids, a: int, b: int, base: int, table: TokenTable, lab: np.ndarray) -> None:
    for t in range(a, b):
        lab[t] = base
    raw = b"".join(table.token_bytes[int(ids[t])] for t in range(a, b))
    regions = _fence_regions(raw)
    if not regions:
        return
    off = 0
    for t in range(a, b):
        n = len(table.token_bytes[int(ids[t])])
        for o, body, hi, close in regions:
            if body <= off < hi:
                lab[t] = F["code_content"]
            elif o <= off < body or hi <= off < close:
                lab[t] = F["scaffold"]  # the fence lines themselves
        off += n


def _label_run(ids, a: int, b: int, table: TokenTable, lab: np.ndarray, gid: np.ndarray, stats: dict) -> None:
    """One generated run [a, b): a message body through its <|im_end|> (or cut at either end)."""
    ws = table.is_ws
    end = b
    if end > a and int(ids[end - 1]) == ID[cf.IM_END]:
        lab[end - 1] = F["turn_end"]
        end -= 1
    i = a
    if i < end and int(ids[i]) == ID[cf.THINK]:
        lab[i] = F["scaffold"]
        j = i + 1
        while j < end and int(ids[j]) != ID[cf.THINK_END]:
            j += 1
        for t in range(i + 1, j):
            lab[t] = F["reasoning_think"]
        t = i + 1
        while t < j and ws[int(ids[t])]:
            lab[t] = F["scaffold"]
            t += 1
        t = j - 1
        while t > i and ws[int(ids[t])]:
            lab[t] = F["scaffold"]
            t -= 1
        if j < end:
            lab[j] = F["scaffold"]
        i = min(j + 1, end)
    else:  # a run that starts inside the think block (cut by a row boundary): a </think> before any <tool_call>
        j = next((t for t in range(i, end) if int(ids[t]) in (ID[cf.THINK_END], ID[cf.TOOL_CALL])), None)
        if j is not None and int(ids[j]) == ID[cf.THINK_END]:
            for t in range(i, j):
                lab[t] = F["reasoning_think"]
            lab[j] = F["scaffold"]
            i = j + 1
    while i < end and ws[int(ids[i])]:
        lab[i] = F["scaffold"]
        i += 1
    call = next((t for t in range(i, end) if int(ids[t]) == ID[cf.TOOL_CALL]), None)
    c = end if call is None else call
    if c > i:
        stop = c
        if call is not None:  # the separator before a call
            while stop > i and ws[int(ids[stop - 1])]:
                stop -= 1
                lab[stop] = F["scaffold"]
        _label_content(ids, i, stop, F["narration"] if call is not None else F["final_answer"], table, lab)
    pos = c
    tools: list[str] = []
    while pos < end:
        if int(ids[pos]) != ID[cf.TOOL_CALL]:
            lab[pos] = F["scaffold"]
            pos += 1
            continue
        lab[pos] = F["call_open"]
        e = next((t for t in range(pos + 1, end) if int(ids[t]) == ID[cf.TOOL_CALL_END]), None)
        stop = end if e is None else e
        S = _Symbols(ids, pos + 1, stop, table)
        v1 = _skip_ws(S.sym, 0) < len(S.sym) and S.sym[_skip_ws(S.sym, 0)] == 123
        ok, stats["next_gid"] = (_parse_v1 if v1 else _parse_v2)(S, stats.get("next_gid", 0))
        tools.append(S.tool)
        stats["calls"] = stats.get("calls", 0) + 1
        if e is None or not ok:
            stats["malformed_calls"] = stats.get("malformed_calls", 0) + 1
        for t in range(pos + 1, stop):
            lab[t] = F["scaffold"]
        per_token: dict[int, bytearray] = {}
        for k, t in enumerate(S.tok):
            if S.emit[k] >= 0:
                per_token.setdefault(t, bytearray()).append(S.emit[k])
        for t in range(pos + 1, stop):
            stats.setdefault("unesc", {})[t] = bytes(per_token.get(t, b""))
        seen = set()
        for k, t in enumerate(S.tok):
            if t in seen or S.role[k] < 0:
                continue
            seen.add(t)
            lab[t] = S.role[k]
            gid[t] = S.gid[k]
        if e is not None:
            lab[e] = F["scaffold"]
        pos = stop + 1 if e is not None else end
    if tools and all(t in FINISH_TOOLS for t in tools):
        # the message only hands the work in (finish, submit, task_complete): its text is the final report, not the
        # thinking aloud before an action
        for t in range(i, c):
            if lab[t] == F["narration"]:
                lab[t] = F["final_answer"]


def document_bytes(ids: np.ndarray, table: TokenTable, unesc: dict[int, bytes], keep_line_numbers: bool = False) -> tuple[bytes, np.ndarray]:
    """The bytes copy detection compares, and (n + 1,) the byte offset where each token starts in them: the document's
    bytes with the JSON escapes of v1 call arguments undone (and control tokens dropped inside a call), so a line copied
    from a tool output (raw newline) into a v1 argument (\\\\n) is the same text whatever the tokenizer did to it; and,
    unless keep_line_numbers, without the line-number prefixes of file views (_LINE_NUMBER, at every line start of the
    document, context and arguments alike), so the lines of a numbered view and the same lines in an argument are the
    same bytes. A token whose bytes are all dropped has length 0."""
    parts = []
    for t in range(len(ids)):
        u = unesc.get(t)
        parts.append(table.token_bytes[int(ids[t])] if u is None else u)
    off = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parts], out=off[1:])
    raw = b"".join(parts)
    if not keep_line_numbers:
        keep = np.ones(len(raw), dtype=bool)
        for m in _LINE_NUMBER.finditer(raw):
            keep[m.start() : m.end()] = False
        if not keep.all():
            before = np.zeros(len(raw) + 1, dtype=np.int64)  # kept bytes before each byte
            np.cumsum(keep, out=before[1:])
            raw, off = np.frombuffer(raw, dtype=np.uint8)[keep].tobytes(), before[off]
    return raw, off


def _copied_mask(raw: bytes, off: np.ndarray, gid: np.ndarray, copy_min: int, max_candidates: int = 64) -> np.ndarray:
    """(n,) bool: tokens covered by a COPIED WINDOW: copy_min consecutive tokens of one argument value (same gid) whose
    bytes (document_bytes) are at least COPY_MIN_BYTES long and occur earlier in the document, the earlier occurrence
    ending before the window's first byte. Candidates are found by the exact 8-byte key of the window's first 8 bytes
    (the earliest max_candidates occurrences are checked)."""
    n, N = len(gid), copy_min
    out = np.zeros(n, dtype=bool)
    if n < N or not (gid >= 0).any() or len(raw) < COPY_MIN_BYTES:
        return out
    K = COPY_MIN_BYTES
    buf = np.frombuffer(raw, dtype=np.uint8).astype(np.uint64)
    m = len(raw) - K + 1
    key = np.zeros(m, dtype=np.uint64)
    for i in range(K):
        key |= buf[i : i + m] << np.uint64(8 * i)
    order = np.argsort(key, kind="stable")  # equal keys keep their position order
    skey = key[order]
    cand = np.nonzero((gid[: n - N + 1] >= 0) & (gid[: n - N + 1] == gid[N - 1 :]))[0]
    d = np.zeros(n + 1, dtype=np.int64)
    last = None  # (window start token, byte offset of its earlier occurrence): the next window usually continues it
    for j in cand:
        a, b = int(off[j]), int(off[j + N])
        w = b - a
        if w < COPY_MIN_BYTES:
            continue
        pat = raw[a:b]
        found = False
        if last is not None and last[0] == j - 1:
            p = last[1] + (a - int(off[j - 1]))
            found = p + w <= a and raw[p : p + w] == pat
            if found:
                last = (j, p)
        if not found:
            k8 = np.uint64(int.from_bytes(pat[:K], "little"))
            lo = np.searchsorted(skey, k8, side="left")
            hi = np.searchsorted(skey, k8, side="right")
            for p in order[lo : min(hi, lo + max_candidates)]:
                p = int(p)
                if p + w > a:
                    break  # later occurrences end after the window starts
                if raw[p : p + w] == pat:
                    found, last = True, (j, p)
                    break
        if found:
            d[j] += 1
            d[j + N] -= 1
    return np.cumsum(d[:n]) > 0


def label_document(ids, assistant, table: TokenTable, copy_min: int = COPY_MIN_TOKENS, stats: dict | None = None) -> np.ndarray:
    """(n,) int8 fine type of every token of one document, -1 where assistant is 0. ids: (n,) token ids of the WHOLE
    document (context included: copy detection reads every earlier token); assistant: (n,) 0/1, 1 where the assistant
    generates the token. A run of generated tokens is one message body; a document (or row piece) that starts in the
    middle of one is read from its first generated token."""
    ids = np.asarray(ids)
    a = np.asarray(assistant).astype(bool)
    n = len(ids)
    lab = np.full(n, -1, dtype=np.int8)
    gid = np.full(n, -1, dtype=np.int64)
    stats = {} if stats is None else stats
    stats["unesc"] = {}  # token index -> the token's bytes with JSON escapes undone, inside tool calls (per document)
    t = 0
    while t < n:
        if not a[t]:
            t += 1
            continue
        u = t
        while u < n and a[u]:
            # a run ends at its <|im_end|>; two messages back to back stay two runs
            u += 1
            if int(ids[u - 1]) == ID[cf.IM_END]:
                break
        _label_run(ids, t, u, table, lab, gid, stats)
        t = u
    eligible = a & ((lab == F["arg"]) | (lab == F["code_arg"]))
    if eligible.any():
        raw, off = document_bytes(ids, table, stats.get("unesc", {}))
        cm = _copied_mask(raw, off, np.where(eligible, gid, -1), copy_min)
        lab[cm & eligible] = F["copied_arg"]
    return lab


# -------------------------------------------------------------------------------------------------------------------
# a second, independent labelling: the segment a loop schedule could read from the tokens already emitted
#
# The types above label the token that is predicted and some of them need the future (copied spans, narration against final
# answer, code). A schedule sets the pass count of position t, whose output predicts token t + 1, before t + 1 exists. So here
# the label belongs to the POSITION t and is a function of ids[:t+1] (and of whether the harness asks the model for a token at
# t + 1) only:
#
#   reasoning   after <think> and up to the position before </think>: inside the pair.
#   content     the position before the first token of a message (the choice to think, call or answer), text outside both
#               pairs (a preamble or a final answer), the position of </think>, and every position from that of
#               </tool_call> up to the next <tool_call> or the <|im_end|>: the choice to call again or to stop.
#   call_head   from the position of <tool_call> through the position of the token that holds the tool name's last byte
#               (the token with the closing quote of the name in v1; with the newline that ends the name in v2), the whole
#               token inclusive even if it holds more bytes after the quote. If the bytes after <tool_call> stop being the
#               start of  {"name": "N"  (v1) or  N\n  (v2), the head ends at the token before the one that breaks it.
#   call_body   every later position of the call, up to the one before the position of </tool_call>.
#   unplaced    anything the state machine cannot place: a run that starts at the first token of a row (cut, so its state is
#               unknown) until a token that decides the state (<think>, </think>, <tool_call>, </tool_call>, <|im_end|>),
#               and a control token where the format does not allow it (<think> inside a call, </think> in content, ...)
#               until the next <|im_end|> or deciding token.
#
# A position whose next token is not generated has no segment (-1). To attach the segment to a stored per-token loss, which is
# indexed by the TARGET token (ce[u] = loss of predicting ids[u] from ids[:u]), use segment_targets: the target u takes the
# segment of position u - 1.

SEGMENTS = ("reasoning", "content", "call_head", "call_body", "unplaced")
SEG = {name: i for i, name in enumerate(SEGMENTS)}
_ST_START, _ST_REASON, _ST_CONTENT, _ST_HEAD, _ST_BODY, _ST_UNPLACED = range(6)
_ST_TO_SEG = {_ST_START: SEG["content"], _ST_REASON: SEG["reasoning"], _ST_CONTENT: SEG["content"], _ST_HEAD: SEG["call_head"],
              _ST_BODY: SEG["call_body"], _ST_UNPLACED: SEG["unplaced"]}
_WS_BYTES = b" \t\n\r"


class _NameScan:
    """Reads the bytes after <tool_call> up to the end of the tool name: v1  ws* { ws* "name" ws* : ws* " name-bytes "  or
    v2  ws* name-bytes \\n. feed() returns 0 (still reading), 1 (the name's last byte was just read; the rest of the data is
    ignored) or -1 (the bytes cannot be the start of either form)."""

    __slots__ = ("st", "k")

    def __init__(self):
        self.st, self.k = 0, 0

    def feed(self, data: bytes) -> int:
        for c in data:
            st = self.st
            if st == 0:  # leading whitespace; "{" opens v1, any other byte is the first byte of a v2 name
                if c in _WS_BYTES:
                    continue
                self.st = 1 if c == 123 else 8
            elif st in (1, 4, 5):  # whitespace, then " (1: the key's quote), : (4), " (5: the value's quote)
                if c in _WS_BYTES:
                    continue
                if c != (58 if st == 4 else 34):
                    return -1
                self.st = {1: 2, 4: 5, 5: 6}[st]
                self.k = 0
            elif st == 2:  # the literal  name
                if c != b"name"[self.k]:
                    return -1
                self.k += 1
                if self.k == 4:
                    self.st = 3
            elif st == 3:  # the key's closing quote
                if c != 34:
                    return -1
                self.st = 4
            elif st == 6:  # inside the name string
                if c == 92:
                    self.st = 7
                elif c == 34:
                    return 1
            elif st == 7:
                self.st = 6
            elif st == 8:  # v2: name bytes up to the newline
                if c == 10:
                    return 1
        return 0


def segment_positions(ids, assistant, table: TokenTable) -> np.ndarray:
    """(n,) int8 SEGMENTS index of every POSITION t whose next token t + 1 is generated, else -1. The label of t is a function
    of ids[:t+1], assistant[:t+1] and assistant[t+1] (the harness asking for a token, not a token's content): replacing
    the tokens after t changes nothing before or at t (tests/test_token_types_segments.py)."""
    ids = np.asarray(ids)
    a = np.asarray(assistant).astype(bool)
    n = len(ids)
    out = np.full(n, -1, dtype=np.int8)
    state, scan, done = None, None, False
    for t in range(n):
        nxt = bool(a[t + 1]) if t + 1 < n else False
        if not a[t]:
            state = None
            if nxt:
                out[t] = SEG["content"]  # the position before a message's first token
            continue
        if t == 0 or not a[t - 1]:
            state = _ST_UNPLACED if t == 0 else _ST_START
        x = int(ids[t])
        if state == _ST_HEAD and done:
            state = _ST_BODY
        if x == ID[cf.IM_END]:
            state = _ST_START
        elif x == ID[cf.THINK]:
            state = _ST_REASON if state in (_ST_START, _ST_UNPLACED, _ST_REASON) else _ST_UNPLACED
        elif x == ID[cf.THINK_END]:
            state = _ST_CONTENT if state in (_ST_REASON, _ST_UNPLACED) else _ST_UNPLACED
        elif x == ID[cf.TOOL_CALL]:
            if state in (_ST_START, _ST_CONTENT, _ST_UNPLACED):
                state, scan, done = _ST_HEAD, _NameScan(), False
            else:
                state = _ST_UNPLACED
        elif x == ID[cf.TOOL_CALL_END]:
            state = _ST_CONTENT if state in (_ST_HEAD, _ST_BODY, _ST_UNPLACED) else _ST_UNPLACED
        elif state == _ST_HEAD:
            if x < N_SPECIAL:
                state = _ST_BODY  # a control token inside the name: not a name
            else:
                r = scan.feed(table.token_bytes[x])
                if r < 0:
                    state = _ST_BODY
                elif r > 0:
                    done = True  # this token is the last of the head
        elif state == _ST_START and x >= N_SPECIAL:
            state = _ST_CONTENT
        if nxt:
            out[t] = _ST_TO_SEG[state]
    return out


def segment_targets(ids, assistant, table: TokenTable) -> np.ndarray:
    """(n,) int8: for every generated token u, the segment of the position u - 1 that predicts it (the stored per-token
    losses are indexed by the target); -1 where assistant is 0. The first token of a row has no predicting position in the
    row: unplaced."""
    a = np.asarray(assistant).astype(bool)
    pos = segment_positions(ids, a, table)
    out = np.full(len(a), -1, dtype=np.int8)
    out[1:] = pos[:-1]
    out[~a] = -1
    if len(a) and a[0]:
        out[0] = SEG["unplaced"]
    return out


COPY_OFFSETS = ("first token", "tokens 2 to 8", "later tokens")


def copy_offsets(lab: np.ndarray) -> np.ndarray:
    """(n,) int32: for a token typed copied_arg, its 0-based offset inside its copied span (a span is a maximal run of
    consecutive tokens typed copied_arg in the document); -1 elsewhere."""
    lab = np.asarray(lab)
    c = lab == F["copied_arg"]
    out = np.full(len(lab), -1, dtype=np.int32)
    if not c.any():
        return out
    idx = np.arange(len(lab))
    start = c & ~np.concatenate([[False], c[:-1]])
    run_start = np.maximum.accumulate(np.where(start, idx, 0))
    out[c] = (idx - run_start)[c]
    return out


def copy_offset_bucket(off: np.ndarray) -> np.ndarray:
    """(n,) int8 index into COPY_OFFSETS: offset 0, offsets 1 to 7 (tokens 2 to 8), offsets 8 and later; -1 for -1."""
    off = np.asarray(off)
    return np.where(off < 0, -1, np.where(off == 0, 0, np.where(off <= 7, 1, 2))).astype(np.int8)


def split_shares(seg: np.ndarray, lab: np.ndarray, off_bucket: np.ndarray, byte_lens: np.ndarray) -> dict:
    """Shares of generated tokens and bytes per SEGMENT (segment_targets) and per COPY_OFFSETS bucket (of all generated tokens
    and of the copied ones), and the cross table: per segment, the share of its tokens in each rolled-up type. Flat arrays
    over the documents, aligned; seg and lab are -1 where the assistant does not generate."""
    seg, lab, off_bucket = np.asarray(seg), np.asarray(lab), np.asarray(off_bucket)
    bl = np.asarray(byte_lens).astype(np.float64)
    on = seg >= 0
    total_t, total_b = int(on.sum()), float(bl[on].sum())
    s_tok = np.bincount(seg[on].astype(np.int64), minlength=len(SEGMENTS))
    s_byt = np.bincount(seg[on].astype(np.int64), weights=bl[on], minlength=len(SEGMENTS))
    ob = off_bucket >= 0
    o_tok = np.bincount(off_bucket[ob].astype(np.int64), minlength=len(COPY_OFFSETS))
    o_byt = np.bincount(off_bucket[ob].astype(np.int64), weights=bl[ob], minlength=len(COPY_OFFSETS))
    ty = coarse(lab)
    both = on & (ty >= 0)
    cross = np.bincount(seg[both].astype(np.int64) * len(TYPES) + ty[both], minlength=len(SEGMENTS) * len(TYPES)).reshape(len(SEGMENTS), len(TYPES))
    n_copied = max(int(ob.sum()), 1)
    return {
        "generated_tokens": total_t, "generated_bytes": int(total_b),
        "segments": {s: {"tokens": int(s_tok[i]), "token_share": s_tok[i] / max(total_t, 1), "byte_share": s_byt[i] / max(total_b, 1)}
                     for i, s in enumerate(SEGMENTS)},
        "copy_offsets": {s: {"tokens": int(o_tok[i]), "token_share": o_tok[i] / max(total_t, 1), "byte_share": o_byt[i] / max(total_b, 1),
                             "share_of_copied": o_tok[i] / n_copied} for i, s in enumerate(COPY_OFFSETS)},
        "cross": {s: {"tokens": int(cross[i].sum()), **{name: cross[i, j] / max(int(cross[i].sum()), 1) for j, name in enumerate(TYPES)}}
                  for i, s in enumerate(SEGMENTS)},
    }


def midway_answer_tokens(ids, assistant, lab: np.ndarray) -> int:
    """How many generated tokens are final_answer or code_content in a message that makes no call and is NOT the
    document's last assistant message. Such a message is an earlier answer of a multi-turn chat, or an action written
    as text that the format did not parse into a call (its text is then typed as an answer, wrongly): an upper bound
    on that mistake, reported next to the shares."""
    ids, a = np.asarray(ids), np.asarray(assistant).astype(bool)
    runs, t, n = [], 0, len(ids)
    while t < n:
        if not a[t]:
            t += 1
            continue
        u = t
        while u < n and a[u]:
            u += 1
            if int(ids[u - 1]) == ID[cf.IM_END]:
                break
        runs.append((t, u))
        t = u
    count = 0
    for t, u in runs[:-1]:
        seg = lab[t:u]
        if not (seg == F["call_open"]).any():
            count += int(((seg == F["final_answer"]) | (seg == F["code_content"])).sum())
    return count


def coarse(lab: np.ndarray) -> np.ndarray:
    """Fine labels to TYPES indices; -1 stays -1."""
    out = np.full(lab.shape, -1, dtype=np.int8)
    ok = lab >= 0
    out[ok] = FINE_TO_TYPE[lab[ok]]
    return out


# -------------------------------------------------------------------------------------------------------------------
# rendering a conversation to ids (tests, and the sample counts)


def conversation_ids(conversation: dict, encode_text, version: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(ids, generated flags) of a rendered conversation, as the shard builder makes them: control
    pieces by id, message text by encode_text(str) -> list[int], one piece at a time."""
    ids, gen = [], []
    for text, g, control in cf.pieces(conversation, version):
        p = [ID[text]] if control else list(encode_text(text))
        ids += p
        gen += [int(g)] * len(p)
    return np.asarray(ids, dtype=np.int64), np.asarray(gen, dtype=np.uint8)


def summarize(labels: list[np.ndarray], byte_lens: list[np.ndarray], copy_extra: dict | None = None) -> dict:
    """Counts of generated tokens and bytes per fine type and per rolled-up type over documents."""
    tok = np.zeros(len(FINE), dtype=np.int64)
    byt = np.zeros(len(FINE), dtype=np.int64)
    for lab, bl in zip(labels, byte_lens):
        on = lab >= 0
        tok += np.bincount(lab[on], minlength=len(FINE))
        byt += np.bincount(lab[on], weights=bl[on], minlength=len(FINE)).astype(np.int64)
    total_t, total_b = int(tok.sum()), int(byt.sum())
    fine = {f: {"tokens": int(tok[i]), "bytes": int(byt[i]), "token_share": tok[i] / max(total_t, 1),
                "byte_share": byt[i] / max(total_b, 1)} for i, f in enumerate(FINE)}
    rolled = {}
    for ty in TYPES:
        idx = [i for i, f in enumerate(FINE) if ROLLUP[f] == ty]
        t, b = int(tok[idx].sum()), int(byt[idx].sum())
        rolled[ty] = {"tokens": t, "bytes": b, "token_share": t / max(total_t, 1), "byte_share": b / max(total_b, 1)}
    return {"generated_tokens": total_t, "generated_bytes": total_b, "fine": fine, "types": rolled}
