"""Generator of the transcoding probe.

Question it answers: can the model move a multi-line string between the two forms it meets in agentic
data, the raw form (tool results, user text, reasoning: real newlines, quotes, backslashes, tabs) and the
JSON-escaped form (the arguments of a tool call in the v1 chat rendering: \\n, \\", \\\\, \\t)? The copy
analysis found verbatim copies that cross this boundary; this probe isolates the transcoding step.

One probe, `transcode`, two directions over the SAME content (a base):
  esc2raw  A tool result is a JSON object whose `content` value is the escaped string. The user asks for
           that string as plain text (decode the escapes: \\n -> line break, \\" -> quote, \\\\ -> one
           backslash, \\t -> tab). The answer is the decoded text. In the current version (probes-transcode-v2)
           the reply opens with a fixed, teacher-forced lead-in (LEADIN_ESC2RAW: a short
           sentence and a code fence opening, where chat-format v1 assistants put raw text) and the answer follows it
           in the same text piece (a tail item, empty suffix), so it no longer opens a fresh assistant reply (a
           v1-trained model expects a tool call, <think> or prose there). `leadin=None` builds the v1 item.
  raw2esc  A tool result (`cat`) shows the raw text. The user asks to save it with the `write_file` tool.
           The answer is the JSON-escaped text between the quotes of the call's `content` argument (the v1
           rendering: chat_format v1, `json.dumps(..., ensure_ascii=False)[1:-1]`), a "tail" item as in
           probe_gen_copy.edit_item: prefix and suffix of the call are fixed context around the answer.
Cells (cell dict): dir {esc2raw, raw2esc} x lines {1, 3, 8, 20}. Group: kind {code, text}. A base (the
content) is rendered in both directions, so the two directions are paired (as edit_old's renderings).
Content: synthetic only (fixed word lists and templates; no training data, no
held-out document). Code lines (python, js, shell, C, SQL) carry quotes, backslashes (regex, Windows
paths, "\\n" written inside a string literal), and tab or space indentation; text lines are dialogue,
paths, tab-separated columns. Every base holds at least one character that needs escaping besides the line
breaks (`"`, `\\` or a tab inside a line), so the answer differs from the source even for one line.
Rules (builder, per tokenizer): raw2esc content may not start with a character whose escaped form would merge with the opening quote of the
argument (boundary_ok), which in practice excludes lines that begin with a quote or backslash; esc2raw (v2): no token may
straddle the end of the lead-in and the start of the answer (the same boundary_ok).
Rules: the first character of the content and the last are not whitespace; no trailing spaces; the answer
(in its target form) occurs 0 times in the context text (the model cannot copy it, it must transcode), and
the source form occurs exactly once.
Floors (`copy-only`, per item and tokenizer, from the builder): the source as it stands (the untranscoded
string) taken as the answer. exact = 0 by construction (asserted: the forms differ), token_acc = share of
answer positions with the same token, lcs_share = (token LCS of the two token sequences) / answer tokens.
"""

from __future__ import annotations

import json

import numpy as np

from ced.eval import probe_gen_copy as gc
from ced.eval import probe_gen_ext as ge
from ced.eval.probe_gen import BASE_SEED

PROBE = "transcode"
PROBE_CODE = 105
DIRS_ = ("esc2raw", "raw2esc")
LINES = (1, 3, 8, 20)
KINDS = ("code", "text")
MARK = "@@TC@@"
ESCAPABLE = ('"', "\\", "\t")

TOOLS = [
    {"name": "get_snippet", "description": "Fetch a saved snippet by id.",
     "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    {"name": "run", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
]

WORDS = ge.VERBS + ge.NOUNS
FIRST = """Ada Ben Cleo Dev Eli Fay Gus Hana Ivo Jun Kai Lena Milo Nia Omar Pia Quin Rae Sam Tess Uma Vik Wren Yara Zed""".split()
EXT = {"python": "py", "javascript": "js", "shell": "sh", "c": "c", "sql": "sql", "json": "json", "text": "txt"}


def _w(rng) -> str:
    return str(ge._pick(rng, WORDS))


def _n(rng) -> str:
    return str(ge._pick(rng, ge.NOUNS))


def _i(rng, hi: int = 99) -> int:
    return int(rng.integers(1, hi))


# Each template builder draws its words once and returns lambdas (one is called): code-like lines with the
# escapable characters of real code (quotes, regex and path backslashes, "\n" inside literals). Not valid programs.
def _py(rng):
    n, w, v = _n(rng), _w(rng), _w(rng)
    return [lambda: f'print("{n}: {v} {_i(rng)} items\\n")',
            lambda: f'{n} = r"C:\\{ge._pick(rng, ge.DIRS)}\\{n}\\{w}.log"',
            lambda: f'raise ValueError(f"bad {n}: {{{n}!r}} (\\"{w}\\" expected)")',
            lambda: f'pattern = re.compile(r"\\d+\\s{n}\\b")',
            lambda: f"{n}[\"{w}\"] = '{v}'",
            lambda: f'f.write("{w}\\t{v}\\n")',
            lambda: f'if name == "{n}":',
            lambda: f'return "{{}}/{n}".format({w})',
            lambda: f'{n}_{w} = {_i(rng)}']


def _js(rng):
    n, w, v = _n(rng), _w(rng), _w(rng)
    return [lambda: f'console.log("{n} \\"{w}\\" ready");',
            lambda: f'const re = /\\/{n}\\/(\\d+)\\/{w}/;',
            lambda: f"const {n} = {{ \"{w}\": '{v}', {v}: {_i(rng)} }};",
            lambda: f'throw new Error(`{n} failed: ${{{w}}}\\n`);',
            lambda: f'if ({n}["{w}"] === "{v}") return;']


def _sh(rng):
    n, w, v = _n(rng), _w(rng), _w(rng)
    return [lambda: f'echo "done: $HOME/{n}" | grep -E \'^\\w+\\s{w}\'',
            lambda: f"printf '%s\\t%s\\n' \"${n}\" \"${w}\"",
            lambda: f'{n}="$(cat "/tmp/{w}.txt")"',
            lambda: f'sed -i "s/\\"{w}\\"/\\"{v}\\"/g" {n}.json',
            lambda: f'if [ "${n}" = "{w}" ]; then']


def _c(rng):
    n, w, v = _n(rng), _w(rng), _w(rng)
    return [lambda: f'printf("%s: %d\\n", {n}, {w});',
            lambda: f'fputs("{n}: \\"{w}\\"\\r\\n", stderr);',
            lambda: f'char *{n} = "C:\\\\{w}\\\\{v}";',
            lambda: f"if ({n}[0] == '\\\\') {{",
            lambda: f"char sep = '\\t'; /* {w} */"]


def _sql(rng):
    n, w, v = _n(rng), _w(rng), _w(rng)
    return [lambda: f'SELECT * FROM {n} WHERE name = \'O\'\'{w}\' AND tag = "{v}";',
            lambda: f'UPDATE {n} SET note = "line1\\nline2" WHERE id = {_i(rng)};',
            lambda: f'-- {w} "{v}" (see C:\\{n}\\schema)']


def _text(rng):
    n, w, v, p = _n(rng), _w(rng), _w(rng), str(ge._pick(rng, FIRST))
    return [lambda: f'{p} said "{w} the {n}" and left.',
            lambda: f'The {n} was "{v}" for {_i(rng)} days (see C:\\Users\\{p}\\{n}).',
            lambda: 'Note: type \\n to start a new line, \\t for a tab.',
            lambda: f'{p}\t{n}\t{_i(rng)}\t"{w}"',
            lambda: f'It is {p}\'s {n}; "{v}" is not allowed here.',
            lambda: f'Open the {n} folder, then pick "{w}" and press Enter.',
            lambda: f'Total: {_i(rng, 500)} {n}s \\ {_i(rng, 500)} {v}s',
            lambda: f'He wrote: "{w}\\{v}" and nothing else.',
            lambda: f'Remember the {n} and the {w} before {p} arrives.']


_CODE = {"python": _py, "javascript": _js, "shell": _sh, "c": _c, "sql": _sql}


def gen_content(rng: np.random.Generator, kind: str, n_lines: int) -> tuple[str, dict]:
    """The raw multi-line string and its meta (kind, lang). Retries inside until the rules hold."""
    for _ in range(200):
        if kind == "code":
            lang = str(ge._pick(rng, list(_CODE)))
            unit = "\t" if rng.random() < 0.3 else "    "
            depth, lines = 0, []
            for k in range(n_lines):
                tmpl = _CODE[lang](rng)
                body = tmpl[int(rng.integers(len(tmpl)))]()
                if k and rng.random() < 0.35:
                    depth = max(0, min(3, depth + int(rng.integers(-1, 2))))
                lines.append((unit * depth if k else "") + body)
        else:
            lang, lines = "text", []
            for _k in range(n_lines):
                tmpl = _text(rng)
                lines.append(tmpl[int(rng.integers(len(tmpl)))]())
        if n_lines >= 8 and rng.random() < 0.5:  # a blank line inside, now and then: "\n\n" in the escaped form
            lines[int(rng.integers(1, n_lines - 1))] = ""
        text = "\n".join(lines)
        if text[0].isspace() or text[-1].isspace() or any(x != x.rstrip() for x in lines):
            continue
        if not any(c in text for c in ESCAPABLE):
            continue
        return text, {"kind": kind, "lang": lang}
    raise RuntimeError("transcode: no content")


def escape(text: str) -> str:
    """The JSON string-literal body of `text`, as chat_format v1 writes tool-call arguments."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def rng_tc(n_lines: int, index: int, attempt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, PROBE_CODE, n_lines, index, attempt]))


def base_content(n_lines: int, index: int, attempt: int = 0) -> dict:
    """The content shared by the two directions. The kind alternates with the index (half code, half text)."""
    rng = rng_tc(n_lines, index, attempt)
    kind = KINDS[index % 2]
    raw, meta = gen_content(rng, kind, n_lines)
    stem = f"{_n(rng)}_{_w(rng)}"
    ext = EXT[meta["lang"]]
    return {"raw": raw, "esc": escape(raw), "kind": kind, "lang": meta["lang"], "n_lines": n_lines, "attempt": attempt,
            "snippet_id": int(rng.integers(100, 900)), "src": f"/tmp/{stem}.{ext}", "dest": f"/repo/{ge._pick(rng, ge.DIRS)}/{stem}.{ext}"}


# The fixed, teacher-forced start of the esc2raw reply before the raw answer (probes-transcode-v2).
# Context, not scored. raw2esc has none: its answer already follows the tool-call header (the JSON prefix of the call).
LEADIN_ESC2RAW = "Here is the decoded text:\n```\n"

QUESTION_ESC2RAW = ("Give me the `content` string of that result as plain text. Decode the JSON escapes: each \\n becomes a real line break, "
                    "\\\" a plain double quote, \\\\ a single backslash and \\t a tab. Output only the decoded text, nothing else.")


def transcode_item(n_lines: int, direction: str, index: int, attempt: int = 0, base: dict | None = None,
                   leadin: str | None = LEADIN_ESC2RAW) -> gc.ConvItem:
    """One item. `leadin` (esc2raw only): the fixed start of the reply before the raw answer (a tail prefix with an empty
    suffix); None gives the v1 item, whose answer opens the reply."""
    assert direction in DIRS_ and n_lines in LINES
    b = base or base_content(n_lines, index, attempt)
    raw, esc = b["raw"], b["esc"]
    bid = f"{PROBE}/lines={n_lines}/{index:04d}"
    tail = None
    if direction == "esc2raw":
        obj = json.dumps({"id": b["snippet_id"], "lang": b["lang"], "content": raw}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": gc.SYSTEM},
            {"role": "user", "content": f"Fetch snippet {b['snippet_id']} with the get_snippet tool."},
            {"role": "assistant", "content": "I'll fetch it.", "tool_calls": [{"name": "get_snippet", "arguments": {"id": b["snippet_id"]}}]},
            {"role": "tool", "content": obj},
            {"role": "user", "content": QUESTION_ESC2RAW},
        ]
        pieces = gc.render_messages(messages, TOOLS) + gc.GEN_HEADER
        answer = raw
        if leadin:
            tail = {"prefix": leadin, "suffix": ""}
    else:
        leadin = None
        messages = [
            {"role": "system", "content": gc.SYSTEM},
            {"role": "user", "content": (f"Read {b['src']} with `cat`, then save the same text to {b['dest']} with the write_file tool "
                                         "(arguments `path` and `content`). The content must be exactly the file's text; the file has no final newline "
                                         "(the line break after the `cat` output belongs to the tool result, not to the file).")},
            {"role": "assistant", "content": "I'll read the file first.", "tool_calls": [{"name": "run", "arguments": {"cmd": f"cat {b['src']}"}}]},
            {"role": "tool", "content": raw},
            {"role": "assistant", "content": "Now I'll save it.",
             "tool_calls": [{"name": "write_file", "arguments": {"path": b["dest"], "content": MARK}}]},
        ]
        per_msg = gc.cf.message_pieces({"messages": messages, "tools": TOOLS}, 1)
        pieces = [(t, bool(c)) for _, ps in per_msg[:-1] for t, _, c in ps]
        final = [(t, bool(c)) for t, _, c in per_msg[-1][1]]
        at = next(i for i, (t, c) in enumerate(final) if not c and MARK in t)
        text = final[at][0]
        i = text.index(MARK)
        assert text[i - 1] == '"' and text[i + len(MARK)] == '"'
        pieces += final[:at]
        answer = esc
        tail = {"prefix": text[:i], "suffix": text[i + len(MARK):]}
    meta = {"base": bid, "kind": b["kind"], "lang": b["lang"], "raw": raw, "n_escapable": sum(raw.count(c) for c in ESCAPABLE),
            "n_chars_raw": len(raw), "n_chars_esc": len(esc), "attempt": b["attempt"], "leadin": leadin}
    return gc.ConvItem(id=f"{bid}/{direction}", probe=PROBE, cell={"dir": direction, "lines": n_lines}, group={"kind": b["kind"]},
                       context_pieces=[[t, c] for t, c in pieces], answer=answer, tail=tail, meta=meta)


def source_form(item: gc.ConvItem) -> str:
    """The untranscoded string: the form the source shows (esc2raw: the escaped body; raw2esc: the raw text)."""
    return item.meta["raw"] if item.cell["dir"] == "raw2esc" else escape(item.meta["raw"])


def check_transcode_item(item: gc.ConvItem) -> None:
    raw = item.meta["raw"]
    esc = escape(raw)
    ctx = gc.context_text(item)
    ans, d = item.answer, item.cell["dir"]
    assert ans == (raw if d == "esc2raw" else esc), item.id
    assert ans != source_form(item), item.id  # the forms differ: transcoding is needed
    assert ans not in ctx, f"{item.id}: the answer occurs in the context"
    assert not ans[0].isspace() and not ans[-1].isspace(), item.id
    assert json.loads('"' + esc + '"') == raw, item.id
    assert raw.count("\n") + 1 == item.cell["lines"], item.id
    assert any(c in raw for c in ESCAPABLE), item.id
    assert ctx.count(source_form(item)) == 1, f"{item.id}: the source form is not present exactly once"
    if d == "raw2esc":
        assert item.tail["prefix"].endswith('"content": "') and item.tail["suffix"].startswith('"}'), item.id
        assert item.tail["prefix"].count("write_file") == 1 and not item.meta.get("leadin"), item.id
    else:
        lead = item.meta.get("leadin")
        assert item.tail == ({"prefix": lead, "suffix": ""} if lead else None), f"{item.id}: lead-in frame"
        assert list(item.context_pieces[-1]) == ["assistant\n", False], item.id
    if d == "esc2raw":
        assert ctx.rindex("Give me the `content`") > ctx.index(source_form(item)), item.id
