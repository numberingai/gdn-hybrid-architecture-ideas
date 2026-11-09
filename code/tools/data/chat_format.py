"""The provisional chat format (ChatML-like, Hermes/Qwen-style tool calls), used by every stage until
the real format is decided.

A conversation is {"tools": [<JSON schema>, ...] (optional), "messages": [...]}, each message
{"role": "system" | "user" | "assistant" | "tool", "content": str,
 "reasoning": str (assistant, optional), "tool_calls": [{"name": str, "arguments": {...}}] (assistant)}.
It renders as

    <|im_start|>system\n{system text}[\n\n# Tools\n\n<tools>\n{schema}\n...</tools>]<|im_end|>\n
    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n[<think>\n{reasoning}\n</think>\n\n]{content}[\n<tool_call>\n{json}\n</tool_call>...]<|im_end|>\n
    <|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n

The model generates an assistant message's body and its <|im_end|>; the header
"<|im_start|>assistant\n" and the newline after <|im_end|> come from the harness.

Control tokens come only from the template (2026-10-05): `pieces` returns the rendered conversation
as (text, generated, control) pieces built from the structure, never by searching the text. A
control piece is exactly one special token, encoded by id; every other piece is message text,
encoded with special-token recognition off, so a literal "<|endoftext|>" or "<tools>" inside code, a
tool output or a message stays text. Text pieces are encoded separately, so generated tokens never
share a token with harness text.
"""

from __future__ import annotations

import json
import re

IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
TOOL_CALL, TOOL_CALL_END = "<tool_call>", "</tool_call>"
TOOL_RESPONSE, TOOL_RESPONSE_END = "<tool_response>", "</tool_response>"
THINK, THINK_END, TOOLS, TOOLS_END = "<think>", "</think>", "<tools>", "</tools>"
SPECIAL_TOKENS = [EOT, IM_START, IM_END, TOOL_CALL, TOOL_CALL_END, TOOL_RESPONSE, TOOL_RESPONSE_END,
                  THINK, THINK_END, TOOLS, TOOLS_END] + [f"<|reserved_{i}|>" for i in range(21)]

# Tool-call syntax and leftovers of other chat formats (XML-style, function=... calls,
# minimax, reserved tokens of other tokenizers). A chat document whose message content, reasoning or
# tool-call arguments contain any of these is not in our format: the builders drop it.
FOREIGN_MARKUP = re.compile(r"<\|tool_call|<\|tool_calls_section|<function_calls>|<invoke name=|<function=|<parameter="
                            r"|<\|reserved_token_|<minimax:|<tool_name>|<use_tool>")
# Control strings that must never be literal text in an assistant message's content (outside the
# reasoning field the template renders): our own think and tool-call markup.
LITERAL_CONTROL_IN_ASSISTANT = re.compile(r"</?think>|</?tool_call>")
# An assistant message that claims to be a named commercial model: the gate drops such samples. The string match is deliberately plain.
CLAIMS_NAMED_COMMERCIAL_MODEL = re.compile(r"\bI(?:'m| am) Claude\b|\bClaude \d")
# A tool response that reports a failed call: an off-schema call followed by one is error recovery.
TOOL_ERROR = re.compile(r"error|not found|does not exist|unknown tool", re.I)
# The final sentence of a last assistant message that announces an action it never takes.
# A closing offer of further help is not an action: the negative lookahead keeps it out.
ANNOUNCES_ACTION = re.compile(r"^(?:(?:Now let me|Let me)(?!\s+know\b)|I'll)\b")

# Chat format v2 (future builds only). Version 1 is the default and is byte-identical to the format
# the shards and eval sets of the reported runs were built with. Argument delimiters are template control tokens
# (reserved ids, so text can never forge them).
ARG_STR, ARG_JSON, ARG_END = "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"
V2_CONTROLS = (ARG_STR, ARG_JSON, ARG_END)

Piece = tuple[str, bool, bool]  # (text, generated, control)

# A file view with a line number on every line ("cat -n": right-aligned number, a tab; or "N→").
_NUMBERED = re.compile(r"^ *(\d+)(\t|\u2192)")
_CATN_PREFACE = re.compile(r"^.*`cat -n`.*:$")
LINE_HEADER = "[line numbers removed; first line is {n}]\n"
_LINE_HEADER_RE = re.compile(r"^\[line numbers removed; first line is (\d+)\]\n")


def strip_line_numbers(text: str) -> tuple[str, str] | None:
    """If `text` is a file view with a line number on every line, return (head, body): `head` is the
    optional one-line preface plus the header stating the first line number, `body` the lines
    without prefixes. Else None; a partial match is never stripped.
    Detection (a view needs the `cat -n` preface or a number right-aligned to width >= 6; zero-padded numbers are refused): optionally a first line containing `cat -n` and ending in ":"; then at least two lines
    (a final empty piece from a trailing newline is allowed), every one of which is
    `<spaces><n><TAB or arrow><rest>`, with n consecutive (+1 per line) and one separator style."""
    lines = text.split("\n")
    preface = ""
    if lines and _CATN_PREFACE.match(lines[0]):
        preface = lines[0] + "\n"
        lines = lines[1:]
    trailing = bool(lines) and lines[-1] == ""
    if trailing:
        lines = lines[:-1]
    if len(lines) < 2:
        return None
    first = None
    sep = None
    body = []
    for i, line in enumerate(lines):
        m = _NUMBERED.match(line)
        if not m:
            return None
        if m.group(1).startswith("0") and len(m.group(1)) > 1:
            return None  # zero-padded: the width would be lost
        if i == 0 and not preface and m.end(1) < 6:
            return None  # no cat -n preface and narrower than a real view (width 6): could be a table
        n = int(m.group(1))
        if first is None:
            first, sep = n, m.group(2)
        elif n != first + i or m.group(2) != sep:
            return None
        body.append(line[m.end():])
    return preface + LINE_HEADER.format(n=first), "\n".join(body) + ("\n" if trailing else "")


def unstrip_line_numbers(head: str, body: str, sep: str = "\t") -> str:
    """Inverse of `strip_line_numbers` for the harness side (number width is not kept: width 6)."""
    m = _LINE_HEADER_RE.search(head[head.find("[line numbers removed"):])
    first = int(m.group(1))
    preface = head[: head.find("[line numbers removed")]
    lines = body.split("\n")
    trailing = lines[-1] == ""
    if trailing:
        lines = lines[:-1]
    return preface + "\n".join(f"{first + i:6d}{sep}{l}" for i, l in enumerate(lines)) + ("\n" if trailing else "")


def _call_pieces_v2(call: dict) -> list[Piece]:
    name, args = call["name"], call.get("arguments", {})
    if "\n" in name:
        raise ValueError("tool name with a newline")
    out: list[Piece] = [(TOOL_CALL, True, True), (f"\n{name}\n", True, False)]
    for key, value in args.items():
        if not isinstance(key, str) or "\n" in key:
            raise ValueError("argument name must be a string without a newline")
        if isinstance(value, str):
            out += [(ARG_STR, True, True), (f"{key}\n", True, False), (value, True, False)]
        else:
            out += [(ARG_JSON, True, True), (f"{key}\n", True, False),
                    (json.dumps(value, ensure_ascii=False, separators=(",", ":")), True, False)]
        out += [(ARG_END, True, True), ("\n", True, False)]
    out.append((TOOL_CALL_END, True, True))
    return out


def parse_tool_call_v2(ps: list[tuple[str, bool]]) -> dict:
    """Parse one v2 call from (text, control) pieces (or a decoded token stream where control tokens
    are known by id) between <tool_call> and </tool_call> inclusive. Exact inverse of the renderer.
    Raises ValueError on malformed input (the harness treats that as an invalid call)."""
    merged: list[tuple[str, bool]] = []
    for t, c in ps:
        if not c and merged and not merged[-1][1]:
            merged[-1] = (merged[-1][0] + t, False)
        elif t or c:
            merged.append((t, c))
    if len(merged) < 3 or merged[0] != (TOOL_CALL, True) or merged[-1] != (TOOL_CALL_END, True):
        raise ValueError("not a tool call")
    inner = merged[1:-1]
    if not inner[0][0].startswith("\n") or inner[0][1]:
        raise ValueError("missing name")
    head = inner[0][0]
    if not head.endswith("\n") or len(head) < 2:
        raise ValueError("missing name")
    name = head[1:-1]
    if "\n" in name:
        raise ValueError("bad name")
    args: dict = {}
    i = 1
    while i < len(inner):
        kind, is_ctl = inner[i]
        if not is_ctl or kind not in (ARG_STR, ARG_JSON):
            raise ValueError("expected argument start")
        i += 1
        text = ""
        if i < len(inner) and not inner[i][1]:
            text = inner[i][0]
            i += 1
        if i >= len(inner) or inner[i] != (ARG_END, True):
            raise ValueError("argument not closed")
        i += 1
        key, nl, value = text.partition("\n")
        if not nl:
            raise ValueError("argument without name")
        if key in args:
            raise ValueError("duplicate argument")
        args[key] = value if kind == ARG_STR else json.loads(value)
        if i < len(inner) and not inner[i][1]:  # the newline after ARG_END
            if inner[i][0] != "\n":
                raise ValueError("junk between arguments")
            i += 1
    return {"name": name, "arguments": args}


def _tool_response_text_pieces_v2(content: str) -> list[Piece]:
    stripped = strip_line_numbers(content)
    if stripped:
        head, body = stripped
        return [("\n" + head, False, False), (body, False, False), ("\n", False, False)]
    return [("\n", False, False), (content, False, False), ("\n", False, False)]


def _body_pieces(m: dict, version: int = 1) -> list[Piece]:
    """An assistant message's body (without <|im_end|>), all generated."""
    out: list[Piece] = []
    if m.get("reasoning"):
        out += [(THINK, True, True), (f"\n{m['reasoning']}\n", True, False), (THINK_END, True, True), ("\n\n", True, False)]
    content = m.get("content") or ""
    out.append((content, True, False))
    calls = m.get("tool_calls") or []
    for i, call in enumerate(calls):
        sep = ("\n" if content else "") if i == 0 else "\n"
        if version == 2:
            out += [(sep, True, False)] + _call_pieces_v2(call)
            continue
        args = json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False)
        out += [(sep, True, False), (TOOL_CALL, True, True), (f"\n{args}\n", True, False), (TOOL_CALL_END, True, True)]
    return out


def pieces(conversation: dict, version: int = 1) -> list[Piece]:
    """The rendered conversation as (text, generated, control) pieces; their concatenation is
    `render`. Empty text pieces are left out."""
    return [p for _, ps in message_pieces(conversation, version) for p in ps]


def message_pieces(conversation: dict, version: int = 1) -> list[tuple[str, list[Piece]]]:
    """[(role, pieces of that message)] in order, the tools block counted in the (possibly inserted)
    system message; each message's pieces run from its <|im_start|> through the newline after its
    <|im_end|>. Empty text pieces are left out."""
    messages = list(conversation["messages"])
    tools = conversation.get("tools") or []
    tool_block: list[Piece] = []
    inserted = False
    if tools:
        schemas = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tool_block = [(TOOLS, False, True), (f"\n{schemas}\n", False, False), (TOOLS_END, False, True)]
        if not (messages and messages[0]["role"] == "system"):
            messages.insert(0, {"role": "system", "content": ""})
            inserted = True
    result = []
    for k, m in enumerate(messages):
        role = m["role"]
        assert role in ("system", "user", "assistant", "tool"), role
        out: list[Piece] = [(IM_START, False, True), (f"{role}\n", False, False)]
        if role == "assistant":
            out += _body_pieces(m, version) + [(IM_END, True, True)]
        elif role == "tool":
            body = [(f"\n{m['content']}\n", False, False)] if version == 1 else _tool_response_text_pieces_v2(m["content"])
            out += [(TOOL_RESPONSE, False, True)] + body + [(TOOL_RESPONSE_END, False, True), (IM_END, False, True)]
        elif k == 0 and tool_block:
            # "{system}\n\n# Tools\n\n<tools>...</tools>", or just the block when there was no system message
            head = "" if inserted else (m["content"] or "") + "\n\n"
            out += [(head + "# Tools\n\n", False, False)] + tool_block + [(IM_END, False, True)]
        else:
            out += [(m["content"] or "", False, False), (IM_END, False, True)]
        out.append(("\n", False, False))
        result.append((role, [p for p in out if p[0]]))
    return result


def closing_pieces(role: str) -> int:
    """How many pieces end a non-assistant message after its content (kept when the content is
    truncated): </tool_response>, <|im_end|>, newline for tool; <|im_end|>, newline otherwise."""
    return 3 if role == "tool" else 2


def segments(conversation: dict, version: int = 1) -> list[tuple[str, bool]]:
    """[(text, generated)]: consecutive pieces with the same flag joined; their concatenation is
    `render`. For reading only: encode with `pieces` (a literal special string in a message must
    not become a control token)."""
    out: list[tuple[str, bool]] = []
    for text, gen, _ in pieces(conversation, version):
        if out and out[-1][1] == gen:
            out[-1] = (out[-1][0] + text, gen)
        else:
            out.append((text, gen))
    return out


def render(conversation: dict, version: int = 1) -> str:
    return "".join(text for text, _, _ in pieces(conversation, version))


def foreign_markup(conversation: dict) -> str | None:
    """The first foreign tool-call marker in any message's content, reasoning or tool-call
    arguments, else None."""
    for m in conversation["messages"]:
        fields = [m.get("content") or "", m.get("reasoning") or ""]
        fields += [json.dumps(c.get("arguments"), ensure_ascii=False) for c in m.get("tool_calls") or []]
        for text in fields:
            hit = FOREIGN_MARKUP.search(text)
            if hit:
                return hit.group(0)
    return None


def _last_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if p.strip()]
    return parts[-1] if parts else ""


def tool_name(tool) -> str | None:
    """The name of one entry of a conversation's `tools`: `name` (the shape our normalisers write) or, for an
    OpenAI-wrapped schema {"type": "function", "function": {"name": ...}}, `function.name`. None for an entry that has neither (or is not a mapping), or whose name is not a string."""
    if not isinstance(tool, dict):
        return None
    wrapped = tool.get("function")
    name = tool.get("name") or (wrapped.get("name") if isinstance(wrapped, dict) else None)
    return name if isinstance(name, str) else None


def chat_gate(conversation: dict, version: int = 1) -> str | None:
    """Why a chat document is not clean training or eval data, else None:
    - foreign tool-call markup or other formats' leftovers (`foreign_markup`);
    - a literal <think>, </think>, <tool_call> or </tool_call> in an assistant message's content;
    - an assistant message in which the assistant claims to be a named commercial model;
    - a tool call without a string name and an object of arguments;
    - a call to a tool not in the document's schema (a schema names its tool as `name` or, OpenAI-wrapped, as
      `function.name`: `tool_name`), unless the next tool response reports an error (error recovery is kept);
    - a last assistant message whose final sentence announces an action (ANNOUNCES_ACTION) but that makes no call."""
    if foreign_markup(conversation):
        return "foreign_tool_call_markup"
    msgs = conversation["messages"]
    names = {tool_name(t) for t in conversation.get("tools") or []} - {None}
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        content = m.get("content") or ""
        if LITERAL_CONTROL_IN_ASSISTANT.search(content):
            return "literal_control_markup_in_assistant"
        if CLAIMS_NAMED_COMMERCIAL_MODEL.search(content) or CLAIMS_NAMED_COMMERCIAL_MODEL.search(m.get("reasoning") or ""):
            return "assistant_claims_named_commercial_model"
        for call in m.get("tool_calls") or []:
            if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                return "invalid_tool_call"
            if version == 2 and ("\n" in call["name"] or any(not isinstance(a, str) or "\n" in a for a in call["arguments"])):
                return "invalid_tool_call"  # v2 puts names on their own line
            if call["name"] not in names:
                responses = []
                for later in msgs[k + 1:]:
                    if later["role"] != "tool":
                        break
                    responses.append(later.get("content") or "")
                if not any(TOOL_ERROR.search(r) for r in responses):
                    return "off_schema_tool_call"
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last is not None and not last.get("tool_calls") and ANNOUNCES_ACTION.search(_last_sentence(last.get("content") or "")):
        return "announced_action_without_call"
    return None
