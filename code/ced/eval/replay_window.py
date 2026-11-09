"""The replay window of the restricted regime, by name: one definition for the three scorers.

The restricted regime runs CED v2's decoder at the scored positions, at each document's first decoder.n_sinks
positions, and at the replay positions before each span start. A span start is a scored position whose previous
position in the same document is not scored: the position that predicts the first token of an assistant turn, of an
answer or of a drawn span. Replay and sink positions run the decoder and write decoder K/V; they are never scored,
so every regime scores the same positions.

A window's name is what `--replay-window` takes in scripts/eval_long.py, scripts/eval_probes.py and scripts/rescore.py,
and what the run scripts append to "replay" for a directory or a file ("replay128", "replayturn", "replayturn128"):

    "<W>"       an integer W >= 1: the min(W, available) positions immediately before the span start
    "turn"      the whole last turn before the span start, capped at its last 512 positions (REPLAY_TURN_MAX)
    "turn<N>"   the whole last turn before the span start, capped at its last N positions

The rule of "turn<N>", exactly: with s a span start and b the first position of the last turn before it, the replay
positions are max(b, s - N) .. s - 1. So the replay never starts before the turn does, and never holds more than the N
positions nearest the span start. It is the turn regime's positions when the turn has N positions or fewer, and
replay N's positions when it is longer; at every span start its positions are a subset of both.

What the last turn is, per kind of scored unit (`available` never leaves the document in any of them):

- An agent trajectory, and any chat document (ced/eval/long_context.py replay_positions): the last message before the
  assistant turn, read from the chat format's <|im_start|> tokens. b is the second-to-last <|im_start|> at or before
  s: the last one opens the assistant turn itself, so the replay holds the whole last message with its header, its
  <|im_end|> and newline, and the assistant header's tokens before s. Two messages before an assistant turn (a user
  message, then a tool result): only the last one.
- A multi-fact item and a needle item (uuid_needle, multikey; ced/eval/probe_turn.py): the question, from its first
  word to the answer, the lead-in included ("What is the access code for N? The access code for N is"). The blank
  line before it is not part of it. The question's last token is the first scored position, so the replay is the
  min(N, question tokens - 1) positions before it (`item_turn_window`).
- No turn boundary in reach. The "turn" regime's rule, unchanged: when fewer than two <|im_start|> lie at or before s
  in the document's part of the row (a row that starts inside the last message or inside the assistant turn; a
  document with no chat tokens at all, such as a web or code document of the 8k eval sets), b is the document's first
  position in the row. For a plain document "turn<N>" is therefore replay N, position for position. A probe item
  whose question length is not known (no question_tokens in the set, and no text items and tokenizer to compute it
  from) is refused, as the "turn" regime refuses it.
"""

from __future__ import annotations

import re

REPLAY_TURN = "turn"  # the whole last turn before a span start
REPLAY_TURN_MAX = 512  # the cap of "turn", in positions
TURN128 = "turn128"  # the candidate regime of 2026-10-11: the whole last turn, capped at 128 positions
_TURN_N = re.compile(r"^turn([1-9][0-9]*)$")


def parse_replay_window(value, turn_max: int | None = None) -> tuple[int | str, int | None]:
    """(window, cap) of a window's name or value: (W, None) for an integer W >= 1 (an int, or its digits as text);
    ("turn", cap) for "turn" (cap: turn_max, default REPLAY_TURN_MAX) and for "turn<N>" (cap N). turn_max given
    together with "turn<N>" must be N. ValueError for anything else (a bool, 0, a negative number, another word)."""
    if isinstance(value, bool):
        raise ValueError(f"replay window {value!r}: an integer >= 1, 'turn' or 'turn<N>'")
    if isinstance(value, str):
        text = value.strip()
        if text == REPLAY_TURN:
            cap = REPLAY_TURN_MAX if turn_max is None else int(turn_max)
            if cap < 1:
                raise ValueError(f"replay window 'turn': the cap must be >= 1, not {turn_max!r}")
            return REPLAY_TURN, cap
        m = _TURN_N.match(text)
        if m:
            cap = int(m.group(1))
            if turn_max is not None and int(turn_max) != cap:
                raise ValueError(f"replay window {text!r} is the last turn capped at {cap} positions: a cap of {turn_max} contradicts it")
            return REPLAY_TURN, cap
        if text.isdigit() and int(text) >= 1:
            return int(text), None
    elif isinstance(value, int) and value >= 1:
        return value, None
    raise ValueError(f"replay window {value!r}: an integer >= 1, 'turn' or 'turn<N>'")


def replay_window_name(window, turn_max: int | None = None) -> str:
    """The name of a window: "128" for 128; "turn" for the last turn at the default cap (REPLAY_TURN_MAX, or no cap
    given); "turn<N>" for the last turn capped at N. The inverse of `parse_replay_window`."""
    window, cap = parse_replay_window(window, turn_max)
    if window != REPLAY_TURN:
        return str(window)
    return REPLAY_TURN if cap == REPLAY_TURN_MAX else f"{REPLAY_TURN}{cap}"


def is_turn(value) -> bool:
    """A window name of the last-turn kind ("turn", "turn<N>")."""
    return isinstance(value, str) and (value.strip() == REPLAY_TURN or bool(_TURN_N.match(value.strip())))


def item_turn_window(question_tokens: int, turn_max: int) -> int:
    """The replay window of one probe item under the last-turn rule: the question less its last token (which is the
    first scored position), capped at turn_max. 0 for a question of one token."""
    assert turn_max >= 1, turn_max
    return min(int(turn_max), max(0, int(question_tokens) - 1))
