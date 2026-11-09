"""The last turn of a probe item, for the replay window "turn" / "turn<N>" (ced/eval/replay_window.py): its question.

A needle item (uuid_needle, multikey) and a multi-fact item (mf_chain, mf_concat) are a haystack, a blank line, and a
question that ends in the answer's lead-in:

    ... haystack ...

    What is the access code for N? The access code for N is| <answer>
    Who is the supervisor of N0? ... The supervisor of N0 is| <answer>

The turn is the question, from its first word to the answer; the blank line before it is not part of it. Its last
token is the first scored position, so the replay runs min(cap, question tokens - 1) positions
(replay_window.item_turn_window), and the decoder then runs from the question's first token on when the cap allows.

The question's token count is the item record's "question_tokens". The multi-fact builder stores it; the sets built
before that, and every needle set, get it here from the set's text items (<set>/text/<probe>.jsonl.gz) and tokenizer
(the path in <set>/<tok>/manifest.json), each count checked against the item's stored tokens: the question tokenized
on its own must be the context's last tokens (probe_gen_multifact.question_token_count; it is under the tokenizer's split
pattern, since the question follows a blank line and starts with a letter). Other probes (path_copy, depth, the
copying probes) have no question defined here and are refused.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ced.eval import probe_gen as g
from ced.eval import probe_gen_multifact as mf
from ced.eval.replay_window import item_turn_window

NEEDLES = ("uuid_needle", "multikey")
TURN_PROBES = NEEDLES + tuple(mf.PROBES)


def question_text(probe: str, context: str) -> str:
    """The question of an item of `probe`: the context from the question's first word to its end."""
    if probe in mf.PROBES:
        return mf.question_text(probe, context)
    if probe in NEEDLES:
        return g.needle_question_text(context)
    raise ValueError(f"no last turn is defined for probe {probe!r}: the replay window 'turn' / 'turn<N>' is for {TURN_PROBES}")


def attach_questions(sets: list, directory: Path, text_dir: Path | None = None, tokenizer: str | None = None,
                     what: str = "--replay-window turn", text_flag: str = "--probes-text",
                     tokenizer_flag: str = "--probes-tokenizer") -> str:
    """Make sure every item record of `sets` (EncodedSet files of `directory`) carries "question_tokens". Sets whose
    items all carry it are left alone. The others get it from their text items (<text_dir>/<probe>.jsonl.gz, default
    <directory>/../text) tokenized with the set's tokenizer (`tokenizer`, default the path in
    <directory>/manifest.json), every count checked against the item's stored tokens. Returns where the counts came
    from ("the set", or the text directory and tokenizer). ValueError (naming the flag to pass) when a probe has no
    question or an input is missing; AssertionError when a question is not its context's last tokens."""
    for es in sets:
        if es.probe not in TURN_PROBES:
            question_text(es.probe, "")  # raises: no last turn for this probe
    need = [es for es in sets if not all("question_tokens" in it for it in es.items)]
    if not need:
        return "the set"
    directory = Path(directory)
    text_dir = Path(text_dir) if text_dir else directory.parent / "text"
    if tokenizer is None:
        manifest = directory / "manifest.json"
        tokenizer = json.loads(manifest.read_text()).get("tokenizer", {}).get("path") if manifest.exists() else None
    if tokenizer is None or not Path(tokenizer).exists():
        raise ValueError(f"{what}: the items carry no question_tokens and the set's tokenizer ({tokenizer}) is not there: "
                         f"pass {tokenizer_flag}")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    tok.encode_special_tokens = True  # text is text, as the builders encode it
    for es in need:
        f = text_dir / f"{es.probe}.jsonl.gz"
        if not f.exists():
            raise ValueError(f"{what}: {f} (the text items of {es.probe}) is not there: pass {text_flag}")
        wanted = {rec["id"] for rec in es.items}
        questions = {}
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                it = json.loads(line)
                if it["id"] in wanted:
                    questions[it["id"]] = question_text(it["probe"], it["context"])
        missing = wanted - set(questions)
        if missing:
            raise ValueError(f"{what}: {f} has no text item for {len(missing)} of {len(wanted)} items of {es.probe} "
                             f"(e.g. {sorted(missing)[0]}): pass {text_flag}")
        mf.attach_question_tokens(es, questions, lambda text: tok.encode(text, add_special_tokens=False).ids)
    return f"text items in {text_dir}, tokenizer {tokenizer}"


def turn_windows(sets: list, turn_max: int) -> list[list[int]]:
    """score_sets' replay_windows under the last-turn rule: per set, each item's own window, the question less its
    last token, capped at turn_max. Every item must carry "question_tokens" (`attach_questions`)."""
    return [[item_turn_window(rec["question_tokens"], turn_max) for rec in es.items] for es in sets]
