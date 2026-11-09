"""The tokenizers the data was built with, known by the sha256 of their tokenizer.json, and a check
that training data and eval sets match the model's vocabulary.

Shard sets record the tokenizer in meta.json ("tokenizer_sha256", written by the shard builder) and
eval-set directories in tokenizer_sha256.txt (written by the eval-set builder; neither builder is part of this
release). Token ids alone cannot tell the tokenizers apart (64k ids are valid 128k ids),
so a model trained or scored on the wrong set would run without an error; the trainer calls
`check_tokenizer` on its shards and eval sets and refuses a known mismatch.
"""

from __future__ import annotations

from pathlib import Path

# sha256 of tokenizer.json -> vocabulary size (the model's vocab_size for data built with it)
TOKENIZER_VOCAB_BY_SHA256 = {
    "947ace9845b089fa06a7aca97d81807fa12141d6d10a3adc639f2e87f7746657": 128000,  # <data volume>
    "36220f3e66fe458cf96036d90fdbd6c6af403f5360243bb785586b4da918f686": 64000,  # <data volume> (v2)
}

EVAL_TOKENIZER_FILE = "tokenizer_sha256.txt"  # next to an eval set's .pt files


def check_tokenizer(sha256: str | None, vocab_size: int, what: str) -> str:
    """Raise ValueError when `sha256` is a known tokenizer whose vocabulary is not `vocab_size`.
    Data without a recorded tokenizer, or with one not in the table, is not checked. Returns a
    one-line note of what was checked."""
    if not sha256:
        return f"{what}: no tokenizer recorded, not checked"
    expect = TOKENIZER_VOCAB_BY_SHA256.get(sha256)
    if expect is None:
        return f"{what}: tokenizer {sha256[:12]} not in the known table, not checked"
    if expect != vocab_size:
        raise ValueError(f"{what} was built with tokenizer {sha256[:12]} ({expect} entries), but the model's "
                         f"vocabulary is {vocab_size}: use the data of the model's tokenizer")
    return f"{what}: tokenizer {sha256[:12]} matches the vocabulary {vocab_size}"


def eval_set_tokenizer(path: str | Path) -> str | None:
    """The tokenizer sha256 recorded next to an eval set file (its directory's tokenizer_sha256.txt), or None."""
    f = Path(path).parent / EVAL_TOKENIZER_FILE
    return f.read_text().strip() if f.exists() else None
