"""Tokenized evaluation documents.

bpb (bits per byte) is the cross-entropy in bits divided by the UTF-8 bytes of the text the
scored tokens cover, so it compares across tokenizers. Every evaluation document therefore carries,
for each token, the number of UTF-8 bytes it covers.

The tokenizer is not decided yet (Phase 1). Evaluation code takes any `TokenizedDoc`;
`byte_tokenize` (one token per UTF-8 byte, vocabulary 256) serves the CPU tests and the tiny models.

The report's shards and eval sets (the table `load_tokenizer` of tools/build_probes.py returns) use the per-token
own-length convention: each token counts the bytes of its own vocabulary string (in byte-level BPE
one byte per character of the string; a special token counts the UTF-8 bytes of its literal text). A
character split over several tokens is then counted in pieces, but per document the total equals the
UTF-8 bytes of the text, so document- and set-level bpb are the same under either convention.
`hf_tokenize` below attributes whole characters instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenizedDoc:
    ids: list[int]
    byte_lens: list[int]  # UTF-8 bytes covered by each token; sum = bytes of the document text

    def __post_init__(self):
        assert len(self.ids) == len(self.byte_lens)


def byte_tokenize(text: str) -> TokenizedDoc:
    data = text.encode("utf-8")
    return TokenizedDoc(ids=list(data), byte_lens=[1] * len(data))


def hf_tokenize(tokenizer):
    """Wrap an HF `tokenizers.Tokenizer` (anything whose encode() gives .ids and character
    .offsets) as a text -> TokenizedDoc function.

    Byte lengths sum to the bytes of the text. Byte-level BPE splits a multi-byte character over
    several tokens that all report the character's offsets; counting len(text[a:b]) for each would
    count that character once per token (on "héllo 日本語 🙂 world" a ByteLevel BPE gives 59 counted
    bytes for 27 real ones). Here each character's bytes go to the first token that reaches it, and
    text after the last token goes to the last token.
    """

    def encode(text: str) -> TokenizedDoc:
        enc = tokenizer.encode(text, add_special_tokens=False)
        byte_lens, covered = [], 0
        for _, end in enc.offsets:
            end = max(end, covered)
            byte_lens.append(len(text[covered:end].encode("utf-8")))
            covered = end
        if byte_lens:
            byte_lens[-1] += len(text[covered:].encode("utf-8"))
        return TokenizedDoc(ids=list(enc.ids), byte_lens=byte_lens)

    return encode
