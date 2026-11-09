"""Long-context grounded-QA probe: score probe items with Looped-CED or the non-looped control.

Why this exists
    It asks whether a shallower encoder hurts reading. The probe items come in gold/swap pairs
    that differ only in whether the context holds what the answer needs. This module scores the
    answer tokens at decoder positions, the way Looped-CED generates them, and scores the control on
    exactly the same tokens.

Layout of one item
    The context, the question and the answer are tokenized separately and concatenated (every
    separator lives inside the strings), giving ids = [context][question][a_1 .. a_m]. The model
    reads ids[:-1] and predicts ids[1:]:

        position  ... q_last  a_1  ...  a_{m-1}        (a_m is never an input)
        target    ...  a_1    a_2  ...   a_m
        dec_mask  ...  True   True ...   True           every earlier position: False

    Context and question positions are context positions (Looped-CED: encoder only, they offer
    memory KV); the m positions that PREDICT answer tokens are decoder positions. So dec_mask is
    True exactly at the last m inputs. The control ignores dec_mask and is scored at the same m
    positions.

Batches are LEFT-padded. The padding is a document of its own (doc id 0; the item is doc 1) of
context positions, and both models isolate documents exactly (document-masked attention, GDN and
conv state reset, RoPE positions restart at each document), so a padded row scores like the item
alone (checked in float64). Left padding puts every row's answer in the
same last columns, so the scored block is a fixed-shape slice, and every prompt ends in the same
column, so greedy generation runs batched.

Metrics per item
    answer bpb      bits of the answer tokens / UTF-8 bytes of the answer string
    exact match     greedy decoding from the prompt reproduces the answer's token ids
                    (LoopedCED.generate; prefill + extend for the control). Computed for gold items.
    tf_greedy       the teacher-forced argmax equals the answer token at every answer position. For
                    a fixed answer length this is the same event as greedy exact match, so the two
                    must agree; a disagreement means the cached path and the teacher-forced path
                    differ (the summary counts them). exact_match="teacher_forced" uses it instead
                    of generating, at no extra cost.
Aggregates per (kind, bin), and per source / depth inside a bin: the mean answer bpb with its
standard error, exact match with its SE, and the context gain = bpb(gold) - bpb(swap) over pairs
with its paired SE. Negative context gain means the gold context lowered the answer's loss.

Usage
    python -m ced.eval.qa_probe --checkpoint <run a> --probe <probe dir> --out a.json
    python -m ced.eval.qa_probe --checkpoint <run b> --probe <probe dir> --out b.json \\
        --compare a.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch

from ced.eval.tokenized import TokenizedDoc, byte_tokenize

Tokenize = Callable[[str], TokenizedDoc]
KINDS = ("copy", "kv", "passage_qa")
GROUP_FIELDS = ("source", "depth_target")  # finer groups inside a (kind, bin)


# --------------------------------------------------------------------------------------------
# items and tokenization
# --------------------------------------------------------------------------------------------


@dataclass
class ProbeItem:
    kind: str
    context: str
    question: str
    answer: str
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def condition(self) -> str:
        return self.meta.get("condition", "gold")


def load_items(path: str | Path, kinds: Iterable[str] = KINDS, max_pairs_per_bin: int | None = None) -> list[ProbeItem]:
    """Items from one JSONL file, or from <dir>/{copy,kv,passage_qa}.jsonl.

    max_pairs_per_bin keeps the first N pairs of every (kind, bin), in file order (smoke runs).
    """
    path = Path(path)
    files = [path] if path.is_file() else [path / f"{k}.jsonl" for k in kinds if (path / f"{k}.jsonl").exists()]
    items, seen = [], defaultdict(list)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                item = ProbeItem(r["kind"], r["context"], r["question"], r["answer"], r.get("meta", {}))
                if item.kind not in kinds:
                    continue
                if max_pairs_per_bin is not None:
                    pairs = seen[(item.kind, item.meta.get("bin"))]
                    pair = item.meta.get("pair_id", item.meta.get("id"))
                    if pair not in pairs:
                        if len(pairs) >= max_pairs_per_bin:
                            continue
                        pairs.append(pair)
                items.append(item)
    return items


def cached_tokenizer(directory: str | Path) -> Tokenize:
    """The probe builder's token cache (`--token-cache`) as a Tokenize function.

    It lets the probe run where the HF `tokenizers` library is not installed. Only the strings of that probe are in it; anything else raises KeyError.
    """
    from ced.data.shard_format import ShardSet

    directory = Path(directory)
    shards = ShardSet(directory)
    assert shards.byte_lens is not None, f"{directory}: the token cache needs byte_lens.bin"
    index = json.loads((directory / "index.json").read_text())

    def tokenize(text: str) -> TokenizedDoc:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in index:
            raise KeyError(f"string not in the token cache {directory}: {text[:60]!r}")
        a, b = int(shards.doc_starts[index[key]]), int(shards.doc_starts[index[key] + 1])
        return TokenizedDoc(ids=shards.tokens[a:b].tolist(), byte_lens=shards.byte_lens[a:b].tolist())

    return tokenize


@dataclass
class EncodedItem:
    ids: list[int]  # context + question + answer token ids
    n_answer: int
    answer_bytes: int  # UTF-8 bytes of the answer string (the bpb denominator)

    @property
    def n_prompt(self) -> int:
        return len(self.ids) - self.n_answer

    @property
    def answer_ids(self) -> list[int]:
        return self.ids[self.n_prompt :]

    def fingerprint(self) -> str:
        """Identifies the exact scored tokens, so two models' scores can be checked to be paired."""
        h = hashlib.blake2b(digest_size=8)
        h.update(json.dumps([self.ids, self.n_answer]).encode())
        return h.hexdigest()


def encode_item(item: ProbeItem, tokenize: Tokenize) -> EncodedItem:
    context, question, answer = tokenize(item.context), tokenize(item.question), tokenize(item.answer)
    assert answer.ids, f"empty answer in {item.meta.get('id')}"
    assert context.ids or question.ids, f"empty prompt in {item.meta.get('id')}"
    return EncodedItem(
        ids=context.ids + question.ids + answer.ids,
        n_answer=len(answer.ids),
        answer_bytes=len(item.answer.encode("utf-8")),
    )


# --------------------------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------------------------

PAD_DOC, ITEM_DOC = 0, 1


@dataclass
class ProbeBatch:
    tokens: torch.Tensor  # (B, L) inputs ids[:-1], left-padded
    doc_ids: torch.Tensor  # (B, L) PAD_DOC on padding, ITEM_DOC on the item
    dec_mask: torch.Tensor  # (B, L) True exactly at the positions that predict answer tokens
    targets: torch.Tensor  # (B, L) ids[1:] (pad_id on padding)
    n_answer: torch.Tensor  # (B,)

    @property
    def answer_width(self) -> int:
        return int(self.n_answer.max())

    def answer_block(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(targets, mask) of the last answer_width columns, which hold every row's answer."""
        A = self.answer_width
        cols = torch.arange(A, device=self.n_answer.device)
        mask = cols[None, :] >= (A - self.n_answer)[:, None]
        return self.targets[:, -A:], mask


def left_pad(rows: list[list[int]], pad_id: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens, doc_ids): rows right-aligned; padding gets PAD_DOC, the row ITEM_DOC."""
    L = max(len(r) for r in rows)
    tokens = torch.full((len(rows), L), pad_id, dtype=torch.long)
    docs = torch.full((len(rows), L), PAD_DOC, dtype=torch.long)
    for b, r in enumerate(rows):
        tokens[b, L - len(r) :] = torch.tensor(r, dtype=torch.long)
        docs[b, L - len(r) :] = ITEM_DOC
    return tokens.to(device), docs.to(device)


def make_batch(encoded: list[EncodedItem], pad_id: int = 0, device=None) -> ProbeBatch:
    tokens, docs = left_pad([e.ids[:-1] for e in encoded], pad_id, device)
    targets, _ = left_pad([e.ids[1:] for e in encoded], pad_id, device)
    L = tokens.shape[1]
    n_answer = torch.tensor([e.n_answer for e in encoded], dtype=torch.long, device=tokens.device)
    dec_mask = torch.arange(L, device=tokens.device)[None, :] >= (L - n_answer)[:, None]
    return ProbeBatch(tokens=tokens, doc_ids=docs, dec_mask=dec_mask, targets=targets, n_answer=n_answer)


def make_prompt_batch(encoded: list[EncodedItem], pad_id: int = 0, device=None):
    """Left-padded prompts (context + question) for generation; the last column is the decoder
    position that predicts a_1."""
    tokens, docs = left_pad([e.ids[: e.n_prompt] for e in encoded], pad_id, device)
    dec_mask = torch.zeros_like(tokens, dtype=torch.bool)
    dec_mask[:, -1] = True
    return tokens, docs, dec_mask


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------


def is_looped(model: torch.nn.Module) -> bool:
    """Looped-CED (v1): the model with a generate() method."""
    return model.cfg.arch == "looped_ced"


def has_slots(model: torch.nn.Module) -> bool:
    """A model whose main head runs at decoder slots (Looped-CED, CED v2, the second-pass control)."""
    from ced.model.control import SLOT_ARCHES

    return model.cfg.arch in SLOT_ARCHES


@torch.no_grad()
def score_batch(model: torch.nn.Module, batch: ProbeBatch, T: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced answer loss. Returns (answer nats (B,), tf_greedy (B,) bool).

    Models with decoder slots (Looped-CED, CED v2, the second-pass control) run with capacity = the
    widest answer in the batch, so the decoder works on that many slots instead of L; no decoder
    position may be dropped.
    """
    A = batch.answer_width
    out = model(batch.tokens, batch.doc_ids, batch.dec_mask, T=T, capacity=A)
    if has_slots(model):
        assert int(out.n_dropped.sum()) == 0, "a decoder position was dropped"
        assert torch.equal(out.kept, batch.dec_mask)
    logits = out.logits[:, -A:].to(torch.float64)  # (B, A, V)
    targets, mask = batch.answer_block()
    ce = torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets[..., None]).squeeze(-1)
    nats = torch.where(mask, ce, torch.zeros_like(ce)).sum(1)
    tf_greedy = ((logits.argmax(-1) == targets) | ~mask).all(1)
    return nats, tf_greedy


@torch.no_grad()
def greedy_continuation(model: torch.nn.Module, encoded: list[EncodedItem], T: int | None = None, pad_id: int = 0) -> torch.Tensor:
    """(B, A) greedy tokens after each prompt, A = the widest answer. LoopedCED.generate for
    Looped-CED; a prefill + extend loop for the control."""
    device = next(model.parameters()).device
    prompt, docs, dec = make_prompt_batch(encoded, pad_id, device)
    A = max(e.n_answer for e in encoded)
    if is_looped(model):
        tokens, _ = model.generate(prompt, docs, dec, A, T=T)
        return tokens
    out = model.prefill(prompt, docs, dec)
    new = []
    for i in range(A):
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        new.append(tok)
        if i + 1 < A:
            out = model.extend(tok, docs[:, -1:], torch.ones_like(tok, dtype=torch.bool), out.cache)
    return torch.cat(new, dim=1)


@dataclass
class ItemScore:
    id: str
    kind: str
    bin: str
    condition: str
    pair_id: str
    group: dict  # finer grouping fields from meta (source, depth_target)
    n_tokens: int  # context + question + answer tokens
    n_answer: int
    answer_bytes: int
    answer_nats: float
    tf_greedy: bool
    exact_match: bool | None  # None when not generated (swap items, or exact_match="off")
    fingerprint: str

    @property
    def bpb(self) -> float:
        return self.answer_nats / math.log(2) / self.answer_bytes


def batches_by_length(lengths: list[int], batch_size: int, max_batch_tokens: int | None) -> list[list[int]]:
    """Indices grouped into batches of similar length (longest first), at most batch_size rows and,
    if given, at most max_batch_tokens = rows x longest row."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out, cur = [], []
    for i in order:
        width = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) >= batch_size or (max_batch_tokens and (len(cur) + 1) * width > max_batch_tokens)):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def score_items(
    model: torch.nn.Module,
    items: list[ProbeItem],
    tokenize: Tokenize,
    *,
    T: int | None = None,
    batch_size: int = 1,
    max_batch_tokens: int | None = None,
    exact_match: str = "generate",
    pad_id: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[ItemScore]:
    """Score every item; returns one ItemScore per item, in the order of `items`.

    exact_match: "generate" (greedy decoding for gold items), "teacher_forced" (use tf_greedy), or
    "off". Swap items are never generated: their exact match would only measure guessing.
    """
    assert exact_match in ("generate", "teacher_forced", "off")
    model.eval()
    device = next(model.parameters()).device
    encoded = [encode_item(it, tokenize) for it in items]
    scores: list[ItemScore | None] = [None] * len(items)
    batches = batches_by_length([len(e.ids) for e in encoded], batch_size, max_batch_tokens)
    for done, idx in enumerate(batches):
        enc = [encoded[i] for i in idx]
        nats, tf = score_batch(model, make_batch(enc, pad_id, device), T=T)
        em: list[bool | None] = [None] * len(idx)
        gen_rows = [j for j, i in enumerate(idx) if items[i].condition != "swap"]
        if exact_match == "teacher_forced":
            em = [bool(tf[j]) if j in gen_rows else None for j in range(len(idx))]
        elif exact_match == "generate" and gen_rows:
            continuation = greedy_continuation(model, [enc[j] for j in gen_rows], T=T, pad_id=pad_id).cpu()
            for r, j in enumerate(gen_rows):
                answer = torch.tensor(enc[j].answer_ids, dtype=torch.long)
                em[j] = bool(torch.equal(continuation[r, : enc[j].n_answer], answer))
        for j, i in enumerate(idx):
            item, e = items[i], encoded[i]
            scores[i] = ItemScore(
                id=item.id, kind=item.kind, bin=str(item.meta.get("bin", "all")), condition=item.condition,
                pair_id=item.meta.get("pair_id", item.id),
                group={k: item.meta[k] for k in GROUP_FIELDS if k in item.meta},
                n_tokens=len(e.ids), n_answer=e.n_answer, answer_bytes=e.answer_bytes,
                answer_nats=float(nats[j]), tf_greedy=bool(tf[j]), exact_match=em[j], fingerprint=e.fingerprint(),
            )
        if progress is not None:
            progress(done + 1, len(batches))
    return scores


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------


def mean_se(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and standard error (sample standard deviation / sqrt(n))."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def group_keys(s: ItemScore) -> list[str]:
    keys = [f"{s.kind}/all", f"{s.kind}/{s.bin}"]
    for name, value in sorted(s.group.items()):
        keys.append(f"{s.kind}/{s.bin}/{name}={value}")
    return keys


def summarize(scores: list[ItemScore]) -> dict[str, dict]:
    """Per group: answer bpb (gold items), exact match, swap bpb, context gain over pairs."""
    groups: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        for key in group_keys(s):
            groups[key].append(s)
    out = {}
    for key in sorted(groups):
        members = groups[key]
        gold = [s for s in members if s.condition != "swap"]
        swap = {s.pair_id: s for s in members if s.condition == "swap"}
        bpb, bpb_se = mean_se([s.bpb for s in gold])
        swap_bpb, swap_se = mean_se([s.bpb for s in swap.values()])
        gains = [s.bpb - swap[s.pair_id].bpb for s in gold if s.pair_id in swap]
        gain, gain_se = mean_se(gains)
        em_values = [float(s.exact_match) for s in gold if s.exact_match is not None]
        em, em_se = mean_se(em_values)
        out[key] = {
            "n": len(gold),
            "mean_tokens": sum(s.n_tokens for s in gold) / len(gold) if gold else None,
            "answer_bpb": bpb, "answer_bpb_se": bpb_se,
            "answer_nats_per_token": (sum(s.answer_nats for s in gold) / sum(s.n_answer for s in gold)) if gold else None,
            "exact_match": em, "exact_match_se": em_se, "n_exact_match": len(em_values),
            "tf_greedy": mean_se([float(s.tf_greedy) for s in gold])[0],
            "em_tf_disagreements": sum(1 for s in gold if s.exact_match is not None and s.exact_match != s.tf_greedy),
            "swap_answer_bpb": swap_bpb, "swap_answer_bpb_se": swap_se,
            "n_pairs": len(gains), "context_gain": gain, "context_gain_se": gain_se,
        }
    return out


def paired_difference(a: list[ItemScore], b: list[ItemScore]) -> dict[str, dict]:
    """Per group, model a minus model b on the same items: answer bpb (gold items) and context gain.

    Items are matched by id and must have identical fingerprints (the same scored tokens)."""
    by_id = {s.id: s for s in b}
    pairs = []
    for s in a:
        t = by_id.get(s.id)
        if t is None:
            continue
        if s.fingerprint != t.fingerprint:
            raise ValueError(f"item {s.id} was scored on different tokens in the two evaluations")
        pairs.append((s, t))
    swap_a = {s.pair_id: s for s, _ in pairs if s.condition == "swap"}
    swap_b = {t.pair_id: t for _, t in pairs if t.condition == "swap"}
    groups: dict[str, list[tuple[ItemScore, ItemScore]]] = defaultdict(list)
    for s, t in pairs:
        if s.condition != "swap":
            for key in group_keys(s):
                groups[key].append((s, t))
    out = {}
    for key in sorted(groups):
        members = groups[key]
        d_bpb, d_bpb_se = mean_se([s.bpb - t.bpb for s, t in members])
        d_gain, d_gain_se = mean_se([
            (s.bpb - swap_a[s.pair_id].bpb) - (t.bpb - swap_b[t.pair_id].bpb)
            for s, t in members if s.pair_id in swap_a and t.pair_id in swap_b
        ])
        out[key] = {"n": len(members), "answer_bpb_diff": d_bpb, "answer_bpb_diff_se": d_bpb_se,
                    "context_gain_diff": d_gain, "context_gain_diff_se": d_gain_se}
    return out


def scores_to_json(scores: list[ItemScore]) -> list[dict]:
    return [asdict(s) for s in scores]


def scores_from_json(rows: list[dict]) -> list[ItemScore]:
    return [ItemScore(**r) for r in rows]


# --------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------


def resolve_tokenizer(spec: str, probe: Path) -> Tokenize:
    """'auto' (the probe's token cache, or bytes if the probe was built with bytes), 'bytes', or a
    token-cache directory. Other tokenizers: call score_items from code with any Tokenize function
    (without the `tokenizers` library, build the probe with --token-cache instead)."""
    if spec == "auto":
        if (probe / "token_cache" / "index.json").exists():
            return cached_tokenizer(probe / "token_cache")
        manifest = probe / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("tokenizer") == "bytes":
            return byte_tokenize
        raise SystemExit(f"{probe} has no token cache and was not built with bytes; rebuild it with --token-cache")
    if spec == "bytes":
        return byte_tokenize
    return cached_tokenizer(spec)


def load_checkpoint_model(checkpoint: str | Path, device: str):
    """The model of a trainer checkpoint (a .pt file, a run directory, or its checkpoints/ dir)."""
    from ced.config import TrainConfig, from_dict
    from ced.model.control import build_model

    path = Path(checkpoint)
    for pointer in (path / "checkpoints" / "latest", path / "latest"):
        if pointer.exists():
            path = pointer.parent / pointer.read_text().strip()
            break
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainConfig, state["config"])
    model = build_model(cfg.model).to(getattr(torch, cfg.dtype))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), path


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Score a checkpoint on the long-context grounded-QA probe.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", required=True, help="probe directory (or one JSONL file)")
    p.add_argument("--tokenizer", default="auto")
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--max-pairs-per-bin", type=int, default=None)
    p.add_argument("--T", type=int, default=None, help="loop count (default: the config's t_eval)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-batch-tokens", type=int, default=None)
    p.add_argument("--exact-match", choices=["generate", "teacher_forced", "off"], default="generate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write settings, summary and per-item scores as JSON")
    p.add_argument("--compare", default=None, help="an earlier --out JSON: add the paired difference (this - that)")
    args = p.parse_args(argv)

    probe = Path(args.probe)
    model, path = load_checkpoint_model(args.checkpoint, args.device)
    items = load_items(probe, args.kinds.split(","), args.max_pairs_per_bin)
    tokenize = resolve_tokenizer(args.tokenizer, probe if probe.is_dir() else probe.parent)
    scores = score_items(
        model, items, tokenize, T=args.T, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens,
        exact_match=args.exact_match, progress=lambda i, n: print(f"\rbatch {i}/{n}", end="", flush=True),
    )
    print()
    result = {
        "settings": {"checkpoint": str(path), "probe": str(probe), "T": args.T, "exact_match": args.exact_match,
                     "n_items": len(items)},
        "summary": summarize(scores),
        "scores": scores_to_json(scores),
    }
    if args.compare:
        other = scores_from_json(json.loads(Path(args.compare).read_text())["scores"])
        result["compare"] = {"against": args.compare, "groups": paired_difference(scores, other)}
    def fmt(value, se=None):
        text = "-" if value is None else f"{value:.4f}"
        return text + (f" ± {se:.4f}" if se is not None else "")

    for key, g in result["summary"].items():
        if key.count("/") == 1:
            print(f"{key:22s} n={g['n']:4d}  bpb={fmt(g['answer_bpb'], g['answer_bpb_se'])}  "
                  f"gain={fmt(g['context_gain'], g['context_gain_se'])}  em={fmt(g['exact_match'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
