"""Multiple-choice and cloze scoring for the small general regression set (ARC-Easy, PIQA,
WinoGrande, BoolQ by log-likelihood; LAMBADA by target-word bpb and greedy accuracy).

Every (item, choice) pair is one document: context tokens followed by the choice tokens. The positions
that predict choice tokens are decoder positions (like assistant tokens in role rows), so Looped-CED,
CED v2 and the second-pass control score the answer with their main head (the decoder or pass 2 runs
at the choice positions) and the control scores the same tokens with its single head; CED v2 with T >=
2 also reports the loop-1 readout's gold_bpb (loop1_gold_bpb, section 10). Documents are packed WHOLE
into rows of L tokens with per-document masking (`pack_whole_documents`: unlike the bpb packer, no
document is cut across rows, so every choice is scored with its full context), and per-choice sums are
read back by document index. Exception: CED v2 with decoder self-KV writes K/V at its decoder
positions that later ones read, so the choice-position run would be the restricted regime (no own keys
in the context). Its decoder runs at every valid target instead (capacity L, as span_bpb.evaluate runs
every CED v2), the regime it trained in, and the choice positions are read out of it. For every other
decoder arch the choice-position run is unchanged.

Per task: acc (argmax of the summed log-likelihood), acc_norm (argmax of log-likelihood per UTF-8
byte of the choice), gold_bpb (bits per byte of the gold choice given the context), and for cloze
tasks greedy_acc (every target token is the argmax). The prompt formats follow the common
lm-evaluation-harness conventions (`format_item`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ced.eval.span_bpb import EvalRows, RoleDoc, build_whole_role_rows  # noqa: F401 (EvalRows re-exported)

LN2 = math.log(2.0)


@dataclass
class Item:
    task: str
    id: str
    context: str
    choices: list[str]
    gold: int
    kind: str = "mc"  # "mc" or "cloze" (one choice: the target)


def format_item(task: str, row: dict, index: int) -> Item:
    if task == "arc-easy":
        labels = list(row["choices"]["label"])
        return Item(task, row["id"], f"Question: {row['question']}\nAnswer:",
                    [" " + t for t in row["choices"]["text"]], labels.index(row["answerKey"]))
    if task == "piqa":
        return Item(task, str(index), f"Question: {row['goal']}\nAnswer:", [" " + row["sol1"], " " + row["sol2"]], int(row["label"]))
    if task == "boolq":
        return Item(task, str(index), f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", [" no", " yes"], int(bool(row["answer"])))
    if task == "winogrande":
        # Partial scoring: the context differs per option, the continuation after the blank is scored.
        i = row["sentence"].index("_")
        cont = " " + row["sentence"][i + 1 :].strip()
        contexts = [row["sentence"][:i] + row["option1"], row["sentence"][:i] + row["option2"]]
        return Item(task, str(index), "\x00".join(contexts), [cont, cont], int(row["answer"]) - 1)
    if task == "lambada":
        text = row["text"]
        ctx, _, word = text.rpartition(" ")
        return Item(task, str(index), ctx, [" " + word], 0, kind="cloze")
    raise ValueError(task)


def build_documents(items: list[Item], encode, byte_len) -> tuple[list[RoleDoc], list[tuple[int, int, int]]]:
    """Documents for every (item, choice) and their (item index, choice index, choice bytes).

    encode: str -> list of token ids; byte_len: list of ids -> list of UTF-8 byte lengths."""
    docs, index = [], []
    for k, item in enumerate(items):
        contexts = item.context.split("\x00") if "\x00" in item.context else [item.context] * len(item.choices)
        for c, (ctx, choice) in enumerate(zip(contexts, item.choices)):
            a, b = encode(ctx), encode(choice)
            ids = a + b
            docs.append(RoleDoc(ids=ids, byte_lens=byte_len(ids), assistant=[0] * len(a) + [1] * len(b)))
            index.append((k, c, sum(byte_len(b))))
    return docs, index


def pack_whole_documents(docs: list[RoleDoc], seq_len: int) -> EvalRows:
    """Role rows (decoder positions = positions whose next token is a choice token), with every
    document inside one row (`ced.eval.span_bpb.build_whole_role_rows`)."""
    return build_whole_role_rows(docs, seq_len)


@torch.no_grad()
def score(model, items: list[Item], docs: list[RoleDoc], index, seq_len: int = 8192, batch_size: int = 1,
          autocast_dtype=None, fused_varlen: bool = False) -> dict:
    """Per-task metrics. Uses the chunked float32 head and the model's decoder at choice tokens."""
    return score_items(model, items, docs, index, seq_len, batch_size, autocast_dtype, fused_varlen)[0]


@torch.no_grad()
def score_items(model, items: list[Item], docs: list[RoleDoc], index, seq_len: int = 8192, batch_size: int = 1,
                autocast_dtype=None, fused_varlen: bool = False) -> tuple[dict, list[dict]]:
    """(per-task metrics, per-item results). Per item: task, id, gold, the summed nats and UTF-8 bytes
    of every choice, the predictions by acc and acc_norm, and for cloze items whether the target is
    greedy. Saved with the aggregates so models can be compared item by item (paired and McNemar
    tests). At 60M, accuracy is a sanity check; gold_bpb is compared."""
    rows = pack_whole_documents(docs, seq_len)
    scores = score_choice_rows(model, rows, len(docs), batch_size, autocast_dtype, fused_varlen)
    return choice_results(items, index, scores)


@torch.no_grad()
def score_choice_rows(model, rows: EvalRows, n_docs: int, batch_size: int = 1, autocast_dtype=None,
                      fused_varlen: bool = False, starts=None) -> dict:
    """score_items' forward passes over the packed rows (pack_whole_documents of the n_docs documents): the batches of
    batch_size rows that begin at `starts` (default: every batch, range(0, len(rows), batch_size)). Returns {"nats":
    (n_docs,) float64, each document's summed nats at its scored positions, "nats_loop1": the same for CED v2's loop-1
    readout (zeros otherwise), "greedy_ok": (n_docs,) bool, every scored target the argmax, "scored": (n_docs,) bool, the
    documents with a scored position in these batches, "loop1": whether the loop-1 readout was scored}. Every document is
    whole inside one row (pack_whole_documents), so its numbers come from one batch alone: the sharded evals (E1,
    ced/train/sharded_eval.py) score each rank's batches so and join them with join_choice_rows."""
    from ced.eval.span_bpb import _device
    from ced.model.control import LOOPED_ARCHES, SLOT_ARCHES

    inner = model
    while hasattr(inner, "module") or hasattr(inner, "_orig_mod"):
        inner = getattr(inner, "module", None) or getattr(inner, "_orig_mod")
    arch = inner.cfg.arch
    slots = arch in SLOT_ARCHES  # the main head at the choice positions
    # decoder self-KV: the decoder at every valid target (see the module docstring), read at the choices
    every = arch == "ced_v2" and bool(inner.cfg.decoder.self_kv)
    T = inner.cfg.loop.t_eval if arch in LOOPED_ARCHES else None
    loop1 = arch == "ced_v2" and T >= 2
    device = _device(model)
    nats = torch.zeros(n_docs, dtype=torch.float64)
    nats_loop1 = torch.zeros(n_docs, dtype=torch.float64)
    greedy_ok = torch.ones(n_docs, dtype=torch.bool)
    touched = torch.zeros(n_docs, dtype=torch.bool)
    was = model.training
    model.eval()
    try:
        for start in (range(0, len(rows), batch_size) if starts is None else starts):
            batch = rows[start : start + batch_size]
            tokens, doc_ids, dec = (x.to(device) for x in (batch.tokens, batch.doc_ids, batch.dec_mask))
            varlen = None
            if fused_varlen:
                from ced.train.varlen import to_device, varlen_for_batch

                v1 = arch == "looped_ced"  # only v1's decoder has GDN layers over the slots
                host = varlen_for_batch(batch.doc_ids, batch.dec_mask if v1 else None, rows.capacity if v1 else None)
                varlen = to_device(host, device)
            run, capacity = (batch.target_valid.to(device), tokens.shape[1]) if every else (dec, rows.capacity)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                ce, top, ce_loop1 = _scored_ce_and_argmax(model, inner, slots, tokens, doc_ids, run,
                                                          batch.targets.to(device), T, capacity, varlen, loop1)
            scored = (batch.target_valid & batch.dec_mask)
            d = batch.doc_ids.clamp(max=n_docs - 1)  # padding rows carry id n_docs; they are never scored
            nats.index_add_(0, d[scored], ce.cpu()[scored])
            if ce_loop1 is not None:
                nats_loop1.index_add_(0, d[scored], ce_loop1.cpu()[scored])
            wrong = scored & (top.cpu() != batch.targets)
            greedy_ok[d[wrong]] = False
            touched[d[scored]] = True
    finally:
        model.train(was)
    return {"nats": nats, "nats_loop1": nats_loop1, "greedy_ok": greedy_ok, "scored": touched, "loop1": loop1}


def join_choice_rows(parts: list, rows: EvalRows | None = None) -> dict:
    """score_choice_rows over every batch, put together from its results on disjoint sets of batches (parts, each a
    score_choice_rows result over the same rows; together every batch once): each document's numbers taken from the
    part that scored it (exactly, no arithmetic: a document is whole inside one row, so one part has it), the defaults
    (0 nats, greedy) for a document no part scored, as one call over every batch leaves them. ValueError when two parts
    scored one document, or (with rows: the packed rows the parts were scored on) when a document with a scored position
    in rows was scored by no part (a batch left out: its documents would keep 0 nats and count as the likeliest choice)."""
    first = parts[0]
    out = {"nats": torch.zeros_like(first["nats"]), "nats_loop1": torch.zeros_like(first["nats_loop1"]),
           "greedy_ok": torch.ones_like(first["greedy_ok"]), "scored": torch.zeros_like(first["scored"]),
           "loop1": first["loop1"]}
    for p in parts:
        if p["loop1"] != first["loop1"] or p["nats"].shape != first["nats"].shape:
            raise ValueError("join_choice_rows: the parts are not of one set and one model")
        if bool((out["scored"] & p["scored"]).any()):
            raise ValueError("join_choice_rows: a document was scored by two parts")
        own = p["scored"]
        for k in ("nats", "nats_loop1", "greedy_ok"):
            out[k] = torch.where(own, p[k], out[k])
        out["scored"] = out["scored"] | own
    if rows is not None:  # every document with a scored position in rows was scored by a part
        want = torch.zeros_like(out["scored"])
        want[rows.doc_ids[rows.target_valid & rows.dec_mask]] = True  # padding (id n_docs) is never scored
        if bool((want & ~out["scored"]).any()):
            missed = (want & ~out["scored"]).nonzero().flatten().tolist()  # the failure path only
            raise ValueError(f"join_choice_rows: documents {missed[:8]} ({len(missed)} in all) have scored positions in "
                             f"the rows and were scored by no part")
    return out


def choice_results(items: list[Item], index, scores: dict) -> tuple[dict, list[dict]]:
    """score_items' (per-task metrics, per-item results) from the documents' scores (score_choice_rows over every batch,
    or join_choice_rows of its parts)."""
    nats, nats_loop1, greedy_ok, loop1 = scores["nats"], scores["nats_loop1"], scores["greedy_ok"], scores["loop1"]
    out: dict[str, dict] = {}
    by_task: dict[str, list[int]] = {}
    for k, item in enumerate(items):
        by_task.setdefault(item.task, []).append(k)
    per_choice: dict[tuple[int, int], tuple[float, int, bool]] = {}
    gold_doc: dict[tuple[int, int], int] = {}
    for doc, (k, c, nbytes) in enumerate(index):
        per_choice[(k, c)] = (float(nats[doc]), nbytes, bool(greedy_ok[doc]))
        gold_doc[(k, c)] = doc
    per_item: list[dict] = []
    for task, ks in by_task.items():
        acc = acc_norm = greedy = 0
        gold_nats = gold_bytes = gold_nats_loop1 = 0.0
        for k in ks:
            item = items[k]
            choices = [per_choice[(k, c)] for c in range(len(item.choices))]
            lls = [-x[0] for x in choices]
            norm = [-x[0] / max(1, x[1]) for x in choices]
            pred = max(range(len(lls)), key=lls.__getitem__)
            pred_norm = max(range(len(norm)), key=norm.__getitem__)
            acc += int(pred == item.gold)
            acc_norm += int(pred_norm == item.gold)
            g = per_choice[(k, item.gold)]
            gold_nats += g[0]
            gold_nats_loop1 += float(nats_loop1[gold_doc[(k, item.gold)]])
            gold_bytes += g[1]
            greedy += int(g[2])
            rec = {"task": task, "id": item.id, "gold": item.gold, "nats": [x[0] for x in choices],
                   "bytes": [x[1] for x in choices], "pred": pred, "pred_norm": pred_norm}
            if item.kind == "cloze":
                rec["greedy"] = g[2]
            per_item.append(rec)
        n = len(ks)
        res = {"n": n, "gold_bpb": gold_nats / (LN2 * gold_bytes) if gold_bytes else float("nan")}
        if loop1:
            res["loop1_gold_bpb"] = gold_nats_loop1 / (LN2 * gold_bytes) if gold_bytes else float("nan")
        if items[ks[0]].kind == "mc":
            res.update(acc=acc / n, acc_norm=acc_norm / n)
        else:
            res.update(greedy_acc=greedy / n)
        out[task] = res
    return out, per_item


def _cp_ce_and_argmax(model, inner, tokens, doc_ids, dec, targets, varlen):
    """_scored_ce_and_argmax of a Copy-Paste model (model.cp_head; ced/model/cp_head.py): CE and argmax of its
    next-token distribution (log p over V of the mixture, or p_LM alone under score "lm") at every position."""
    from ced.model.cp_head import head_logprobs_at

    out = model(tokens, doc_ids, dec, return_hidden=True, varlen=varlen)
    B, L = tokens.shape
    index = torch.arange(L, device=tokens.device).expand(B, L)
    ces, tops = [], []
    for s in range(0, L, 1024):  # (B, 1024, V) log p at a time
        lp = head_logprobs_at(inner, out.hidden_aux, tokens, doc_ids, index[:, s : s + 1024], pointer=out.cp_pointer)
        ces.append(-lp.gather(-1, targets[:, s : s + 1024, None]).squeeze(-1))
        tops.append(lp.argmax(-1))
    return torch.cat(ces, 1).double(), torch.cat(tops, 1), None


def _scored_ce_and_argmax(model, inner, slots, tokens, doc_ids, dec, targets, T, capacity, varlen, loop1=False):
    """Like span_bpb._scored_ce, plus the argmax token at every position (for greedy accuracy):
    (ce, argmax, loop-1 readout ce or None). slots: a model with a main head at decoder slots
    (Looped-CED, CED v2, the second-pass control), run with the decoder at the rows' decoder
    positions (the choice tokens); loop1: also CED v2's loop-1 readout at those positions."""
    from ced.model.capacity import gather_slots, scatter_slots
    from ced.model.cp_head import cp_head_of
    from ced.model.norms import compute_dtype

    if cp_head_of(inner) is not None:  # a Copy-Paste head: CE and argmax of its next-token distribution
        return _cp_ce_and_argmax(model, inner, tokens, doc_ids, dec, targets, varlen)
    W = inner.head_weight()

    def head(h, t):
        ces, tops = [], []
        flat_h, flat_t = h.reshape(-1, h.shape[-1]), t.reshape(-1)
        for s in range(0, flat_h.shape[0], 1024):
            with torch.autocast(device_type=h.device.type, enabled=False):
                dtype = compute_dtype(flat_h)  # float32 or wider, as in the training loss
                logits = torch.nn.functional.linear(flat_h[s : s + 1024].to(dtype), W.to(dtype))
                lse = torch.logsumexp(logits, -1)
                ces.append(lse - logits.gather(-1, flat_t[s : s + 1024, None]).squeeze(-1))
                tops.append(logits.argmax(-1))
        return torch.cat(ces).reshape(t.shape).double(), torch.cat(tops).reshape(t.shape)

    if slots:
        kwargs = {} if T is None else {"T": T}
        if loop1:
            kwargs["return_drafts"] = True
        out = model(tokens, doc_ids, dec, capacity=capacity, return_hidden=True, varlen=varlen, **kwargs)
        assert int(out.n_dropped.sum()) == 0
        t_slots = gather_slots(targets, out.slot_index, fill=0)
        ce_aux, top_aux = head(out.hidden_aux, targets)
        ce_s, top_s = head(out.hidden_dec, t_slots)
        ce_loop1 = scatter_slots(ce_aux, out.slot_index, head(out.hidden_drafts[0], t_slots)[0]) if loop1 else None
        return scatter_slots(ce_aux, out.slot_index, ce_s), scatter_slots(top_aux, out.slot_index, top_s), ce_loop1
    out = model(tokens, doc_ids, dec, return_hidden=True, varlen=varlen)
    return (*head(out.hidden_aux, targets), None)
