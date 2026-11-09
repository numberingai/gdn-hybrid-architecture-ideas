"""Token Superposition Training (TST): bags of tokens for the first part of a run, then plain next-token training.

Switch rules: "Never re-initialise embeddings or the head. Keep the LR schedule continuous. Reset Muon momentum and Adam
moments at the switch. No EMA or merge windows before the switch." Method: arXiv 2605.06546 (Peng, Gigant, Quesnelle,
"Efficient Pre-Training with Token Superposition"; its section 3 and appendix A listings 1-3); prior art PLT, arXiv
2407.12665.

Off by default (TrainConfig.tst.schedule []): the trainer builds nothing of this module and every run is the run it was.

The schedule (TSTConfig.schedule): [[start_0 = 0, s_0], [start_1, s_1], ...]. Phase p runs the optimizer steps
start_p .. start_{p+1} - 1 at bag size s_p. A step is the base's step in shape: B = data.batch_size rows of L =
data.seq_len POSITIONS, the same micro-batches, the same model call; only what a position holds changes.

THE DATA (one packed stream, ced/train/shards.py). The base reads the stream in rows of L: its step k reads the stream's
rows k B .. k B + B - 1 of L tokens, row r being tokens [r L, r L + L] (L inputs and the last one's target). Under TST,
step k reads B s_k such rows ("base rows"), starting at

    R(k) = sum over j < k of B s_j                    (base rows read before step k; R(k) = k B with every s = 1)

and turns each s consecutive base rows into one BAG ROW: bag row b of step k covers the stream's tokens
[(R(k) + b s) L, (R(k) + b s) L + L s + s): L bags of s input tokens (position j: tokens j s .. j s + s - 1 of the
piece) and, as targets, the next bag of every position (tokens (j + 1) s .. (j + 1) s + s - 1). Two neighbouring bag
rows overlap by one bag (the last position's target bag is the next row's first input bag), as two base rows overlap by
one token. So the input tokens of the whole run are the stream from 0 to R(steps) L in order: NO TOKEN IS READ TWICE AS
AN INPUT AND NONE IS SKIPPED, across every switch (tests/test_tst.py). With s = 1 a bag row is the base row, read by
the base's own ShardPacker.rows: [[0, 1]] is the base bit for bit. A next-token phase after a bag phase reads the base
rows from R(k) on with that same function (positions of the data's own decoder mask are drawn per row index, as always).

A bag position's document is the document of its LAST input token (doc_ids of the row, non-decreasing; a bag that
crosses a document start belongs to the later document); target token i of position j is valid iff it is in that
document (the base's target_valid rule per token: at s = 1 the base's rule exactly). The valid targets of a bag are a
prefix of it (documents are contiguous), so a position has a valid target iff its first target is valid: that is the
batch's target_valid, which the head selection and the decoder-position sampler read as always.

THE LOSS of a bag phase (one logit row per position, the normal head; ced/train/fused_head.py fused_linear_multi_ce or
ced/train/chunked_loss.py token_multi_ce):

    loss = sum_{b,j,i} v_bji w_i (ce_bji + z lse_bj^2) / sum_{b,j,i} v_bji w_i       ce_bji = lse_bj - logit_bj[t_bji]

w = bag_weights(s): 1/s each (the paper's uniform MCE, eq. 3), or for s >= 8 w_i proportional to 1/i (i = 1 .. s)
normalised to sum 1 (the paper's power law, appendix D); v the validity above. With every target valid this is the mean
over positions of the paper's MCE (plus the base's z-loss on the position's lse); the denominator is the batch's, from
the host, as the base's n_valid. (The paper's listing averages each target index over its own valid count; the two
differ only where a bag crosses a document end, which the paper does not mask.)

THE SWITCH (Trainer.train_step, at the start of the step start_p, p > 0, before its batch): ONE call on the optimizer (none with
tst.keep_optimizer_state, THE TST CORE below), reset_optimizer_moments: optimizer.reset_moments() when the optimizer defines it, else today's reset:
every Muon momentum buffer and every AdamW exp_avg, exp_avg_sq and step is dropped (the optimizer's state is emptied; the next update creates them as
at a run's first update, so AdamW's bias correction restarts at t = 1); nothing else changes: the weights (the embedding and the head included), the
LR (a function of the step), the RNG states, the data position. A checkpoint written before the switch step resumes into it; one written after carries
the new state: a stop anywhere is the uninterrupted run bit for bit. The step's metrics carry "tst_switch" and fit() writes {"event": "tst_switch"}.

further options (TSTConfig; each off by default):
(a) gdn_decay_rescale: at a switch from s to s', A_log += ln(s' / s) in every GDN layer: g = -exp(A_log) softplus(.)
    becomes g s' / s for every input, so the decay per token, exp(g / s) for a position of s tokens, is unchanged.
(b) token_scale_rope: in a bag phase the capped-rotary layers rotate at s x (document position) with the cap in tokens
    (ced/model/attention.py GatedAttention._forward_token_scale). The paper left RoPE unchanged ("more stable").
(c) staged switches: a schedule of three or more entries (e.g. s 6 -> 2 -> 1); nothing else.
(d) phase1_exclude_sources: phase 0 reads stream 1 = the documents whose source is not excluded, in the order of the
    base's epoch-0 permutation (the per-seed one); the later phases read stream 2 = the rest: the piece of the
    document phase 0 stopped in, the other pieces of documents phase 0 began, then the documents of both queues that
    phase 0 did not touch, merged by tokens in proportion to the queues' token totals over the whole set (the ladder
    mix): a document's place is its queue's tokens before it over the queue's total. No document is read twice, none
    is skipped (tests/test_tst.py). Each stream is read within its one epoch (refused otherwise).
(e) aux_multi_hot K: in next-token phases from the first one after a bag phase, the logits at t are also scored against
    t + 2 .. t + K: loss = the base's + lambda(k) sum_{t, k>=2} v ce / ((K - 1) n_valid), lambda falling linearly from
    aux_weight at that phase's first step S to 0 at the decay start D = int(steps (1 - decay_fraction)):
    lambda(k) = aux_weight (D - k) / (D - S). At lambda = 0 (k >= D) the step is the base's step, the same code.

THE REDESIGN FOR THE GDN HYBRID (2026-10-16). TSTConfig keys, each off by default; with all of them off nothing below
runs and every run is today's TST bit for bit (tests/test_tst_redesign.py, against main's tree 140015a6).
(E) gdn_resolution_gates: in a bag phase of size s the model's forward gets gdn_scale = s (and gdn_beta_power with
    "decay_beta"): every GDN layer's kernel then takes g = s x the per-token log decay and beta = 1 - (1 - beta)^s
    (ced/model/gdn.py GatedDeltaNet.gates), so A_log / dt_bias / in_proj_a / in_proj_b always describe ONE token and the
    switch needs no surgery. The s is the step's phase's: each stage of a cascade gets its own. Replaces (a); the two
    together are refused.
(G) gdn_conv_reinit: at each switch INTO next-token (s' = 1; not at a switch between two bag sizes), every GDN layer's
    short-conv weight (conv_weight, nothing else) is drawn again from its own init U(-1/sqrt(K), 1/sqrt(K)), layer by
    layer in model order, from counter_generator(seed, switch step, CONV_REINIT_STREAM): a pure function of the run's
    seed and the step (a resume into the switch draws the same numbers; the global RNG is not used). Its optimizer state
    is reset by the switch's one call with every other moment.
(L) bag_lr_power p: in a bag phase of size s the LR of every group is the schedule's times s^p (lr_factor).
(R) rewarm_steps N: from each switch into next-token at step S, the LR of every group is the schedule's times
    (k - S + 1) / N for S <= k < S + N (lr_factor; the form of the loop switch's ramp, trainer.switch_ramp).
(output-only) output_only: a phase with s > 1 reads the BASE's rows (one token a position; the stream advances B rows a
    step, as the base's) and each position's logits are scored against its next s tokens t + 1 .. t + s with the bag
    loss above (bag_weights(s), the same multi-target head); its inputs are the base's batch at every step, so a
    next-token phase after it reads the base's own rows at the base's own step.

THE TST CORE. TSTConfig keys, each off by default; with both off every run is today's TST bit for bit (tests/test_tst_core.py,
against main's tree 495b6b6a).
(bag_input) "mean" (default): a bag position is the mean of its s embeddings; "sum_sqrt": their sum over sqrt(s). Both are
    formed in float32 (the embedding table's dtype; "sum_sqrt" raises a lower-precision table to float32 itself) before embed_norm (ced/model/control.py
    embed_tokens). RMSNorm follows the pooling in every config we train, so the two are the same input up to the norm's eps.
    The model's forward gets bag_input in a bag phase only when it is not "mean" (TSTPlan.bag_kwargs).
(keep_optimizer_state) True: the switch does not reset the optimizer (no Muon momentum, no AdamW m, v or step is touched) at any
    switch of the schedule (main.tex:190, which says nothing of the optimizer's state). The switch's event then reads
    "optimizer_states_reset": 0, "keep_optimizer_state": True and "optimizer_states_kept": the number of optimizer states (the count a
    reset would have reset); fit() logs "optimizer state KEPT". Refused with gdn_conv_reinit (the re-drawn conv weights would carry the
    old weights' moments), with optim.ours.m4 (M4 acts inside the reset call, ced/config.py OursConfig.m4, and would never fire) and with
    a schedule that has no switch. In a stage the key is carried with the rest of the source's TST keys (inert there: every switch is
    before the stage's first update). The keep-moments test arm is this key: runs/tst-keep-60m.yaml, tests/test_tst_keep.py.
(schedules of three or more phases) were always built (TSTSchedule); the redesign's options act per phase: E takes each
    bag phase's own s (g_bag = s x the per-token log decay, gdn_kwargs), L its s^p, G and R act at the switch into next-token
    only.
(the stage) a stage may now carry a schedule of any number of phases and the redesign's options E, G, L and R (they are inert
    there: every bag phase and every switch is before the stage's first update, and R's window ends before it).

DATA PARALLELISM (ced/train/data_parallel.py TST_DP; tests/test_dp_tst.py). Rank r of W reads bag rows r B/W .. (r + 1) B/W - 1 of the
step's B (batch_at first / count: stream base rows R(k) + (r B/W) s onward, the positions drawn for the global row index). The bag loss's
denominator is the GLOBAL batch's: host_counts gives each rank's histograms of the number of valid targets of a position (the integers
the ranks add in Trainer._dp_batch's one host reduction) and weights(batch, step, summed) forms the totals from the sum, the same on
every rank at every world size. The switch is a function of the step, made by every rank at the same update (the step number is added in
the same reduction and a rank elsewhere halts every rank by name); under ZeRO-1 each owner resets (or, with keep_optimizer_state, keeps)
its own tensors' state and the event's count is the ranks' counts added. With one process nothing of this runs: batch_at(step) and
weights(batch, step) are what they were, bit for bit.

A STAGE AFTER THE LAST SWITCH (TST arm, sections 3, 12.3 and 14; tests/test_tst_stage.py). A TST run's stage C
(TrainConfig.stage "continue": resume_from the source's kept checkpoint of step S, data_start_step D, ced/train/trainer.py "A
stage that continues another run") carries the source's schedule unchanged, with every switch of the schedule before S (so the
stage contains none: TSTPlan.switch never acts in it), the last phase next-token (s 1), and no option. It is built in two
forms, which differ in the data only:
- On a NEW shard set (D > 0; the trainer's stage_data_relation then requires D = S). Its update D + j reads the new set's batch j
  exactly as the base's stage C does: the base rows from (step - D) B (first_row counts the rows since D, not the source's bag
  rows) and the decoder and aux positions drawn at the data step step - D (Trainer._build_batch's k). Checked when the plan is
  built (D is S).
- On the source's OWN shard set (D = 0; tests/test_tst_stage_own_set.py). The stream goes on where the source is
  (stage_data_relation: a stage on the source's set keeps the source's data_start_step, and a new set needs D = S), so update k
  reads the rows and draws the positions the uninterrupted run draws at k: first_row counts the bag rows since step 0 and the data
  step is the step, the identity this module has for every run that is not a stage. A stage then differs from a stop and resume of
  the source in the keys of STAGE_FREE alone (optim.decay_fraction, steps): the stable run of a WSD pair continues a checkpoint
  with no decay, a branch continues the stable run's with the budget's decay, and each is the uninterrupted run of its schedule
  (the LR of every update is the whole run's: Trainer.resume_from_checkpoint checks the source's updates). The source's step S is
  known when its checkpoint is read, not when the plan is built, so what the first form checks at the construction (every switch
  before S, the last phase next-token, the re-warm window over) is checked by check_stage_source(S), which
  Trainer.resume_from_checkpoint calls after the config and data checks and before any state is loaded. A stage that would start
  before the last switch or inside a bag phase would restart the bag phase's data or reset the optimizer's moments at the switch
  inside the stage: refused, as in the first form. The option (e) matters most here: its weight anneals to 0 at int(steps (1 -
  decay_fraction)), which the stage changes, so its updates would follow another anneal than the source's. For every TST run that
  is not a stage first_row and data_step are the identity (D = 0, nothing checked), so every other TST run is the same bit for
  bit. Refused by name: a switch at or after the stage's first update (D, or S on the source's set), a stage that would start
  inside a bag phase, a re-warm window that ends after it, and the further options ((a) to (e)) and output-only in a stage (E, G,
  L, R, bag_input, keep_optimizer_state and a schedule of three or more phases are carried since the TST core,
  tests/test_tst_core.py); by the trainer's stage_data_relation, data_start_step above 0 on the source's own set and 0 on a new
  one; init_from and a plain resume_from (no stage) stay refused, as do shape_schedule and loop_switch_step.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from ced.config import BAG_INPUTS
from ced.train.batch_check import BatchError
from ced.train.data import Batch, counter_generator
from ced.train.shards import ShardPacker

POWER_LAW_FROM = 8
SWITCH_EVENT = "tst_switch"
# (E) TSTConfig.gdn_resolution_gates: off; the decay and the write strength (the derivation's form); the decay alone
GATES = ("none", "decay_beta", "decay")
# (G) the counter stream of the short-conv re-initialisation (ced/train/data.py counter_generator: streams 0-4, 1000 + b,
# 2000 + b and 10,000,000 + L are the data's and the loop's; this one is G's alone)
CONV_REINIT_STREAM = 7_000_001


def bag_weights(s: int) -> list[float]:
    """The weights of the s targets of a bag (sum 1): uniform below POWER_LAW_FROM, else 1/i normalised (i = 1 .. s)."""
    if s < POWER_LAW_FROM:
        return [1.0 / s] * s
    h = sum(1.0 / i for i in range(1, s + 1))
    return [(1.0 / i) / h for i in range(1, s + 1)]


@dataclass(frozen=True)
class Phase:
    index: int
    start: int  # its first optimizer step
    end: int    # one past its last
    s: int      # bag size (1: next-token)


class TSTSchedule:
    """TSTConfig.schedule, validated: phases, the bag size of a step, the switches, the stream position."""

    def __init__(self, entries, steps: int):
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError(f"tst.schedule {entries!r}: a non-empty list of [start_step, s]")
        pairs = []
        for e in entries:
            if (not isinstance(e, (list, tuple)) or len(e) != 2 or any(type(x) is not int for x in e)):
                raise ValueError(f"tst.schedule entry {e!r}: [start_step, s], two integers")
            pairs.append((int(e[0]), int(e[1])))
        if pairs[0][0] != 0:
            raise ValueError(f"tst.schedule {entries!r}: the first phase starts at step 0")
        for (a, sa), (b, sb) in zip(pairs, pairs[1:]):
            if not a < b:
                raise ValueError(f"tst.schedule {entries!r}: the starts must increase")
            if sa == sb:
                raise ValueError(f"tst.schedule {entries!r}: two neighbouring phases with s = {sa}: a switch that changes "
                                 "nothing would still reset the optimizer's moments; merge them")
        if pairs[-1][0] >= steps:
            raise ValueError(f"tst.schedule {entries!r}: a phase starts at or after the run's end ({steps} steps)")
        if any(s < 1 for _, s in pairs):
            raise ValueError(f"tst.schedule {entries!r}: bag sizes s >= 1")
        ends = [b for b, _ in pairs[1:]] + [steps]
        self.phases = [Phase(i, a, e, s) for i, ((a, s), e) in enumerate(zip(pairs, ends))]
        self.steps = steps
        self._starts = [p.start for p in self.phases]

    @property
    def bags(self) -> bool:
        """Whether any phase reads bags (s > 1)."""
        return any(p.s > 1 for p in self.phases)

    def phase_at(self, step: int) -> Phase:
        import bisect

        return self.phases[bisect.bisect_right(self._starts, step) - 1]

    def s_at(self, step: int) -> int:
        return self.phase_at(step).s

    def switch_at(self, step: int) -> tuple[Phase, Phase] | None:
        """(the phase before, the phase that starts) when `step` is a later phase's first step; else None."""
        p = self.phase_at(step)
        return (self.phases[p.index - 1], p) if p.index > 0 and p.start == step else None

    def rows_before(self, step: int, batch_size: int, since: int = 0, output_only: bool = False) -> int:
        """Base rows of L read by the steps since <= j < step: sum of batch_size x s_j (the module docstring's R); with
        output_only (TSTConfig.output_only) every step reads batch_size rows, as the base's (its inputs are tokens)."""
        out = 0
        for p in self.phases:
            a, b = max(p.start, since), min(p.end, step)
            if b > a:
                out += (b - a) * batch_size * (1 if output_only else p.s)
        return out


# ----------------------------------------------------------------------------------------------
# reading the stream


def bag_row(packer: ShardPacker, first_row: int, s: int) -> dict:
    """One bag row (module docstring): the stream's tokens [first_row L, first_row L + L s + s) as L positions of s
    input tokens, their next bags as targets, the positions' documents (the last input token's) and the validity of
    every target (the target token is in the position's document)."""
    L = packer.L
    x = packer._stream(first_row * L, L * s + s)
    tokens = torch.from_numpy(x["tokens"])
    doc = torch.from_numpy(x["doc"])
    inputs = tokens[: L * s].view(L, s)
    targets = tokens[s:].view(L, s)
    bag_doc = doc[s - 1 : L * s : s]
    valid = doc[s:].view(L, s) == bag_doc[:, None]
    return {"tokens": inputs, "doc_ids": bag_doc, "multi_targets": targets, "multi_valid": valid}


def extra_targets(packer: ShardPacker, first_row: int, K: int) -> dict:
    """The K targets of every position of base row `first_row` (option (e)): tokens t + 1 .. t + K of the stream and
    whether each is in the position's document. Entry 0 is the row's own target and target_valid."""
    L = packer.L
    x = packer._stream(first_row * L, L + K)
    tok = torch.from_numpy(x["tokens"])
    doc = torch.from_numpy(x["doc"])
    idx = torch.arange(L)[:, None] + 1 + torch.arange(K)[None, :]  # (L, K): t + 1 + k
    return {"multi_targets": tok[idx], "multi_valid": doc[idx] == doc[:L, None]}


class OrderedPacker(ShardPacker):
    """A ShardPacker whose stream is ONE epoch of a given document order, after an optional prefix of document pieces
    (option (d)): the layout of chat_whole over the order (row boundaries at multiples of L of this stream), the rows,
    positions and flags of the base's functions. Reading past the stream's end is refused (it would need an epoch the
    order does not define)."""

    def __init__(self, shards, cfg, span_fraction: float = 0.25, capacity: int | None = None, order=(), prefix=()):
        super().__init__(shards, cfg, span_fraction, capacity=capacity)
        self._order = np.asarray(order, dtype=np.int64)
        self._prefix = [tuple(int(v) for v in p) for p in prefix]

    def _epoch(self, epoch: int) -> dict[str, np.ndarray]:
        if epoch != 0:
            raise ValueError("this stream is one epoch of its document order: it has no epoch beyond the first")
        if 0 not in self._epoch_cache:
            p_doc = np.array([p[0] for p in self._prefix], dtype=np.int64)
            p_off = np.array([p[1] for p in self._prefix], dtype=np.int64)
            p_len = np.array([p[2] for p in self._prefix], dtype=np.int64)
            start = int(p_len.sum())
            if self.cfg.chat_whole:
                doc, offset, length = self._chat_whole_layout(self._order, start=start)
            else:
                doc, offset = self._order, np.zeros_like(self._order)
                length = np.diff(self.shards.doc_starts)[self._order]
            doc, offset, length = (np.concatenate([p_doc, doc]), np.concatenate([p_off, offset]),
                                   np.concatenate([p_len, length]))
            cum = np.concatenate([[0], np.cumsum(length)])
            self._epoch_cache = {0: {"doc": doc, "offset": offset, "length": length, "cum": cum}}
        return self._epoch_cache[0]

    @property
    def n_stream_tokens(self) -> int:
        return int(self._epoch(0)["cum"][-1])

    def _stream(self, start: int, length: int):
        if start + length > self.n_stream_tokens:
            raise ValueError(f"tst.phase1_exclude_sources: a read of stream tokens [{start}, {start + length}) passes "
                             f"the end of its one-epoch stream ({self.n_stream_tokens} tokens)")
        return super()._stream(start, length)


def base_permutation(n_docs: int, seed: int) -> np.ndarray:
    """The base packer's epoch-0 document permutation for the data seed (ShardPacker._epoch: counter stream 3)."""
    return torch.randperm(n_docs, generator=counter_generator(seed, 0, stream=3)).numpy()


def proportional_merge(order_a: np.ndarray, order_b: np.ndarray, lengths: np.ndarray, total_a: int, total_b: int
                       ) -> np.ndarray:
    """The documents of two queues in one order, by tokens in proportion total_a : total_b: a document's key is its
    queue's tokens before it over its queue's total; stable sort by (key, queue A first, place in its queue). When a
    queue has fewer documents left than its share, the other continues alone after it."""
    def keys(order, total):
        before = np.concatenate([[0], np.cumsum(lengths[order])[:-1]]) if len(order) else np.zeros(0)
        return before.astype(np.float64) / max(1, total)

    ka, kb = keys(order_a, total_a), keys(order_b, total_b)
    key = np.concatenate([ka, kb])
    queue = np.concatenate([np.zeros(len(order_a), np.int64), np.ones(len(order_b), np.int64)])
    place = np.concatenate([np.arange(len(order_a)), np.arange(len(order_b))])
    idx = np.lexsort((place, queue, key))
    return np.concatenate([order_a, order_b])[idx]


@dataclass
class SplitStreams:
    """Option (d)'s two streams and what they hold (for the logs)."""

    first: OrderedPacker
    rest: OrderedPacker
    info: dict


def split_streams(shards, data_cfg, span_fraction: float, capacity: int | None, exclude: list[str],
                  phase1_tokens: int, phase1_extra: int) -> SplitStreams:
    """Option (d) (module docstring): stream 1 = the not-excluded documents in the base permutation's order; stream 2
    = the rest after phase 0 has read stream 1's first phase1_tokens input tokens (and phase1_extra more as targets)."""
    names = list(shards.meta.get("sources") or [])
    if shards.doc_source is None or not names:
        raise ValueError("tst.phase1_exclude_sources needs the shard set's per-document sources (doc_source.npy and "
                         "meta.json 'sources'): this set has none")
    unknown = [n for n in exclude if n not in names]
    if unknown:
        raise ValueError(f"tst.phase1_exclude_sources {unknown!r}: not sources of this shard set ({names})")
    excluded = np.isin(shards.doc_source, [names.index(n) for n in exclude])
    lengths = np.diff(shards.doc_starts).astype(np.int64)
    perm = base_permutation(shards.n_docs, data_cfg.seed)
    order_a, order_b = perm[~excluded[perm]], perm[excluded[perm]]
    first = OrderedPacker(shards, data_cfg, span_fraction, capacity=capacity, order=order_a)
    lay = first._epoch(0)
    cum, n1 = lay["cum"], int(lay["cum"][-1])
    if phase1_tokens + phase1_extra > n1:
        raise ValueError(f"tst.phase1_exclude_sources: the first phase reads {phase1_tokens + phase1_extra} tokens and the "
                         f"not-excluded documents hold {n1} (one epoch of them; a second is not built)")
    i0 = int(np.searchsorted(cum, phase1_tokens, side="right") - 1)
    within = phase1_tokens - int(cum[i0])
    touched = np.zeros(shards.n_docs, dtype=bool)
    touched[lay["doc"][:i0]] = True
    prefix = []
    if within > 0:
        touched[lay["doc"][i0]] = True
        prefix.append((int(lay["doc"][i0]), int(lay["offset"][i0]) + within, int(lay["length"][i0]) - within))
    # the other pieces of documents phase 0 began (chat_whole's fill tails), from the first piece phase 0 did not read
    # into: i0 itself when phase 0 ended exactly at its start (a tail that starts at the cut; review, 2026-10-15: it was
    # skipped before), else the piece after it
    later = np.arange(i0 + 1 if within > 0 else i0, len(lay["doc"]))
    later = later[touched[lay["doc"][later]]]
    prefix += [(int(lay["doc"][j]), int(lay["offset"][j]), int(lay["length"][j])) for j in later]
    remaining_a = order_a[~touched[order_a]]
    total_a, total_b = int(lengths[~excluded].sum()), int(lengths[excluded].sum())
    merged = proportional_merge(remaining_a, order_b, lengths, total_a, total_b)
    rest = OrderedPacker(shards, data_cfg, span_fraction, capacity=capacity, order=merged, prefix=prefix)
    info = {"sources": names, "excluded": list(exclude), "tokens_first": n1, "tokens_rest": rest.n_stream_tokens,
            "total_not_excluded": total_a, "total_excluded": total_b, "phase1_tokens": phase1_tokens,
            "docs_touched_by_phase1": int(touched.sum()), "prefix_pieces": len(prefix)}
    return SplitStreams(first=first, rest=rest, info=info)


# ----------------------------------------------------------------------------------------------
# the switch's one call on the optimizer


RESET_HOOK = "reset_moments"


def reset_optimizer_moments(optimizer) -> int:
    """What the switch does to the optimizer (module docstring, THE SWITCH): ONE call, and nothing else of the
    optimizer is touched here. Returns the number of tensors whose state was reset (the event's
    "optimizer_states_reset").

    - An optimizer that defines optimizer.reset_moments() (RESET_HOOK; a method taking no argument and returning
      that count): that method, alone.
    - Every other optimizer (torch.optim.AdamW; MuonWithAdamW too until the merge of 2026-10-16, which gave it a
      reset_moments that zeroes every state tensor in place, equal to this emptying bit for bit): its whole state is
      emptied, so every Muon momentum buffer and every AdamW exp_avg, exp_avg_sq and step is created afresh by the next
      update, as at a run's first (AdamW's bias correction restarts at t = 1). For these two optimizers this equals
      zeroing every state tensor in place (zeros, then the same update), bit for bit (tests/test_tst.py)."""
    hook = getattr(optimizer, RESET_HOOK, None)
    if callable(hook):
        return int(hook())
    n_state = sum(1 for st in optimizer.state.values() if st)
    optimizer.state.clear()
    return n_state


def gdn_layers(model) -> list:
    """The model's GDN layers (GatedDeltaNet), in module order."""
    from ced.model.gdn import GatedDeltaNet

    return [m for m in model.modules() if isinstance(m, GatedDeltaNet)]


def conv_init_draws(model, seed: int, step: int) -> list[torch.Tensor]:
    """(G) What the re-initialisation at switch `step` writes into each GDN layer's conv_weight, in module order: the
    layer's own init (ced/model/gdn.py GatedDeltaNet.__init__: U(-1/sqrt(K), 1/sqrt(K)), K = conv_kernel) drawn in float32
    on the CPU from ONE counter generator of (seed, step, CONV_REINIT_STREAM), layer after layer. A pure function of the
    three numbers: the global RNG is neither read nor advanced."""
    g = counter_generator(seed, step, stream=CONV_REINIT_STREAM)
    out = []
    for m in gdn_layers(model):
        bound = 1.0 / math.sqrt(m.cfg.conv_kernel)
        out.append(torch.empty(tuple(m.conv_weight.shape), dtype=torch.float32).uniform_(-bound, bound, generator=g))
    return out


@torch.no_grad()
def reinit_gdn_conv(model, seed: int, step: int) -> int:
    """(G) Write conv_init_draws into every GDN layer's conv_weight (in place, on its device and in its dtype). Returns
    the number of layers."""
    layers = gdn_layers(model)
    for m, w in zip(layers, conv_init_draws(model, seed, step)):
        m.conv_weight.copy_(w.to(device=m.conv_weight.device, dtype=m.conv_weight.dtype))
    return len(layers)


# ----------------------------------------------------------------------------------------------
# the plan the trainer holds


class TSTPlan:
    """What the trainer does for TST (module docstring): the schedule, the streams, the batches, the switch, the loss
    weights. Built by Trainer.__init__ only when tst.schedule is set."""

    def __init__(self, trainer):
        cfg = trainer.cfg
        t = cfg.tst
        self.cfg, self.t = cfg, t
        self.sched = TSTSchedule(t.schedule, cfg.steps)
        self.B, self.L = cfg.data.batch_size, cfg.data.seq_len
        # data_start_step D: 0 for every run but a stage after the last switch (module docstring), whose new shard set's
        # stream starts at its beginning at update D (stream_of, data_step)
        self.D = cfg.data_start_step
        self.output_only = t.output_only is True  # the redesign's output-only TST (validated in _check)
        self._check(trainer)
        self.positions = trainer.positions
        self.packer = trainer.data
        self.split = None
        if t.phase1_exclude_sources:
            p0 = self.sched.phases[0]
            tokens1 = (p0.end - p0.start) * self.B * p0.s * self.L
            self.split = split_streams(trainer.data.shards, cfg.data, cfg.data.span_fraction, trainer.capacity,
                                       list(t.phase1_exclude_sources), tokens1, p0.s)
            last = self.sched.phases[-1]
            need = self.sched.rows_before(cfg.steps, self.B, since=p0.end) * self.L + last.s + max(0, t.aux_multi_hot - 1)
            if need > self.split.rest.n_stream_tokens:
                raise ValueError(f"tst.phase1_exclude_sources: the later phases read {need} tokens of the rest's stream, "
                                 f"which holds {self.split.rest.n_stream_tokens}")
        # (e): the anneal's start (the first next-token step after a bag phase) and end (the decay start)
        self.aux_from = self.aux_until = None
        if t.aux_multi_hot:
            after = [p for p in self.sched.phases if p.s == 1 and p.index > 0 and self.sched.phases[p.index - 1].s > 1]
            self.aux_from = after[0].start
            self.aux_until = int(cfg.steps * (1.0 - cfg.optim.decay_fraction))
            if self.aux_until <= self.aux_from:
                raise ValueError(f"tst.aux_multi_hot: the decay starts at step {self.aux_until}, not after the first "
                                 f"next-token step after a bag phase ({self.aux_from}): nothing to anneal")

    # -- validation -------------------------------------------------------------------------------

    def _check(self, trainer) -> None:
        """Refuse every form TST is not built and tested for, by name."""
        cfg, t, m = self.cfg, self.t, self.cfg.model
        what = "tst (Token Superposition Training)"
        if m.arch != "control" or m.loop.pass2_gate != "none" or m.loop.recur_layers or m.encoder.loops != 1:
            raise ValueError(f"{what} is built for arch 'control' with one pass through the stack (no second pass, no "
                             "looped middle, encoder.loops 1)")
        if m.encoder.attn_res not in ("none", "block_entry"):
            raise ValueError(f"{what} is built and tested for model.encoder.attn_res 'none' (the base) and 'block_entry' "
                             f"(R1-entry), not {m.encoder.attn_res!r}")
        if m.encoder.attn_res_heads != 1 or m.encoder.attn_res_query != "static":
            # the architecture arms on R1-entry (2026-10-16): nothing tests TST with them
            raise ValueError(f"{what} is not built together with model.encoder.attn_res_heads {m.encoder.attn_res_heads!r} "
                             f"or model.encoder.attn_res_query {m.encoder.attn_res_query!r} (R1-entry's own readers only)")
        if cfg.shape_schedule or cfg.loop_switch_step:
            raise ValueError(f"{what} is not built together with shape_schedule or loop_switch_step")
        # A stage (TrainConfig.stage "continue"; the trainer has checked its keys and requires resume_from with it): built
        # after the schedule's last switch only (module docstring, A STAGE AFTER THE LAST SWITCH; _check_stage).
        stage = cfg.stage != "none" or bool(cfg.data_start_step)
        # init_from is built with TST for one case: TrainConfig.freeze_trunk's post-hoc add-on stage (ced/train/freeze_trunk.py), whose
        # schedule is [[0, 1]] (the base bit for bit; freeze_trunk's own check refuses any schedule with a bag phase)
        if (cfg.init_from and not cfg.freeze_trunk) or (cfg.resume_from and not stage):
            raise ValueError(f"{what} is not built for resume_from or init_from (a run trains its phases from its own "
                             "start; a stop resumes from its own checkpoints); a stage after the schedule's last switch is "
                             "built: stage 'continue' with resume_from and data_start_step")
        if stage:
            self._check_stage()
        if cfg.optim.schedule_steps is not None:
            raise ValueError(f"{what}: the LR schedule is the run's own (optim.schedule_steps None)")
        if cfg.data.decoder_positions == "answer_spans":
            raise ValueError(f"{what} is not built for data.decoder_positions 'answer_spans'")
        bags = self.sched.bags
        if (bags or t.phase1_exclude_sources or t.aux_multi_hot) and cfg.data.kind != "shards":
            raise ValueError(f"{what} reads the packed shard stream (data.kind 'shards'): synthetic data has none")
        if (bags or t.phase1_exclude_sources or t.aux_multi_hot) and type(trainer.data) is not ShardPacker:
            # the stream functions this module reads (_stream, rows, _epoch) are ShardPacker's own; another loader (e.g.
            # a later loader over two shard sets) is not built or tested with TST until it is (review, 2026-10-15)
            raise ValueError(f"{what} reads one ShardPacker's stream; this run's loader is {type(trainer.data).__name__}")
        if (bags or t.aux_multi_hot) and not cfg.head_select:
            raise ValueError(f"{what}: the bag loss and the multi-hot aux are built on the selected head (head_select true)")
        options = t.gdn_decay_rescale or t.token_scale_rope or t.phase1_exclude_sources or t.aux_multi_hot
        if options and len(self.sched.phases) < 2:
            raise ValueError(f"{what}: the wave B options act at or after a switch; this schedule has none")
        if t.token_scale_rope:
            from ced.model.attention import rope_cap

            cap = rope_cap(m.encoder.attention)
            if cap is None:
                raise ValueError(f"{what}: tst.token_scale_rope scales the capped-rotary layers' positions; this model's "
                                 "global layers have no rotary distance cap (model.encoder.attention.rope_max_distance)")
            low = [p.s for p in self.sched.phases if p.s > 1 and cap // p.s < 1]
            if low:
                raise ValueError(f"{what}: tst.token_scale_rope at s {low} leaves no near key under the cap {cap}")
        K = t.aux_multi_hot
        if type(K) is not int or K < 0 or K == 1:
            raise ValueError(f"tst.aux_multi_hot {K!r}: 0 (off) or the horizon K >= 2 (targets t + 2 .. t + K)")
        if K:
            if not (isinstance(t.aux_weight, (int, float)) and not isinstance(t.aux_weight, bool) and t.aux_weight > 0
                    and math.isfinite(t.aux_weight)):
                raise ValueError(f"tst.aux_weight {t.aux_weight!r}: a finite weight above 0")
            if not any(p.s == 1 and p.index > 0 and self.sched.phases[p.index - 1].s > 1 for p in self.sched.phases):
                raise ValueError(f"{what}: tst.aux_multi_hot acts in the next-token phase after a bag phase; this schedule "
                                 "has none")
        if not isinstance(t.phase1_exclude_sources, (list, tuple)) or not all(isinstance(n, str) for n in t.phase1_exclude_sources):
            raise ValueError(f"tst.phase1_exclude_sources {t.phase1_exclude_sources!r}: a list of source names")
        self._check_redesign(trainer)
        # ANVIL-II's tail windows (its EMA blend, and with optim.anvil2_tavg / anvil2_tail_ema its tail average and
        # TailEMA; ced/train/matrix_optim.py, which reset_moments keeps) must open at or after the last switch
        # (2026-10-16). (A host-side plan without an optimizer, scripts/tst_gpu_check.py's expected hashes, has no
        # window to check.)
        anvil = getattr(getattr(trainer, "optimizer", None), "anvil", None)
        if anvil is not None and len(self.sched.phases) > 1 and any(
                g.get("rule") == "anvil2" for g in trainer.optimizer.param_groups):
            from ced.train.matrix_optim import anvil_points

            total, switch = trainer.total_steps, self.sched.phases[-1].start
            pts = anvil_points(total, anvil["points"])
            windows = {"the EMA blend": pts["ema"], "TAVG": pts["tavg"] if anvil["tavg"] else 0,
                       "TailEMA": pts["tail"] if anvil["tail_ema"] else 0}
            early = {k: total - w for k, w in windows.items() if w and total - w < switch}
            if early:
                raise ValueError(f"{what}: no EMA or merge window before the switch (step {switch}); ANVIL-II's windows "
                                 f"open at {early} (optim.anvil2_points {anvil['points']!r}, {total} steps)")

    def _check_stage(self) -> None:
        """A stage (TrainConfig.stage "continue"; module docstring, A STAGE AFTER THE LAST SWITCH): built only after the schedule's
        last switch, with the last phase next-token and with the schedule alone, on a new shard set (data_start_step D > 0, the
        source's step: checked here) or on the source's own (D = 0: the source's step is not known until its checkpoint is read, so
        the same checks are made by check_stage_source). Everything else is refused by name."""
        cfg, t = self.cfg, self.t
        D = cfg.data_start_step
        what = "tst (Token Superposition Training) in a stage (stage 'continue')"
        if cfg.stage != "continue" or type(D) is not int or D < 0:
            raise ValueError(f"{what} is built with stage 'continue' and data_start_step an integer >= 0 (D > 0: the source's step, "
                             "a new shard set's stream from its beginning; 0: the source's own set, its stream going on); "
                             f"stage {cfg.stage!r} with data_start_step {D!r}")
        # Since the TST core a stage carries a schedule of any number of phases and the redesign's E, G, L and R (and the core's two
        # keys): all of them act in bag phases or at a switch, which are before the stage's first update (below), so they are inert
        # in the stage. The further options and output-only stay refused (nothing tests them there; (e) would act in the stage).
        on = [name for name, v in (("gdn_decay_rescale (a)", t.gdn_decay_rescale), ("token_scale_rope (b)", t.token_scale_rope),
                                   ("phase1_exclude_sources (d)", bool(t.phase1_exclude_sources)),
                                   ("aux_multi_hot (e)", bool(t.aux_multi_hot)), ("output_only", bool(t.output_only))) if v]
        if on:
            raise ValueError(f"{what}: {on} not built in a stage (nothing tests them there); a TST stage carries the "
                             "schedule and the redesign's E, G, L, R alone")
        if D > 0:  # a new shard set (or a stage of a stage on one): D is the source's step
            self._check_stage_start(D, f"data_start_step {D}")

    def check_stage_source(self, step: int) -> None:
        """A stage on the source's own shard set (data_start_step 0; module docstring): the checks of the first update, which are
        D's for a stage on a new set and so are made when the plan is built, made with the source's step `step`. Called by
        Trainer.resume_from_checkpoint once the source's checkpoint is read and its config and data are checked, and before any of
        its state is loaded. ValueError by name. A stage with data_start_step above 0 was checked at the construction."""
        if self.cfg.data_start_step == 0:
            self._check_stage_start(step, f"the source's step {step} on its own shard set, data_start_step 0")

    def _check_stage_start(self, first: int, where: str) -> None:
        """The stage's first update `first` (data_start_step D on a new set, the source's step S on its own): every switch of the
        schedule is before it, it is in the schedule's last phase, a next-token one, and the re-warm window is over. `where` names
        what `first` is in the message."""
        t, phases = self.t, self.sched.phases
        what = "tst (Token Superposition Training) in a stage (stage 'continue')"
        inside = [p.start for p in phases[1:] if p.start >= first]
        if inside:
            raise ValueError(f"{what}: the schedule switches at step(s) {inside}, at or after the stage's first update ({where}); "
                             "a switch inside a stage is not built: a TST stage continues a run after its last switch")
        last = self.sched.phase_at(first)  # every switch is before it: the schedule's last phase
        if last.s != 1:
            raise ValueError(f"{what}: the stage would start inside a bag phase (s {last.s} from step {last.start}); a TST "
                             "stage continues a run in its last phase, a next-token one (s 1)")
        if type(t.rewarm_steps) is int and t.rewarm_steps > 0 and last.index > 0 and last.start + t.rewarm_steps > first:
            raise ValueError(f"{what}: tst.rewarm_steps {t.rewarm_steps} from the last switch (step {last.start}) ends after the "
                             f"stage's first update ({where}); the re-warm window must be over before the stage")

    def _check_redesign(self, trainer) -> None:
        """The redesign's keys (module docstring, THE REDESIGN): their values, where each acts, and the combinations
        that are refused by name."""
        t, phases = self.t, self.sched.phases
        what = "tst (Token Superposition Training)"
        E, G, p, N = t.gdn_resolution_gates, t.gdn_conv_reinit, t.bag_lr_power, t.rewarm_steps
        if E not in GATES:
            raise ValueError(f"tst.gdn_resolution_gates {E!r}: one of {GATES}")
        if not isinstance(G, bool):
            raise ValueError(f"tst.gdn_conv_reinit {G!r}: true or false")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"tst.bag_lr_power {p!r}: a number in [0, 1] (0: off; 0.5: the LR x sqrt(s) in a bag phase)")
        if type(N) is not int or N < 0:
            raise ValueError(f"tst.rewarm_steps {N!r}: an integer >= 0 (0: off)")
        if not isinstance(t.output_only, bool):
            raise ValueError(f"tst.output_only {t.output_only!r}: true or false")
        bag = [ph for ph in phases if ph.s > 1]
        to_token = [ph for ph in phases if ph.s == 1 and ph.index > 0]  # neighbours differ: each follows a bag phase
        # the TST core's two keys (module docstring, THE TST CORE)
        if t.bag_input not in BAG_INPUTS:
            raise ValueError(f"tst.bag_input {t.bag_input!r}: one of {BAG_INPUTS}")
        if not isinstance(t.keep_optimizer_state, bool):
            raise ValueError(f"tst.keep_optimizer_state {t.keep_optimizer_state!r}: true or false")
        if t.bag_input != "mean" and not bag:
            raise ValueError(f"{what}: tst.bag_input {t.bag_input!r} acts in bag phases (s > 1); this schedule has none")
        if t.keep_optimizer_state and G:
            raise ValueError(f"{what}: tst.keep_optimizer_state keeps every optimizer state across the switch, and "
                             "tst.gdn_conv_reinit draws the conv weights again at it (their moments would be the old weights'); "
                             "set one of them")
        if t.keep_optimizer_state:
            if len(phases) < 2:
                raise ValueError(f"{what}: tst.keep_optimizer_state acts at a switch; this schedule has none")
            ours = getattr(getattr(getattr(trainer, "cfg", None), "optim", None), "ours", None)
            if getattr(ours, "m4", False):
                raise ValueError(f"{what}: tst.keep_optimizer_state makes no call on the optimizer at the switch, and M4 "
                                 "(optim.ours.m4) acts inside that call: it would never fire; set one of them")
        if (E != "none" or p) and not bag:
            raise ValueError(f"{what}: tst.gdn_resolution_gates and tst.bag_lr_power act in bag phases (s > 1); this "
                             "schedule has none")
        if (G or N) and not to_token:
            raise ValueError(f"{what}: tst.gdn_conv_reinit and tst.rewarm_steps act at a switch into next-token (s = 1); "
                             "this schedule has none")
        short = [(ph.start, ph.end - ph.start) for ph in to_token if N > ph.end - ph.start]
        if short:
            raise ValueError(f"{what}: tst.rewarm_steps {N} does not fit the next-token phase(s) (start, steps) {short}")
        if E != "none" and t.gdn_decay_rescale:
            raise ValueError(f"{what}: tst.gdn_resolution_gates (E) replaces tst.gdn_decay_rescale (a): E keeps A_log the "
                             "per-token rate in every phase, (a) moves A_log at the switch; set one of them")
        model = getattr(trainer, "model", None)
        if (E != "none" or G) and model is not None and not gdn_layers(model):
            raise ValueError(f"{what}: tst.gdn_resolution_gates and tst.gdn_conv_reinit act on GDN layers; this model has none")
        if t.output_only:
            if not bag:
                raise ValueError(f"{what}: tst.output_only scores the next s tokens in phases with s > 1; this schedule has none")
            on = [name for name, v in (("gdn_resolution_gates", E != "none"), ("gdn_conv_reinit", G),
                                       ("gdn_decay_rescale", t.gdn_decay_rescale), ("token_scale_rope", t.token_scale_rope),
                                       ("phase1_exclude_sources", bool(t.phase1_exclude_sources)),
                                       ("aux_multi_hot", bool(t.aux_multi_hot)), ("bag_input", t.bag_input != "mean")) if v]
            if on:
                raise ValueError(f"{what}: tst.output_only reads token rows (one token a position, the base's stream); "
                                 f"{on} act on bag positions or on the bag stream and are not built with it")

    # -- what a step reads ------------------------------------------------------------------------

    def stream_of(self, phase: Phase) -> tuple[ShardPacker, int]:
        """(the packer of the phase's stream, the step from which that stream's rows are counted): the run's packer from
        data_start_step D (0 for every run but a stage after the last switch, whose new shard set's stream starts at its
        beginning at update D; module docstring), or one of option (d)'s two streams (never in a stage)."""
        if self.split is None:
            return self.packer, self.D
        first = self.sched.phases[0]
        return (self.split.first, 0) if phase.index == 0 else (self.split.rest, first.end)

    def first_row(self, step: int) -> int:
        """The first base row step `step` reads, in its stream (module docstring: R(k), counted from the stream's first
        step; in a stage after the last switch (step - D) B, the base's row of the data step; with output_only every step
        reads B rows, the base's)."""
        _, since = self.stream_of(self.sched.phase_at(step))
        return self.sched.rows_before(step, self.B, since=since, output_only=self.output_only)

    def data_step(self, step: int) -> int:
        """Where update `step` draws its decoder and aux positions: step - data_start_step, as Trainer.data_step (the step
        itself for every run but a stage after the last switch). ValueError for an update before the stream starts: a
        stage never trains those (they are the source's)."""
        k = step - self.D
        if k < 0:
            raise ValueError(f"update {step} is before this run's data stream starts (data_start_step {self.D}): those "
                             "updates are the source's")
        return k

    def s_in(self, phase: Phase) -> int:
        """The tokens one INPUT position of the phase holds: its s, or 1 under output_only (token rows)."""
        return 1 if self.output_only else phase.s

    def lr_factor(self, step: int) -> float | None:
        """(L) and (R) (module docstring, THE REDESIGN): the factor on the schedule's LR of `step`, every group; None
        where neither acts (the LR is then the schedule's own number, not that number times 1.0).
        - a bag phase of size s with bag_lr_power p > 0: s^p;
        - the first rewarm_steps N steps of a next-token phase that follows a bag phase (from its start S):
          (step - S + 1) / N."""
        phase = self.sched.phase_at(step)
        if phase.s > 1:
            return float(phase.s) ** self.t.bag_lr_power if self.t.bag_lr_power else None
        N = self.t.rewarm_steps
        if N and phase.index > 0 and step - phase.start < N:
            return (step - phase.start + 1) / N
        return None

    def gdn_kwargs(self, step: int) -> dict:
        """(E) The model forward's GDN arguments at `step`: {"gdn_scale": s, "gdn_beta_power": ...} in a bag phase with
        gdn_resolution_gates on (s the phase's own: each stage of a cascade has its own), else {} (the forward as
        always)."""
        s = self.sched.s_at(step)
        if s == 1 or self.t.gdn_resolution_gates == "none":
            return {}
        return {"gdn_scale": s, "gdn_beta_power": self.t.gdn_resolution_gates == "decay_beta"}

    def bag_kwargs(self, step: int) -> dict:
        """The model forward's bag argument at `step`: {"bag_input": "sum_sqrt"} in a bag phase with tst.bag_input
        "sum_sqrt", else {} (the forward as always: "mean" is the forward's default)."""
        if self.sched.s_at(step) == 1 or self.t.bag_input == "mean":
            return {}
        return {"bag_input": self.t.bag_input}

    def aux_lambda(self, step: int) -> float:
        """Option (e)'s weight at `step` (0.0 off, outside next-token phases, before the anneal's start and from the
        decay start on)."""
        if not self.t.aux_multi_hot or self.sched.s_at(step) != 1 or step < self.aux_from or step >= self.aux_until:
            return 0.0
        return self.t.aux_weight * (self.aux_until - step) / (self.aux_until - self.aux_from)

    def batch_at(self, step: int, first: int = 0, count: int | None = None) -> Batch:
        """The training batch of `step` (module docstring), with the configured decoder / aux positions of its data step
        (the step itself but in a stage after the last switch: step - D, as the base's stage draws them).
        first, count: DATA PARALLELISM (ced/train/data_parallel.py D2; Trainer._build_batch): a rank's rows first .. first +
        count - 1 of the step's B rows. Row b of the step's batch is the stream's base row R(k) + b x s_in (s_in: the bag size,
        1 for the next-token phase and under output_only), so the rank reads from base row R(k) + first x s_in on; and the
        positions of row b are drawn from the generator of row first + b (PositionSampler.apply's row0), as for the whole
        batch. With the defaults (the whole batch) every line is the one it was."""
        phase = self.sched.phase_at(step)
        packer, _ = self.stream_of(phase)
        k = self.data_step(step)
        r0 = self.first_row(step)
        n = self.B if count is None else count
        if phase.s == 1:
            batch = packer.rows(r0 + first, n)  # the base's own rows: with every s = 1 the base's batch(step) exactly
            K = self.t.aux_multi_hot
            if K and self.aux_lambda(step) > 0.0:
                extra = [extra_targets(packer, r0 + first + b, K) for b in range(n)]
                batch = dataclasses.replace(batch, multi_targets=torch.stack([e["multi_targets"] for e in extra]),
                                            multi_valid=torch.stack([e["multi_valid"] for e in extra]))
            return self.positions.apply(batch, k, row0=first)
        if self.output_only:
            # output-only TST: the base's rows (its inputs, documents and positions), each position's next s tokens as
            # targets (entry 0 is the row's own target and target_valid: extra_targets)
            batch = packer.rows(r0 + first, n)
            extra = [extra_targets(packer, r0 + first + b, phase.s) for b in range(n)]
            batch = dataclasses.replace(batch, multi_targets=torch.stack([e["multi_targets"] for e in extra]),
                                        multi_valid=torch.stack([e["multi_valid"] for e in extra]))
            return self.positions.apply(batch, k, row0=first)
        rows = [bag_row(packer, r0 + (first + b) * phase.s, phase.s) for b in range(n)]
        mt = torch.stack([r["multi_targets"] for r in rows])
        mv = torch.stack([r["multi_valid"] for r in rows])
        batch = Batch(tokens=torch.stack([r["tokens"] for r in rows]), targets=mt[..., 0].clone(),
                      target_valid=mv[..., 0].clone(), doc_ids=torch.stack([r["doc_ids"] for r in rows]),
                      dec_mask=torch.zeros(n, self.L, dtype=torch.bool), multi_targets=mt, multi_valid=mv)
        return self.positions.apply(batch, k, row0=first)

    def weights(self, batch: Batch, step: int, summed: list[int] | None = None) -> tuple[torch.Tensor, dict]:
        """(w (K,) float64 per-target weights of the step, host totals). A bag phase: bag_weights(s), and
        "w_total" = sum of v w over the batch (the loss's denominator), "w_dec" / "w_ctx" over the decoder-mask / other
        positions (the metrics' split). The multi-hot aux: [1, lam / (K - 1), ...] (the main target's weight 1; the
        denominator is n_valid as the base's), "n_aux" = the valid aux targets.
        summed: DATA PARALLELISM (Trainer._dp_batch): host_counts of every rank's rows summed over the ranks. The totals are
        then the GLOBAL batch's, as the single process's are, so each rank's micro-batch loss is its local sum over the global
        denominator; w itself is a function of the step. Without it (every run in one process) the totals are those of the
        batch given, computed as they always were."""
        s = self.sched.s_at(step)
        v = batch.multi_valid.to(torch.float64)
        if s > 1:
            w = torch.tensor(bag_weights(s), dtype=torch.float64)
            if summed is not None:
                w_ctx, w_dec = (float(torch.tensor(h, dtype=torch.float64) @ self._prefix_weights(w))
                                for h in (summed[:s + 1], summed[s + 1:2 * (s + 1)]))
                return w, {"w_total": w_dec + w_ctx, "w_dec": w_dec, "w_ctx": w_ctx}
            per_pos = (v * w).sum(-1)
            dec = batch.dec_mask.to(torch.float64)
            return w, {"w_total": float(per_pos.sum()), "w_dec": float((per_pos * dec).sum()),
                       "w_ctx": float((per_pos * (1.0 - dec)).sum())}
        K, lam = v.shape[-1], self.aux_lambda(step)
        w = torch.tensor([1.0] + [lam / (K - 1)] * (K - 1), dtype=torch.float64)
        return w, {"n_aux": int(batch.multi_valid[..., 1:].sum()) if summed is None else int(summed[0]), "lambda": lam}

    @staticmethod
    def _prefix_weights(w: torch.Tensor) -> torch.Tensor:
        """(s + 1,) float64: entry k is the weight a position with k valid targets carries, w_1 + ... + w_k (entry 0: 0). The
        valid targets of a position are a prefix of its bag (module docstring), so this is sum_i v_i w_i."""
        return torch.cat([torch.zeros(1, dtype=w.dtype), torch.cumsum(w, 0)])

    @property
    def host_width(self) -> int:
        """The most integers host_counts returns at any step of the schedule: the fixed width of the TST part of the step's host
        reduction, so that a rank at another update still sends a vector of the same length as the others (and the step-number
        check, not a length mismatch, is what stops the ranks)."""
        return max([2 * (p.s + 1) if p.s > 1 else (1 if self.t.aux_multi_hot else 0) for p in self.sched.phases])

    def host_count_size(self, step: int) -> int:
        """How many integers host_counts returns at `step`: a function of the step alone, so that a rank whose batch failed
        the loader's validation (and has none to count) still sends the right number to the other ranks' sum. A bag phase:
        two histograms of s + 1; a next-token phase with the multi-hot aux on: 1; otherwise 0 (no multi-target batch)."""
        s = self.sched.s_at(step)
        if s > 1:
            return 2 * (s + 1)
        return 1 if self.t.aux_multi_hot and self.aux_lambda(step) > 0.0 else 0

    def host_counts(self, batch: Batch, step: int) -> list[int]:
        """DATA PARALLELISM: the integers of this rank's rows from which weights(..., summed) makes the global batch's totals;
        the ranks add them (Trainer._dp_batch, in its one host reduction). A bag phase (the batch's s > 1): the histograms of
        the number k of valid targets of a position (0 .. s), the positions on the batch's decoder mask last: [positions
        off the mask with k = 0 .. s] + [positions on it with k = 0 .. s]. The sum of two ranks' histograms is the histogram
        of their rows together, exactly, so every rank gets the same totals whatever the world size (float sums of the
        ranks' float totals would depend on the order of the sum). The multi-hot aux: [the valid aux targets]. Empty when the
        batch has no multi-target tensors. BatchError if a position's valid targets are not a prefix of its bag (the
        histogram would weight it wrongly; the loader's documents are contiguous, so this does not happen)."""
        if batch.multi_targets is None:
            return []
        if self.sched.s_at(step) == 1:
            return [int(batch.multi_valid[..., 1:].sum())]
        s = self.sched.s_at(step)
        v = batch.multi_valid
        if bool((v[..., 1:] & ~v[..., :-1]).any()):
            raise BatchError("the valid targets of a TST bag position are not a prefix of its bag",
                             {"field": "multi_valid", "rule": "prefix"})
        k = v.sum(-1).reshape(-1) + (s + 1) * batch.dec_mask.reshape(-1).to(torch.long)
        return torch.bincount(k, minlength=2 * (s + 1)).tolist()

    # -- the switch -------------------------------------------------------------------------------

    @torch.no_grad()
    def switch(self, trainer, step: int) -> dict | None:
        """At a later phase's first step (before its batch): reset the optimizer's state (not with
        tst.keep_optimizer_state, which leaves it), with (a) rescale every GDN layer's A_log, and with (G), at a switch into
        next-token, draw every GDN layer's short-conv weight again from its init (after the reset, which covers its
        moments). Returns the event's record, None at every other step."""
        sw = self.sched.switch_at(step)
        if sw is None:
            return None
        before, after = sw
        keep = self.t.keep_optimizer_state  # the state goes on across the switch (THE TST CORE)
        n_state = 0 if keep else reset_optimizer_moments(trainer.optimizer)
        event = {"event": SWITCH_EVENT, "step": step, "from_s": before.s, "to_s": after.s, "phase": after.index,
                 "optimizer_states_reset": n_state}
        if keep:
            event["keep_optimizer_state"] = True
            event["optimizer_states_kept"] = sum(1 for st in trainer.optimizer.state.values() if st)  # the count a reset would have reset
        # DATA PARALLELISM: the switch is a function of the step, which every rank has reached together (Trainer._dp_batch checks
        # it), so every rank resets or keeps its optimizer's state at the same update with no message. A replicated optimizer
        # holds all of the state on every rank (the count is the single process's); the sharded one (ZeRO-1) holds each tensor's on
        # its owner only, so the event's count is the ranks' counts added: no collective here (a rank at another update would not
        # reach it); the count rides in Trainer._dp_batch's one host reduction, which patches the event in place.
        if getattr(trainer, "_z1", None) is not None:
            key = "optimizer_states_kept" if keep else "optimizer_states_reset"
            trainer._tst_switch_pending = (event, key, int(event[key]))
        report = None if keep else getattr(trainer.optimizer, "reset_report", None)
        if report is not None:  # our optimizer (OptimConfig.ours with m2 or m3): M2's noise estimate before, what M4 did
            event["optimizer_report"] = report
        if self.t.gdn_decay_rescale:
            from ced.model.gdn import GatedDeltaNet

            shift = math.log(after.s / before.s)
            layers = [m for m in trainer.model.modules() if isinstance(m, GatedDeltaNet)]
            for m in layers:
                m.A_log.add_(shift)
            event["gdn_decay_rescale"] = {"A_log_shift": shift, "layers": len(layers)}
        if self.t.gdn_conv_reinit and after.s == 1:  # (G): into next-token only (module docstring)
            n = reinit_gdn_conv(trainer.model, trainer.cfg.seed, step)
            event["gdn_conv_reinit"] = {"layers": n, "seed": trainer.cfg.seed, "stream": CONV_REINIT_STREAM}
        return event

    def describe(self) -> dict:
        """The run's TST plan for its start records (Trainer.fit): the phases, the options, option (d)'s streams."""
        t = self.t
        out = {"phases": [[p.start, p.end, p.s] for p in self.sched.phases],
               "options": {"gdn_decay_rescale": t.gdn_decay_rescale, "token_scale_rope": t.token_scale_rope,
                           "phase1_exclude_sources": list(t.phase1_exclude_sources), "aux_multi_hot": t.aux_multi_hot,
                           "aux_weight": t.aux_weight if t.aux_multi_hot else None,
                           "gdn_resolution_gates": t.gdn_resolution_gates, "gdn_conv_reinit": t.gdn_conv_reinit,
                           "bag_lr_power": t.bag_lr_power, "rewarm_steps": t.rewarm_steps, "output_only": t.output_only}}
        # the TST core's keys appear in the record only when they are not at their defaults, so that every earlier run's start
        # record is the record it was
        core = {k: getattr(t, k) for k, d in (("bag_input", "mean"), ("keep_optimizer_state", False)) if getattr(t, k) != d}
        if core:
            out["options"].update(core)
        if self.split is not None:
            out["split"] = self.split.info
        if t.aux_multi_hot:
            out["aux_anneal"] = [self.aux_from, self.aux_until]
        return out

    def data_state(self) -> dict:
        """What the run's data identity adds under TST (Trainer.data_state): the phases' bag sizes and starts, which
        place every step in the stream, and option (d)'s sources. Absent when the stream is the base's."""
        out = {}
        if self.sched.bags:
            out["tst_schedule"] = [[p.start, p.s] for p in self.sched.phases]
        if self.t.phase1_exclude_sources:
            out["tst_phase1_exclude_sources"] = list(self.t.phase1_exclude_sources)
        if self.output_only:  # output-only TST reads the base's rows: another place in the stream than bags of s
            out["tst_output_only"] = True
        return out
