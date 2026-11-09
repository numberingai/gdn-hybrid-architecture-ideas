"""E1, row E1 ("the eval sets' rows split over ranks and the per-row sums gathered (rows are independent: exact), instead
of 7 GPUs waiting"). "Part E1".

OFF BY DEFAULT: ced/train/data_parallel.py ENV_SHARD_EVALS ("CED_DP_SHARD_EVALS" = 1), read under data parallelism only.
Without it rank 0 scores every eval row while the other ranks wait at a barrier (D4, Trainer._lead_duty); one process is
never touched by it.

What is split. The periodic and final evals on the eval sets (TrainConfig.eval.sets; ced/train/periodic_eval.py
run_eval_sets) run one forward a batch: evaluate's batches of eval.batch_size rows (ced/eval/span_bpb.py), and score_items'
batches of its packed rows (ced/eval/multiple_choice.py; final eval only). Rank r of W takes the contiguous block of
batches batch_block(n_rows, batch_size, r, W): the batches r nb / W .. (r + 1) nb / W - 1 of nb, so the blocks are
disjoint, cover every batch once, in order, and differ in size by at most one batch (a rank may get none when nb < W).
Each batch is the single process's batch (the same rows from the same start), so its numbers are the single process's:
a row is scored within its batch alone, and the model is the same on every rank (the replicas are equal: D3).

What is gathered (one gather a set, to rank 0 on the host group, DataParallel.gather_to_lead: gloo, CPU tensors):
- an eval set of rows: per batch, evaluate's result on that batch alone (a SpanEval: the per-row and per-segment sums, the
  fused check's running sums of that batch);
- a multiple-choice set: score_choice_rows over the rank's batches (each document's nats, loop-1 nats and greedy flag,
  with the documents it scored).

How rank 0 joins. span_bpb.join_batches puts the batches' sums in row order and forms the totals, bpbs, clusters and the
fingerprint with evaluate's own code over all the rows (span_bpb._span_eval), and folds the fused check's sums batch by
batch in evaluate's order; multiple_choice.join_choice_rows takes each document's numbers from the part that scored it (a
document is whole inside one row) and multiple_choice.choice_results makes the items' records. So the records rank 0
writes are the single-process eval's bit for bit on the same weights (tests/test_sharded_evals.py: in one process over the
architectures and regimes, and at W = 2 and 4 with gloo through Trainer.fit). On a GPU they are as equal as two
single-process evals of the same weights are: the same batches through the same kernels, on another device of the node.

Cost and memory: every rank loads every eval set (the trainer's cache, as rank 0 alone did before): at L = 8,192 a row's
four int64 and two bool fields are 279 KB, so 1,000 rows are 279 MB of host memory a rank (modelled). What travels is per
row and per segment: about 8 KB a one-row batch pickled, most of it the tensors' own overhead (measured on the tests'
sets; "Part E1"). A rank's eval forwards run between two training steps, as rank 0's did.
"""

from __future__ import annotations

from ced.eval.multiple_choice import choice_results, join_choice_rows, pack_whole_documents, score_choice_rows
from ced.eval.span_bpb import evaluate, join_batches


def batch_block(n_rows: int, batch_size: int, rank: int, world: int) -> list[int]:
    """The first rows of rank `rank`'s batches when n_rows rows are scored in batches of batch_size rows over `world`
    ranks: batches rank nb / world .. (rank + 1) nb / world - 1 of nb = ceil(n_rows / batch_size) (module docstring)."""
    if batch_size < 1 or not 0 <= rank < world:
        raise ValueError(f"batch_block: batch size {batch_size}, rank {rank} of {world}")
    nb = -(-n_rows // batch_size)
    return [k * batch_size for k in range(rank * nb // world, (rank + 1) * nb // world)]


def evaluate_sharded(model, rows, dp, batch_size: int = 8, **kwargs):
    """span_bpb.evaluate(model, rows, batch_size=batch_size, **kwargs) with the batches split over the ranks (module
    docstring): every rank calls it with the same arguments; rank 0 gets the joined SpanEval, the others None. dp: this
    rank's place (rank, world) and its gather (gather_to_lead; data_parallel.DataParallel)."""
    starts = batch_block(len(rows), batch_size, dp.rank, dp.world)
    parts = [(s, evaluate(model, rows[s : s + batch_size], batch_size=batch_size, **kwargs)) for s in starts]
    got = dp.gather_to_lead(parts)
    if got is None:
        return None
    return join_batches(rows, [p for rank_parts in got for p in rank_parts], batch_size)


def score_items_sharded(model, items, docs, index, dp, seq_len: int = 8192, batch_size: int = 1, autocast_dtype=None,
                        fused_varlen: bool = False):
    """multiple_choice.score_items with the packed rows' batches split over the ranks (module docstring): every rank calls
    it with the same arguments; rank 0 gets (per-task metrics, per-item results), the others None."""
    rows = pack_whole_documents(docs, seq_len)
    starts = batch_block(len(rows), batch_size, dp.rank, dp.world)
    part = score_choice_rows(model, rows, len(docs), batch_size, autocast_dtype, fused_varlen, starts=starts)
    got = dp.gather_to_lead(part)
    if got is None:
        return None
    return choice_results(items, index, join_choice_rows(got, rows))
