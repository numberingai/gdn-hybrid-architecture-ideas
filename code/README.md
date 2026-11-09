# Code for "Four Architecture Ideas for Gated DeltaNet–Attention Hybrids"

Code for the report [Four Architecture Ideas for Gated DeltaNet–Attention Hybrids](../architecture-report.pdf) by Jihoo Lee: a hybrid language model whose
trunk mixes Gated DeltaNet (GDN) layers with global-attention layers. The Python package is `ced`. The repository's own README is [../README.md](../README.md).

This folder holds the reference PyTorch implementation of the model, the training code that goes with it (trainer, optimizer, data loader, checkpoint and
resume), the evaluation code, the CPU tests of the parts below, and the configuration and launcher files for the 60M and 200M training runs of the report.
The kernels shipped here are the ones the shipped configs use. Other kernels that were used in our own experiments are not released, and the shipped
configs do not need them.

**What is not released.** The tokenizer (a 64k byte-level BPE tokenizer, described but not shipped) and the held-out evaluation sets are not released, so the
report's exact numbers cannot be reproduced from this repository alone. Training and the Based recall suite run from public data (see "Data you must bring").

## What is in it

| Part | Where |
|---|---|
| Trunk: hybrid GDN + global attention (capped RoPE and NoPE layers, Scalable-Softmax temperature, gates, norms, tied embeddings) | `ced/model/encoder.py`, `attention.py`, `gdn.py`, `norms.py`, `rope.py`, `control.py` (the non-looped model), `ced/config.py` (every key) |
| R1-entry: depth mixing at each block's entry | `ced/model/attn_res.py` |
| GDN input mixes: the write mix `wsl` and its variants `qmix`, `korig`, `vonly`, `kqmix` | `ced/model/gdn_arms.py` |
| TST: the token-superposition training schedule (bag phase, switch, loss) | `ced/train/tst.py`, wired in `ced/train/trainer.py` |
| Grid KV sharing in the global attention layers (two axes, and the third axis) | `ced/model/attention.py` (`GridSpec`, `assemble_grid`) |
| Trainer, optimizer (Muon + AdamW), data loader, checkpoint and resume | `ced/train/`, `ced/data/`, `scripts/train.py` |
| Cached generation (prefill, extend) | `ced/model/control.py` (`prefill`, `extend`), `ced/infer/engine.py` |
| Configs and run lists | `configs/v2/`, `runs/*.yaml` (`runs/examples.yaml`: the entry used below) |
| Local launcher | `scripts/run_entry.py` |
| Evaluation code: held-out bpb, multi-key and needle retrieval probes, long-context suite, RULER subset, Based recall suite, copy loss; the builders of the probe sets | `ced/eval/`, `scripts/rescore.py`, `eval_probes.py`, `eval_long.py`, `recall_ruler_score.py`, `recall_based_score.py`, `tools/build_*.py`, `tools/data/build_recall_based.py`; commands and input formats in [`EVAL.md`](EVAL.md) |
| Chat format: the pure-Python renderer of the chat template and the filter of chat documents (read by the token-type evaluation, `ced/eval/token_types.py`) | `tools/data/chat_format.py` |
| Tests (CPU) | `tests/` |

The slow, readable pure-PyTorch path is the reference. The runs used two fused paths on a GPU: the Triton GDN kernel (`ced/kernels/gdn_triton.py`, with its
CPU reference in `ced/model/gdn_chunked.py`) and the fused linear cross-entropy head (`ced/train/fused_head.py`, Triton kernels inside the file, a
PyTorch fallback beside them). A fast path is accepted only where its logits match the reference.

## Unused experimental code, imported by the core files

The modules below are experimental code of closed lines of work. No shipped config uses them. They are here only because the trainer, the model or the
evaluation modules import them at module level or at every model build; they are not part of the architecture the report is about, and they are not
maintained.

- Copy-Paste head: `ced/model/cp_head.py`, `ced/model/cp_trunk.py`, `ced/model/cp_aux.py`, `ced/train/freeze_trunk.py`.
- Looped and decoder models: `ced/model/decoder.py`, `looped_ced.py`, `loop.py`, `pass_schedule.py`, `recur_depth.py`.
- Multi-token-prediction drafter and GDN update-rule variants: `ced/model/mtp.py`, `gdn_kda.py`, `gdn_erase.py`, `gdn_negeig.py`, `gdn_rules.py`,
  `gdn_dprod.py`, `gdn_dprod_c.py`, `gdn_dprod_place.py`, `gdn_dprod_sparse.py`, `gdn_dprod_sparse_fast.py`.
- Entmax attention: `ced/model/entmax.py`.
- Linear depth memory mixer: `ced/model/ldm.py`.
- Low-precision and memory experiments: `ced/model/fp8.py`, `mxfp8.py`, `nvfp4.py`, `fused_proj.py`, `act_checkpoint.py`.
- Pre-pass of an alternative attention row term, kept for its value check: `ced/kernels/attn_row_term.py`.

Some options of `ced/config.py` belong to code that is not part of this release (matrix-optimizer experiments, the CED v2 and second-pass models,
Transformer-Engine FP8, the `once` mode of `gdn.edges`, the loop readout, length-scaled reads in the probe scorer, CuTe-DSL attention kernels). Setting one of
them fails with an `ImportError` or a refusal message; no shipped config sets one.

## Setup

Python 3.11 or newer, and PyTorch 2.14 or newer. From this folder:

```
pip install -e .            # or: pip install -e .[test,gpu,eval,logging]
```

or put the folder on `PYTHONPATH` and run the scripts from it. Extras: `gpu` (Triton, for the GDN kernel), `eval` (`pyarrow`, for the Based recall builder),
`logging` (`wandb`), `test` (`pytest`). The model, the trainer and the tests run on a CPU; training at the report's scale needs GPUs.

## Training a run

A run needs a directory `<data>` that holds tokenised shard sets (and, optionally, held-out evaluation sets) under the names the runs files use, and a
directory `<runs>` for run directories and checkpoints.

```
python scripts/run_entry.py runs/examples.yaml --list
python scripts/run_entry.py runs/examples.yaml --name f-60m-s1 --data-root <data> --run-root <runs> --dry-run
python scripts/run_entry.py runs/examples.yaml --name f-60m-s1 --data-root <data> --run-root <runs> [--nproc 8]
```

`f-60m-s1` is model F of the report at 60M parameters, seed 1: the settings of the entry `abl-f-mxl-a-s1` of `runs/ablation-60m.yaml`. The other runs files
(`runs/*.yaml`) hold the entries of the 60M and 200M runs of the report (the 60M ablation set, the scale checks, the TST runs, the 200M arms, the grid KV
runs); use them in the same way. A 200M run has two entries: stage A, then stage C, which resumes from stage A's checkpoint (`after`, `resume_from`). Training is
resumable: start the same command again after an interruption. The runs trained on earlier snapshots of the code base than this one; this repository does not
claim bitwise reproduction of those runs, only the configurations and the architecture (the parameter counts match the report's).

### Data you must bring

- **A tokenizer.** The code takes token ids; the shipped configs use a vocabulary of 64,000 (`model.vocab_size`). The report's tokenizer is not released.
- **Shard sets** in the format of `ced/data/shard_format.py`, one directory per set (`${DATA_ROOT}/phase0-v2-t64k/train` in `runs/examples.yaml`). Build one from
  your own tokenised text with the shipped `ShardWriter`:

  ```python
  import numpy as np
  from ced.data.shard_format import ShardWriter

  w = ShardWriter("<data>/phase0-v2-t64k/train", token_dtype="uint16")   # ids below 65,536
  for ids, nbytes in tokenised_documents:    # ids: token ids of one document; nbytes: UTF-8 bytes each token covers (for bpb)
      w.add_arrays(np.asarray(ids), np.asarray(nbytes, dtype=np.uint8))
  w.close({"tokenizer_sha256": "<sha256 of your tokenizer file>"})
  ```

  `DATA_CARD.md` names the public sources of the report's sets and their mix shares. The mix itself (the builders) is not part of this repository.
- **Held-out evaluation sets** (optional for training; the launcher passes the files `web.pt`, `code.pt`, `agent.pt`, `general_web.pt` of `eval_dir` that exist,
  for the periodic evaluation). The format is in `ced/eval/span_bpb.py` and `EVAL.md`. The report's sets are not released.

The Based recall suite (`scripts/recall_based_score.py`, `tools/data/build_recall_based.py`) needs only public downloads and a tokenizer; see `EVAL.md`.

## Evaluation

The evaluation scripts, their input formats and the commands are in [`EVAL.md`](EVAL.md). The tokenizer and the held-out evaluation sets are not released, so
the report's exact numbers cannot be reproduced from this repository alone; the builders take any document pool and tokenizer and make sets of the same design.

## Tests

```
pip install -e .[test]
python -m pytest -q
```

The tests run on a CPU in float64 where they compare logits. They check, among other things: causality (a change at position j never changes an
output before j), equality of teacher-forced logits with cached generation, checkpoint resume (a run stopped before, at and after a TST switch and
resumed equals the uninterrupted run, bit for bit), the grid KV sharing, and the parameter counts of the shipped runs. There are no tests of the evaluation code.

## Licence

The project's own code is MIT, see `LICENSE` (Copyright (c) 2026 Jihoo Lee). The RULER-derived parts of `ced/eval/recall_ruler.py` are under the Apache License 2.0,
whose text is in `LICENSES/Apache-2.0.txt`. Code derived from other projects keeps their notices: see `THIRD_PARTY_NOTICES.md`.
