# Evaluation

**What is and is not released.** The tokenizer (a 64k byte-level BPE) and the held-out evaluation sets are not released, so the report's exact numbers cannot
be reproduced from this repository alone. Training and the Based recall suite run from public data. The evaluation code is here with documented input
formats: the scorers take any tokenizer of the right kind and any document pool, and the builders make sets of the same design from them. Numbers from
other tokenizers and other documents are not comparable with the report's.

All commands run from the directory that holds `ced/`, `scripts/` and `tools/`, with the package importable (`pip install -e .`, or `PYTHONPATH=.`). Run names
such as `f-60m-s1` are examples. `<...>` marks what you bring or choose.

## What you bring

| Placeholder | What it is |
|---|---|
| `<tokenizer.json>` | A Hugging Face `tokenizers` file of a byte-level BPE with a 64,000-entry vocabulary. Probe-set directories are named after the tokenizer (`t64k`); `scripts/eval_probes.py` refuses a set whose name implies another vocabulary size than the model's (`--allow-vocab-mismatch` overrides). |
| `<checkpoint>` | A trainer checkpoint (`<run>/checkpoints/step_<N>.pt`, written by `scripts/train.py` or `scripts/run_entry.py`). The scorers rebuild the model from the config stored in it. Most scorers also accept a run directory (they follow `checkpoints/latest`). |
| `<held-out docs>` | Two files `web.docs.jsonl.gz` and `general_web.docs.jsonl.gz`: gzip JSON lines `{"id": ..., "text": ...}`, one document per line. They are the haystacks of the retrieval and copy builders and must not be in the training data. |
| `<eval sets>` | For held-out bpb: `web.pt`, `code.pt`, `agent.pt`, `general_web.pt`, in the `EvalRows` format below. |

The end-of-text token `<|endoftext|>` must exist in the tokenizer (the Based recall suite assumes id 0 for it). CPU is enough for small tests of every command; the scorers at 32k tokens and longer are meant for a GPU.

## 1. Held-out bpb (four sets)

Input: four `EvalRows` files. `ced/eval/span_bpb.py` defines and writes them (`EvalRows.save`, `EvalRows.load`). A file is one `torch.save` dictionary:

| field | shape, type | meaning |
|---|---|---|
| `tokens` | (N, L) int64 | the inputs; L is the row length (8,192 for the report's sets) |
| `targets` | (N, L) int64 | the next tokens; row i of the stream overlaps row i+1 by one token |
| `target_valid` | (N, L) bool | False where the target belongs to the next document and in the padding |
| `target_bytes` | (N, L) int64 | UTF-8 bytes covered by each target token (a token's own vocabulary string; the table `load_tokenizer` of `tools/build_probes.py` returns) |
| `doc_ids` | (N, L) int64 | document index, increasing along a row |
| `dec_mask` | (N, L) bool | the scored positions (decoder positions) |
| `capacity` | int | decoder slots a row gets |
| `kind` | str | `"spans"` (fixed-seed spans, text) or `"roles"` (assistant tokens of chat trajectories) |
| `fingerprint` | str | hash of the fields above; `EvalRows.load` checks it, so write files with `EvalRows.save`, not by hand |
| `doc_group`, `special_ids`, `target_class` | optional | cluster id per document; special-token ids; per-target token class (uint8) |

Build them from your own tokenised documents with `ced.eval.span_bpb.build_eval_rows(docs, seq_len, capacity, spans, seed)` for text sets,
`build_role_eval_rows` or `build_whole_role_rows` for chat trajectories, with `docs` a list of `ced.eval.tokenized.TokenizedDoc(ids, byte_lens)`
(`RoleDoc` adds the per-token assistant flags). Text documents end with the `<|endoftext|>` token. The report's four sets are web, code, agent
(chat trajectories, `kind="roles"`) and general_web. Optionally put the tokenizer's sha256 in `tokenizer_sha256.txt` next to the files.

```
python scripts/rescore.py --checkpoint <checkpoint> \
    --set web=<eval sets>/web.pt --set code=<eval sets>/code.pt --set agent=<eval sets>/agent.pt --set general_web=<eval sets>/general_web.pt \
    --out <out>/rescore_final.json [--resume]
```

`<out>/rescore_final.json` holds `results[<set>]["dec_bpb"]` for each set: bits per byte at the scored positions, sum of nats / (ln 2 x sum of target bytes).
The four-set bpb is the mean of the four `dec_bpb` values. `--resume` keeps each finished set in `<out>.units/`.

## 2. Multi-key and UUID needles, 8k to 64k and 128k

Build (CPU):

```
python tools/build_probes.py --out <probes> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz \
    --probes uuid_needle,multikey --workers 4
python tools/build_probes.py --out <probes-128k> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz \
    --probes uuid_needle,multikey --lengths 131072 --uuid-per-depth 20 --multikey-per-cell 100 --workers 4
```

The first call builds 8k, 16k, 32k and 64k with 300 multi-key items per cell. The report's 8k to 64k sets were built with a second tokenizer as an extra length
measure (add `--tokenizer t128k=<other tokenizer.json>`; an item's length is then the larger of its two token counts). The output is `<probes>/t64k/<probe>.pt`
(`ced.eval.probes.EncodedSet`: token ids, answer spans and per-item records), `<probes>/text/<probe>.jsonl.gz` (the text items) and manifests with the sha256 of every file.

Score:

```
python scripts/eval_probes.py --checkpoint <checkpoint> --probes <probes>/t64k --probe-names uuid_needle,multikey --out <out>/probes.json [--resume]
python scripts/eval_probes.py --checkpoint <checkpoint> --probes <probes-128k>/t64k --out <out>/needles-128k.json [--resume]
```

`<out>/probes.json` holds `summary[<probe>]["length=<L>,n_keys=<K>"]` with `exact` (every answer token is the argmax under teacher forcing), `first_acc`,
`nats_per_token` and their standard errors, and every item's result under `items`. The ten retrieval cells are `multikey` with 4 and 16 keys at 8,192, 16,384,
32,768, 65,536 and 131,072 tokens; each cell's value is `exact`.

## 3. Multi-fact

```
python tools/build_probes_multifact.py --out <multifact> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --per-cell 60 --workers 4
python tools/build_probes_multifact.py --out <multifact-128k> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --lengths 131072 --per-cell 60 --workers 4
python scripts/eval_long.py --checkpoint <checkpoint> --data <any existing directory> --out <out>/multifact --parts multifact --multifact <multifact>/t64k
python scripts/eval_long.py --checkpoint <checkpoint> --data <any existing directory> --out <out>/multifact-128k --parts multifact --multifact <multifact-128k>/t64k
```

The probes are `mf_chain` and `mf_concat` with k = 2, 4 and 8 facts at 8k, 16k, 32k and 64k (default) or 128k. `--data` is read only by the other parts of
`eval_long.py`, so any existing directory works for `--parts multifact`. `<out>/multifact/summary.json` and `summary.md` hold, per probe, k and length (`summary["multifact"][<probe>]["k=<k>"][<length>]`), `exact` (every fact is right),
`fact_acc` (share of facts right), `nats` (nats per fact token), `first_nats` and `rest_nats`, with standard errors, and the same per layout (spread, front, back).

## 4. Long-context suite

`python scripts/eval_long.py --checkpoint <checkpoint> --data <long-context sets> --out <out> [--parts bpb,agent,multifact,gdn]` runs four parts. `multifact` needs only
the sets of section 3. The others read `<data>/web.pt`, `code.pt`, `code_repo.pt`, `agent.pt` and `gdn_joined.pt`: whole tokenised documents in the `LongDocs` format of
`ced/eval/long_context.py` (`LongDocs.from_docs`: token ids, byte lengths, offsets, per-document metadata, and for chat sets the assistant flags). No builder for these
sets is included.

## 5. The 200M multi-key items

`configs/eval/recall-items-plan.json` lists 16 cells (multi-key needles with 4 and 16 keys at 8k to 128k, and the `pcr_niahk` items at 2k, 4k and 8k; 8,000 items in
all) and splits them into three scoring groups. Each cell's first items are those of the smaller sets of section 2.

```
python tools/build_recall_items.py build --plan configs/eval/recall-items-plan.json --out <items> --tokenizer-t64k <tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --workers 4
python scripts/eval_probes.py --checkpoint <checkpoint> --probes <items>/groups/g1/t64k --out <out>/probes/f-200m-s1/g1.json --resume --taus ""
    (the same for g2 and g3)
python tools/analysis/recall_items_values.py run --plan configs/eval/recall-items-plan.json --run f-200m-s1 \
    --probes-dir <out>/probes/f-200m-s1 --out <out>/values/f-200m-s1.values.json
```

`build` runs three raw builds (`tools/build_probes.py`, `tools/build_report_pcr.py`) and then `split`. Add `--tokenizer-t128k` for the second length tokenizer of the
8k to 64k cells, and `--old-root` to compare the first items of each cell with the old sets (see the docstring of `tools/build_recall_items.py`). The values file holds
`cells["items|<probe>|<cell>|exact"]` and `n_items` per cell; `recall_items_values.py target-check` compares the sampling error with the spread between seeds.

## 6. RULER subset

Six tasks of RULER (multi-key, multi-value and multi-query needles, variable tracking) at 2,048, 4,096 and 8,192 tokens, with a haystack of held-out documents.

```
python tools/build_recall_ruler.py --out <ruler items> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --per-cell 500 --workers 4
python scripts/recall_ruler_score.py --checkpoint <run>/checkpoints/step_00007996.pt --items <ruler items> --tokenizer <tokenizer.json> \
    --out <out>/ruler --run f-200m-s1 --n-per-cell 500
```

Output: `<out>/ruler/f-200m-s1.values.json` (`cells["ruler|<task>|<length>|recall"]`, RULER's string-match recall), `.items.jsonl` (every generation) and `.meta.json`.
Defaults: `--scorer fast`, reference attention, chunked GDN kernel. `ced/eval/recall_ruler.py` states the deviations from RULER and carries its attribution.

## 7. Based recall suite (SQuAD completion, SWDE, FDA)

This one needs no held-out documents of ours. The builder downloads three pinned files from the Hugging Face hub (`hazyresearch/based-squad`, `based-swde`, `based-fda`,
validation splits; revision and sha256 are in `ced/eval/recall_based.py`), checks their sha256, and needs `pyarrow` (`pip install -e .[eval]`). SQuAD 2.0 is CC BY-SA 4.0;
SWDE and FDA state no licence, so the items are downloaded for scoring and never redistributed.

```
python tools/data/build_recall_based.py --raw <raw> --out <based items> --tasks squad,swde,fda
python scripts/recall_based_score.py score --out <out>/based --items <based items> --tokenizer <tokenizer.json> --final-step 7996 --run f-200m-s1=<run dir>
```

The protocol is that of the reference harness (prompt, 48 greedy tokens, `contains` metric, left truncation); `ced/eval/recall_based.py` lists the points where it
differs. Output: `<out>/based/f-200m-s1.values.json` with `cells["based|<task>|all|acc"]`, and per-item records. `recall_based_score.py values` recomputes the values file with an
exclusion list.

## 8. Copy loss and retrieval inside the training length

```
python tools/build_report_pcr.py --out <pcr> --tokenizer t64k=<tokenizer.json> \
    --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --workers 4
python scripts/eval_probes.py --checkpoint <checkpoint> --probes <pcr>/t64k --probe-names pcr_copy2,pcr_copy3,pcr_fresh --resume --out <out>/pcr-copy.json
python scripts/eval_probes.py --checkpoint <checkpoint> --probes <pcr>/t64k --probe-names pcr_niah1,pcr_niahk,pcr_vt --resume --out <out>/pcr-retrieval.json
```

The copy sets repeat a random token sequence two or three times (`pcr_copy2`, `pcr_copy3`) or follow it with a fresh one (`pcr_fresh`). The copy loss by position in the copy is
`ced.eval.report_pcr.copy_table(record["items"])` applied to the scorer's output; `ced/eval/report_pcr.py` defines the position buckets. The retrieval sets are single-key
and multi-key needles and variable tracking at 2,048, 4,096 and 8,192 tokens.
