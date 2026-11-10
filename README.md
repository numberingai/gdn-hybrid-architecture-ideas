# Four Architecture Ideas for Gated DeltaNet–Attention Hybrids

Jihoo Lee, Seoul National University

Report (PDF): [architecture-report.pdf](architecture-report.pdf)
Blog post: https://numbering.ai/blog/four-architecture-ideas

Four architecture ideas for hybrids of Gated DeltaNet and softmax attention, tested on 60M and 200M models:

1. Block-entry depth mixing: a learned softmax mix over the embedding and earlier blocks' outputs, applied where each block after the first begins and before the output, rather than before every sublayer as in Block Attention Residuals.
2. Long context: beyond the training length, retrieval rose and fell in a ~20k-token cycle, which I traced to one rotary frequency. I trained with Jianlin Su's ReRoPE distance cap at 2,048 and left one attention layer without positions, as in Su's HWFA, Cohere's RNoPE and Llama 4's iRoPE.
3. The inputs a Gated DeltaNet layer writes to its state (k, v, β, α) and its query each get their own depth mix.
4. Grid KV sharing: at the cache and cost of grouped-query attention (GQA), r × c distinct keys from r + c cached parts.

Ideas 1–3 and two published methods (token-superposition training, tied embeddings) make up the full model. At 200M it has 2.1% lower bits per byte than its base (two seeds per model). Grid KV sharing, tested separately, had slightly lower bits per byte than GQA at equal cache at 200M (three seeds, within the noise) but lower multi-key retrieval, and trailed GQA with a quarter less cache.

Status: v2 (November 2026), with the results of every run and the code. Two planned runs were dropped: the second seed of one GDN-mix variant and a quarter-learning-rate run of the full model.

## Code

The code is in [`code/`](code/), under the MIT licence (`code/LICENSE`). Adapted third-party code and its notices are listed in `code/THIRD_PARTY_NOTICES.md`; the parts adapted from RULER stay under Apache-2.0 (`code/LICENSES/Apache-2.0.txt`). The folder holds the reference PyTorch implementation of the model, the training code, the configs and launcher files of the 60M and 200M runs, and the evaluation code.

```
cd code
pip install -e .        # Python 3.11 or newer
python scripts/run_entry.py runs/examples.yaml --name f-60m-s1 --data-root <data> --run-root <runs>  # the full model at 60M, seed 1
```

`code/README.md` lists the data a run needs; `code/EVAL.md` gives the evaluation commands and their input formats. The tokenizer and the held-out evaluation sets are not released, so the report's exact numbers can't be reproduced from this repository alone; training and the Based recall suite need only public data and a tokenizer of your own.
