# Four Architecture Ideas for Gated DeltaNet–Attention Hybrids

Jihoo Lee, Seoul National University

Report (PDF): [architecture-report.pdf](architecture-report.pdf)
Blog post: https://numbering.ai/blog/four-architecture-ideas

Four architecture ideas for hybrids of Gated DeltaNet and softmax attention, tested on 60M and 200M models:

1. Block-entry depth mixing: a learned softmax mix over the embedding and earlier blocks' outputs, applied where each block after the first begins and before the output, rather than before every sublayer as in Block Attention Residuals.
2. Long context: beyond the training length, retrieval rose and fell in a ~20k-token cycle, which I traced to one rotary frequency. I trained with Jianlin Su's ReRoPE distance cap at 2,048 and left one attention layer without positions, as in Su's HWFA, Cohere's RNoPE and Llama 4's iRoPE.
3. The inputs a Gated DeltaNet layer writes to its state (k, v, β, α) and its query each get their own depth mix.
4. Grid KV sharing: at the cache and cost of grouped-query attention (GQA), r × c distinct keys from r + c cached parts.

Ideas 1–3 and two published methods (token-superposition training, tied embeddings) make up the full model. At 200M it has 2.1% lower bits per byte than its base (two seeds per model). Grid KV sharing, tested separately, matched GQA at equal cache at 200M but trailed it with a quarter less cache.

Status: v1. Some runs were still in progress when this was written; their results, the code and an arXiv version will come with v2.
