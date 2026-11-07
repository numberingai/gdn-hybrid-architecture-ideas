# Four Architecture Ideas for Gated DeltaNet–Attention Hybrids

Jihoo Lee, Seoul National University

Report (PDF): [architecture-report.pdf](architecture-report.pdf)
Blog post: https://numbering.ai/blog/four-architecture-ideas

Four architecture ideas for hybrids of Gated DeltaNet and softmax attention, tested on 60M and 200M models:

1. Block-entry depth mixing: a depth mix read only where a block of layers begins.
2. Capped RoPE with a NoPE layer: rotary distance capped at 2,048 in training and at inference, with the deepest attention layer left without positions.
3. Separate write and read mixes for Gated DeltaNet: what a layer writes and what it queries each read their own depth mix.
4. Grid KV sharing: at the cache and cost of grouped-query attention, r × c distinct keys from r + c cached parts.

Ideas 1–3 and two published methods (token-superposition training, tied embeddings) make up the full model. At 200M, two seeds each, it has 2.1% lower bits per byte than its base.

Status: some runs were still going when this version was written; their results, the code, and an arXiv version will follow with the next version.
