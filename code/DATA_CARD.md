# Data card

The 60M and 200M runs in `runs/*.yaml` train on the three shard sets named below. This card gives the sources of each set and their mix shares. It is all
that this repository says about the data: no data, no data-building code and no counts are included. The held-out evaluation sets are not released
either, and neither is the tokenizer (see `README.md`).

All three sets are tokenised with one 64k byte-level BPE tokenizer (not included). Shares are shares of tokens. Nominal shares are the targets of the
mix; "realised" shares are measured on the built set.

## Sources

| source (short name) | dataset | licence on the dataset card | link |
|---|---|---|---|
| fineweb-edu | HuggingFaceFW/fineweb-edu (one Common Crawl dump) | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| dclm | mlfoundations/dclm-baseline-1.0-parquet | CC-BY-4.0 | https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet |
| finepdfs-edu | HuggingFaceFW/finepdfs-edu (English) | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu |
| stack-edu | common-pile/stackv2_edu_filtered | per-file permissive licences | https://huggingface.co/datasets/common-pile/stackv2_edu_filtered |
| finemath | HuggingFaceTB/finemath (finemath-4plus) | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceTB/finemath |
| swallow-math-v2 | tokyotech-llm/swallow-math-v2 | Apache-2.0 | https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2 |
| nemotron-specialized | nvidia/Nemotron-Pretraining-Specialized-v1 (NVIDIA): subsets RQA, STEM-SFT (general reasoning), InfiniByte-Reasoning | CC-BY-4.0, CC-BY-SA-4.0, GFDL, other (per its card) | https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1 |
| other source | other source, 2 %, not named | - | - |
| open-swe-traces | nvidia/Open-SWE-Traces (NVIDIA) | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/Open-SWE-Traces |
| toucan | Agent-Ark/Toucan-1.5M (SFT config) | Apache-2.0 | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M |
| nemotron-agentic | nvidia/Nemotron-SFT-Agentic-v2 (NVIDIA) | CC-BY-4.0, Apache-2.0, MIT | https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2 |

CC-BY-4.0 text: https://creativecommons.org/licenses/by/4.0 . The licence column is what the dataset cards print. The card of the Nemotron-Pretraining-Specialized
dataset adds that a model distributed from it may be subject to the Qwen, DeepSeek and Phi-4 terms. Some of the sources are model-generated public
data (SwallowMath-v2, the Nemotron subsets, the agent trajectories).

## Set 1: `phase0-v2-t64k`

Stage A of every 200M run, and the only set of the 60M runs.

| source | category | nominal share (%) | realised share (%) |
|---|---|---:|---:|
| fineweb-edu | web | 19.8 | 19.816 |
| dclm | web | 13.2 | 13.242 |
| finepdfs-edu | PDF | 5.0 | 5.026 |
| stack-edu | code | 30.0 | 30.133 |
| finemath | math | 7.2 | 7.130 |
| swallow-math-v2 | math | 4.8 | 4.720 |
| nemotron-specialized and the other source | STEM and reasoning | 12.0 | 11.871 |
| open-swe-traces | agent | 3.6 | 3.657 |
| toucan | agent | 2.0 | 2.008 |
| nemotron-agentic | agent | 2.4 | 2.399 |

Category shares: web 33, PDF 5, code 30, math 12, STEM and reasoning 12, agent 8 (%). Inside a category: web FineWeb-Edu 60 : DCLM 40; math FineMath 60 : SwallowMath 40; agent Open-SWE-Traces 45 :
Toucan 25 : Nemotron-agentic 30 (nominal, set by the mix).

Splits inside a source (nominal, %):

- stack-edu, by language: Python 50, JavaScript 8, Java 7, C++ 7, TypeScript 6, C 5, C# 3, Go 3, Rust 3, Shell 3, Markdown 2, PHP 2, Ruby 1.
- the STEM and reasoning source (nemotron-specialized and the other source): RQA 41.7, STEM-SFT (general reasoning) 33.3, other source (not named) 16.7,
  InfiniByte-Reasoning 8.3.
- swallow-math-v2: textbook 50, Q&A 50.
- nemotron-agentic: tool calling 55, interactive agent 35, search 10.

## Set 2: `phase-c-200m-t64k`

Stage C of every 200M run (the last 30 % of the run). Its documents are drawn from set 1 at random, source by source, to these nominal quotas:

| source | category | nominal share (%) |
|---|---|---:|
| stack-edu | code | 35.0 |
| finemath | math | 9.0 |
| swallow-math-v2 | math | 6.0 |
| nemotron-specialized and the other source | STEM and reasoning | 15.0 |
| open-swe-traces | agent | 6.75 |
| toucan | agent | 3.75 |
| nemotron-agentic | agent | 4.5 |
| fineweb-edu | web | 10.42 |
| dclm | web | 6.95 |
| finepdfs-edu | PDF | 2.63 |

Category shares: code 35, math 15, STEM and reasoning 15, agent 15, web and PDF together 20 (%). Inside a category: math 60 : 40 as above; agent 45 : 25 : 30 as above; web and PDF in
the ratio of set 1 (33 : 5), web split 60 : 40.

## Set 3: `phase0-v3-t64k`

Read only by the long scale-check runs, after set 1 (`data.union_order: sets_in_order`). Same sources, same tokenizer, and documents that set 1 does not
hold. Nominal shares of the new set: those of set 1's realised column for every source except the agent category, whose shares are open-swe-traces 4.260, toucan 1.008 and nemotron-agentic 2.795 (%), so that
the three agent sources together stay at the share of set 1 (8.064 %). Taken together with set 1, toucan is 1.300 % of the tokens.
