# Third-party notices

The project's own code is under the MIT licence (`LICENSE`, Copyright (c) 2026 Jihoo Lee). The RULER-derived parts of `ced/eval/recall_ruler.py` are under the
Apache License 2.0 (`LICENSES/Apache-2.0.txt`, see "NVIDIA RULER" below).

Some files were written with code or numerical recipes of other open-source projects in front of them, or contain parts adapted from them. Their copyright
notices and licence texts are given here, once per upstream project, as their licences require. The notices already written in the headers of the files are
kept there too.

| upstream | licence | files |
|---|---|---|
| flash-linear-attention | MIT | `ced/kernels/gdn_triton.py`, `ced/kernels/gdn_c2.py`, `ced/kernels/gdn_edges.py`, `ced/kernels/gdn.py` |
| depth-memory | MIT | `ced/model/ldm.py` |
| Polar Express | MIT | `ced/train/muon.py` |
| modded-nanogpt (and a fork of it) | MIT | `ced/model/cp_head.py` |
| lm-evaluation-harness, Based evaluation harness | MIT | `ced/eval/multiple_choice.py`, `ced/eval/recall_based.py` |
| NVIDIA RULER | Apache-2.0 | `ced/eval/recall_ruler.py` |

## flash-linear-attention (fla-core 0.5.2)

Source: https://github.com/fla-org/flash-linear-attention

Derived files:

- `ced/kernels/gdn_triton.py` (Triton chunk kernels of the gated delta rule: the per-value-head arithmetic of each kernel is fla's, the
  organisation differs; the file's header says so),
- `ced/kernels/gdn_c2.py` and `ced/kernels/gdn_edges.py` (autograd wrappers that launch kernels made of the statements of `gdn_triton.py`),
- `ced/kernels/gdn.py` (the orchestration of the same function: fla's `chunk_gated_delta_rule` for this code's call).

Copyright notice, as printed in the `LICENSE` file of the upstream repository (default branch, read on 2026-11-07):

```
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
```

The headers of the derived files print the same line followed by "and contributors"; they are kept as they are.

MIT licence text (the upstream licence, with the notice above):

```
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## depth-memory (Linear Depth Memory)

Source: https://github.com/davidgonmar/depth-memory, commit `9a6865feeb43`, MIT licence (as named in the header of the derived file).

Derived file: `ced/model/ldm.py` (the delta-rule write, the read and the address networks of the paper's reference implementation, re-written for
this code base's stack; the file's header lists the source files it follows).

Copyright notice, copied verbatim from the `LICENSE` file of that repository:

```
Copyright (c) 2026 David Gonzalez Martinez
```

MIT licence text (the upstream licence, with the notice above):

```
MIT License

Copyright (c) 2026 David Gonzalez Martinez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Polar Express

Source: the paper "The Polar Express" (arXiv 2505.16932, v5; the listing "Code for Polar Express" in its appendix) and its reference repository,
https://github.com/NoahAmsel/PolarExpress.

Derived file: `ced/train/muon.py` (the coefficient tuples and the iteration of the listing, `POLAR_EXPRESS` and `polar_express`, are taken from the
listing, with its safety factor of 1.01; the file's header says so).

Copyright notice, as printed in the `LICENSE` file of the upstream repository:

```
Copyright (c) 2025 Noah Amsel
```

MIT licence text (the upstream licence, with the notice above):

```
MIT License

Copyright (c) 2025 Noah Amsel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## modded-nanogpt

Source: a fork of https://github.com/KellerJordan/modded-nanogpt (https://github.com/NathanGodey/modded-nanogpt, pull request 379), which carries the upstream
MIT licence unchanged.

Derived file: `ced/model/cp_head.py` (the initialisation ranges, the sink key and the query and key normalisation with its gain follow the fork's code, as the
file's header lists). It is experimental code of a closed line of work (see `README.md`, "Unused experimental code").

Copyright notice, as printed in the `LICENSE` file of the upstream repository:

```
Copyright (c) 2024 Keller Jordan
```

MIT licence text (the upstream licence, with the notice above):

```
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## lm-evaluation-harness and the Based evaluation harness

Sources: https://github.com/EleutherAI/lm-evaluation-harness and its fork https://github.com/HazyResearch/based-evaluation-harness (which keeps
EleutherAI's licence text).

Derived files:

- `ced/eval/recall_based.py` (the protocol of the SQuAD-completion, SWDE and FDA tasks of the Based evaluation harness: prompt formats, the stop rule, the
  truncation rule and the contains-the-answer metric),
- `ced/eval/multiple_choice.py` (the multiple-choice scoring conventions of lm-evaluation-harness).

Copyright notice, as printed in the `LICENSE` file of the upstream repositories:

```
Copyright (c) 2020 EleutherAI
```

MIT licence text (the upstream licence, with the notice above):

```
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA RULER (Apache License 2.0)

Source: https://github.com/NVIDIA/RULER (commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`), "RULER: What's the Real Context Size of Your Long-Context
Language Models?" (Hsieh et al., 2024).

Adapted file: `ced/eval/recall_ruler.py`. It adapts RULER's task templates, its needle and noise haystack generators, its variable-tracking chains and its
string-match scoring. It is a modified adaptation: a re-implementation with numpy and this project's helpers (no file of RULER is copied), and the changes
from RULER are listed under DEVIATIONS in the file's header. The parts of that file that are derived from RULER are licensed under the Apache License,
Version 2.0, not under the MIT licence of the rest of this repository; the full text is in `LICENSES/Apache-2.0.txt` and applies to those parts. The rest of
the file, and every other file of this repository, is under the MIT licence of `LICENSE`.

Copyright notice, as printed in the file headers of the upstream repository:

```
Copyright (c) 2024, NVIDIA CORPORATION
```

The upstream repository has no NOTICE file.

## Algorithms implemented from papers

Other modules implement algorithms from papers and cite them by their arXiv numbers in their headers; for example the optimizer (Moonlight, arXiv 2502.16982;
Polar Express, arXiv 2505.16932), the token-superposition schedule (arXiv 2605.06546), Scalable-Softmax (arXiv 2501.19399), the gated delta rule (arXiv 2412.06464),
the depth mixing of `ced/model/attn_res.py` (arXiv 2603.15031), the NVFP4 emulation of `ced/model/nvfp4.py` (arXiv 2509.25149, arXiv 2601.22813) and the loop injection
of `ced/model/loop.py` (arXiv 2604.12946). No source code of other projects beyond what is listed above is known to be copied into this repository.
