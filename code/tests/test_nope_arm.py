"""The NoPE arm on control (a): the 60M run ss-a-s1 with the rotary fraction at 0 on the
encoder's attention, which in control (a) is the global layers only."""
import copy
import sys
from pathlib import Path

import torch
import yaml

from ced.config import load_yaml, tiny_control_config
from ced.model.attention import GatedAttention
from ced.model.control import build_model

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "nope-60m.yaml"




def test_nope_control_is_causal_and_cache_consistent():
    """Tiny control (a) with Scalable-Softmax and no rotary part: the same parameters as with it, no earlier logit
    moves when later tokens change, and a forward in two chunks through the cache equals the full forward."""
    torch.manual_seed(0)
    base = tiny_control_config()
    base.encoder.attention.temperature = "ssmax"
    nope = copy.deepcopy(base)
    nope.encoder.attention.rope_fraction = 0.0
    mb, m = build_model(base), build_model(nope).eval()
    assert set(mb.state_dict()) == set(m.state_dict())
    assert all(a.cfg.rope_fraction == 0.0 for a in m.modules() if isinstance(a, GatedAttention))
    B, L, cut = 2, 96, 60
    tok = torch.randint(0, base.vocab_size, (B, L))
    doc = torch.zeros(B, L, dtype=torch.long)
    doc[1, 40:] = 1
    with torch.no_grad():
        full = m(tok, doc).logits
        tok2 = tok.clone()
        tok2[:, cut:] = torch.randint(0, base.vocab_size, (B, L - cut))
        assert torch.equal(m(tok2, doc).logits[:, :cut], full[:, :cut])
        o1 = m(tok[:, :50], doc[:, :50], return_cache=True)
        o2 = m(tok[:, 50:], doc[:, 50:], cache=o1.cache, return_cache=True)
        assert torch.allclose(torch.cat([o1.logits, o2.logits], dim=1), full, atol=1e-4)
