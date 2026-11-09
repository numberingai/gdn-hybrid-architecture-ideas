"""The GDN memory-rule options a layer carries (GDNConfig.negeig.kda.erase.dprod: the GDN memory-rule screen's arms), and the refusal
of a tool that reads a GDN layer in its own way, outside the layer's forward, and is not built for an option: it refuses by name
instead of reading another layer under this one's name.

The tools (each calls `refuse` with its name and the options it reads correctly):
- ced/eval/gdn_diag.py gdn_inputs (the StateWalk of the GDN state diagnostics, the lce- suite's "gdn" part; also ced/eval/gdn_wipe.py)
  and ced/infer/engine.py (the decode engine) walk the state themselves: one decay a value head and one write a token, from
  GatedDeltaNet.gates and in_proj_qkv. They read negative eigenvalues (gates() hands them beta = 2 sigmoid(b), and the walk is the
  same rule at any beta) and nothing else.
- ced/eval/gdn_memory.py (the per-head memory length: the decay and the median write strength from gates()), ced/eval/gate_floor.py
  and ced/eval/gdn_force.py (the evaluation-only floor and cap, which act per value head in gates()) read the gates of one write a
  token. They read negative eigenvalues and nothing else.

Why each option is refused (REASONS): KDA's decay is per key channel (gates() refuses it already, by the same words); the erase adds
a step a token that gates() does not give (gates() refuses it already, by the same words); Gated DeltaProduct writes n_h steps a token
with beta = 2 sigmoid(b) on every step, so a tool that reads gates()' one beta = sigmoid(b) and in_proj_qkv's one key and value
reads Gated DeltaNet in its place: nothing refused it before this module (found at the merge of the arm branches, gdnrule-integrate's
review). A layer with no option on is never refused (every model before these keys). Python 3.11 compatible; no torch.
"""

from __future__ import annotations

OPTIONS = ("negeig", "kda", "erase", "dprod")  # GDNConfig's memory-rule sub-sections, the arms' order of building
REASONS = {
    "negeig": "its write strength is beta = 2 sigmoid(b) in (0, 2) (GDNConfig.negeig.beta_max 2; ced/model/gdn_negeig.py)",
    "kda": ("its decay is per key channel (GDNConfig.kda.decay 'channel'; ced/model/gdn_kda.py): one decay a value head, key channel "
            "and token, where the tool reads one decay a value head (GatedDeltaNet.gates)"),
    "erase": ("it erases its state along a direction of its own (GDNConfig.erase.mode 'eda'; ced/model/gdn_erase.py): a step a token "
              "that the tool, reading GatedDeltaNet.gates, leaves out"),
    "dprod": ("it is Gated DeltaProduct (GDNConfig.dprod.n_h {n}; ced/model/gdn_dprod.py): n_h delta-rule steps a token, each with its "
              "own key, value and beta = 2 sigmoid(b) in (0, 2), where the tool reads one write a token (GatedDeltaNet.gates' beta = "
              "sigmoid(b), in_proj_qkv's one key and value) and would read Gated DeltaNet in its place"),
}


def options_on(gdn_cfg) -> dict:
    """{option: its setting in words} for every memory-rule option on in this GDNConfig (empty: none, the layer as it was). Read from
    the config with getattr defaults, so a config object without a sub-section (a duck-typed one in a test) has it off."""
    on = {}
    m = getattr(getattr(gdn_cfg, "negeig", None), "beta_max", 1)
    if m != 1:
        on["negeig"] = f"negeig.beta_max {m}"
    d = getattr(getattr(gdn_cfg, "kda", None), "decay", "scalar")
    if d != "scalar":
        on["kda"] = f"kda.decay {d!r}"
    e = getattr(getattr(gdn_cfg, "erase", None), "mode", "none")
    if e != "none":
        on["erase"] = f"erase.mode {e!r}"
    n = getattr(getattr(gdn_cfg, "dprod", None), "n_h", 1)
    if n != 1:
        on["dprod"] = f"dprod.n_h {n}"
    return on


def refuse(gdn_cfg, what: str, reads=("negeig",), where: str = "this GDN layer") -> None:
    """ValueError when the GDNConfig has a memory-rule option on that `what` (the tool, in words) does not read: the tool, the
    option(s) by key and why. `reads`: the options the tool reads correctly (each tool names its own; module docstring)."""
    bad = [k for k in options_on(gdn_cfg) if k not in reads]
    if not bad:
        return
    n = getattr(getattr(gdn_cfg, "dprod", None), "n_h", 1)
    why = "; ".join(REASONS[k].format(n=n) for k in bad)
    raise ValueError(f"{what} is not built for {where} with {', '.join(options_on(gdn_cfg)[k] for k in bad)}: {why}. It refuses by name "
                     "rather than read another layer (ced/model/gdn_rules.py)")


def refuse_model(cfg, what: str, reads=("negeig",)) -> None:
    """refuse() on a ModelConfig's GDN sections: the encoder's, and the decoder's when it has one. Also refused, by name: a model
    with a GDN shape of each layer's own (model.encoder.gdn_shape_by_layer): the tools that call this read one GDN shape
    (model.encoder.gdn's) for the whole stack, and would size every layer's state and projections from it."""
    for side in ("encoder", "decoder"):
        sec = getattr(cfg, side, None)
        gdn = getattr(sec, "gdn", None)
        if gdn is not None:
            refuse(gdn, what, reads, where=f"the {side}'s GDN layers (model.{side}.gdn)")
    refuse_shape_table(cfg, what)


def refuse_shape_table(cfg, what: str) -> None:
    """ValueError when the ModelConfig has a GDN shape of each layer's own (model.encoder.gdn_shape_by_layer): for a tool that reads ONE GDN
    shape (model.encoder.gdn's) for the whole stack. It sizes nothing from a layer: it refuses by name rather than read another layer."""
    table = getattr(getattr(cfg, "encoder", None), "gdn_shape_by_layer", None)
    if table:
        raise ValueError(f"{what} is not built for a model with model.encoder.gdn_shape_by_layer {dict(table)!r}: it reads one GDN shape "
                         "(model.encoder.gdn's) for every layer and would size each layer from it. It refuses by name rather than read "
                         "another layer (ced/model/gdn_rules.py)")
