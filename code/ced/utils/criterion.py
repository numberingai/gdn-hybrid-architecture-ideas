"""One reading of a numeric check under a bound of the form "err <= FACTOR x err_ref + FLOOR" (or its max form).

A check compares a candidate's error `err` (against a ground truth) with the error `ref` of a reference path against the same ground
truth. Its bound is one of

    "sum"   FACTOR x ref + FLOOR                      (the trunk copy's and qmix's gradient checks, FA4's rule)
    "max"   max(FACTOR x ref, FLOOR)                  (the mix kernels' "2 x today's, floor 1e-6", the band checks' tolerances)

optionally capped (`ceil`: min(bound, ceil), the head checks) and optionally scaled (`scale`: the bound times a value, the data-parallel
check's "(10 x rel + 1e-6) x |value|"). A ratio rule without a floor ("sum" with floor 0, `no_floor`: the GDN kernels' "1.10 x today's", a
seed band, the "1.1 x flex's") is the band form: its lines print "(no floor)" where the others print their floor. A line PASSES BY THE RATIO
ALONE when err <= FACTOR x ref (the floor was not needed) and PASSES ONLY THROUGH THE FLOOR when err <= bound but err > FACTOR x ref (the
floor carried it). This module only reads and formats: the PASS / FAIL rule of every check is `Rule.passes`, the comparison that check always
made (`ratio_form` keeps the tune checks' spelling, err / bound <= 1.0, which gives the same verdict for finite numbers). Pure Python (math
only): the checks, the tools and the tests import it without torch. Python 3.11 compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ONLY_FLOOR = "PASS ONLY THROUGH THE FLOOR"  # the words a line that the floor carried ends with (grep for them)
BY_RATIO = "ok (ratio alone)"  # the tail of a line that passed without the floor


def num(x: float) -> str:
    """A rule number as it is written in the rules: 4, 0.02, 1e-6 (not 1e-06), 1e-4 (not 0.0001)."""
    s = f"{x:g}" if not math.isfinite(x) or x == 0 or abs(x) >= 1e-3 else f"{x:.6e}"
    if "e" not in s:
        return s
    mantissa, exp = s.split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def _ratio(err: float, ref: float) -> float:
    """err / ref; 0 / 0 is 0 (nothing to explain), x / 0 is inf, NaN stays NaN."""
    if math.isnan(err) or math.isnan(ref):
        return math.nan
    if ref == 0.0:
        return 0.0 if err == 0.0 else math.inf
    return err / ref


def over(err: float, bound: float) -> bool:
    """err is over its bound: True when err > bound AND when err or the bound is NaN. "A NaN difference fails": `err > bound` is False for a
    NaN, so a check that FAILS on `err > bound` passes a NaN; this is `not (err <= bound)`, the form `Rule.passes` has always had (its
    negation)."""
    return not (err <= bound)


def nan_max(values, default: float = math.nan) -> float:
    """max() over errors that keeps a NaN: any NaN in `values` makes the result NaN. Python's max() drops a NaN that is not first (every
    `x > nan` is False), so `max(errors) <= bound` and `worst = max(worst, e)` pass a NaN difference. `default` for no values."""
    xs = [float(v) for v in values]
    if not xs:
        return default
    return math.nan if any(math.isnan(x) for x in xs) else max(xs)


@dataclass(frozen=True)
class Rule:
    """The bound of a check: factor x ref + floor (form "sum") or max(factor x ref, floor) (form "max"), capped by `ceil` when given.

    source  where the floor was fixed, quoted in the verdict ("fixed 2026-10-22", "the plan's bound, ...").
    err_name / ref_name  the words of the rule text ("err", "err_ref"; "diff", "band").
    ratio_form  the verdict is read as err / bound <= 1.0 (write_mix_check, entry_tune_check, qmix_fused_check spell it so)."""

    factor: float
    floor: float
    form: str = "sum"
    ceil: float | None = None
    ratio_form: bool = False
    source: str = ""
    err_name: str = "err"
    ref_name: str = "err_ref"

    def __post_init__(self):
        if self.form not in ("sum", "max"):
            raise ValueError(f"Rule.form {self.form!r}: 'sum' or 'max'")

    def bound(self, ref: float, scale: float | None = None) -> float:
        """The largest err that passes. NaN when ref is NaN (nothing passes a NaN bound)."""
        if math.isnan(ref):
            return math.nan
        b = self.factor * ref + self.floor if self.form == "sum" else max(self.factor * ref, self.floor)
        if self.ceil is not None:
            b = min(b, self.ceil)
        return b if scale is None else b * scale

    def ratio_bound(self, ref: float, scale: float | None = None) -> float:
        """What the ratio alone allows: factor x ref (times the scale)."""
        b = self.factor * ref
        return b if scale is None else b * scale

    def passes(self, err: float, ref: float, scale: float | None = None) -> bool:
        """The check's verdict on one tensor: the comparison the check always made (NaN never passes)."""
        b = self.bound(ref, scale)
        return bool(err / b <= 1.0) if self.ratio_form else bool(err <= b)

    def read(self, label: str, err: float, ref: float, scale: float | None = None) -> "Reading":
        return Reading(label=label, err=err, ref=ref, rule=self, scale=scale)

    @property
    def no_floor(self) -> bool:
        """A ratio rule without a floor (form "sum", floor 0: err <= factor x ref), the band form of the checks whose bound is
        a factor of a measured reference alone (the GDN kernels' "1.10 x today's", a seed band). Its bound is factor x ref
        bit for bit (x + 0.0 is x), and no line can pass only through a floor."""
        return self.form == "sum" and self.floor == 0.0

    def floor_words(self) -> str:
        """The floor as the compact forms name it: 'floor 1e-6', or 'no floor' for the band form (never 'floor 0')."""
        return "no floor" if self.no_floor else f"floor {num(self.floor)}"

    def formula(self, scale: bool = False) -> str:
        """The bound's formula as a line prints it: '4 x ref + floor 1e-6' ('1.1 x ref (no floor)' for a rule without one)."""
        f = (f"{num(self.factor)} x ref (no floor)" if self.no_floor
             else f"{num(self.factor)} x ref + floor {num(self.floor)}" if self.form == "sum"
             else f"max({num(self.factor)} x ref, floor {num(self.floor)})")
        if self.ceil is not None:
            f = f"min({f}, ceiling {num(self.ceil)})"
        return f"({f}) x |value|" if scale else f

    def describe(self) -> str:
        """The rule as a verdict names it: 'err <= 4 x err_ref + 1e-6' ('err <= 1.1 x err_ref' without a floor)."""
        f = (f"{num(self.factor)} x {self.ref_name}" if self.no_floor
             else f"{num(self.factor)} x {self.ref_name} + {num(self.floor)}" if self.form == "sum"
             else f"max({num(self.factor)} x {self.ref_name}, {num(self.floor)})")
        if self.ceil is not None:
            f = f"min({f}, {num(self.ceil)})"
        return f"{self.err_name} <= {f}"

    def named(self) -> str:
        """The rule with its floor and where the floor was fixed: 'err <= 4 x err_ref + 1e-6 (the floor 1e-6, fixed 2026-10-22)'
        ('err <= 1.1 x err_ref (no floor, ...)' without one)."""
        src = f", {self.source}" if self.source else ""
        return f"{self.describe()} (no floor{src})" if self.no_floor else f"{self.describe()} (the floor {num(self.floor)}{src})"


@dataclass(frozen=True)
class Reading:
    """One compared tensor: err, the reference's err, and what the rule makes of them."""

    label: str
    err: float
    ref: float
    rule: Rule
    scale: float | None = None

    @property
    def ratio(self) -> float:
        """err / ref; under a scale (a rule relative to a value) ref is relative, so the ratio is err / (ref x scale)."""
        return _ratio(self.err, self.ref if self.scale is None else self.ref * self.scale)

    @property
    def bound(self) -> float:
        return self.rule.bound(self.ref, self.scale)

    @property
    def passes(self) -> bool:
        return self.rule.passes(self.err, self.ref, self.scale)

    @property
    def by_ratio_alone(self) -> bool:
        """Passes, and err <= factor x ref: the floor was not needed."""
        return self.passes and bool(self.err <= self.rule.ratio_bound(self.ref, self.scale))

    @property
    def only_through_floor(self) -> bool:
        """Passes, but err > factor x ref: the floor carried the line."""
        return self.passes and not self.by_ratio_alone

    @property
    def of_bound(self) -> float:
        """err / bound (a value over 1 fails)."""
        b = self.bound
        return math.nan if math.isnan(self.err) or math.isnan(b) else (self.err / b if b else (0.0 if self.err == 0.0 else math.inf))

    def status(self) -> str:
        if not self.passes:
            return "OVER"
        return ONLY_FLOOR if self.only_through_floor else BY_RATIO

    def text(self) -> str:
        """'err 1.595e-06  ref 3.786e-07  ratio 4.21 (> 4)  bound 2.514e-06 = 4 x ref + floor 1e-6  PASS ONLY THROUGH THE FLOOR'
        (the label goes before it: f"{label}: " + text())."""
        r = self.ratio
        if math.isnan(r):
            cmp_ = "(not comparable)"
        else:
            cmp_ = f"({'<=' if self.err <= self.rule.ratio_bound(self.ref, self.scale) else '>'} {num(self.rule.factor)})"
        shown = "nan" if math.isnan(r) else ("inf" if math.isinf(r) else f"{r:.2f}")
        ref = f"{self.ref:.3e}" if self.scale is None else f"{self.ref:.3e} x |value| {self.scale:.3e}"
        return (f"err {self.err:.3e}  ref {ref}  ratio {shown} {cmp_}  bound {self.bound:.3e} = "
                f"{self.rule.formula(self.scale is not None)}  {self.status()}")

    def line(self) -> str:
        return f"{self.label}: {self.text()}"

    def short(self, with_err: bool = False, formula: bool = True) -> str:
        """The compact form, for the lines a delivered log cuts at 300 to 330 characters: 'ratio 1.62 of 4, bound 2.27e-06 = 4 x ref + floor 1e-6'
        (formula=False: 'bound 2.27e-06 (floor 1e-6)', 'bound 2.20e-03 (no floor)' for the band form; with_err: 'err 5.14e-07 ref 3.17e-07 '
        before)."""
        r = self.ratio
        shown = "nan" if math.isnan(r) else ("inf" if math.isinf(r) else f"{r:.2f}")
        head = f"err {self.err:.2e} ref {self.ref:.2e} " if with_err else ""
        form = f" = {self.rule.formula(self.scale is not None)}" if formula else f" ({self.rule.floor_words()})"
        return f"{head}ratio {shown} of {num(self.rule.factor)}, bound {self.bound:.2e}{form}"

    def as_dict(self) -> dict:
        """For a RECORD / JSON: nothing is rounded."""
        return {"label": self.label, "err": self.err, "ref": self.ref, "ratio": self.ratio, "bound": self.bound,
                "passes": self.passes, "by_ratio_alone": self.by_ratio_alone, "only_through_floor": self.only_through_floor}


def worst(readings) -> "Reading | None":
    """The reading closest to (or furthest over) its bound: the largest err / bound; a NaN reading is the worst (it cannot pass)."""
    best, key = None, None
    for rd in readings:
        k = math.inf if math.isnan(rd.of_bound) else rd.of_bound
        if key is None or k > key:
            best, key = rd, k
    return best


@dataclass
class Tally:
    """The readings of a check, and what the check's verdict says about them."""

    rule: Rule
    readings: list = field(default_factory=list)

    def read(self, label: str, err: float, ref: float, scale: float | None = None) -> Reading:
        rd = self.rule.read(label, err, ref, scale)
        self.readings.append(rd)
        return rd

    def add(self, rd: Reading) -> Reading:
        self.readings.append(rd)
        return rd

    @property
    def n(self) -> int:
        return len(self.readings)

    @property
    def n_ratio(self) -> int:
        return sum(1 for r in self.readings if r.by_ratio_alone)

    @property
    def floor_only(self) -> list:
        return [r.label for r in self.readings if r.only_through_floor]

    @property
    def over(self) -> list:
        return [r.label for r in self.readings if not r.passes]

    def tail(self, limit: int = 8) -> str:
        """'59 of 60 lines by the ratio alone, 1 only through the floor: C1 ... k_far' (and 'k over the bound: ...' when some fail)."""
        return counts_text(self.n, self.n_ratio, self.floor_only, self.over, limit)

    def verdict_text(self, limit: int = 8) -> str:
        """The rule with its floor, then the counts: 'under err <= 4 x err_ref + 1e-6 (the floor 1e-6, fixed 2026-10-22): 59 of 60 ...'."""
        return f"under {self.rule.named()}: {self.tail(limit)}"

    def records(self) -> list:
        return [r.as_dict() for r in self.readings]

    def extend(self, readings) -> None:
        self.readings.extend(readings)

    def summary(self) -> dict:
        """The counts as JSON (a record that outlives the readings): {'n', 'n_ratio', 'floor_only': [...], 'over': [...]}."""
        return {"n": self.n, "n_ratio": self.n_ratio, "floor_only": self.floor_only, "over": self.over}


def merge_summaries(summaries) -> dict:
    """The sum of several Tally.summary() dicts (a verdict over pieces that each tallied their own cells)."""
    out = {"n": 0, "n_ratio": 0, "floor_only": [], "over": []}
    for s in summaries:
        out["n"] += s["n"]
        out["n_ratio"] += s["n_ratio"]
        out["floor_only"] += list(s["floor_only"])
        out["over"] += list(s["over"])
    return out


def verdict_of(rule: Rule, summary: dict, limit: int = 8) -> str:
    """Tally.verdict_text() from a summary dict: 'under <rule> (the floor ..): n of m lines by the ratio alone, k only through the floor: ...'."""
    return f"under {rule.named()}: {counts_text(summary['n'], summary['n_ratio'], summary['floor_only'], summary['over'], limit)}"


def _names(names, limit: int) -> str:
    shown = list(names)[:limit]
    more = len(names) - len(shown)
    return "; ".join(shown) + (f"; and {more} more" if more > 0 else "")


def counts_text(n: int, n_ratio: int, floor_only, over=(), limit: int = 8) -> str:
    """The counts a verdict names: lines by the ratio alone, lines only through the floor (named), lines over the bound (named)."""
    parts = [f"{n_ratio} of {n} lines by the ratio alone"]
    floor_only = list(floor_only)
    parts.append(f"{len(floor_only)} only through the floor" + (f": {_names(floor_only, limit)}" if floor_only else ""))
    over = list(over)
    if over:
        parts.append(f"{len(over)} over the bound: {_names(over, limit)}")
    return ", ".join(parts)


def cell_short(rule: Rule, readings, with_err: bool = False, formula: bool = True) -> str:
    """cell_text in the compact form (Reading.short) for the lines a delivered log cuts: the worst tensor with its ratio and its floor bound, and
    'floor-only k/n' when some tensors passed only through the floor (nothing is added when none did; the formula names the floor)."""
    readings = list(readings)
    w = worst(readings)
    if w is None:
        return f"no tensor read ({rule.floor_words()})"
    fo = sum(1 for r in readings if r.only_through_floor)
    return f"worst {w.label} at {w.of_bound:.2f}: {w.short(with_err, formula)}" + (f", floor-only {fo}/{len(readings)}" if fo else "")


def cell_text(rule: Rule, readings, limit: int = 3) -> str:
    """One line for a cell of many tensors: the worst tensor with its ratio and bound, then the counts. The ratio and the floor bound are
    on the line even when the cell has fifty tensors."""
    readings = list(readings)
    w = worst(readings)
    if w is None:
        return f"no tensor read (rule {rule.describe()}, {rule.floor_words()})"
    ratio_alone = sum(1 for r in readings if r.by_ratio_alone)
    floor_only = [r.label for r in readings if r.only_through_floor]
    over = [r.label for r in readings if not r.passes]
    return (f"worst {w.label} at {w.of_bound:.2f} of its bound: {w.text()}; "
            f"{counts_text(len(readings), ratio_alone, floor_only, over, limit)} (rule {rule.describe()}, {rule.floor_words()})")
