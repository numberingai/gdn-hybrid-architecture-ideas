"""The runs files of this export and the launcher scripts/run_entry.py (written for the export).

  - every shipped runs file parses; every entry's config file exists and every `set` item applies to it (scripts/train.py's rule: an
    unknown key is an error); every `after` names an entry of some runs file; no entry holds a cluster path or a placeholder other
    than ${DATA_ROOT} / ${RUN_ROOT};
  - the model of a sample of entries builds on the meta device with exactly the parameter counts the project's report prints for it;
  - the launcher builds the command the project's own launcher ran, minus the copying.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_entry import build_command, expand, load_runs, main  # noqa: E402
from train import apply_override  # noqa: E402

from ced.config import load_yaml  # noqa: E402
from ced.utils.params import build_on_meta, parameter_breakdown  # noqa: E402

RUNS = sorted((REPO / "runs").glob("*.yaml"))


def entries():
    for f in RUNS:
        spec = load_runs(f)
        for run in spec["runs"]:
            yield f.stem, spec, run


def test_runs_files_exist_and_parse():
    assert len(RUNS) >= 20
    for f in RUNS:
        spec = load_runs(f)
        names = [r["name"] for r in spec["runs"]]
        assert len(names) == len(set(names)), f"{f.name}: a name twice"
        assert set(spec) <= {"shards", "eval_dir", "runs"}, f"{f.name}: unexpected file-level keys {set(spec)}"


def test_no_cluster_paths_or_unknown_placeholders():
    for f in RUNS:
        text = f.read_text()
        assert not re.search(r"(^|[\s\"'=\[,])/(data|home|mnt|export)\b", text), f.name
        assert set(re.findall(r"\$\{(\w+)\}", text)) <= {"DATA_ROOT", "RUN_ROOT"}, f.name


def test_after_names_an_entry():
    every = {r["name"] for _, _, r in entries()}
    for stem, _, run in entries():
        if run.get("after"):
            assert run["after"] in every, (stem, run["name"], run["after"])
            assert "after_step" in run or "resume_from" in " ".join(map(str, run.get("set", []))), (stem, run["name"])


def test_every_entry_applies_to_its_config():
    cache: dict[str, object] = {}
    n = 0
    for stem, spec, run in entries():
        cfg_path = REPO / run["config"]
        assert cfg_path.is_file(), (stem, run["name"])
        cfg = load_yaml(cfg_path)
        for item in run.get("set", []):
            assert "=" in str(item), (stem, run["name"], item)
            apply_override(cfg, str(item))
        n += 1
    assert n >= 130


# the counts the project's report prints (total, non-embedding), counted on the meta device
COUNTS = {
    ("ablation-60m", "abl-f-mxl-a-s1"): (99_730_728, 66_962_728),
    ("ablation-60m", "abl-base-mxl-a-s1"): (129_274_124, 63_738_124),
    ("examples", "f-60m-s1"): (99_730_728, 66_962_728),  # the README's example: the same settings as abl-f-mxl-a-s1
    ("ladder-200m-base", "base200-s1"): (306_533_556, 208_229_556),
    ("ladder-200m-r1e", "r1e200-s1"): (306_544_308, 208_240_308),
    ("arch-s5qmix-200m", "s5qmix200-s1"): (267_769_564, 218_617_564),
    ("fminus200-nomix", "fminus200-nomix-s1"): (267_714_228, 218_562_228),
}


@pytest.mark.parametrize("key", sorted(COUNTS))
def test_parameter_counts_of_the_runs(key):
    runs, name = key
    run = next(r for r in load_runs(REPO / "runs" / f"{runs}.yaml")["runs"] if r["name"] == name)
    cfg = load_yaml(REPO / run["config"])
    for item in run.get("set", []):
        apply_override(cfg, str(item))
    b = parameter_breakdown(build_on_meta(cfg.model))
    assert (b["total"], b["non-embedding"]) == COUNTS[key]


def test_every_entry_builds_on_the_meta_device():
    """One entry of each distinct (config, model-key set): the model builds (the key combination is accepted by the code)."""
    seen = set()
    for stem, spec, run in entries():
        keys = tuple(sorted(str(s).split("=", 1)[0] for s in run.get("set", []) if str(s).split("=", 1)[0].startswith(("model.", "tst."))))
        sig = (run["config"], keys, tuple(str(s) for s in run.get("set", []) if str(s).startswith("model.")))
        if sig in seen:
            continue
        seen.add(sig)
        cfg = load_yaml(REPO / run["config"])
        for item in run.get("set", []):
            apply_override(cfg, str(item))
        assert parameter_breakdown(build_on_meta(cfg.model))["total"] > 0, (stem, run["name"])


ROOTS = {"DATA_ROOT": "/d", "RUN_ROOT": "/r"}


def test_launcher_command_for_a_stage_a_entry():
    spec = load_runs(REPO / "runs" / "ablation-60m.yaml")
    run = next(r for r in spec["runs"] if r["name"] == "abl-f-mxl-a-s1")
    cmd = build_command(spec, run, ROOTS, python="python", check_eval=False)
    assert cmd[:6] == ["python", "scripts/train.py", "--config", "configs/v2/control-a-60m.yaml", "--run-dir", "/r/abl-f-mxl-a-s1"]
    joined = " ".join(cmd)
    assert "--set seed=1 --set data.seed=1001 --set data.shard_dir=/d/phase0-v2-t64k/train" in joined
    assert "eval.sets=" in joined and "/d/eval/phase0-v2-t64k/web.pt" in joined
    assert all("${" not in c for c in cmd)


def test_launcher_resume_from_and_shard_list():
    spec = load_runs(REPO / "runs" / "scale-checks-60m.yaml")
    stage_c = [r for r in spec["runs"] if any(str(s).startswith("resume_from=") for s in r.get("set", []))]
    assert stage_c
    cmd = build_command(spec, stage_c[0], ROOTS, python="python", check_eval=False)
    assert any(c.startswith("resume_from=/r/") and c.endswith(".pt") for c in cmd)
    listed = [r for r in spec["runs"] if isinstance(r.get("shards"), list)]
    assert listed
    cmd = build_command(spec, listed[0], ROOTS, python="python", check_eval=False)
    extra = [c for c in cmd if c.startswith("data.extra_shard_dirs=")]
    assert len(extra) == 1 and extra[0].startswith('data.extra_shard_dirs=["/d/') and "phase0-v3-t64k" in extra[0]


def test_launcher_refuses_a_missing_root_and_lists_entries(capsys):
    spec = load_runs(REPO / "runs" / "ablation-60m.yaml")
    run = spec["runs"][0]
    with pytest.raises(SystemExit):
        build_command(spec, run, {"DATA_ROOT": None, "RUN_ROOT": "/r"}, python="python")
    with pytest.raises(SystemExit):
        expand("${DATA_ROOT}/x", {"DATA_ROOT": None})
    assert main([str(REPO / "runs" / "ablation-60m.yaml"), "--list"]) == 0
    assert "abl-f-mxl-a-s1" in capsys.readouterr().out
    assert main([str(REPO / "runs" / "ablation-60m.yaml"), "--name", "abl-f-mxl-a-s1", "--data-root", "/d", "--run-root", "/r", "--dry-run"]) == 0
    assert "scripts/train.py" in capsys.readouterr().out
