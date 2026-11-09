"""Turn one entry of a runs file (runs/*.yaml) into a local training command, and run it or print it.

    python scripts/run_entry.py runs/ablation-60m.yaml --list
    python scripts/run_entry.py runs/ablation-60m.yaml --name abl-f-mxl-a-s1 \
        --data-root /path/to/data --run-root /path/to/runs --dry-run
    python scripts/run_entry.py runs/ablation-60m.yaml --name abl-f-mxl-a-s1 --data-root ... --run-root ... --nproc 8

A runs file holds a few file-level keys (shards, eval_dir) and a list of entries (name, config, seed, set, and for a stage that
continues another run: after, after_step, and a resume_from override inside `set`). The command it builds is the one the
project's own launcher ran, minus the copying to node-local disk:

    python scripts/train.py --config <config> --run-dir <run-root>/<name> --set seed=<seed> --set data.seed=<1000 + seed>
        --set data.shard_dir=<first shard set> [--set data.extra_shard_dirs=[<further sets>]] --set eval.sets={...} --set <override> ...

Placeholders in a runs file: ${DATA_ROOT} (the directory that holds the shard sets and the eval sets) and ${RUN_ROOT} (where run
directories are made); they come from --data-root / --run-root or from the environment variables of the same names. A stage-C
entry resumes from the checkpoint of its stage-A entry (`resume_from`): run the stage-A entry first, with the same --run-root.

The data (shard sets, held-out eval sets) is not part of this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
# The held-out text sets a run scores during training, by name -> file in the eval directory. Only the files that exist are passed.
EVAL_SETS = {"web": "web.pt", "code": "code.pt", "agent": "agent.pt", "general_web": "general_web.pt"}
PLACEHOLDER = re.compile(r"\$\{(DATA_ROOT|RUN_ROOT)\}")


def load_runs(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("runs"), list):
        raise SystemExit(f"{path}: not a runs file (a mapping with a `runs` list)")
    return spec


def expand(value, roots: dict[str, str | None]):
    """Replace ${DATA_ROOT} / ${RUN_ROOT} in a string (or every string of a list); refuse one whose root was not given."""
    if isinstance(value, list):
        return [expand(v, roots) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match) -> str:
        root = roots.get(m.group(1))
        if not root:
            raise SystemExit(f"{m.group(0)} is used by this entry: pass --{m.group(1).lower().replace('_', '-')} or set the environment variable")
        return root.rstrip("/")

    return PLACEHOLDER.sub(sub, value)


def find_entry(spec: dict, name: str) -> dict:
    for run in spec["runs"]:
        if run.get("name") == name:
            return run
    raise SystemExit(f"no entry named {name!r}; --list shows them")


def shard_sets(spec: dict, run: dict) -> list:
    """The entry's shard sets: its own `shards` key, else the file's; a string is one set, a list is read as one stream."""
    value = run.get("shards", spec.get("shards"))
    if value is None:
        raise SystemExit(f"{run['name']}: no `shards` key (neither on the entry nor in the file)")
    sets = value if isinstance(value, list) else [value]
    if not sets or not all(isinstance(s, str) and s for s in sets) or len(set(sets)) != len(sets):
        raise SystemExit(f"{run['name']}: shards {value!r} is a shard set directory or a list of distinct ones")
    return sets


def build_command(spec: dict, run: dict, roots: dict[str, str | None], nproc: int = 1, python: str | None = None,
                  extra_sets: tuple[str, ...] = (), check_eval: bool = True) -> list[str]:
    """The argv of the training process for one entry."""
    run_dir = f"{(roots.get('RUN_ROOT') or '').rstrip('/')}/{run['name']}"
    if not roots.get("RUN_ROOT"):
        raise SystemExit("--run-root (or RUN_ROOT) is needed: the run directory is <run-root>/<name>")
    seed = int(run["seed"])
    first, *more = [expand(s, roots) for s in shard_sets(spec, run)]
    cmd = [python or sys.executable] if nproc <= 1 else ["torchrun", "--nproc_per_node", str(nproc)]
    cmd += ["scripts/train.py", "--config", run["config"], "--run-dir", run_dir,
            "--set", f"seed={seed}", "--set", f"data.seed={1000 + seed}", "--set", f"data.shard_dir={first}"]
    if more:
        cmd += ["--set", f"data.extra_shard_dirs={json.dumps(more)}"]
    eval_dir = spec.get("eval_dir")
    if eval_dir:
        eval_dir = expand(eval_dir, roots)
        present = {k: f"{eval_dir}/{v}" for k, v in EVAL_SETS.items() if not check_eval or (Path(eval_dir) / v).is_file()}
        if present:
            cmd += ["--set", "eval.sets=" + json.dumps(present)]  # JSON is a YAML flow mapping
    for item in run.get("set", []):
        cmd += ["--set", str(expand(str(item), roots))]
    for item in extra_sets:
        cmd += ["--set", item]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", help="a runs file, e.g. runs/ablation-60m.yaml")
    ap.add_argument("--list", action="store_true", help="print the entries' names and exit")
    ap.add_argument("--name", help="the entry to run")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"), help="replaces ${DATA_ROOT} (default: $DATA_ROOT)")
    ap.add_argument("--run-root", default=os.environ.get("RUN_ROOT"), help="replaces ${RUN_ROOT}; run directories are <run-root>/<name> (default: $RUN_ROOT)")
    ap.add_argument("--nproc", type=int, default=1, help="processes on this node: >1 starts torchrun")
    ap.add_argument("--set", dest="extra", action="append", default=[], metavar="KEY=VALUE", help="a further scripts/train.py override (after the entry's own)")
    ap.add_argument("--dry-run", action="store_true", help="print the command and do not run it")
    a = ap.parse_args(argv)
    spec = load_runs(a.runs)
    if a.list:
        for run in spec["runs"]:
            after = f"  (after {run['after']} step {run.get('after_step', 'end')})" if run.get("after") else ""
            print(f"{run['name']}{after}")
        return 0
    if not a.name:
        raise SystemExit("--name <entry> is needed (or --list)")
    run = find_entry(spec, a.name)
    roots = {"DATA_ROOT": a.data_root, "RUN_ROOT": a.run_root}
    cmd = build_command(spec, run, roots, nproc=a.nproc, extra_sets=tuple(a.extra), check_eval=not a.dry_run)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())
