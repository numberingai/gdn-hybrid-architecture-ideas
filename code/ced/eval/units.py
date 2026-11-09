"""Resumable scoring: a unit of at most about 10 minutes of GPU work keeps its own result file.

A long scoring pass (scripts/eval_probes.py, scripts/rescore.py) is cut into units: a run of rows of one row shape
(closed on a token bound or a time bound: ced/eval/probes.py score_sets), or one eval set. When a unit has finished,
its result is written to <units dir>/<unit>-<key hash>.json (under a temporary name, flushed, then renamed: a
half-written file never has the final name). A rerun of the same command computes the same key for each unit and
reads the file instead of scoring it; a stop (a SIGTERM, a deadline, an eviction) loses the unit in flight
and nothing else.

The key holds everything that changes a number: the checkpoint (path, step, a content fingerprint), the settings of the
pass, the sets (their sha256, their order), the unit's place (the row shape and its first row, or the eval set), the
device, the torch version, the code (code_fingerprint: the sha256 of every Python source of the ced package and of
the script, so a unit scored by other model or scoring code is never read: a rerun under another snapshot whose ced/
differs scores everything again), the CED_* environment switches (number_env: CED_GDN_IMPL, CED_FLA_SOLVE_PRECISION,
...) and UNIT_FORMAT, the version of this file format. Where a unit of rows ends is not in the key (a time bound
depends on the clock): the file holds its rows' members and item fingerprints next to the records, and the reader
takes it only when those are the rows its own plan has at that place (load's `check`).
A unit file is used only if it parses, its stored key equals the key asked for, its payload checksum holds and the
check passes; any other file (truncated, edited, another key, other rows) is ignored and the unit is scored again and
overwritten. Nothing is ever deleted. Disabled (the default of both scripts): no unit is read or written, and the
scripts write the files they wrote before this module existed, with the same content.

Writers never share a temporary file (write_atomic: its name carries the process id and a random part), so two processes
that score the same unit at the same time each rename a complete file into place and the last one wins.

Enable with the scripts' --resume or the environment variable CED_SCORE_RESUME=1 (an older code snapshot ignores the
variable); CED_SCORE_FORCE=1 or --force scores every unit again and overwrites its file.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

UNIT_FORMAT = 1  # the unit file's format: raise it when the payload's layout changes (the code that scores is in the key by itself)
ENV_RESUME, ENV_FORCE = "CED_SCORE_RESUME", "CED_SCORE_FORCE"
ENV_NOT_NUMBERS = (ENV_RESUME, ENV_FORCE, "CED_CODE_SHA")  # CED_* variables that change no number (number_env)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def temporary_name(path: Path) -> Path:
    """A name next to `path` that no other writer uses: <name>.<pid>.<random>.tmp (two processes writing the same file
    must not share it: one would rename the other's half-written file into place)."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def write_atomic(path: Path | str, text: str) -> None:
    """Write `text` to `path` so that no reader sees a partial file: a temporary name of this writer's own, flush,
    fsync, rename. A writer that is killed leaves at worst its temporary file, which no reader looks at."""
    path = Path(path)
    tmp = temporary_name(path)
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def move_atomic(src: Path | str, dst: Path | str) -> None:
    """Move a finished file to `dst`, also across file systems, so that `dst` is never a partial
    copy: copied under a temporary name next to it, then renamed."""
    import shutil

    dst = Path(dst)
    tmp = temporary_name(dst)
    try:
        shutil.move(str(src), tmp)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def code_fingerprint(*scripts: Path | str) -> str:
    """The code a unit's numbers come from: sha256 over every Python source of the ced package (the model, the
    kernels, the scorers; by relative path and content) and the given script files. Two snapshots with the same
    ced/ and script give the same value, whatever else differs (docs, infra, tests)."""
    package = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for f in sorted(package.rglob("*.py")):
        h.update(f.relative_to(package).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    for f in scripts:
        h.update(Path(f).name.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()[:16]


def number_env() -> dict:
    """The CED_* environment variables of this process, less the units' own switches and the snapshot's name: the
    switches that pick a kernel or a numeric path (ced/kernels/gdn.py CED_GDN_IMPL and CED_GDN_KEEP, ced/model/gdn.py
    CED_GDN_PREP and CED_SECOND_PASS_READ, ced/kernels/gdn_triton.py CED_FLA_SOLVE_PRECISION, and any added later)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("CED_") and k not in ENV_NOT_NUMBERS}


def file_fingerprint(path: Path | str, edge: int = 1 << 20) -> dict:
    """Cheap identity of a (large) checkpoint file: its size and the sha256 of its first and last MiB. A torch zip
    checkpoint ends with the central directory (a CRC32 per tensor record), so a changed tensor changes the tail."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(edge))
        f.seek(max(0, size - edge))
        h.update(f.read(edge))
    return {"size": size, "edges_sha256": h.hexdigest()}


def key_hash(key) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


class UnitStore:
    """The unit files of one scoring pass in `directory`. `base` is the pass-wide part of every key; `scripts`: the
    script files whose code the numbers also depend on (code_fingerprint). enabled=False: load() finds nothing and
    save() writes nothing (the directory is not even created)."""

    def __init__(self, directory: Path | str, base: dict, enabled: bool = True, force: bool = False, scripts=()):
        self.dir, self.base, self.enabled, self.force = Path(directory), base, enabled, force
        self.reused, self.scored = 0, 0
        # of the units read from files (the caller adds them up): the seconds they took when they were scored, and
        # the largest peak of allocated CUDA memory their scoring processes had reached
        self.reused_seconds, self.reused_peak_bytes = 0.0, 0
        self.notes: list[str] = []
        self.code = code_fingerprint(*scripts) if enabled else None
        self.env = number_env()

    def key(self, unit_key: dict) -> dict:
        return {"format": UNIT_FORMAT, "code": self.code, "env": self.env, **self.base, "unit": unit_key}

    def _file(self, name: str, key: dict) -> Path:
        return self.dir / f"{re.sub(r'[^A-Za-z0-9_.=-]+', '_', name)}-{key_hash(key)[:16]}.json"

    def load(self, name: str, unit_key: dict, check=None):
        """The payload saved for this unit and key, or None (disabled, forced, absent, truncated, another key, or
        refused by `check`: a function of the payload that says whether it is what the caller expects there)."""
        if not self.enabled or self.force:
            return None
        key = self.key(unit_key)
        f = self._file(name, key)
        if not f.is_file():
            return None
        try:
            doc = json.loads(f.read_text())
            payload = doc["payload"]
            ok = doc["key"] == json.loads(json.dumps(key, sort_keys=True, default=str)) and \
                doc["sha256"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            ok = ok and (check is None or bool(check(payload)))
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
            ok = False
        if not ok:
            self.notes.append(f"unit {name}: {f.name} is damaged or for another key or other rows; scored again")
            return None
        self.reused += 1
        return payload

    def save(self, name: str, unit_key: dict, payload) -> None:
        self.scored += 1
        if not self.enabled:
            return
        key = self.key(unit_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        doc = {"key": json.loads(json.dumps(key, sort_keys=True, default=str)),
               "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), "payload": payload}
        write_atomic(self._file(name, key), json.dumps(doc))
