"""Checkpoint resume: a run that stops and resumes matches an uninterrupted run bit for bit.

Covers the in-process path (new Trainer objects loaded from the checkpoint file), the command-line
path across separate processes, and a SIGTERM (spot eviction notice) in the middle of a run.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from ced.config import DataConfig, TrainConfig, save_yaml, tiny_control_config, tiny_model_config
from ced.train.trainer import Trainer

REPO = Path(__file__).resolve().parent.parent


def small_config(arch: str, steps: int = 8) -> TrainConfig:
    model = tiny_model_config() if arch == "looped_ced" else tiny_control_config()
    return TrainConfig(
        model=model,
        data=DataConfig(seq_len=32, batch_size=2),
        steps=steps,
        log_every=1000,
        ckpt_every_steps=1000,
        ckpt_every_minutes=1e9,
    )


def assert_same_state(a: Trainer, b: Trainer):
    assert a.step == b.step
    for (name, pa), (_, pb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(pa, pb), f"parameter {name} differs"
    sa, sb = a.optimizer.state_dict(), b.optimizer.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for i in sa["state"]:
        for key in sa["state"][i]:
            assert torch.equal(sa["state"][i][key], sb["state"][i][key]), f"optimizer state {i}/{key} differs"


@pytest.mark.parametrize("arch", ["looped_ced", "control"])
def test_resume_in_process_matches_uninterrupted(arch, tmp_path):
    cfg = small_config(arch)
    straight = Trainer(cfg, tmp_path / "straight")
    straight_losses = [straight.train_step()["loss"] for _ in range(8)]

    first = Trainer(cfg, tmp_path / "resumed")
    losses = [first.train_step()["loss"] for _ in range(4)]
    path = first.save_checkpoint()
    del first
    torch.manual_seed(999)  # disturb the global RNG: the checkpoint must restore everything needed
    second = Trainer.from_checkpoint(path)
    losses += [second.train_step()["loss"] for _ in range(4)]

    assert losses == straight_losses
    assert_same_state(second, straight)


def run_cli(config_path, run_dir, *extra, wait=True):
    cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(config_path), "--run-dir", str(run_dir), *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if wait:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def final_state(run_dir) -> Trainer:
    trainer = Trainer.from_checkpoint(Path(run_dir) / "checkpoints" / (Path(run_dir) / "checkpoints" / "latest").read_text().strip())
    return trainer


def test_resume_across_processes_matches_uninterrupted(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    save_yaml(small_config("looped_ced"), cfg_path)
    run_cli(cfg_path, tmp_path / "straight", "--threads", "1")
    run_cli(cfg_path, tmp_path / "resumed", "--threads", "1", "--until-step", "3")
    out = run_cli(cfg_path, tmp_path / "resumed", "--threads", "1")
    assert "resumed from" in out.stdout
    assert_same_state(final_state(tmp_path / "resumed"), final_state(tmp_path / "straight"))


@pytest.mark.slow
def test_sigterm_saves_and_resume_matches(tmp_path):
    """Send SIGTERM mid-run (the spot eviction notice): the run checkpoints and exits, and the
    resumed run ends bit-identical to an uninterrupted one."""
    cfg = small_config("looped_ced", steps=40)
    cfg.metrics_jsonl = True  # the step records tell this test when the run is under way
    cfg_path = tmp_path / "cfg.yaml"
    save_yaml(cfg, cfg_path)
    run_cli(cfg_path, tmp_path / "straight", "--threads", "1")

    proc = run_cli(cfg_path, tmp_path / "evicted", "--threads", "1", wait=False)
    # Mid-run: once two training steps are recorded. A fixed sleep ("import + a few steps", 6 s) is too short on a
    # loaded VM: the import alone took 12 s at load 13, so the signal arrived before the trainer's handler existed.
    metrics = tmp_path / "evicted" / "metrics.jsonl"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and proc.poll() is None:
        if metrics.exists() and metrics.read_text().count('"loss"') >= 2:
            break
        time.sleep(0.02)
    proc.send_signal(signal.SIGTERM)
    output, _ = proc.communicate(timeout=120)
    assert "SIGTERM received" in output, output
    stopped_at = final_state(tmp_path / "evicted").step
    assert 0 < stopped_at < 40

    run_cli(cfg_path, tmp_path / "evicted", "--threads", "1")
    assert_same_state(final_state(tmp_path / "evicted"), final_state(tmp_path / "straight"))


class _BrokenWandb:
    def log(self, *args, **kwargs):
        raise ConnectionResetError("wandb-core is gone")

    def finish(self, *args, **kwargs):
        raise ConnectionResetError("wandb-core is gone")


def test_wandb_failure_never_stops_training_or_saving(tmp_path):
    trainer = Trainer(small_config("looped_ced", steps=4), tmp_path / "run")
    trainer._wandb = _BrokenWandb()  # as if W&B died (e.g. killed by the eviction SIGTERM)
    messages = []
    trainer.fit(log=messages.append)
    assert trainer.step == 4
    assert final_state(tmp_path / "run").step == 4
    assert any("continuing without W&B" in m for m in messages)


def test_resume_refuses_a_changed_config_unless_allowed(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    save_yaml(small_config("looped_ced", steps=6), cfg_path)
    run_cli(cfg_path, tmp_path / "run", "--threads", "1", "--until-step", "3")
    refused = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(cfg_path), "--run-dir", str(tmp_path / "run"),
         "--threads", "1", "--steps", "12"],
        capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"),
    )
    assert refused.returncode != 0 and "steps: checkpoint 6 -> requested 12" in refused.stderr
    assert final_state(tmp_path / "run").step == 3
    run_cli(cfg_path, tmp_path / "run", "--threads", "1", "--steps", "12", "--allow-config-change")
    resumed = final_state(tmp_path / "run")
    assert resumed.step == 12 and resumed.cfg.steps == 12
