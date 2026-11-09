"""TrainConfig.freeze_trunk: add-on CP as a post-hoc stage on a finished trunk.

WHAT THE KEY DOES. With freeze_trunk true every tensor that is not the add-on head's (a parameter whose name does not start with cp_head.PREFIX;
the tied embedding included) gets requires_grad False when the Trainer is built. The trainer's own step then trains the head alone: the trunk's
forward builds no autograd graph (nothing of its activations is saved), no trunk tensor gets a gradient or an optimizer state (the stock Muon and
AdamW skip a tensor with no gradient before creating its state), the logged grad_norm (the trunk's own clip) is 0 and cp_grad_norm is the head's.
Nothing else in the step is changed: the model stays in training mode (it is the joint run's code path; the mixture head refuses eval mode), the
batch, the losses, the head's clip and its optimizer groups are the joint run's. The equivalence rests on the add-on head's detached sources: the
head's loss never reaches the trunk, so training the head on the finished trunk is the joint run's head objective at the trunk's last state. The
CPU tests are tests/test_addon_posthoc_freeze.py (the frozen stage through the trainer's own step, its equality with the joint run's head step at
the same trunk under err <= 4 x err_ref + 1e-6, resume, causality of the head's path) and tests/test_freeze_trunk_off.py (the key off is main's
tree).

THE START. The stage is a new run (step 0, a fresh optimizer) from init_from's model weights (Trainer.init_from_checkpoint). A source equal to
this run's model with model.cp_head off is the growth this key accepts: the head's tensors (every state-dict name under cp_head.) are the only
ones the source lacks, they stay as built (own-generator init) and are recorded in init_info["grown"]; every trunk tensor is loaded strictly (names,
shapes). Without the key the old refusal stands ("the source's model config differs"). A source that already has the head loads whole, as any
init_from, and the trunk is frozen.

THE CHECKSUM. trunk_sha256 hashes every state-dict entry outside the head (name, dtype, shape, bytes; parameters and buffers) in name order. It
is taken once when the trunk is loaded (init_from_checkpoint, or the resume's load_state_dict from the file's record) and kept as the start; every
save recomputes it, refuses to write a checkpoint whose trunk differs from the start (TrunkChanged), and records both in the checkpoint
("freeze_trunk": trunk_sha256, start_sha256, tensors); every load recomputes it from the loaded model and refuses a file whose trunk does not
hash to its record (a file edited after the save). init_info["freeze_trunk"] also keeps the source file's own trunk hash, so the composed final's trunk
is shown equal to the source's by two numbers.

THE PER-STEP CHECK. Before the clip: no trunk tensor has a gradient and none requires one (check_grads); after the optimizer step: no trunk tensor
has optimizer state (check_state). A trunk tensor touched raises TrunkTouched before the step is written to the record. A step before the trunk
is loaded raises too (check_loaded): a frozen random trunk would train a head on nothing. After the process's first update the trunk is hashed
once (after_first_step): an in-place write by a part of the step that is neither a gradient nor optimizer state is then seen on the update
that made it, not at the next save. The clamps that follow an update (project_layer_scalars, floor_gates_) are not run (nothing trains them).

REFUSED BY NAME (check_config, called first in Trainer.__init__): freeze_trunk without init_from; with resume_from (a branch of another run's
whole state, the trunk's optimizer state included); without model.cp_head.enabled and train_mode "addon"; with a tst.schedule other than empty or
[[0, 1]] (a bag phase trains the trunk); with data parallelism, cuda_graphs, activation_checkpoint, shape_schedule, precision_switch, the sharded
optimizer (optimizer_sharding), wgrad_accum (its persistent float32 gradients would attach to the frozen weights),
model.encoder.gdn_qmix_select_step (its choice writes the trunk's tensors in a step) or log_zero_wgrad_every (it counts the trunk's weight
gradients). A resume also checks that the file's trunk record is the trunk its init_from record started from, and that its optimizer holds no
state for a trunk tensor; a start refuses a source whose tensors the model cannot hold as they are (another dtype).

NOT HERE. Head-only rolling saves (build, optional: only if <data volume> is tight) are not built: the stage's checkpoints are the stock ones (the
model state, the head's optimizer state, step, generators), which every scoring tool reads unchanged.
"""

from __future__ import annotations

import hashlib

import torch

from ced.model.cp_head import PREFIX as HEAD_PREFIX

# tst.schedule values that mean "next-token steps only": no schedule, or TST at s = 1 from step 0 (the base bit for bit, TSTConfig)
NEXT_TOKEN_SCHEDULES = ([], [[0, 1]])


class TrunkTouched(RuntimeError):
    """A frozen trunk tensor has a gradient, an optimizer state, or requires a gradient: the stage is not training the head alone."""


class TrunkChanged(RuntimeError):
    """The trunk's checksum is not the one taken when it was loaded (or not the one a checkpoint recorded)."""


def check_config(cfg, data_parallel: bool) -> None:
    """ValueError, by name, for a config freeze_trunk is not built for. Off: only the key's type. cfg: the TrainConfig; data_parallel:
    whether the process is one rank of a data-parallel run (Trainer's dp is not None)."""
    key = cfg.freeze_trunk
    if type(key) is not bool:
        raise ValueError(f"freeze_trunk {key!r}: true or false")
    if not key:
        return
    head = cfg.model.cp_head
    schedule = [list(entry) for entry in cfg.tst.schedule]
    refused = [
        (not cfg.init_from, "freeze_trunk needs init_from: the finished trunk's checkpoint is where the stage starts (its model weights)"),
        (bool(cfg.resume_from), f"freeze_trunk is not built together with resume_from {cfg.resume_from!r}: the stage starts from "
                                "init_from's model weights, a fresh optimizer and step 0, not from another run's whole training state"),
        (not (head.enabled and head.train_mode == "addon"),
         "freeze_trunk needs model.cp_head.enabled with model.cp_head.train_mode 'addon': only the add-on head's loss never reaches the "
         f"trunk (enabled {head.enabled!r}, train_mode {head.train_mode!r})"),
        (schedule not in NEXT_TOKEN_SCHEDULES, f"freeze_trunk is not built together with tst.schedule {cfg.tst.schedule!r}: a bag phase trains "
                                              "the trunk (built: [] or [[0, 1]])"),
        (data_parallel, "freeze_trunk is not built together with data parallelism (a world size above 1)"),
        (cfg.cuda_graphs != "none", f"freeze_trunk is not built together with cuda_graphs {cfg.cuda_graphs!r}"),
        (cfg.activation_checkpoint != "none", f"freeze_trunk is not built together with activation_checkpoint {cfg.activation_checkpoint!r}"),
        (bool(cfg.shape_schedule), f"freeze_trunk is not built together with shape_schedule {cfg.shape_schedule!r}"),
        (cfg.precision_switch is not False, f"freeze_trunk is not built together with precision_switch {cfg.precision_switch!r}"),
        (cfg.optimizer_sharding != "replicated", f"freeze_trunk is not built together with the sharded optimizer (optimizer_sharding "
                                                 f"{cfg.optimizer_sharding!r})"),
        (cfg.wgrad_accum != "autograd", f"freeze_trunk is not built together with wgrad_accum {cfg.wgrad_accum!r}: its persistent float32 "
                                        "gradients would attach to the frozen weights"),
        (bool(cfg.model.encoder.gdn_qmix_select_step), "freeze_trunk is not built together with model.encoder.gdn_qmix_select_step "
                                                       f"{cfg.model.encoder.gdn_qmix_select_step!r}: the choice writes the trunk's tensors in a step"),
        # the count reads every linear's weight gradient, which a frozen trunk does not have (the step would end in a
        # TypeError at its first logged step, after the forward and backward)
        (bool(cfg.log_zero_wgrad_every), f"freeze_trunk is not built together with log_zero_wgrad_every {cfg.log_zero_wgrad_every!r}: it counts the "
                                         "trunk's weight gradients, which a frozen trunk does not have"),
    ]
    for bad, why in refused:
        if bad:
            raise ValueError(why)


def is_trunk(name: str) -> bool:
    """A state-dict or parameter name outside the add-on head."""
    return not name.startswith(HEAD_PREFIX)


def trunk_sha256(items) -> tuple[str, int]:
    """(sha256 hex, tensor count) over (name, tensor) pairs outside the head, in name order: each entry's name, dtype, shape and bytes. The
    source file's state dict and the model's give the same digest where the dtype is the same."""
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(((n, t) for n, t in items if is_trunk(n)), key=lambda item: item[0]):
        t = tensor.detach().contiguous()
        if t.is_meta:  # the entry check builds trainers on the meta device (scripts/entry_check.py not_here reads this text)
            raise RuntimeError(f"freeze_trunk's checksum reads the value of {name}, which meta tensors do not have")
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.reshape(-1).view(torch.uint8).cpu().numpy())
        count += 1
    return digest.hexdigest(), count


class TrunkGuard:
    """The frozen trunk of one Trainer: freezes the parameters outside the head, takes the start checksum, and checks the trunk each step
    and at each save and load (module docstring). model: the built model; its optimizer is passed to check_state."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.params = [(n, p) for n, p in model.named_parameters() if is_trunk(n)]
        self.head_params = [(n, p) for n, p in model.named_parameters() if not is_trunk(n)]
        if not self.head_params:
            raise ValueError("freeze_trunk: the model has no parameter under cp_head. (nothing would train)")
        for _, p in self.params:
            p.requires_grad_(False)
        self.start: str | None = None  # the trunk's sha256 when it was loaded (None: not loaded yet)
        self.stepped = False  # whether this process's first update has been checked (after_first_step)

    # -- the checksum ---------------------------------------------------------------------------------------------------------

    def checksum(self) -> tuple[str, int]:
        return trunk_sha256(self.model.state_dict().items())

    def set_start(self) -> tuple[str, int]:
        """The trunk has been loaded (init_from_checkpoint): its checksum is the start."""
        sha, count = self.checksum()
        self.start = sha
        return sha, count

    def record(self) -> dict:
        """The checkpoint's "freeze_trunk" entry; TrunkChanged when the trunk is not the one that was loaded."""
        self.check_loaded()
        sha, count = self.checksum()
        if sha != self.start:
            raise TrunkChanged(f"freeze_trunk: the trunk's sha256 {sha} is not the one it had when it was loaded ({self.start}): a save of "
                               "a trunk that moved is refused")
        return {"trunk_sha256": sha, "start_sha256": self.start, "tensors": count}

    def verify_load(self, saved: dict | None, init_info: dict | None = None, optimizer: torch.optim.Optimizer | None = None) -> None:
        """A load (Trainer.load_state_dict, after the model and the optimizer are loaded): the checkpoint must be a freeze_trunk run's, and its
        trunk must hash to its record, which must be the start's; the start becomes the file's. init_info (the file's "init_from" record) names
        the trunk the stage started from: the file's record must be that one (a file whose trunk and record were both replaced, by another
        stage's, is not this stage's). optimizer (the one just loaded): no trunk tensor has state in it (the file's optimizer holds the head's
        state only; check_state refuses it here and not at the first update)."""
        if saved is None:
            raise ValueError("the checkpoint is not a freeze_trunk run's (no 'freeze_trunk' entry): this config has freeze_trunk true")
        sha, count = self.checksum()
        if saved["trunk_sha256"] != saved["start_sha256"]:
            raise TrunkChanged(f"the checkpoint's trunk record is inconsistent: trunk_sha256 {saved['trunk_sha256']} is not its start "
                               f"{saved['start_sha256']}")
        if sha != saved["trunk_sha256"] or count != saved["tensors"]:
            raise TrunkChanged(f"the loaded trunk ({count} tensors, sha256 {sha}) is not the checkpoint's record ({saved['tensors']} tensors, "
                               f"sha256 {saved['trunk_sha256']}): the file was changed after it was written")
        started = ((init_info or {}).get("freeze_trunk") or {}).get("trunk_sha256")
        if started != saved["start_sha256"]:
            raise TrunkChanged(f"the checkpoint's trunk record ({saved['start_sha256']}) is not the trunk its init_from record says the stage "
                               f"started from ({started}): the file is not the continuation of one freeze_trunk run")
        if optimizer is not None:
            self.check_state(optimizer)
        self.start = saved["start_sha256"]

    def describe(self) -> dict:
        """The start record of metrics.jsonl: what is frozen and what trains, and the trunk's start checksum."""
        return {"trunk_tensors": len(self.params), "trunk_parameters": sum(p.numel() for _, p in self.params),
                "head_tensors": len(self.head_params), "head_parameters": sum(p.numel() for _, p in self.head_params),
                "start_sha256": self.start}

    # -- the checks -----------------------------------------------------------------------------------------------------------

    def after_first_step(self) -> None:
        """Once a process, after its first update: the trunk hashes to its start. The per-step checks see gradients and optimizer state only;
        an in-place write by another part of the step (a clamp, a decay, a kernel that updates its operand) is seen at the next save, which
        on a preemptible machine may be minutes of GPU time later, and an entry check of a few updates never saves. One hash of the trunk (the save's
        cost) here puts it on the update that wrote. Later updates are the save's."""
        if not self.stepped:
            self.stepped = True
            sha, _ = self.checksum()
            if sha != self.start:
                raise TrunkChanged(f"freeze_trunk: the trunk's sha256 {sha} is not the one it had when it was loaded ({self.start}) after this "
                                   "process's first update: a part of the step wrote to a trunk tensor in place")

    def check_loaded(self) -> None:
        if self.start is None:
            raise ValueError("freeze_trunk: the trunk has not been loaded (init_from_checkpoint, or a resume from a freeze_trunk run's "
                             "checkpoint): a frozen random trunk would train a head on nothing")

    def check_grads(self) -> None:
        """Before the clip: no trunk tensor has a gradient, none requires one."""
        bad = [n for n, p in self.params if p.grad is not None or p.requires_grad]
        if bad:
            raise TrunkTouched(f"freeze_trunk: {len(bad)} trunk tensors have a gradient or require one (first: {bad[:3]})")

    def check_state(self, optimizer: torch.optim.Optimizer) -> None:
        """After the optimizer step: no trunk tensor has optimizer state."""
        held = {id(p) for p in optimizer.state}
        bad = [n for n, p in self.params if id(p) in held]
        if bad:
            raise TrunkTouched(f"freeze_trunk: {len(bad)} trunk tensors have optimizer state (first: {bad[:3]})")
