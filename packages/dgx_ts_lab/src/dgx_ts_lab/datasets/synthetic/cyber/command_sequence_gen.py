"""Synthetic command-stream generator with realistic patterns + injections.

Produces ``(token_ids, labels, aux, tokenizer)`` that feed a
``CommandSequenceDataset``. Patterns modeled:

    routine        high-frequency telemetry pulls + heartbeats
    scheduled      mode changes, payload activations on a periodic schedule
    operator       ad-hoc operator-issued config changes (lower frequency)

Injection patterns (labeled True):

    priv_escalation   user A issues a command class only user B should issue
    flooding          same command repeated abnormally fast
    replay            recent sub-sequence repeated verbatim
    timing            command issued at a forbidden time (e.g., eclipse)
    sequence          ordered command sequence that's syntactically valid
                      but operationally meaningless
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dgx_ts_core.data import Subsystem
from dgx_ts_core.registry import DATASET_REGISTRY

from ...cyber._tokenizer import CommandTokenizer
from ...cyber.command_sequence import CommandSequenceDataset


# Realistic-sounding opcode classes. Kept synthetic to stay air-gap clean.
ROUTINE_OPCODES = ("TELEM_REQUEST", "HEARTBEAT_TX", "TIME_SYNC")
SCHEDULED_OPCODES = ("MODE_SET", "PAYLOAD_ACTIVATE", "PAYLOAD_DEACTIVATE")
OPERATOR_OPCODES = (
    "EPS_LOAD_SHED",
    "ADCS_PARAM_UPDATE",
    "RW_SPEEDUP",
    "RW_SLOWDOWN",
    "COMMS_RESET",
    "PAYLOAD_PARAM_UPDATE",
)
PRIV_OPCODES = ("FW_UPLOAD", "ADCS_OVERRIDE", "SAFE_MODE_EXIT")

ALL_OPCODES = ROUTINE_OPCODES + SCHEDULED_OPCODES + OPERATOR_OPCODES + PRIV_OPCODES

# Parameter vocabulary — small synthetic set.
PARAM_VOCAB = (
    "0", "1", "5", "10", "25", "100",
    "MODE_SUN", "MODE_ECLIPSE", "MODE_PAYLOAD",
    "CHAN_A", "CHAN_B", "SAFE", "NORMAL",
)


def _default_tokenizer() -> CommandTokenizer:
    return CommandTokenizer(opcodes=list(ALL_OPCODES), param_values=list(PARAM_VOCAB))


def _pick_command(rng: np.random.Generator, class_: str) -> tuple[str, list[str]]:
    if class_ == "routine":
        op = rng.choice(ROUTINE_OPCODES)
        params: list[str] = []
    elif class_ == "scheduled":
        op = rng.choice(SCHEDULED_OPCODES)
        params = [rng.choice(("MODE_SUN", "MODE_ECLIPSE", "MODE_PAYLOAD"))]
    elif class_ == "operator":
        op = rng.choice(OPERATOR_OPCODES)
        params = [rng.choice(("1", "5", "10", "25", "100")), rng.choice(("CHAN_A", "CHAN_B"))]
    elif class_ == "priv":
        op = rng.choice(PRIV_OPCODES)
        params = [rng.choice(("SAFE", "NORMAL"))]
    else:
        op = "TELEM_REQUEST"
        params = []
    return str(op), [str(p) for p in params]


def generate_command_sequence(
    n_commands: int = 10_000,
    seed: int = 0,
    routine_rate: float = 0.6,
    scheduled_rate: float = 0.15,
    operator_rate: float = 0.24,
    priv_rate: float = 0.01,
    # Injection rates
    inject_priv_escalation_rate: float = 0.002,
    inject_flooding_rate: float = 0.001,
    inject_replay_rate: float = 0.001,
    inject_sequence_anomaly_rate: float = 0.0015,
    tokenizer: CommandTokenizer | None = None,
) -> dict[str, Any]:
    """Generate a command stream + token IDs + labels + aux.

    Returns a dict ready to pass as kwargs to CommandSequenceDataset.
    """
    rng = np.random.default_rng(seed)
    if tokenizer is None:
        tokenizer = _default_tokenizer()

    # ── 1. Lay down baseline commands ───────────────────────────────────
    class_choices = ["routine", "scheduled", "operator", "priv"]
    class_probs = np.array([routine_rate, scheduled_rate, operator_rate, priv_rate])
    class_probs = class_probs / class_probs.sum()

    commands: list[tuple[str, list[str]]] = []
    injection_type_per_cmd: list[int] = []   # 0 == normal
    for _ in range(n_commands):
        cls = str(rng.choice(class_choices, p=class_probs))
        commands.append(_pick_command(rng, cls))
        injection_type_per_cmd.append(0)

    # ── 2. Inject anomalies ─────────────────────────────────────────────
    n_priv_inj = int(n_commands * inject_priv_escalation_rate)
    n_flood = int(n_commands * inject_flooding_rate)
    n_replay = int(n_commands * inject_replay_rate)
    n_seq = int(n_commands * inject_sequence_anomaly_rate)

    # Priv escalation: random ops issue PRIV_OPCODES (would normally be admin-only)
    for _ in range(n_priv_inj):
        i = int(rng.integers(0, n_commands))
        commands[i] = _pick_command(rng, "priv")
        injection_type_per_cmd[i] = 1

    # Flooding: same command 10–30 times in a row
    for _ in range(n_flood):
        start = int(rng.integers(0, n_commands - 30))
        cmd = _pick_command(rng, "operator")
        burst_len = int(rng.integers(10, 30))
        for j in range(start, min(start + burst_len, n_commands)):
            commands[j] = cmd
            injection_type_per_cmd[j] = 2

    # Replay: pick a random 5-cmd subsequence and paste it 50 cmds later
    for _ in range(n_replay):
        src = int(rng.integers(50, n_commands - 50))
        dst = src + int(rng.integers(20, 80))
        for k in range(5):
            if dst + k < n_commands:
                commands[dst + k] = commands[src + k]
                injection_type_per_cmd[dst + k] = 3

    # Sequence anomaly: ordered priv+operator combination at random spot
    for _ in range(n_seq):
        i = int(rng.integers(0, n_commands - 3))
        commands[i] = (PRIV_OPCODES[0], ["SAFE"])
        commands[i + 1] = (OPERATOR_OPCODES[0], ["100", "CHAN_A"])
        commands[i + 2] = (PRIV_OPCODES[1], ["NORMAL"])
        for k in range(3):
            injection_type_per_cmd[i + k] = 4

    # ── 3. Tokenize ──────────────────────────────────────────────────────
    all_tokens: list[int] = []
    token_labels: list[bool] = []
    token_injection_type: list[int] = []
    for cmd, inj_type in zip(commands, injection_type_per_cmd):
        opcode, params = cmd
        toks = tokenizer.encode_command(opcode, params)
        all_tokens.extend(toks)
        is_anom = inj_type > 0
        token_labels.extend([is_anom] * len(toks))
        token_injection_type.extend([inj_type] * len(toks))

    tokens = np.asarray(all_tokens, dtype=np.int64)
    labels = np.asarray(token_labels, dtype=np.bool_)
    aux = {"injection_type": np.asarray(token_injection_type, dtype=np.int64)}

    return {
        "tokens": tokens,
        "labels": labels,
        "aux": aux,
        "tokenizer": tokenizer,
        "name": "synth_command_sequence",
        "subsystem": Subsystem.OBDH,
    }


@DATASET_REGISTRY.register("synth_command_sequence")
def _create(**kwargs: Any) -> CommandSequenceDataset:
    """Factory: builds the synth stream then returns a CommandSequenceDataset."""
    gen_kwargs = {
        k: v for k, v in kwargs.items()
        if k in (
            "n_commands", "seed", "routine_rate", "scheduled_rate", "operator_rate",
            "priv_rate", "inject_priv_escalation_rate", "inject_flooding_rate",
            "inject_replay_rate", "inject_sequence_anomaly_rate",
        )
    }
    ds_kwargs = generate_command_sequence(**gen_kwargs)
    return CommandSequenceDataset(**ds_kwargs)
