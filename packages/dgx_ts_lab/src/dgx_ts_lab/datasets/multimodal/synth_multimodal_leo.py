"""Synthetic multi-modal LEO data generator.

Produces correlated (telemetry, commands, logs) aligned to 1 Hz. The key
property: anomalies surface across modalities. A bus-voltage glitch in
telemetry generates an ERROR log + may have been preceded by an operator
command. The cross-modal MAE pretraining exploits this correlation.

Composition:
    telemetry  → LayeredSyntheticDataset (LEO EPS-style, 6 channels by default)
    commands   → synthesized in lock-step with mode-machine transitions +
                 ad-hoc operator commands, plus a few priv-escalation injections
                 that correlate with telemetry-fault windows
    logs       → triggered by:
                   * routine ops (INFO heartbeats every 60 s)
                   * mode transitions (INFO)
                   * fault injections (ERROR / FATAL with severity ∝ magnitude)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dgx_ts_core.data import Channel, Subsystem, Units
from dgx_ts_core.registry import DATASET_REGISTRY

from ..cyber._tokenizer import CommandTokenizer
from ._log_tokenizer import LogSeverity, LogTokenizer
from .event_bucketer import CommandEventBucketer, LogEventBucketer
from .multimodal_dataset import MultiModalDataset

_DEFAULT_OPCODES = ("TELEM_REQUEST", "MODE_SET", "PAYLOAD_ACTIVATE", "EPS_LOAD_SHED", "FW_UPLOAD")
_DEFAULT_PARAMS = ("0", "1", "MODE_SUN", "MODE_ECLIPSE", "SAFE", "NORMAL")
_DEFAULT_LOG_SOURCES = ("OBDH", "EPS", "TCS", "COMMS", "PAYLOAD")
_DEFAULT_LOG_CODES = ("HEARTBEAT", "MODE_TRANSITION", "FAULT_DETECTED", "SAFE_MODE_ENTERED")


def generate_multimodal_leo(
    n_seconds: int = 3600,
    seed: int = 0,
    n_telemetry_channels: int = 6,
    command_rate_per_hour: float = 30.0,
    log_heartbeat_period_s: float = 60.0,
    fault_correlated_priv_rate: float = 0.05,    # fraction of fault windows that get a priv cmd
) -> dict[str, Any]:
    """Generate aligned (telemetry, commands, logs) for ``n_seconds``.

    Returns a dict ready to pass as kwargs to MultiModalDataset.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Telemetry via a tiny LayeredSynth ─────────────────────────
    from ..synthetic.layered import (
        LayeredSyntheticDataset,
        faults,
        modes,
        noise,
        physics,
    )

    tel_channels = tuple(
        Channel(
            name=f"tel_{i}",
            units=Units.DIMENSIONLESS,
            subsystem=Subsystem.EPS,
            sample_rate_hz=1.0,
            description=f"Synthetic telemetry channel {i}",
        )
        for i in range(n_telemetry_channels)
    )
    components: list = [
        modes.ModeMachine(period_s=600.0, eclipse_fraction=0.35),
    ]
    for i in range(n_telemetry_channels):
        components.append(
            physics.OrbitalSinusoid(f"tel_{i}", amplitude=1.0, period_s=600.0)
        )
        components.append(noise.GaussianNoise(f"tel_{i}", std=0.05))
    # Inject some faults across channels
    for i in range(min(2, n_telemetry_channels)):
        components.append(
            faults.PointFault(f"tel_{i}", rate_per_hour=10.0, magnitude=5.0)
        )

    tel_ds = LayeredSyntheticDataset(
        channels=tel_channels,
        components=components,
        n_samples=n_seconds,
        sample_rate_hz=1.0,
        seed=seed,
    )
    telemetry_arr = tel_ds._data                                   # (T, C_tel)
    tel_labels = tel_ds._labels                                    # (T,) bool
    fault_log = tel_ds._fault_log

    # ── 2. Commands ─────────────────────────────────────────────────
    cmd_tokenizer = CommandTokenizer(
        opcodes=list(_DEFAULT_OPCODES), param_values=list(_DEFAULT_PARAMS)
    )
    n_routine_cmds = int(command_rate_per_hour * n_seconds / 3600.0)
    cmd_times: list[float] = list(rng.uniform(0, n_seconds, size=n_routine_cmds))
    cmd_opcodes: list[int] = []
    cmd_params: list[int] = []
    for _ in range(n_routine_cmds):
        op = str(rng.choice(_DEFAULT_OPCODES[:4]))   # routine ops only
        p = str(rng.choice(_DEFAULT_PARAMS))
        cmd_opcodes.append(cmd_tokenizer._opcode_to_id.get(op, 0))
        cmd_params.append(cmd_tokenizer._param_to_id.get(p, 0))

    # Fault-correlated priv-escalation commands
    for f in fault_log:
        if rng.random() < fault_correlated_priv_rate:
            # Drop an FW_UPLOAD command shortly BEFORE the fault window
            t_priv = max(0.0, float(f["start"]) - rng.uniform(1.0, 5.0))
            cmd_times.append(t_priv)
            cmd_opcodes.append(cmd_tokenizer._opcode_to_id["FW_UPLOAD"])
            cmd_params.append(cmd_tokenizer._param_to_id["NORMAL"])

    cmd_times_arr = np.asarray(cmd_times, dtype=np.float64)
    cmd_opcodes_arr = np.asarray(cmd_opcodes, dtype=np.int64)
    cmd_params_arr = np.asarray(cmd_params, dtype=np.int64)
    # Sort by time so bucketer's "last_*" overwrites are chronological
    order = np.argsort(cmd_times_arr)
    cmd_times_arr = cmd_times_arr[order]
    cmd_opcodes_arr = cmd_opcodes_arr[order]
    cmd_params_arr = cmd_params_arr[order]

    cmd_bucketer = CommandEventBucketer(sample_rate_hz=1.0)
    commands_arr = cmd_bucketer.bucket(
        cmd_times_arr, cmd_opcodes_arr, cmd_params_arr, n_bins=n_seconds
    )

    # ── 3. Logs ─────────────────────────────────────────────────────
    log_tokenizer = LogTokenizer(
        sources=list(_DEFAULT_LOG_SOURCES), codes=list(_DEFAULT_LOG_CODES)
    )
    log_times: list[float] = []
    log_severities: list[int] = []
    log_sources: list[int] = []
    # Routine heartbeats
    n_hb = int(n_seconds / log_heartbeat_period_s)
    for i in range(n_hb):
        log_times.append(float(i * log_heartbeat_period_s))
        log_severities.append(int(LogSeverity.INFO))
        log_sources.append(log_tokenizer.source_id("OBDH"))
    # Fault-correlated ERROR + FATAL logs
    for f in fault_log:
        log_times.append(float(f["start"]))
        sev = LogSeverity.ERROR if abs(float(f.get("magnitude", 0))) < 10 else LogSeverity.FATAL
        log_severities.append(int(sev))
        log_sources.append(log_tokenizer.source_id("EPS"))

    log_times_arr = np.asarray(log_times, dtype=np.float64)
    log_severities_arr = np.asarray(log_severities, dtype=np.int64)
    log_sources_arr = np.asarray(log_sources, dtype=np.int64)
    order = np.argsort(log_times_arr)
    log_times_arr = log_times_arr[order]
    log_severities_arr = log_severities_arr[order]
    log_sources_arr = log_sources_arr[order]

    log_bucketer = LogEventBucketer(sample_rate_hz=1.0)
    logs_arr = log_bucketer.bucket(
        log_times_arr, log_severities_arr, log_sources_arr, n_bins=n_seconds
    )

    return {
        "telemetry": telemetry_arr,
        "commands": commands_arr,
        "logs": logs_arr,
        "telemetry_channels": tel_channels,
        "labels": tel_labels,
        "sample_rate_hz": 1.0,
        "name": "synth_multimodal_leo",
        "subsystem": Subsystem.EPS,
    }


@DATASET_REGISTRY.register("synth_multimodal_leo")
def _create(**kwargs: Any) -> MultiModalDataset:
    gen_keys = (
        "n_seconds", "seed", "n_telemetry_channels", "command_rate_per_hour",
        "log_heartbeat_period_s", "fault_correlated_priv_rate",
    )
    gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_keys}
    ds_kwargs = generate_multimodal_leo(**gen_kwargs)
    return MultiModalDataset(**ds_kwargs)
