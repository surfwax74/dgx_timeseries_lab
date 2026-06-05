"""Synthetic operator-activity generator with impersonation injection.

Builds activity-window features ready for ActivityWindowDataset. Each
operator has a distinctive behavior fingerprint (preferred hours, command
rate, diversity, session length). Impersonation injection swaps one
operator's traffic for another's at random windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dgx_ts_core.data import Subsystem
from dgx_ts_core.registry import DATASET_REGISTRY

from ...cyber.activity_window import ActivityWindowDataset


@dataclass
class OperatorPersona:
    """A single operator's behavior fingerprint."""

    operator_id: int
    name: str
    # Hours of typical activity (24h clock)
    active_hour_center: float = 14.0
    active_hour_spread: float = 4.0
    # Command rate (per minute) — lognormal
    log_command_rate_mean: float = 0.6      # exp(0.6) ≈ 1.8 cmds/min
    log_command_rate_std: float = 0.3
    # Command diversity (Shannon entropy in nats over command-class dist)
    diversity_mean: float = 1.5
    diversity_std: float = 0.3
    # Session length in minutes (lognormal)
    log_session_length_mean: float = 3.5    # exp(3.5) ≈ 33 min
    log_session_length_std: float = 0.4
    # Login frequency (per hour)
    login_freq_mean: float = 2.0
    login_freq_std: float = 0.5


DEFAULT_PERSONAS: list[OperatorPersona] = [
    OperatorPersona(0, "alice",   active_hour_center=9.0,  log_command_rate_mean=0.8, diversity_mean=1.8, log_session_length_mean=3.0),
    OperatorPersona(1, "bob",     active_hour_center=14.0, log_command_rate_mean=0.4, diversity_mean=1.2, log_session_length_mean=4.0),
    OperatorPersona(2, "carol",   active_hour_center=22.0, log_command_rate_mean=1.1, diversity_mean=2.2, log_session_length_mean=2.5),
]


def _sample_persona_window(
    rng: np.random.Generator, persona: OperatorPersona, window_idx: int
) -> np.ndarray:
    """Sample one activity-window feature vector consistent with the persona."""
    # Time-of-day for this window (window_idx = minute index → hour)
    hour = (window_idx / 60.0) % 24.0
    # Persona is more active near their center hour; scale rates accordingly
    hour_distance = min(abs(hour - persona.active_hour_center),
                        24.0 - abs(hour - persona.active_hour_center))
    activity_factor = float(np.exp(-(hour_distance / persona.active_hour_spread) ** 2))

    login_freq = max(
        0.0,
        rng.normal(persona.login_freq_mean, persona.login_freq_std) * activity_factor,
    )
    command_rate = max(
        0.0,
        float(np.exp(rng.normal(persona.log_command_rate_mean, persona.log_command_rate_std)))
        * activity_factor,
    )
    diversity = max(
        0.0,
        rng.normal(persona.diversity_mean, persona.diversity_std),
    )
    session_length = max(
        0.0,
        float(np.exp(rng.normal(persona.log_session_length_mean, persona.log_session_length_std))),
    )
    hod_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hod_cos = float(np.cos(2 * np.pi * hour / 24.0))

    return np.asarray(
        [login_freq, command_rate, diversity, hod_sin, hod_cos, session_length],
        dtype=np.float32,
    )


def generate_operator_traffic(
    n_windows: int = 5_000,
    seed: int = 0,
    personas: list[OperatorPersona] | None = None,
    impersonation_rate: float = 0.02,
) -> dict[str, Any]:
    """Generate activity-window features + per-window operator_id + labels."""
    rng = np.random.default_rng(seed)
    personas = personas or DEFAULT_PERSONAS
    if not personas:
        raise ValueError("at least one persona required")

    # Assign a claimed operator to each window (uniform among personas).
    claimed_op_ids = rng.integers(0, len(personas), size=n_windows).astype(np.int64)
    # Actual operator generating the window — usually the claimed one,
    # except where impersonation kicks in.
    actual_op_ids = claimed_op_ids.copy()

    n_impers = int(n_windows * impersonation_rate)
    impers_idx = rng.choice(n_windows, size=n_impers, replace=False)
    for i in impers_idx:
        # Pick a DIFFERENT actual operator for this window
        choices = [op for op in range(len(personas)) if op != claimed_op_ids[i]]
        actual_op_ids[i] = rng.choice(choices)

    features = np.empty((n_windows, 6), dtype=np.float32)
    for i in range(n_windows):
        features[i] = _sample_persona_window(rng, personas[actual_op_ids[i]], i)

    # Label = True at impersonation windows (claimed != actual)
    labels = (claimed_op_ids != actual_op_ids).astype(np.bool_)

    return {
        "features": features,
        "operator_ids": claimed_op_ids,    # the claim — what the model gets at inference
        "labels": labels,
        "sample_rate_hz": 1.0 / 60.0,
        "name": "synth_operator_traffic",
        "subsystem": Subsystem.OBDH,
    }


@DATASET_REGISTRY.register("synth_operator_traffic")
def _create(**kwargs: Any) -> ActivityWindowDataset:
    gen_keys = ("n_windows", "seed", "impersonation_rate")
    gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_keys}
    ds_kwargs = generate_operator_traffic(**gen_kwargs)
    return ActivityWindowDataset(**ds_kwargs)
