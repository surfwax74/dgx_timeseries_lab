"""Phase 8 synthetic cybersecurity-flavored data generators.

    command_sequence_gen.py   realistic ops command stream with injection
                              (priv escalation, flooding, replay, timing,
                              sequence anomalies)
    operator_traffic_gen.py   per-operator behavior fingerprints with
                              impersonation injection

Both register convenience factories that produce ready-to-use
`CommandSequenceDataset` / `ActivityWindowDataset` instances.
"""

from . import command_sequence_gen, operator_traffic_gen  # noqa: F401  registrations
from .command_sequence_gen import generate_command_sequence
from .operator_traffic_gen import generate_operator_traffic

__all__ = [
    "command_sequence_gen",
    "generate_command_sequence",
    "generate_operator_traffic",
    "operator_traffic_gen",
]
