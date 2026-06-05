"""Phase 8 cybersecurity-flavored datasets.

Three new `TelemetryDataset` implementations:

    CommandSequenceDataset   discrete command-token stream
                             (multi-token: [<CMD>, opcode, param1, ...])
    ActivityWindowDataset    rolling statistical windows over operator activity
                             (login rate, command rate, command diversity, etc.)
    SideChannelDataset       generic adapter: takes ANY existing TelemetryDataset
                             + a "behavior summary" function, derives a new
                             dataset with rolling stats — apply behavior-style
                             AD to EPS, ADCS, anywhere.

The CommandTokenizer in ``_tokenizer.py`` is the single source of truth for
how commands map to token IDs; reused by Phase 11's procedure synthesizer.
"""

from . import (  # noqa: F401  side-effect: register all three
    activity_window,
    command_sequence,
    side_channel,
)
from ._tokenizer import CommandTokenizer
from .activity_window import ActivityWindowDataset
from .command_sequence import CommandSequenceDataset
from .side_channel import SideChannelDataset

__all__ = [
    "ActivityWindowDataset",
    "CommandSequenceDataset",
    "CommandTokenizer",
    "SideChannelDataset",
    "activity_window",
    "command_sequence",
    "side_channel",
]
