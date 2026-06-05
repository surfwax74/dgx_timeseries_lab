"""Log event tokenizer — small, structured, deterministic.

Per Phase 10 locked decision: no free-text bodies. Logs are structured
events with `(severity, source, code)`. Each field has a small enum vocab
that gets tokenized to integer IDs.

Vocab structure:
    severity    enum: TRACE / DEBUG / INFO / WARN / ERROR / FATAL
    source      enum: typically a small set of subsystem names + opt
    code        enum: event-type codes scoped to the source

Designed to be reused by Phase 11 LLM ops co-pilot for log retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class LogSeverity(IntEnum):
    """Standard log severity levels. Numeric ordering matches importance."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5


# Special token IDs reserved for source + code vocabularies
LOG_PAD = 0
LOG_UNK = 1
LOG_N_SPECIAL = 2


@dataclass
class LogTokenizer:
    """Tokenize (severity, source, code) tuples into integer IDs.

    Source + code share a single combined vocab so the model only has to
    learn one embedding table (severity is already numeric / small).
    """

    sources: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)

    _source_to_id: dict[str, int] = field(default_factory=dict, init=False)
    _code_to_id: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        self._source_to_id = {s: i + LOG_N_SPECIAL for i, s in enumerate(self.sources)}
        self._code_to_id = {
            c: i + LOG_N_SPECIAL + len(self.sources) for i, c in enumerate(self.codes)
        }

    @property
    def vocab_size(self) -> int:
        return LOG_N_SPECIAL + len(self.sources) + len(self.codes)

    def severity_id(self, sev: LogSeverity | int) -> int:
        return int(sev)

    def source_id(self, source: str) -> int:
        return self._source_to_id.get(source, LOG_UNK)

    def code_id(self, code: str) -> int:
        return self._code_to_id.get(code, LOG_UNK)

    def add_source(self, name: str) -> int:
        if name in self._source_to_id:
            return self._source_to_id[name]
        self.sources.append(name)
        self._rebuild()
        return self._source_to_id[name]

    def add_code(self, name: str) -> int:
        if name in self._code_to_id:
            return self._code_to_id[name]
        self.codes.append(name)
        self._rebuild()
        return self._code_to_id[name]
