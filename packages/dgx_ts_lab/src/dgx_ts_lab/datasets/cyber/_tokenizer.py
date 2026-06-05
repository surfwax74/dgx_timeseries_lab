"""Shared command tokenizer for cyber + LLM phases.

Multi-token encoding per Phase 8 locked decision:

    each command  →  [<CMD>, opcode_id, param1_id, param2_id, ...]

Reserved special tokens:
    0  <PAD>      padding
    1  <CMD>      command-start marker (precedes every command)
    2  <UNK>      unknown opcode or parameter value
    3  <MASK>     MLM training masks

Vocabulary layout:
    [0..3]                       special tokens (above)
    [4..4+N_opcodes-1]           opcode tokens
    [N_opcodes+4..]              parameter tokens
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


PAD_TOKEN = 0
CMD_TOKEN = 1
UNK_TOKEN = 2
MASK_TOKEN = 3
N_SPECIAL = 4


@dataclass
class CommandTokenizer:
    """Multi-token tokenizer over (opcode, param values) sequences."""

    opcodes: list[str] = field(default_factory=list)
    param_values: list[str] = field(default_factory=list)

    # Built lazily from opcodes/param_values
    _opcode_to_id: dict[str, int] = field(default_factory=dict, init=False)
    _param_to_id: dict[str, int] = field(default_factory=dict, init=False)
    _id_to_token: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        self._opcode_to_id = {op: i + N_SPECIAL for i, op in enumerate(self.opcodes)}
        self._param_to_id = {
            str(p): i + N_SPECIAL + len(self.opcodes)
            for i, p in enumerate(self.param_values)
        }
        self._id_to_token = {
            PAD_TOKEN: "<PAD>",
            CMD_TOKEN: "<CMD>",
            UNK_TOKEN: "<UNK>",
            MASK_TOKEN: "<MASK>",
        }
        for op, idx in self._opcode_to_id.items():
            self._id_to_token[idx] = f"OP[{op}]"
        for p, idx in self._param_to_id.items():
            self._id_to_token[idx] = f"P[{p}]"

    # ── encoding ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + len(self.opcodes) + len(self.param_values)

    def encode_command(self, opcode: str, params: list[str | int | float]) -> list[int]:
        """Encode a single command to its token sequence."""
        tokens = [CMD_TOKEN, self._opcode_to_id.get(opcode, UNK_TOKEN)]
        for p in params:
            tokens.append(self._param_to_id.get(str(p), UNK_TOKEN))
        return tokens

    def encode_stream(
        self, commands: list[tuple[str, list[str | int | float]]]
    ) -> list[int]:
        out: list[int] = []
        for opcode, params in commands:
            out.extend(self.encode_command(opcode, params))
        return out

    def add_opcode(self, opcode: str) -> int:
        if opcode in self._opcode_to_id:
            return self._opcode_to_id[opcode]
        self.opcodes.append(opcode)
        self._rebuild_maps()
        return self._opcode_to_id[opcode]

    def add_param(self, p: str) -> int:
        if str(p) in self._param_to_id:
            return self._param_to_id[str(p)]
        self.param_values.append(str(p))
        self._rebuild_maps()
        return self._param_to_id[str(p)]

    # ── decoding / introspection ─────────────────────────────────────────

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(tid), f"?{tid}") for tid in token_ids]

    def is_special(self, token_id: int) -> bool:
        return 0 <= int(token_id) < N_SPECIAL

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"opcodes": list(self.opcodes), "param_values": list(self.param_values)},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "CommandTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(opcodes=data["opcodes"], param_values=data["param_values"])
