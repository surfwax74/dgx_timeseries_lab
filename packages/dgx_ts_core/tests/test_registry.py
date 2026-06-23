"""Unit tests for the Registry primitive."""

from __future__ import annotations

import pytest
from dgx_ts_core.registry import Registry


class _Thing:
    def __init__(self, x: int = 0) -> None:
        self.x = x


def test_register_and_create_round_trip() -> None:
    reg: Registry[_Thing] = Registry("things")

    @reg.register("alpha")
    def _make(x: int = 7) -> _Thing:
        return _Thing(x=x)

    obj = reg.create("alpha", x=42)
    assert obj.x == 42
    assert reg.list() == ["alpha"]


def test_register_rejects_duplicate_keys() -> None:
    reg: Registry[_Thing] = Registry("things")

    @reg.register("a")
    def _one() -> _Thing:
        return _Thing()

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("a")
        def _two() -> _Thing:
            return _Thing()


def test_get_unknown_key_raises_keyerror_with_available_list() -> None:
    reg: Registry[_Thing] = Registry("things")
    reg.register("alpha")(lambda: _Thing())  # type: ignore[arg-type]
    reg.register("beta")(lambda: _Thing())  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="alpha"):
        reg.get("gamma")
