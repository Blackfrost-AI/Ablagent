"""Executor protocol."""
from __future__ import annotations
from typing import Any, Protocol

class Executor(Protocol):
    name: str
    def run(self, *, stage: str, inputs: list[dict[str, Any]]) -> dict[str, Any]: ...
