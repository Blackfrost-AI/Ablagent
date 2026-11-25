"""Injection-safe local subprocess executor."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any


class LocalExecutor:
    name = "local"

    def __init__(
        self, argv: Sequence[str], *, cwd: str | None = None, env: dict[str, str] | None = None
    ):
        if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError("argv must be a non-empty sequence of strings")
        self.argv = tuple(argv)
        self.cwd = cwd
        self.env = env

    def run(self, *, stage: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        completed = subprocess.run(
            self.argv, cwd=self.cwd, env=self.env, capture_output=True, text=True, check=False
        )
        return {
            "stage": stage,
            "executor": self.name,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
