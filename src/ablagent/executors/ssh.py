"""SSH executor using an argv-only remote command.

Credentials are supplied by the caller's SSH agent/config; this module never
accepts passwords or constructs a shell command from untrusted input.
"""
from __future__ import annotations
import subprocess
import re
import shlex
from typing import Any, Sequence

class SSHExecutor:
    name = "ssh"
    def __init__(self, host: str, argv: Sequence[str], *, user: str | None = None):
        if not re.match(r"^[A-Za-z0-9_.-]+$", host) or host.startswith("-") or (user and not re.match(r"^[A-Za-z0-9_.-]+$", user)):
            raise ValueError("host/user contains unsafe characters")
        if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError("host and non-empty argv strings are required")
        self.host = host
        self.argv = tuple(argv)
        self.user = user
    def run(self, *, stage: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        target = f"{self.user}@{self.host}" if self.user else self.host
        # OpenSSH invokes the remote command through the user's shell. Quote
        # every argv element so metacharacters remain data, never syntax.
        remote_command = " ".join(shlex.quote(arg) for arg in self.argv)
        completed = subprocess.run(["ssh", "--", target, remote_command], capture_output=True, text=True, check=False)
        return {"stage": stage, "executor": self.name, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
