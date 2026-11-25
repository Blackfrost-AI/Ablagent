"""Private, atomic filesystem primitives."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .errors import IntegrityError


def ensure_private_dir(path: Path, *, create: bool = False) -> Path:
    path = Path(path)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError(f"not a real directory: {path}")
    path.chmod(0o700)
    return path.resolve()


def ensure_private_file(path: Path, *, create: bool = False) -> Path:
    path = Path(path)
    if create:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"not a real file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise IntegrityError(f"control file is not private: {path}")
    return path


def atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    parent = path.parent
    ensure_private_dir(parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent, text=False)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def read_stable(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> bytes:
    path = ensure_private_file(Path(path))
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if len(data) > max_bytes or (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IntegrityError(f"file changed during read or exceeds limit: {path}")
    return data


def lock_file(path: Path) -> Iterator[None]:
    """Context manager implemented in store.py to avoid importing fcntl on Windows."""
    raise NotImplementedError
