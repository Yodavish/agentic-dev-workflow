"""Lifecycle helpers for runner-owned SQLite graph checkpoints."""

from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DATABASE_NAME = "checkpoints.sqlite3"
SQLITE_BUSY_TIMEOUT_MS = 5_000


class CheckpointStorageError(RuntimeError):
    """Safe, bounded storage error that does not expose database contents."""


def _ensure_private_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    if current.exists() and not current.is_dir():
        raise CheckpointStorageError("checkpoint data path is not a directory")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise CheckpointStorageError(
                    "checkpoint data path changed during creation"
                ) from None
        except OSError:
            raise CheckpointStorageError("checkpoint data directory could not be created") from None
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise CheckpointStorageError(
            "checkpoint data directory permissions are too broad; directory was not changed"
        )


def _prepare_database_file(path: Path) -> bool:
    created = False
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            metadata = path.lstat()
        except OSError:
            raise CheckpointStorageError("checkpoint database file could not be created") from None
        else:
            os.close(descriptor)
            metadata = path.lstat()
            created = True

    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointStorageError("checkpoint database must be a regular file")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CheckpointStorageError(
            "checkpoint database permissions are too broad; file was not changed"
        )
    return created


def _validate_existing_database(path: Path) -> None:
    """Check an existing file read-only before letting saver setup run."""
    connection: sqlite3.Connection | None = None
    try:
        if path.stat().st_size == 0:
            raise CheckpointStorageError(
                "existing checkpoint database is empty; the file was preserved"
            )
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"checkpoints", "writes"}.issubset(tables):
            raise CheckpointStorageError(
                "existing SQLite file has no LangGraph checkpoint schema; file was preserved"
            )
    except CheckpointStorageError:
        raise
    except sqlite3.Error:
        raise CheckpointStorageError(
            "existing checkpoint database is corrupt or unreadable; file was preserved"
        ) from None
    except OSError:
        raise CheckpointStorageError(
            "existing checkpoint database could not be inspected; file was preserved"
        ) from None
    finally:
        if connection is not None:
            connection.close()


@contextmanager
def open_sqlite_checkpointer(
    data_dir: Path, *, require_existing: bool = False
) -> Iterator[SqliteSaver]:
    """Open the managed saver and close its connection when the context exits.

    The configured directory must already be canonical and absolute. New
    directories and the database file are created with restrictive modes; an
    existing database is never chmodded, removed, or replaced.
    """
    if not isinstance(data_dir, Path) or not data_dir.is_absolute():
        raise CheckpointStorageError("checkpoint data directory must be an absolute path")
    try:
        canonical_dir = data_dir.resolve(strict=False)
        if canonical_dir == Path(canonical_dir.anchor):
            raise CheckpointStorageError("filesystem root cannot be used for checkpoints")
        database_path = canonical_dir / CHECKPOINT_DATABASE_NAME
        if require_existing and not database_path.is_file():
            raise CheckpointStorageError(
                "checkpoint database does not exist; no new database was created"
            )
        _ensure_private_directory(canonical_dir)
        created = _prepare_database_file(database_path)
        if not created:
            _validate_existing_database(database_path)
    except CheckpointStorageError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise CheckpointStorageError("checkpoint storage path could not be validated") from None

    try:
        # SqliteSaver.from_conn_string owns and closes the sqlite3.Connection.
        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            # The factory does not expose sqlite3.connect(timeout=...), so set
            # the equivalent connection-level PRAGMA before graph operations.
            saver.conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
            yield saver
    except sqlite3.Error:
        raise CheckpointStorageError(
            "SQLite checkpoint storage failed; the database was preserved"
        ) from None
    except OSError:
        raise CheckpointStorageError(
            "SQLite checkpoint storage could not be opened; the database was preserved"
        ) from None
