"""Fail-closed Git change inventory and approved-scope validation.

This module reads one complete ``git status --porcelain=v2 -z`` snapshot from a
repository worktree. It is read-only: paths are parsed from NUL-delimited
output and are never passed back to Git as arguments, and no Git command that
can mutate the index or worktree is invoked.

Scope validation is a pure path comparison. It is **not an OS sandbox**, does
not isolate a process, does not prevent a concurrent same-user repository
change, and is not a substitute for process isolation. Reader provenance is an
in-process capability bound to the canonical repository root and the exact
snapshot. Bridging that snapshot requires the same root, and the resulting
complete inventory requires it again before a lock check will trust it.
Unrestricted code in the same process can still tamper with private module
state or call private issuance helpers. A clean repository is an empty
:class:`GitChangeInventory`; acquisition failures raise
:class:`GitInventoryError` instead of returning an empty or partial inventory.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import secrets
import selectors
import shutil
import subprocess
import time
import weakref
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .agent_outputs import (
    AgentOutputValidationError,
    has_file_directory_prefix_conflict,
    portable_path_collision_key,
    validate_relative_repository_path,
)
from .git_safety import DEFAULT_GIT_TIMEOUT_SECONDS, git_environment
from .verification_registry import (
    MAX_INVENTORY_ENTRIES,
    MAX_PATH_BYTES,
    CheckPathChange,
    CheckPathChangeKind,
    CompletePathInventory,
    _issue_complete_path_inventory,
)

MAX_GIT_STATUS_OUTPUT_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024

_MODE_FILE = 0o100644
_MODE_EXECUTABLE = 0o100755
_MODE_SYMLINK = 0o120000
_MODE_GITLINK = 0o160000
_OCTAL_MODE = re.compile(rb"[0-7]{6}\Z")
_CHANGE_CODES = frozenset(b".MTADRCU")
_RENAME_SCORE = re.compile(rb"(?P<kind>[RC])(?P<score>[0-9]{1,3})\Z")
_PROVENANCE_DOMAIN = b"agentic-dev-workflow/git-inventory/v1"
_CAPABILITY_DOMAIN = b"agentic-dev-workflow/git-inventory-capability/v1\x00"
_CAPABILITY_KEY = secrets.token_bytes(32)
_CONFIG_OUTPUT_BYTES = 64 * 1024


class GitInventoryDiagnosticCode(StrEnum):
    """Fixed diagnostics for inventory acquisition and representation."""

    GIT_UNAVAILABLE = "git_unavailable"
    INVALID_REPOSITORY = "invalid_repository"
    INVALID_TIMEOUT = "invalid_timeout"
    INVALID_ARGUMENT = "invalid_argument"
    COMMAND_FAILED = "command_failed"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    INVENTORY_LIMIT = "inventory_limit"
    MALFORMED_OUTPUT = "malformed_output"
    UNMERGED_STATE = "unmerged_state"
    UNREPRESENTABLE_PATH = "unrepresentable_path"
    AMBIGUOUS_PATH = "ambiguous_path"
    PATH_CONFLICT = "path_conflict"
    INVALID_ENTRY = "invalid_entry"
    INVALID_INVENTORY = "invalid_inventory"
    UNSAFE_CONFIGURATION = "unsafe_configuration"
    UNSAFE_INDEX_FLAGS = "unsafe_index_flags"


class ApprovedScopeDiagnosticCode(StrEnum):
    """Fixed diagnostics for approved-scope validation."""

    INVALID_INVENTORY = "invalid_inventory"
    INVALID_APPROVED_PATH = "invalid_approved_path"
    AMBIGUOUS_SCOPE = "ambiguous_scope"
    OUT_OF_SCOPE = "out_of_scope"
    SUBMODULE_SCOPE_UNSUPPORTED = "submodule_scope_unsupported"


class GitInventoryError(ValueError):
    """A complete inventory could not be acquired or represented."""

    def __init__(self, diagnostic_code: GitInventoryDiagnosticCode) -> None:
        self.diagnostic_code = diagnostic_code
        super().__init__(f"Git change inventory failed: {diagnostic_code.value}")


class ApprovedScopeError(ValueError):
    """A change inventory is not valid or is outside the approved scope."""

    def __init__(self, diagnostic_code: ApprovedScopeDiagnosticCode) -> None:
        self.diagnostic_code = diagnostic_code
        super().__init__(f"approved scope validation failed: {diagnostic_code.value}")


class GitEntryType(StrEnum):
    """Path type reported by Git modes, when the mode is available."""

    FILE = "file"
    SYMLINK = "symlink"
    SUBMODULE = "submodule"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GitInventoryEntry:
    """One changed path with validated raw bytes and Git entry metadata."""

    kind: CheckPathChangeKind
    path: bytes
    entry_type: GitEntryType
    previous_path: bytes | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", CheckPathChangeKind(self.kind))
        except (TypeError, ValueError):
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ENTRY) from None
        try:
            object.__setattr__(self, "entry_type", GitEntryType(self.entry_type))
        except (TypeError, ValueError):
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ENTRY) from None
        _validate_represented_path(self.path)
        if self.kind in (CheckPathChangeKind.RENAMED, CheckPathChangeKind.COPIED):
            if type(self.previous_path) is not bytes:
                raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ENTRY)
            _validate_represented_path(self.previous_path)
            if self.previous_path == self.path:
                raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ENTRY)
        elif self.previous_path is not None:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ENTRY)

    @property
    def path_text(self) -> str:
        return self.path.decode("utf-8")

    @property
    def previous_path_text(self) -> str | None:
        if self.previous_path is None:
            return None
        return self.previous_path.decode("utf-8")

    @property
    def is_symlink(self) -> bool:
        return self.entry_type == GitEntryType.SYMLINK

    @property
    def is_submodule(self) -> bool:
        return self.entry_type == GitEntryType.SUBMODULE


@dataclass(frozen=True, slots=True)
class _ReaderProvenance:
    """Opaque, reader-issued capability for one live inventory object."""

    owner_ref: weakref.ReferenceType
    digest: bytes
    signature: bytes
    repository_root: str


@dataclass(frozen=True)
class GitChangeInventory:
    """A complete snapshot; an empty entry tuple means a clean repository."""

    entries: tuple[GitInventoryEntry, ...]
    _provenance: _ReaderProvenance | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_INVENTORY)
        if len(self.entries) > MAX_INVENTORY_ENTRIES:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVENTORY_LIMIT)
        if any(type(entry) is not GitInventoryEntry for entry in self.entries):
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_INVENTORY)

        current_paths = [entry.path for entry in self.entries]
        if len(set(current_paths)) != len(current_paths):
            raise GitInventoryError(GitInventoryDiagnosticCode.AMBIGUOUS_PATH)

        all_ends = set(current_paths)
        for entry in self.entries:
            if entry.previous_path is not None:
                all_ends.add(entry.previous_path)
        end_keys = {portable_path_collision_key(raw.decode("utf-8")) for raw in all_ends}
        if len(end_keys) != len(all_ends):
            raise GitInventoryError(GitInventoryDiagnosticCode.AMBIGUOUS_PATH)

        current_keys = {portable_path_collision_key(raw.decode("utf-8")) for raw in current_paths}
        if has_file_directory_prefix_conflict(current_keys):
            raise GitInventoryError(GitInventoryDiagnosticCode.PATH_CONFLICT)

    @property
    def is_clean(self) -> bool:
        return not self.entries

    def to_complete_path_inventory(
        self, *, repository_root: str | os.PathLike[str] | None = None
    ) -> CompletePathInventory:
        """Bridge to the Stage 2A contract only for the reader's exact snapshot."""
        if not _has_reader_provenance(self) or repository_root is None:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_INVENTORY)
        capability = self._provenance
        try:
            expected_root = Path(repository_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_INVENTORY) from None
        if expected_root.as_posix() != capability.repository_root:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_INVENTORY)
        entries = tuple(
            CheckPathChange(
                kind=entry.kind,
                path=entry.path,
                previous_path=entry.previous_path,
            )
            for entry in self.entries
        )
        return _issue_complete_path_inventory(entries, repository_root=expected_root)


def _snapshot_digest(entries: tuple[GitInventoryEntry, ...]) -> bytes:
    """Bind reader provenance to every represented inventory field."""
    digest = hashlib.sha256()
    digest.update(_PROVENANCE_DOMAIN)
    digest.update(len(entries).to_bytes(8, "big"))
    for entry in entries:
        _update_snapshot_digest(digest, entry.kind.value.encode("ascii"))
        _update_snapshot_digest(digest, entry.path)
        _update_snapshot_digest(digest, entry.entry_type.value.encode("ascii"))
        if entry.previous_path is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            _update_snapshot_digest(digest, entry.previous_path)
    return digest.digest()


def _update_snapshot_digest(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _capability_signature(
    owner: GitChangeInventory, digest: bytes, repository_root: str
) -> bytes:
    """Authenticate one capability for one live inventory object."""
    message = (
        _CAPABILITY_DOMAIN
        + id(owner).to_bytes(8, "big")
        + digest
        + b"\x00"
        + repository_root.encode("utf-8")
    )
    return hmac.new(_CAPABILITY_KEY, message, hashlib.sha256).digest()


def _issue_reader_provenance(
    inventory: GitChangeInventory, repository_root: str | Path
) -> None:
    """Attach a fresh capability; only trusted module paths call this."""
    try:
        resolved_root = Path(repository_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY) from None
    root_text = resolved_root.as_posix()
    digest = _snapshot_digest(inventory.entries)
    capability = _ReaderProvenance(
        owner_ref=weakref.ref(inventory),
        digest=digest,
        signature=_capability_signature(inventory, digest, root_text),
        repository_root=root_text,
    )
    object.__setattr__(inventory, "_provenance", capability)


def _has_reader_provenance(inventory: GitChangeInventory) -> bool:
    """Verify capability origin, owner identity, and the exact snapshot."""
    if type(inventory) is not GitChangeInventory:
        return False
    if type(inventory.entries) is not tuple or any(
        type(entry) is not GitInventoryEntry for entry in inventory.entries
    ):
        return False
    capability = getattr(inventory, "_provenance", None)
    if type(capability) is not _ReaderProvenance:
        return False
    if type(capability.owner_ref) is not weakref.ReferenceType:
        return False
    if capability.owner_ref() is not inventory:
        return False
    if (
        type(capability.digest) is not bytes
        or type(capability.signature) is not bytes
        or type(getattr(capability, "repository_root", None)) is not str
        or not capability.repository_root
    ):
        return False
    try:
        digest = _snapshot_digest(inventory.entries)
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False
    if not hmac.compare_digest(capability.digest, digest):
        return False
    expected = _capability_signature(inventory, digest, capability.repository_root)
    return hmac.compare_digest(capability.signature, expected)


def parse_git_status_porcelain_v2(
    output: bytes,
    *,
    max_entries: int | None = None,
) -> tuple[GitInventoryEntry, ...]:
    """Parse a NUL-delimited porcelain v2 status stream fail-closed."""
    if type(output) is not bytes:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if max_entries is None:
        limit = MAX_INVENTORY_ENTRIES
    elif type(max_entries) is not int or isinstance(max_entries, bool) or max_entries < 1:
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_ARGUMENT)
    else:
        limit = max_entries

    if not output:
        return ()
    if not output.endswith(b"\x00"):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    tokens = output[:-1].split(b"\x00")
    if any(not token for token in tokens):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)

    entries: list[GitInventoryEntry] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith(b"#"):
            index += 1
            continue
        if token.startswith(b"1 "):
            entries.append(_parse_ordinary(token))
            index += 1
        elif token.startswith(b"2 "):
            if index + 1 >= len(tokens):
                raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
            previous_path = tokens[index + 1]
            if _looks_like_record_token(previous_path):
                raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
            entries.append(_parse_renamed_or_copied(token, previous_path))
            index += 2
        elif token.startswith(b"u "):
            raise GitInventoryError(GitInventoryDiagnosticCode.UNMERGED_STATE)
        elif token.startswith(b"? "):
            path = token[2:]
            if not path:
                raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
            entries.append(
                GitInventoryEntry(
                    kind=CheckPathChangeKind.UNTRACKED,
                    path=path,
                    entry_type=GitEntryType.UNKNOWN,
                )
            )
            index += 1
        else:
            raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
        if len(entries) > limit:
            raise GitInventoryError(GitInventoryDiagnosticCode.INVENTORY_LIMIT)
    return tuple(entries)


def read_git_change_inventory(
    repository: str | Path,
    *,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    git_executable: str | None = None,
) -> GitChangeInventory:
    """Read one complete worktree inventory; never mutate the repository."""
    timeout = _validated_timeout(timeout_seconds)
    root = _validated_repository(repository)
    executable = _validated_git_executable(git_executable)
    discovery_environment = _inventory_git_environment(root, ceiling=False)
    _validate_safe_configuration(executable, root, timeout, discovery_environment)
    _validate_canonical_worktree_root(executable, root, timeout, discovery_environment)
    snapshot_environment = _inventory_git_environment(root, ceiling=True)
    _validate_index_flags(executable, root, timeout, snapshot_environment)
    stdout = _run_bounded_git(
        [
            executable,
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotepath=false",
            "-c",
            "status.renames=true",
            "-c",
            "status.relativePaths=false",
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=root,
        timeout_seconds=timeout,
        max_output_bytes=MAX_GIT_STATUS_OUTPUT_BYTES,
        env=snapshot_environment,
    )
    entries = parse_git_status_porcelain_v2(stdout)
    inventory = GitChangeInventory(entries=entries)
    _issue_reader_provenance(inventory, root)
    return inventory


def validate_approved_scope(
    inventory: GitChangeInventory,
    approved_paths: tuple[str, ...] | list[str],
) -> GitChangeInventory:
    """Return a revalidated inventory only if every changed end is approved."""
    if type(inventory) is not GitChangeInventory:
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_INVENTORY)
    try:
        if type(inventory.entries) is not tuple:
            raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_INVENTORY)
        if any(type(entry) is not GitInventoryEntry for entry in inventory.entries):
            raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_INVENTORY)
        entries = tuple(
            GitInventoryEntry(
                kind=entry.kind,
                path=entry.path,
                entry_type=entry.entry_type,
                previous_path=entry.previous_path,
            )
            for entry in inventory.entries
        )
        provenance = getattr(inventory, "_provenance", None)
        if provenance is not None and not _has_reader_provenance(inventory):
            raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_INVENTORY)
        canonical = GitChangeInventory(entries=entries)
        if provenance is not None and type(provenance) is _ReaderProvenance:
            _issue_reader_provenance(canonical, provenance.repository_root)
    except ApprovedScopeError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_INVENTORY) from None

    if any(
        entry.entry_type == GitEntryType.SUBMODULE for entry in canonical.entries
    ):
        raise ApprovedScopeError(
            ApprovedScopeDiagnosticCode.SUBMODULE_SCOPE_UNSUPPORTED
        )

    if type(approved_paths) not in (tuple, list):
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_APPROVED_PATH)
    approved: list[str] = []
    for value in approved_paths:
        try:
            approved.append(validate_relative_repository_path(value))
        except AgentOutputValidationError:
            raise ApprovedScopeError(ApprovedScopeDiagnosticCode.INVALID_APPROVED_PATH) from None

    encoded = [path.encode("utf-8") for path in approved]
    if len(set(encoded)) != len(encoded):
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.AMBIGUOUS_SCOPE)
    keys = [portable_path_collision_key(path) for path in approved]
    if len(set(keys)) != len(keys):
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.AMBIGUOUS_SCOPE)
    if has_file_directory_prefix_conflict(set(keys)):
        raise ApprovedScopeError(ApprovedScopeDiagnosticCode.AMBIGUOUS_SCOPE)

    approved_set = set(encoded)
    for entry in canonical.entries:
        if entry.path not in approved_set:
            raise ApprovedScopeError(ApprovedScopeDiagnosticCode.OUT_OF_SCOPE)
        if entry.kind in (CheckPathChangeKind.RENAMED, CheckPathChangeKind.COPIED):
            if entry.previous_path not in approved_set:
                raise ApprovedScopeError(ApprovedScopeDiagnosticCode.OUT_OF_SCOPE)
    return canonical


def _looks_like_record_token(token: bytes) -> bool:
    """Return true when a token is structurally a porcelain record."""
    if token.startswith(b"1 "):
        try:
            _parse_ordinary(token)
        except GitInventoryError:
            return False
        return True
    if token.startswith(b"2 "):
        fields = token.split(b" ", 9)
        if len(fields) != 10:
            return False
        _, xy, sub, mh, mi, mw, _hh, _hi, score, path = fields
        try:
            _validate_change_codes(xy, allow_rename=True)
            _validate_submodule_field(sub)
            _parse_modes(mh, mi, mw)
        except GitInventoryError:
            return False
        match = _RENAME_SCORE.fullmatch(score)
        return (
            bool(path)
            and match is not None
            and int(match.group("score")) <= 100
            and match.group("kind") in xy
        )
    return token.startswith((b"u ", b"? ", b"# ", b"! "))


def _parse_ordinary(token: bytes) -> GitInventoryEntry:
    fields = token.split(b" ", 8)
    if len(fields) != 9:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    _, xy, sub, mh, mi, mw, _hh, _hi, path = fields
    _validate_change_codes(xy, allow_rename=False)
    _validate_submodule_field(sub)
    modes = _parse_modes(mh, mi, mw)
    if not path:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if b"D" in xy:
        kind = CheckPathChangeKind.DELETED
    elif b"A" in xy:
        kind = CheckPathChangeKind.ADDED
    else:
        kind = CheckPathChangeKind.CHANGED
    return GitInventoryEntry(
        kind=kind,
        path=path,
        entry_type=_entry_type(*modes, sub=sub),
    )


def _parse_renamed_or_copied(token: bytes, previous_path: bytes) -> GitInventoryEntry:
    fields = token.split(b" ", 9)
    if len(fields) != 10:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    _, xy, sub, mh, mi, mw, _hh, _hi, score, path = fields
    _validate_change_codes(xy, allow_rename=True)
    _validate_submodule_field(sub)
    modes = _parse_modes(mh, mi, mw)
    if not path or not previous_path or path == previous_path:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    match = _RENAME_SCORE.fullmatch(score)
    if match is None or int(match.group("score")) > 100:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    score_kind = match.group("kind")
    if score_kind not in xy:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    kind = (
        CheckPathChangeKind.RENAMED
        if score_kind == b"R"
        else CheckPathChangeKind.COPIED
    )
    return GitInventoryEntry(
        kind=kind,
        path=path,
        entry_type=_entry_type(*modes, sub=sub),
        previous_path=previous_path,
    )


def _validate_change_codes(xy: bytes, *, allow_rename: bool) -> None:
    if xy == b".." or len(xy) != 2 or any(code not in _CHANGE_CODES for code in xy):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if b"U" in xy:
        raise GitInventoryError(GitInventoryDiagnosticCode.UNMERGED_STATE)
    if not allow_rename and (b"R" in xy or b"C" in xy):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)


def _validate_submodule_field(sub: bytes) -> None:
    if len(sub) != 4 or sub[:1] not in (b"N", b"S"):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if sub.startswith(b"N") and sub[1:] != b"...":
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if sub.startswith(b"S") and any(flag not in b".CMU" for flag in sub[1:]):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)


def _parse_modes(mh: bytes, mi: bytes, mw: bytes) -> tuple[int, int, int]:
    mh_mode = _parse_mode(mh)
    mi_mode = _parse_mode(mi)
    mw_mode = _parse_mode(mw)
    for mode in (mh_mode, mi_mode, mw_mode):
        if mode not in (0, _MODE_FILE, _MODE_EXECUTABLE, _MODE_SYMLINK, _MODE_GITLINK):
            raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    return mh_mode, mi_mode, mw_mode


def _parse_mode(value: bytes) -> int:
    if _OCTAL_MODE.fullmatch(value) is None:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    return int(value, 8)


def _entry_type(mh: int, mi: int, mw: int, *, sub: bytes) -> GitEntryType:
    selected = next((mode for mode in (mw, mi, mh) if mode != 0), 0)
    if selected in (_MODE_FILE, _MODE_EXECUTABLE):
        entry_type = GitEntryType.FILE
    elif selected == _MODE_SYMLINK:
        entry_type = GitEntryType.SYMLINK
    elif selected == _MODE_GITLINK:
        entry_type = GitEntryType.SUBMODULE
    elif selected == 0:
        entry_type = GitEntryType.UNKNOWN
    else:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    if sub.startswith(b"S") and entry_type != GitEntryType.SUBMODULE:
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    return entry_type


def _validate_represented_path(raw: object) -> None:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > MAX_PATH_BYTES
        or b"\x00" in raw
    ):
        raise GitInventoryError(GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH)
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise GitInventoryError(GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH) from None
    try:
        validate_relative_repository_path(text)
    except AgentOutputValidationError:
        raise GitInventoryError(GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH) from None


def _validated_timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_TIMEOUT)
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite or value <= 0:
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_TIMEOUT)
    return float(value)


def _inventory_git_environment(root: Path, *, ceiling: bool) -> dict[str, str]:
    """Return the minimal, deterministic environment for inventory commands."""
    environment = {
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "XDG_CONFIG_HOME": "/nonexistent",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if ceiling:
        environment["GIT_CEILING_DIRECTORIES"] = str(root.parent)
    return environment


def _validate_safe_configuration(
    executable: str,
    root: Path,
    timeout_seconds: float,
    environment: dict[str, str],
) -> None:
    """Reject repository-local configuration that can hide worktree changes."""
    ignorecase = _run_bounded_git(
        [executable, "--no-pager", "config", "--bool", "--get", "core.ignorecase"],
        cwd=root,
        timeout_seconds=timeout_seconds,
        max_output_bytes=_CONFIG_OUTPUT_BYTES,
        env=environment,
        accepted_returncodes=frozenset({0, 1}),
    ).strip()
    if ignorecase not in (b"", b"false"):
        raise GitInventoryError(GitInventoryDiagnosticCode.UNSAFE_CONFIGURATION)

    worktree = _run_bounded_git(
        [executable, "--no-pager", "config", "--get", "core.worktree"],
        cwd=root,
        timeout_seconds=timeout_seconds,
        max_output_bytes=_CONFIG_OUTPUT_BYTES,
        env=environment,
        accepted_returncodes=frozenset({0, 1}),
    )
    if worktree:
        raise GitInventoryError(GitInventoryDiagnosticCode.UNSAFE_CONFIGURATION)


def _validate_canonical_worktree_root(
    executable: str,
    root: Path,
    timeout_seconds: float,
    environment: dict[str, str],
) -> None:
    """Require the supplied path to be the canonical worktree root."""
    output = _run_bounded_git(
        [
            executable,
            "--no-pager",
            "rev-parse",
            "--path-format=absolute",
            "--is-inside-work-tree",
            "--show-toplevel",
        ],
        cwd=root,
        timeout_seconds=timeout_seconds,
        max_output_bytes=_CONFIG_OUTPUT_BYTES,
        env=environment,
    )
    lines = output.splitlines()
    if len(lines) != 2 or lines[0] != b"true":
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY)
    try:
        actual_root = Path(os.fsdecode(lines[1])).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY) from None
    if actual_root != root:
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY)


def _validate_index_flags(
    executable: str,
    root: Path,
    timeout_seconds: float,
    environment: dict[str, str],
) -> None:
    """Reject index flags that can hide worktree modifications."""
    output = _run_bounded_git(
        [executable, "--no-pager", "ls-files", "-v", "-z", "--"],
        cwd=root,
        timeout_seconds=timeout_seconds,
        max_output_bytes=MAX_GIT_STATUS_OUTPUT_BYTES,
        env=environment,
    )
    if not output:
        return
    if not output.endswith(b"\x00"):
        raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
    for record in output.split(b"\x00")[:-1]:
        if not record:
            raise GitInventoryError(GitInventoryDiagnosticCode.MALFORMED_OUTPUT)
        tag = record[:1]
        if tag == b"S" or tag.islower():
            raise GitInventoryError(GitInventoryDiagnosticCode.UNSAFE_INDEX_FLAGS)


def _validated_repository(repository: object) -> Path:
    try:
        root = Path(repository).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY) from None
    if not root.is_dir():
        raise GitInventoryError(GitInventoryDiagnosticCode.INVALID_REPOSITORY)
    return root


def _validated_git_executable(value: object) -> str:
    if value is None:
        found = shutil.which("git")
        if found is None:
            raise GitInventoryError(GitInventoryDiagnosticCode.GIT_UNAVAILABLE)
        return found
    if type(value) is not str or not value:
        raise GitInventoryError(GitInventoryDiagnosticCode.GIT_UNAVAILABLE)
    return value


class _GitTimeout(Exception):
    pass


class _GitOutputLimit(Exception):
    pass


def _run_bounded_git(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int,
    env: dict[str, str] | None = None,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> bytes:
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=env if env is not None else git_environment(),
            bufsize=0,
        )
    except FileNotFoundError:
        raise GitInventoryError(GitInventoryDiagnosticCode.GIT_UNAVAILABLE) from None
    except OSError:
        raise GitInventoryError(GitInventoryDiagnosticCode.COMMAND_FAILED) from None

    selector = selectors.DefaultSelector()
    stdout = bytearray()
    deadline = time.monotonic() + timeout_seconds
    open_streams = 2
    try:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _GitTimeout
            events = selector.select(remaining)
            if not events:
                raise _GitTimeout
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), _READ_CHUNK_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    open_streams -= 1
                elif key.data == "stdout":
                    stdout.extend(chunk)
                    if len(stdout) > max_output_bytes:
                        raise _GitOutputLimit
                else:
                    # Diagnostic stderr is bounded by reading and discarding it;
                    # it is never reported or retained.
                    pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _GitTimeout
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise _GitTimeout from None
    except _GitTimeout:
        _kill(process)
        raise GitInventoryError(GitInventoryDiagnosticCode.TIMEOUT) from None
    except _GitOutputLimit:
        _kill(process)
        raise GitInventoryError(GitInventoryDiagnosticCode.OUTPUT_LIMIT) from None
    except OSError:
        _kill(process)
        raise GitInventoryError(GitInventoryDiagnosticCode.COMMAND_FAILED) from None
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    if process.returncode not in accepted_returncodes:
        raise GitInventoryError(GitInventoryDiagnosticCode.COMMAND_FAILED)
    return bytes(stdout)


def _kill(process: subprocess.Popen) -> None:
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass
