"""Code-owned verification checks for this repository.

Agent data can name a check ID, but it cannot provide command configuration.
The absolute executables below are inside the future reviewed runner image;
they are not host paths and are never resolved through the host ``PATH``.
No execution backend is configured by this module.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

RUNNER_RUNTIME_PROFILE_ID = "runner-verification-v1"
MAX_INVENTORY_ENTRIES = 100_000
MAX_PATH_BYTES = 4096


class CheckId(StrEnum):
    """Verification checks approved for the agentic-dev-workflow repository."""

    RUNNER_TESTS = "runner.tests"
    RUNNER_COMPILE = "runner.compile"
    RUNNER_DIFF_CHECK = "runner.diff-check"
    RUNNER_LOCK_CHECK = "runner.lock-check"


class CheckApplicability(StrEnum):
    ALWAYS = "always"
    DEPENDENCY_FILES_CHANGED = "dependency_files_changed"


class CheckPathChangeKind(StrEnum):
    """Path-change kinds supplied by a future complete Git inventory."""

    ADDED = "added"
    DELETED = "deleted"
    CHANGED = "changed"
    RENAMED = "renamed"
    COPIED = "copied"
    UNTRACKED = "untracked"


class VerificationRegistryError(ValueError):
    """Fixed diagnostic for an unregistered or modified check definition."""

    def __init__(self) -> None:
        super().__init__("verification check is not registered")


class IncompletePathInventoryError(ValueError):
    """Lock applicability cannot be evaluated from an incomplete inventory."""

    def __init__(self) -> None:
        super().__init__("dependency check applicability requires a complete path inventory")


@dataclass(frozen=True)
class VerificationCheckDefinition:
    """Immutable command and policy for one runner-owned check."""

    check_id: CheckId
    commands: tuple[tuple[str, ...], ...]
    timeout_seconds: int
    max_output_bytes: int
    applicability: CheckApplicability
    working_directory: str
    environment_profile: str
    runtime_profile_id: str
    workspace_read_only: bool
    network_access: bool

    def __post_init__(self) -> None:
        if (
            type(self.check_id) is not CheckId
            or type(self.commands) is not tuple
            or not self.commands
            or any(
                type(command) is not tuple
                or not command
                or any(
                    not isinstance(argument, str)
                    or not argument
                    or "\x00" in argument
                    for argument in command
                )
                or not command[0].startswith("/")
                for command in self.commands
            )
            or not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
            or not isinstance(self.max_output_bytes, int)
            or isinstance(self.max_output_bytes, bool)
            or self.max_output_bytes <= 0
            or type(self.applicability) is not CheckApplicability
            or self.working_directory != "/workspace"
            or self.environment_profile != "runner-checks-v1"
            or self.runtime_profile_id != RUNNER_RUNTIME_PROFILE_ID
            or self.workspace_read_only is not True
            or self.network_access is not False
        ):
            raise ValueError("verification check definition is invalid")


@dataclass(frozen=True)
class CheckPathChange:
    """One path change; rename and copy entries retain both raw-byte endpoints."""

    kind: CheckPathChangeKind
    path: bytes
    previous_path: bytes | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", CheckPathChangeKind(self.kind))
        except (TypeError, ValueError) as error:
            raise ValueError("path inventory entry has an invalid change kind") from error
        if (
            type(self.path) is not bytes
            or not self.path
            or len(self.path) > MAX_PATH_BYTES
            or b"\x00" in self.path
        ):
            raise ValueError("path inventory entry is invalid")
        if self.kind in (CheckPathChangeKind.RENAMED, CheckPathChangeKind.COPIED):
            if (
                type(self.previous_path) is not bytes
                or not self.previous_path
                or len(self.previous_path) > MAX_PATH_BYTES
                or b"\x00" in self.previous_path
            ):
                raise ValueError("rename or copy inventory entry is invalid")
        elif self.previous_path is not None:
            raise ValueError("only a rename or copy entry may contain a previous path")


@dataclass(frozen=True, slots=True)
class _CompleteInventoryProvenance:
    """Opaque capability for one live complete path inventory."""

    owner_ref: weakref.ReferenceType
    digest: bytes
    signature: bytes
    repository_root: str | None


_CAPABILITY_DOMAIN = b"agentic-dev-workflow/complete-path-inventory-capability/v1\x00"
_SNAPSHOT_DOMAIN = b"agentic-dev-workflow/complete-path-inventory-snapshot/v1"
_CAPABILITY_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class CompletePathInventory:
    """Issued path snapshot consumed by pure check applicability.

    Public construction is rejected. Applicability trusts an inventory only
    when its issuance capability matches this object and its entries. A
    repository-bound inventory also requires the same canonical root. This is
    an in-process integrity check, not protection against unrestricted code
    running in this process.
    """

    entries: tuple[CheckPathChange, ...]
    complete: bool
    _provenance: _CompleteInventoryProvenance | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, entries: tuple[CheckPathChange, ...], complete: bool) -> None:
        raise ValueError("path inventory is invalid")


def _update_entries_digest(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _entries_digest(entries: tuple[CheckPathChange, ...]) -> bytes:
    """Bind an issued inventory to every path-change field."""
    if type(entries) is not tuple:
        raise ValueError("path inventory is invalid")
    digest = hashlib.sha256()
    digest.update(_SNAPSHOT_DOMAIN)
    digest.update(len(entries).to_bytes(8, "big"))
    for entry in entries:
        if type(entry) is not CheckPathChange:
            raise ValueError("path inventory is invalid")
        kind = CheckPathChangeKind(entry.kind)
        if type(entry.path) is not bytes:
            raise ValueError("path inventory is invalid")
        _update_entries_digest(digest, kind.value.encode("ascii"))
        _update_entries_digest(digest, entry.path)
        if entry.previous_path is None:
            digest.update(b"\x00")
        else:
            if type(entry.previous_path) is not bytes:
                raise ValueError("path inventory is invalid")
            digest.update(b"\x01")
            _update_entries_digest(digest, entry.previous_path)
    return digest.digest()


def _capability_signature(
    owner: CompletePathInventory,
    digest: bytes,
    repository_root: str | None,
) -> bytes:
    """Authenticate one capability for one live inventory object."""
    if repository_root is None:
        root_bytes = b"\x00"
    else:
        encoded = repository_root.encode("utf-8")
        root_bytes = b"\x01" + len(encoded).to_bytes(8, "big") + encoded
    message = _CAPABILITY_DOMAIN + id(owner).to_bytes(8, "big") + digest + root_bytes
    return hmac.new(_CAPABILITY_KEY, message, hashlib.sha256).digest()


def _canonical_repository_root(repository_root: str | os.PathLike[str]) -> str:
    try:
        resolved = Path(repository_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ValueError("path inventory is invalid") from None
    if not resolved.is_dir():
        raise ValueError("path inventory is invalid")
    return resolved.as_posix()


def _has_issued_provenance(inventory: CompletePathInventory) -> bool:
    """Verify capability origin, owner identity, and the exact entries."""
    if type(inventory) is not CompletePathInventory or inventory.complete is not True:
        return False
    capability = getattr(inventory, "_provenance", None)
    if type(capability) is not _CompleteInventoryProvenance:
        return False
    try:
        owner_ref = capability.owner_ref
        digest = capability.digest
        signature = capability.signature
        repository_root = capability.repository_root
    except AttributeError:
        return False
    if type(owner_ref) is not weakref.ReferenceType or owner_ref() is not inventory:
        return False
    if type(digest) is not bytes or type(signature) is not bytes:
        return False
    if repository_root is not None and (type(repository_root) is not str or not repository_root):
        return False
    try:
        current = _entries_digest(inventory.entries)
        expected = _capability_signature(inventory, current, repository_root)
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False
    return _constant_equal(digest, current) and _constant_equal(signature, expected)


def _constant_equal(left: bytes, right: bytes) -> bool:
    if type(left) is not bytes or type(right) is not bytes or len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _issue_complete_path_inventory(
    entries: tuple[CheckPathChange, ...],
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> CompletePathInventory:
    """Issue a complete inventory only from trusted module paths."""
    inventory = object.__new__(CompletePathInventory)
    if type(entries) is not tuple:
        raise ValueError("path inventory is invalid")
    if len(entries) > MAX_INVENTORY_ENTRIES or any(
        type(entry) is not CheckPathChange for entry in entries
    ):
        raise ValueError("path inventory is invalid")
    bound_root = (
        None if repository_root is None else _canonical_repository_root(repository_root)
    )
    object.__setattr__(inventory, "entries", entries)
    object.__setattr__(inventory, "complete", True)
    try:
        digest = _entries_digest(entries)
        signature = _capability_signature(inventory, digest, bound_root)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("path inventory is invalid") from None
    object.__setattr__(
        inventory,
        "_provenance",
        _CompleteInventoryProvenance(
            owner_ref=weakref.ref(inventory),
            digest=digest,
            signature=signature,
            repository_root=bound_root,
        ),
    )
    return inventory


class VerificationCheckIdError(VerificationRegistryError):
    """Unknown check ID (message intentionally omits the supplied value)."""


_ENVIRONMENT_PROFILE: Mapping[str, str] = MappingProxyType(
    {
        "HOME": "/tmp/home",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/pycache",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "UV_OFFLINE": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
    }
)

_DEFINITIONS: Mapping[CheckId, VerificationCheckDefinition] = MappingProxyType(
    {
        CheckId.RUNNER_TESTS: VerificationCheckDefinition(
            check_id=CheckId.RUNNER_TESTS,
            commands=(
                (
                    "/usr/local/bin/python",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-v",
                ),
            ),
            timeout_seconds=300,
            max_output_bytes=512 * 1024,
            applicability=CheckApplicability.ALWAYS,
            working_directory="/workspace",
            environment_profile="runner-checks-v1",
            runtime_profile_id=RUNNER_RUNTIME_PROFILE_ID,
            workspace_read_only=True,
            network_access=False,
        ),
        CheckId.RUNNER_COMPILE: VerificationCheckDefinition(
            check_id=CheckId.RUNNER_COMPILE,
            commands=(
                ("/usr/local/bin/python", "-m", "compileall", "-q", "src", "tests"),
            ),
            timeout_seconds=120,
            max_output_bytes=128 * 1024,
            applicability=CheckApplicability.ALWAYS,
            working_directory="/workspace",
            environment_profile="runner-checks-v1",
            runtime_profile_id=RUNNER_RUNTIME_PROFILE_ID,
            workspace_read_only=True,
            network_access=False,
        ),
        CheckId.RUNNER_DIFF_CHECK: VerificationCheckDefinition(
            check_id=CheckId.RUNNER_DIFF_CHECK,
            commands=(
                (
                    "/usr/bin/git",
                    "--no-pager",
                    "-c",
                    "core.pager=cat",
                    "-c",
                    "color.ui=false",
                    "diff",
                    "--check",
                    "--no-ext-diff",
                    "--no-textconv",
                ),
                (
                    "/usr/bin/git",
                    "--no-pager",
                    "-c",
                    "core.pager=cat",
                    "-c",
                    "color.ui=false",
                    "diff",
                    "--cached",
                    "--check",
                    "--no-ext-diff",
                    "--no-textconv",
                ),
            ),
            timeout_seconds=60,
            max_output_bytes=256 * 1024,
            applicability=CheckApplicability.ALWAYS,
            working_directory="/workspace",
            environment_profile="runner-checks-v1",
            runtime_profile_id=RUNNER_RUNTIME_PROFILE_ID,
            workspace_read_only=True,
            network_access=False,
        ),
        CheckId.RUNNER_LOCK_CHECK: VerificationCheckDefinition(
            check_id=CheckId.RUNNER_LOCK_CHECK,
            commands=(("/usr/local/bin/uv", "lock", "--check", "--offline"),),
            timeout_seconds=180,
            max_output_bytes=256 * 1024,
            applicability=CheckApplicability.DEPENDENCY_FILES_CHANGED,
            working_directory="/workspace",
            environment_profile="runner-checks-v1",
            runtime_profile_id=RUNNER_RUNTIME_PROFILE_ID,
            workspace_read_only=True,
            network_access=False,
        ),
    }
)
_DEPENDENCY_PATHS = frozenset({b"pyproject.toml", b"uv.lock", b"uv.toml"})


def get_check_definition(check_id: CheckId | str) -> VerificationCheckDefinition:
    """Resolve an allowed ID to its canonical immutable definition."""
    try:
        selected = CheckId(check_id)
    except (TypeError, ValueError) as error:
        raise VerificationCheckIdError() from None
    return _DEFINITIONS[selected]


def validate_registered_check(
    definition: VerificationCheckDefinition,
) -> VerificationCheckDefinition:
    """Accept only the exact registry object, not a caller-built equivalent."""
    if type(definition) is not VerificationCheckDefinition:
        raise VerificationRegistryError()
    canonical = _DEFINITIONS.get(definition.check_id)
    if canonical is not definition:
        raise VerificationRegistryError()
    return canonical


def lock_check_applies(
    inventory: CompletePathInventory,
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether dependency files changed; reject incomplete snapshots.

    ``entries`` must already cover added, deleted, modified, and untracked
    paths from staged and unstaged state. A rename or copy entry contributes
    both its current path and previous path to this decision. A
    repository-bound inventory is accepted only for the canonical root stored
    at issuance. An unbound inventory rejects ``repository_root``.
    """
    if type(inventory) is not CompletePathInventory or inventory.complete is not True:
        raise IncompletePathInventoryError()
    if not _has_issued_provenance(inventory):
        raise IncompletePathInventoryError()
    bound_root = inventory._provenance.repository_root
    if bound_root is None:
        if repository_root is not None:
            raise IncompletePathInventoryError()
    else:
        if repository_root is None:
            raise IncompletePathInventoryError()
        try:
            expected_root = _canonical_repository_root(repository_root)
        except ValueError:
            raise IncompletePathInventoryError() from None
        if expected_root != bound_root:
            raise IncompletePathInventoryError()
    if type(inventory.entries) is not tuple or any(
        type(entry) is not CheckPathChange for entry in inventory.entries
    ):
        raise IncompletePathInventoryError()
    changed_paths: set[bytes] = set()
    for entry in inventory.entries:
        changed_paths.add(entry.path)
        if entry.kind in (CheckPathChangeKind.RENAMED, CheckPathChangeKind.COPIED):
            assert entry.previous_path is not None
            changed_paths.add(entry.previous_path)
    return bool(changed_paths & _DEPENDENCY_PATHS)


def create_sandbox_environment(
    definition: VerificationCheckDefinition,
) -> Mapping[str, str]:
    """Return the fixed environment profile without reading host variables."""
    validate_registered_check(definition)
    if definition.environment_profile != "runner-checks-v1":
        raise VerificationRegistryError()
    return _ENVIRONMENT_PROFILE
