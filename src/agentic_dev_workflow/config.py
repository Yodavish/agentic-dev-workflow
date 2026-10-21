"""Validated configuration for the local workflow runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_REPOSITORY = "/home/dvit/projects/politician-dashboard"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"
REPOSITORY_ENV = "AGENTIC_DEV_WORKFLOW_REPOSITORY"
OTLP_ENDPOINT_ENV = "AGENTIC_DEV_WORKFLOW_OTLP_ENDPOINT"
CODEX_ENABLED_ENV = "AGENTIC_DEV_WORKFLOW_CODEX_ENABLED"
CODEX_EXECUTABLE_ENV = "AGENTIC_DEV_WORKFLOW_CODEX_EXECUTABLE"
DATA_DIR_ENV = "AGENTIC_DEV_WORKFLOW_DATA_DIR"


class ConfigurationError(ValueError):
    """Raised when a configured value is invalid."""


@dataclass(frozen=True)
class Configuration:
    """Resolved repository, telemetry, and runner-owned data settings."""

    repository: Path
    otlp_endpoint: str
    data_dir: Path
    codex_enabled: bool = False
    codex_executable: Path | None = None

    @classmethod
    def from_values(
        cls,
        repository: str | Path | None = None,
        otlp_endpoint: str | None = None,
        codex_enabled: bool | str | None = None,
        codex_executable: str | Path | None = None,
        data_dir: str | Path | None = None,
        *,
        environ: dict[str, str] | None = None,
    ) -> Configuration:
        """Resolve CLI values, environment overrides, and documented defaults."""
        environment = os.environ if environ is None else environ
        repository_value = (
            repository if repository is not None
            else environment.get(REPOSITORY_ENV, DEFAULT_REPOSITORY)
        )
        endpoint_value = (
            otlp_endpoint if otlp_endpoint is not None
            else environment.get(OTLP_ENDPOINT_ENV, DEFAULT_OTLP_ENDPOINT)
        )
        data_dir_value = (
            data_dir
            if data_dir is not None
            else environment.get(DATA_DIR_ENV)
        )
        codex_value = (
            codex_enabled
            if codex_enabled is not None
            else environment.get(CODEX_ENABLED_ENV, "false")
        )
        executable_value = (
            codex_executable
            if codex_executable is not None
            else environment.get(CODEX_EXECUTABLE_ENV)
        )
        if not str(repository_value).strip():
            raise ConfigurationError("repository path must not be empty")
        if not endpoint_value.strip():
            raise ConfigurationError("OTLP endpoint must not be empty")
        if isinstance(codex_value, bool):
            codex_is_enabled = codex_value
        elif isinstance(codex_value, str) and codex_value.strip().lower() in {"true", "false"}:
            codex_is_enabled = codex_value.strip().lower() == "true"
        else:
            raise ConfigurationError(
                f"{CODEX_ENABLED_ENV} must be explicitly true or false"
            )

        executable_path: Path | None = None
        if executable_value is not None:
            if not isinstance(executable_value, (str, Path)) or not str(executable_value).strip():
                raise ConfigurationError(f"{CODEX_EXECUTABLE_ENV} must be an absolute executable path")
            raw_executable = str(executable_value)
            candidate = Path(raw_executable)
            if not candidate.is_absolute() or os.path.normpath(raw_executable) != raw_executable:
                raise ConfigurationError(
                    f"{CODEX_EXECUTABLE_ENV} must be an absolute, lexically normalized path"
                )
            executable_path = candidate

        # Normalize lexically only. Resolving here would touch the filesystem
        # even for commands (such as workflow-demo) that do not target a repo.
        path = Path(os.path.abspath(os.path.expanduser(str(repository_value))))
        resolved_data_dir = resolve_data_dir(data_dir_value, environ=environment)
        _validate_endpoint(endpoint_value)
        return cls(
            repository=path,
            otlp_endpoint=endpoint_value,
            data_dir=resolved_data_dir,
            codex_enabled=codex_is_enabled,
            codex_executable=executable_path,
        )


def resolve_data_dir(
    value: str | Path | None = None, *, environ: dict[str, str] | None = None
) -> Path:
    """Return a canonical runner-owned data path without creating it."""
    environment = os.environ if environ is None else environ
    if value is None:
        xdg_home = environment.get("XDG_DATA_HOME")
        if xdg_home is not None:
            if not xdg_home.strip() or not Path(xdg_home).expanduser().is_absolute():
                raise ConfigurationError(
                    "XDG_DATA_HOME must be a non-empty absolute path"
                )
            root = Path(xdg_home).expanduser()
        else:
            home_value = environment.get("HOME")
            home = Path(home_value).expanduser() if home_value else Path.home()
            if not home.is_absolute():
                raise ConfigurationError("home directory must resolve to an absolute path")
            root = home / ".local" / "share"
        candidate = root / "agentic-dev-workflow"
    else:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigurationError(
                f"{DATA_DIR_ENV} must be a non-empty absolute path"
            )
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ConfigurationError(f"{DATA_DIR_ENV} must be an absolute path")

    try:
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and not resolved.is_dir():
            raise ConfigurationError(f"{DATA_DIR_ENV} must point to a directory")
        if resolved == Path(resolved.anchor):
            raise ConfigurationError(f"{DATA_DIR_ENV} cannot be a filesystem root")
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError(f"{DATA_DIR_ENV} cannot be safely resolved") from None
    return resolved


def _validate_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.port != 0
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and parsed.path.endswith("/v1/traces")
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(
            "OTLP endpoint must be an HTTP(S) URL ending in /v1/traces "
            "without embedded credentials, query, or fragment"
        )
