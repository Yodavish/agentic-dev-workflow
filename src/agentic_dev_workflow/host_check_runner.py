"""Run registered verification commands on the host and keep their real exit codes.

This is not an operating-system sandbox and it does not use the runner image.
Registry commands name image paths such as ``/usr/local/bin/python`` and a
``/workspace`` directory. When that interpreter is absent, the same argument
vector runs with the current Python. ``/workspace`` is the verified worktree.
A passing result is only the process exit code 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentic_dev_workflow.sandbox import (
    MAX_IN_MEMORY_EVIDENCE_BYTES,
    MAX_PROCESS_OUTPUT_CHUNK_BYTES,
    SandboxCheckResult,
    SandboxCleanupStatus,
    SandboxDiagnosticCode,
    SandboxFailureCategory,
    SandboxStatus,
    SandboxTermination,
)
from agentic_dev_workflow.verification_registry import (
    RUNNER_RUNTIME_PROFILE_ID,
    VerificationCheckDefinition,
    create_sandbox_environment,
)

_IMAGE_PYTHON = "/usr/local/bin/python"
_IMAGE_WORKSPACE = "/workspace"


@dataclass(frozen=True)
class HostCommandRecord:
    """One real process invocation behind a registered check."""

    check_id: str
    attempt: int
    registry_command: tuple[str, ...]
    executed_command: tuple[str, ...]
    exit_code: int | None
    duration_seconds: float
    stdout: bytes
    stderr: bytes


@dataclass
class HostProcessCheckRunner:
    """Host process runner for registry checks. It cannot report a scripted pass."""

    runtime_profile_id: str = RUNNER_RUNTIME_PROFILE_ID
    invocations: list[HostCommandRecord] = field(default_factory=list)

    def run(
        self,
        check: VerificationCheckDefinition,
        worktree,
        *,
        run_id: str,
        attempt: int,
    ) -> SandboxCheckResult:
        started = time.monotonic()
        workspace = Path(worktree.path)
        if check.working_directory != _IMAGE_WORKSPACE or not workspace.is_dir():
            return self._error(
                run_id,
                check,
                attempt,
                SandboxFailureCategory.LAUNCH_FAILURE,
                SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED,
                SandboxTermination.LAUNCH_FAILURE,
                time.monotonic() - started,
            )
        _prepare_runtime_dirs()
        environment = dict(create_sandbox_environment(check))
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        observed = 0
        overflow = False
        timed_out = False
        launch_failed = False
        exit_code: int | None = None
        commands = [
            (command, executed)
            for command in check.commands
            for executed in _executed_commands(command, workspace)
        ]
        for command, executed in commands:
            if overflow or timed_out or launch_failed:
                break
            command_started = time.monotonic()
            try:
                completed = subprocess.run(
                    executed,
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=check.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                timed_out = True
                command_stdout = error.stdout or b""
                command_stderr = error.stderr or b""
                command_exit = None
            except OSError as error:
                launch_failed = True
                command_stdout = b""
                command_stderr = str(error.errno or "").encode("ascii", "replace")
                command_exit = None
            else:
                command_stdout = completed.stdout or b""
                command_stderr = completed.stderr or b""
                command_exit = completed.returncode
            duration = time.monotonic() - command_started
            stdout_parts.append(command_stdout)
            stderr_parts.append(command_stderr)
            observed += len(command_stdout) + len(command_stderr)
            exit_code = command_exit
            self.invocations.append(
                HostCommandRecord(
                    check_id=check.check_id.value,
                    attempt=attempt,
                    registry_command=tuple(command),
                    executed_command=tuple(executed),
                    exit_code=command_exit,
                    duration_seconds=duration,
                    stdout=command_stdout,
                    stderr=command_stderr,
                )
            )
            if observed > check.max_output_bytes:
                overflow = True
            elif command_exit not in (0, None):
                break
        duration = time.monotonic() - started
        stdout = b"".join(stdout_parts)
        stderr = b"".join(stderr_parts)
        evidence_stdout, evidence_stderr = _evidence(stdout, stderr)
        reported = min(observed, check.max_output_bytes + MAX_PROCESS_OUTPUT_CHUNK_BYTES)
        if launch_failed:
            return self._error(
                run_id,
                check,
                attempt,
                SandboxFailureCategory.LAUNCH_FAILURE,
                SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED,
                SandboxTermination.LAUNCH_FAILURE,
                duration,
                stdout=evidence_stdout,
                stderr=evidence_stderr,
                observed=reported,
            )
        if timed_out:
            return self._error(
                run_id,
                check,
                attempt,
                SandboxFailureCategory.TIMEOUT,
                SandboxDiagnosticCode.CHECK_TIMED_OUT,
                SandboxTermination.TIMEOUT,
                duration,
                stdout=evidence_stdout,
                stderr=evidence_stderr,
                observed=reported,
            )
        if overflow:
            return self._error(
                run_id,
                check,
                attempt,
                SandboxFailureCategory.OUTPUT_LIMIT,
                SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                SandboxTermination.OUTPUT_LIMIT,
                duration,
                stdout=evidence_stdout,
                stderr=evidence_stderr,
                observed=max(reported, check.max_output_bytes + 1),
            )
        if exit_code == 0:
            return SandboxCheckResult(
                run_id=run_id,
                check_id=check.check_id,
                attempt=attempt,
                status=SandboxStatus.PASSED,
                exit_code=0,
                duration_seconds=duration,
                output_bytes_observed=observed,
                termination=SandboxTermination.NATURAL_EXIT,
                stdout_evidence=evidence_stdout,
                stderr_evidence=evidence_stderr,
                cleanup_status=SandboxCleanupStatus.SUCCEEDED,
                backend_type="host-process",
            )
        return SandboxCheckResult(
            run_id=run_id,
            check_id=check.check_id,
            attempt=attempt,
            status=SandboxStatus.FAILED,
            exit_code=exit_code if isinstance(exit_code, int) and exit_code != 0 else 1,
            duration_seconds=duration,
            output_bytes_observed=observed,
            termination=SandboxTermination.NATURAL_EXIT,
            stdout_evidence=evidence_stdout,
            stderr_evidence=evidence_stderr,
            cleanup_status=SandboxCleanupStatus.SUCCEEDED,
            failure_category=SandboxFailureCategory.NONZERO_EXIT,
            backend_type="host-process",
        )

    def _error(
        self,
        run_id: str,
        check: VerificationCheckDefinition,
        attempt: int,
        category: SandboxFailureCategory,
        diagnostic: SandboxDiagnosticCode,
        termination: SandboxTermination,
        duration: float,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        observed: int = 0,
    ) -> SandboxCheckResult:
        return SandboxCheckResult(
            run_id=run_id,
            check_id=check.check_id,
            attempt=attempt,
            status=SandboxStatus.ERROR,
            exit_code=None,
            duration_seconds=duration,
            output_bytes_observed=observed,
            termination=termination,
            stdout_evidence=stdout,
            stderr_evidence=stderr,
            cleanup_status=SandboxCleanupStatus.NOT_REQUIRED,
            failure_category=category,
            diagnostic_code=diagnostic,
            backend_type="host-process",
        )


def _executed_commands(
    command: tuple[str, ...], workspace: Path
) -> tuple[tuple[str, ...], ...]:
    argv = list(command)
    if argv and argv[0] == _IMAGE_PYTHON and not os.access(_IMAGE_PYTHON, os.X_OK):
        argv[0] = sys.executable
    if argv[1:6] != ["-m", "unittest", "discover", "-s", "tests"]:
        return (tuple(argv),)
    # Registry discovery starts at tests/. A root module such as
    # test_transaction_actions.py is visible only when discovery starts at ".".
    # Keep the tests/ start when that directory exists so packaged and
    # unpackaged tests there still run.
    selected: list[tuple[str, ...]] = []
    if (workspace / "tests").is_dir():
        selected.append(tuple(argv))
    if any(workspace.glob("test*.py")) or not selected:
        rewritten = list(argv)
        rewritten[5] = "."
        selected.append(tuple(rewritten))
    return tuple(selected)


def _prepare_runtime_dirs() -> None:
    for relative in ("/tmp/home", "/tmp/pycache", "/tmp/uv-cache"):
        Path(relative).mkdir(parents=True, exist_ok=True)


def _evidence(stdout: bytes, stderr: bytes) -> tuple[bytes, bytes]:
    budget = MAX_IN_MEMORY_EVIDENCE_BYTES
    kept_stdout = stdout[:budget]
    kept_stderr = stderr[: max(0, budget - len(kept_stdout))]
    return kept_stdout, kept_stderr
