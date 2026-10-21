# Agentic Development Workflow

A local runner foundation for a future feature implementation workflow. This
repository is separate from the Politician Dashboard checkout.

## Architecture and phase boundaries

Phase 2 provides configuration, read-only Git preflight, CLI diagnostics, and
short-lived OpenTelemetry export. Phase 3A adds a `StateGraph` to exercise
workflow routing and checkpoint behavior without any autonomous actions.
LangGraph is used for explicit state, conditional transitions, and checkpoints;
no model provider or LangSmith dependency is present.

The mocked graph is:

```text
START → planner → mock_executor → verifier → retry_decision
                         ↑                         │
                         └──── bounded retry ─────┘
```

All three work nodes are deterministic simulations. The planner returns a
fixed structured plan and does not inspect a repository. The executor changes
only graph state; it runs no commands and writes no files. The verifier derives
its simulated outcome from the selected scenario and explicitly says that no
real tests or checks ran.

`max_retries` counts repair attempts after the initial execution. It is
validated as an integer from 0 through 2, so a run can make at most three
execution attempts. A failed verification retries the mock executor while
repairs remain; otherwise the graph terminates as failed. Attempt history and
simulated failure messages remain in final state.

## Checkpointing

The Python workflow API keeps `InMemorySaver` as its default for fast tests and
explicit saver injection. `workflow-demo --durable` uses the synchronous
`SqliteSaver` from the pinned `langgraph-checkpoint-sqlite==3.1.2` package.
Its stable database filename is `checkpoints.sqlite3`; the default directory
is `${XDG_DATA_HOME:-~/.local/share}/agentic-dev-workflow`. Set
`AGENTIC_DEV_WORKFLOW_DATA_DIR` or pass `--data-dir` to select another absolute
directory; passing `--data-dir` to `workflow-demo` opts that invocation into
durable storage. The configured directory is canonicalized. Newly created
directory and database files use restrictive permissions where supported. Existing
directories or databases with broad POSIX permissions are rejected without
changing their permissions. The database and its WAL/SHM sidecars are never
deleted or reset automatically.
Canonical-path resolution and symlink/permission checks reduce accidental
misconfiguration, but they are not a complete defense against a concurrent
same-user path-swap race. Use a trusted runner-owned directory; this storage
layer does not provide race-free filesystem isolation.

Checkpoint state includes the synthetic specification, plan, run metadata,
attempt history, and errors. Treat the database as local workflow data, not a
secret store. The default location is outside application repositories. If a
custom data directory is chosen, ensure it is runner-owned and protected.

To pause after the planner checkpoint and exit before mock execution, then
resume from that checkpoint in a later process:

```bash
uv run agentic-dev-workflow workflow-demo --run-id demo-001 \
  --scenario retry-success --max-retries 1 --pause-after-planning --no-tracing

uv run agentic-dev-workflow workflow-demo --run-id demo-001 \
  --scenario retry-success --max-retries 1 --resume --no-tracing
```

Pause and resume require an explicit run ID and use SQLite automatically. The
resume operation checks the namespaced thread ID, run ID, SHA-256 specification
hash, checkpoint format version, scenario, and retry limit before running any
node. It also validates the saved plan, state types, graph continuation, and
attempt/retry history against the mock scenario. Missing checkpoints, terminal
runs, corrupt databases, and incompatible state fail without starting a fresh
run. Corrupt or incompatible files are
preserved; stop runner processes before inspecting or backing up the database.
Use SQLite integrity checks or a SQLite backup while the runner is stopped;
avoid selecting serialized checkpoint blobs because they contain workflow
state. The saver owns its table setup and schema migrations; existing files
without the expected LangGraph tables are rejected instead of initialized.
Application state carries format version `1`, and unknown versions are rejected
before node execution. Back up the database before changing the locked saver
version. This phase does not provide a database repair or deletion command.

SQLite checkpointing initially supports serial run/resume operations. SQLite
locking and the configured five-second busy timeout bound lock waits, but this
does not provide multi-process workflow coordination. Do not run two commands
against the same checkpoint database concurrently. Checkpoints persist graph
state and completed-node boundaries; they do not guarantee exactly-once
external side effects if a future node acts externally and the process stops
before its checkpoint is committed.

## Agent adapter contracts (Phase 3B-1)

`adapters` defines typed requests and results for three roles: planner,
executor, and reviewer. Requests carry a run ID, role, task text, structured
context, bounded timeout (1–600 seconds), and correlation metadata. They have
no executable path, shell, argument-vector, or environment fields. Results
carry typed status, structured output, optional exit code, duration, a stable
failure category, and a short diagnostic code rather than unrestricted output.
Reserved process configuration keys are rejected from structured context too.

`AgentAdapter` is a provider-neutral synchronous protocol. `MockAgentAdapter`
returns configurable deterministic success or failure results without I/O.
`CodexCLIAdapter` implements a guarded subprocess adapter (Phase 3B-3), while
`DisabledCodexAdapter` remains an always-rejecting placeholder. The real
adapter is disabled by default, is not exposed through the CLI, and is not
connected to LangGraph. The Phase 3A planner, mock executor, and verifier
remain local deterministic nodes.

The safe telemetry helper emits only run ID, role, attempt, adapter type,
status, failure category, and elapsed duration. It omits task text, context,
structured output, diagnostics, and correlation metadata.

## Declared execution policy

`execution_policy` provides pure validation for a future execution request.
It requires an explicit full 40-character SHA-1 or 64-character SHA-256 hex
base commit, distinct absolute repository and worktree declarations, and a
worktree named `agentic-workflow-<run_id>`. The declaration must require a
clean baseline, explicitly authorize workspace writes, disable network access,
set a timeout from 1 through 3,600 seconds, and cap retries at 2. Automatic
commit, merge, deployment, production database access, and destructive cleanup
are rejected.

These checks validate declared strings and flags only. They do not establish
that the commit exists, that either path exists, that the paths are not
symlinks, or that the worktree is actually isolated. Later Git/filesystem
integration must verify a clean base commit and the real worktree boundary
before any write. No policy check accesses the filesystem or invokes Git.

The Codex adapter checks these declarations in Python and verifies a clean
source baseline plus the manager-owned worktree association before launch. The
worktree may become dirty after execution so a repair attempt can reuse it.
Fields forbidding commits, merges, deployment, production database access, and
destructive cleanup are validated declarations; they are not command-level
enforcement. End-to-end workflow orchestration, human approval UX, and an
OS-enforced sandbox remain future work.

## Setup and tests

Python 3.14 and `uv` are expected. Dependencies are declared in `pyproject.toml`
and locked in `uv.lock`.

```bash
uv sync --locked
uv run agentic-dev-workflow --help
uv run python -m unittest discover -s tests -v
```

The test suite uses temporary Git repositories and in-memory tracing; it does
not require Jaeger or a model.

## CLI

```bash
# Existing Phase 2 diagnostics and dry-run preflight command
uv run agentic-dev-workflow doctor
uv run agentic-dev-workflow preflight
uv run agentic-dev-workflow trace-smoke
uv run agentic-dev-workflow run

# Deterministic mock scenarios; tracing exports to local Jaeger by default
uv run agentic-dev-workflow workflow-demo --scenario success
uv run agentic-dev-workflow workflow-demo --scenario retry-success --max-retries 1
uv run agentic-dev-workflow workflow-demo --scenario exhausted-retries --max-retries 2

# Set an explicit run ID, disable export, or configure another local endpoint
uv run agentic-dev-workflow workflow-demo --scenario success --run-id demo-001 --no-tracing
uv run agentic-dev-workflow workflow-demo --otlp-endpoint http://localhost:4318/v1/traces

# Durable mock run; pause/resume as shown above
uv run agentic-dev-workflow workflow-demo --durable --run-id durable-demo

# Select a repository explicitly for read-only preflight
uv run agentic-dev-workflow preflight --repository /path/to/repository

# Verify an explicitly selected Git baseline (read-only)
uv run agentic-dev-workflow baseline-check \
  --repository /path/to/disposable-repository \
  --base-commit <full-40-or-64-character-commit-id>

# Explicitly create, then explicitly clean up, a managed worktree
uv run agentic-dev-workflow worktree --action create \
  --source-repository /path/to/disposable-repository \
  --base-commit <full-commit-id> --run-id demo-001 \
  --worktree-root /path/to/runner-owned/worktrees
uv run agentic-dev-workflow worktree --action cleanup \
  --source-repository /path/to/disposable-repository \
  --base-commit <full-commit-id> --run-id demo-001 \
  --worktree-root /path/to/runner-owned/worktrees
```

`workflow-demo` does not load repository configuration and never accesses the
Dashboard path. It prints a concise JSON summary and exits 0 on simulated
success, 1 on simulated verification failure, and 2 for invalid options.
`--no-tracing` suppresses telemetry export. If enabled, graph and node spans
use the `agentic-dev-workflow` service and include run ID, phase, node, attempt,
outcome, and mock-mode attributes. Specification text, plan content, source,
diffs, credentials, and environment values are not span attributes. Search
Jaeger for `workflow.mock-run` and inspect its child spans.

## Controlled Codex CLI adapter (Phase 3B-3)

The installed CLI was inspected locally and reported `codex-cli 0.163.0`.
Its `codex exec --help` confirms non-interactive execution, `--cd`,
`--sandbox read-only|workspace-write`, `--ask-for-approval on-request`,
`--json`, and `--output-schema`. It also supports `--oss --local-provider
ollama`. This adapter pins runtime acceptance to that exact version and fails
closed for missing, unlaunchable, or different versions. Recheck the installed
help and review the adapter before upgrading Codex.

Every adapter run uses local Ollama mode, an isolated temporary Codex home,
`--ignore-user-config`, `--ephemeral`, no web search, and a minimal child
environment that excludes caller credentials and API-key variables. Set
`AGENTIC_DEV_WORKFLOW_CODEX_EXECUTABLE` to an absolute, normalized executable
path; the adapter does not search ambient `PATH` for Codex. Planner
and reviewer roles use the `read-only` sandbox; executor uses `workspace-write`.
Every role requests `on-request` approval. It never uses bypass flags, adds
directories, or accepts an executable, working directory, sandbox, approval
setting, or command arguments from the request. The request prompt is sent on
stdin and is not written to logs or telemetry. An absolute path and matching
version string do not authenticate the binary: the configured file, its
interpreter/runtime, and command resolution through the child `PATH` remain
local trust assumptions.

The CLI help does not provide an execution timeout option. The Python wrapper
uses one deadline for version probing and agent execution, with bounded
termination and stream-cleanup grace. It terminates the Codex process group on
POSIX; on Windows it fails closed because this implementation cannot verify
process-tree cleanup. POSIX cleanup kills descendants that remain in the
Codex process group; a process that deliberately creates a new session can
escape that group, so this is not a substitute for an OS sandbox. It caps
combined stdout/stderr capture at 1 MiB by default
(constructor limits: 1 KiB through 16 MiB). It parses JSONL completion events
and validates a bounded Codex-specific response with exactly `summary` and
`items` fields. The structured message is limited to 64 KiB; summaries to 2,048
characters; and items to 64 strings of at most 512 characters each. The
generic `AgentResult.structured_output` mapping contract is unchanged, but the
Codex response shape is intentionally narrower than the earlier arbitrary
object. The schema constrains shape and size only; it does not detect secrets.
Returned structured values are model-generated, untrusted, and potentially
sensitive. Full JSONL output and stderr are not returned; rejected-output
diagnostics and telemetry omit their values.

Executor runs use strict CLI configuration overrides to disable workspace
network access, set additional writable roots to an empty list, and exclude
`/tmp` and `TMPDIR` from workspace writes. Startup update checks and Codex
analytics are also disabled. These CLI sandbox settings constrain Codex's
execution environment; they are not a complete host security boundary. The
sandbox and analytics keys are listed in the
[Codex configuration reference](https://developers.openai.com/codex/config-reference).

The prompt asks Codex not to commit, merge, deploy, access production databases,
or perform destructive cleanup. These instructions are advisory. Python
validates that corresponding policy flags are false, but this phase does not
independently block those actions. `on-request` approval and the CLI sandbox
remain Codex controls, not a separate runner approval UI or OS sandbox.

Codex execution requires `AGENTIC_DEV_WORKFLOW_CODEX_ENABLED=true` through
validated configuration and direct construction of `CodexCLIAdapter` with an
`ExecutionPolicy` and `WorktreeManager`. The adapter then verifies the clean
source baseline, manager ownership record, linked Git worktree, selected
commit, and run ID before launch. Worktree checks are repeated after the
version probe, but are not race-free against a concurrent same-user process.
The manifest and marker prevent accidental reuse, not a malicious same-user
process. Strong isolation requires an OS sandbox or stronger architecture.
No existing CLI command or LangGraph node
constructs it, so opt-in configuration alone does not start an agent.

Tests exercise process handling with a fake Codex executable and temporary
Git fixtures only. They do not invoke the installed Codex agent or call a
model. The installed adapter is deliberately not run against an application
repository in this phase. Local Ollama inference avoids a paid model API, but
requires an available local Ollama service and compatible model; hardware and
electricity costs still apply. The Codex sandbox and Git worktree are not a
complete OS security sandbox, and configured Git filters remain a concern.

## Git baseline verification and worktrees (Phase 3B-2)

`baseline-check` requires an explicit source repository and full commit ID. It
resolves the repository and selected commit, reports the current HEAD and
staged, unstaged, and untracked paths, and rejects a dirty source by default.
It is read-only. Git calls use argument lists, an explicit working directory,
bounded timeouts, and bounded diagnostic output; command output is not printed.
Git subprocesses also ignore inherited `GIT_DIR`/worktree/index overrides and
disable optional index refreshes.

The `worktree` command requires an explicit `create` or `cleanup` action, source
repository, full base commit, safe run ID, and absolute runner-owned root. It
generates the destination itself as `agentic-workflow-<run-id>`, refuses dirty
baselines and existing/unowned destinations, and verifies the detached HEAD,
Git common directory, registration, and clean status. Cleanup requires the
root ownership marker, a matching per-run record, Git registration, and a
clean worktree; it never force-removes or recursively deletes a caller path.
Failed creation is reported with partial-state information and is not
automatically cleaned up.

Path validation and ownership verification are distinct: absolute/canonical
path checks identify where a destination resolves, while the manager record
and Git's worktree registry establish that it was registered for the given
source and run. These local records protect against accidental reuse, not a
hostile user who can rewrite both records. Creating/removing a linked worktree
necessarily changes Git administrative metadata in the source repository's
common Git directory; this runner verifies that source files, index, HEAD,
refs, and status remain unchanged/clean. It does not claim that a Git worktree
is an OS security sandbox.
Git checkout can still invoke repository-configured filters. The worktree
manager disables Git hooks, but callers should use only trusted repositories
until those checkout operations run under a reviewed OS sandbox.

All integration tests create temporary Git repositories. No real user project
is used by these tests. The Dashboard is outside this phase's code and test
scope. Phase 3B-2 had no real Codex adapter; Phase 3B-3 adds one that remains
opt-in and disconnected from the CLI and LangGraph. Before autonomous workflow
execution, a later phase must add explicit approval UX, independent OS sandbox
enforcement, bounded orchestration/retries, and review of worktree changes.

## Jaeger

The existing local Jaeger v2.22.0 setup can be started with:

```bash
docker run --rm --name agentic-jaeger \
  -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.22.0
```

Open <http://localhost:16686>. The default trace endpoint is
`http://localhost:4318/v1/traces`. Use `--no-tracing` for an offline demo.

## Safety and limitations

- Phase 3A does not access, inspect, or modify the Dashboard repository.
- Existing CLI commands and LangGraph workflows do not invoke Codex or a model.
  The separate adapter can only run after explicit validated opt-in and
  direct construction; this phase's tests use a fake executable and no model.
  No agent shell task or real verification is wired into autonomous workflow
  execution, and there is no automated commit. The Phase 3B-2 worktree command
  remains the only CLI Git mutation.
- `run` remains a dry-run/preflight command only. Future implementation must
  require an explicitly selected base commit and use a separate worktree.
- `doctor` checks local tools and endpoint reachability; it does not verify
  model access, account entitlements, license terms, or API billing.
- Jaeger all-in-one uses volatile local storage. Traces are for local
  development, not durable retention.
- The standalone `trace_smoke_test.py` remains available as an exporter check.
