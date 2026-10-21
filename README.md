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

The graph uses LangGraph's `InMemorySaver` by default. Every run gets a
run-specific thread ID (`agentic-dev-workflow:<run_id>`). Tests pause before
the executor, read the saved checkpoint, verify its run ID and SHA-256 hash of
the specification, then resume that thread. Completed nodes are not repeated.
The resume helper rejects a changed run ID, thread ID, or specification.

In-memory checkpoints exist only in the current process and are not durable
across restarts. Phase 3A does not add SQLite or any runner-owned data files.

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
`DisabledCodexAdapter` always rejects the request. No real Codex adapter is
enabled, and the Phase 3A LangGraph nodes do not use these adapters implicitly.
The planner, mock executor, and verifier remain local deterministic nodes.

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

Before real execution, a later phase must define and review an explicitly
bounded Codex adapter, verify the selected commit and dedicated worktree,
enforce human approval and operating-system permissions, test timeout and
cancellation behavior, and preserve the existing telemetry allowlist. This
phase supplies contracts only; it does not authorize or perform execution.

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
scope. No real Codex adapter is enabled; the disabled placeholder remains
disabled. Before any real agent runs, a later phase must add explicit process
permissions/sandbox enforcement, approval gates, bounded execution and
cancellation, and actual isolated-worktree checks under the reviewed policy.

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
- There is no Codex or other agent invocation, model call, agent shell
  execution, real verification, or automated commit. The explicit Phase 3B-2
  worktree command is the only added Git mutation; it operates on the
  explicitly selected disposable source and run-owned worktree only.
- `run` remains a dry-run/preflight command only. Future implementation must
  require an explicitly selected base commit and use a separate worktree.
- `doctor` checks local tools and endpoint reachability; it does not verify
  model access, account entitlements, license terms, or API billing.
- Jaeger all-in-one uses volatile local storage. Traces are for local
  development, not durable retention.
- The standalone `trace_smoke_test.py` remains available as an exporter check.
