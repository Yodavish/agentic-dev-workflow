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
```

`workflow-demo` does not load repository configuration and never accesses the
Dashboard path. It prints a concise JSON summary and exits 0 on simulated
success, 1 on simulated verification failure, and 2 for invalid options.
`--no-tracing` suppresses telemetry export. If enabled, graph and node spans
use the `agentic-dev-workflow` service and include run ID, phase, node, attempt,
outcome, and mock-mode attributes. Specification text, plan content, source,
diffs, credentials, and environment values are not span attributes. Search
Jaeger for `workflow.mock-run` and inspect its child spans.

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
- There is no Codex or other agent invocation, model call, shell execution,
  file mutation, Git mutation, real verification, or automated commit.
- `run` remains a dry-run/preflight command only. Future implementation must
  require an explicitly selected base commit and use a separate worktree.
- `doctor` checks local tools and endpoint reachability; it does not verify
  model access, account entitlements, license terms, or API billing.
- Jaeger all-in-one uses volatile local storage. Traces are for local
  development, not durable retention.
- The standalone `trace_smoke_test.py` remains available as an exporter check.
