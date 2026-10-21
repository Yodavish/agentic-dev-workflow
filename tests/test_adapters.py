import unittest

from agentic_dev_workflow.adapters import (
    MAX_AGENT_TIMEOUT_SECONDS,
    AdapterError,
    AgentFailureCategory,
    AgentRequest,
    AgentResult,
    AgentRole,
    AgentStatus,
    DisabledCodexAdapter,
    MockAgentAdapter,
    invoke_adapter,
    safe_telemetry_attributes,
)


def request(**changes):
    values = {
        "run_id": "run-123",
        "role": AgentRole.EXECUTOR,
        "task": "Implement the synthetic feature",
        "structured_context": {"spec": "private task details"},
        "timeout_seconds": 30,
        "correlation_metadata": {"trace_key": "corr-7"},
    }
    values.update(changes)
    return AgentRequest(**values)


class _InvalidAdapter:
    adapter_type = "invalid-test-double"

    def invoke(self, agent_request):
        return {"status": "success"}


class AgentAdapterTests(unittest.TestCase):
    def test_valid_request_accepts_typed_role_and_structured_context(self) -> None:
        agent_request = request()
        self.assertEqual(agent_request.role, AgentRole.EXECUTOR)
        self.assertEqual(agent_request.timeout_seconds, 30)

    def test_request_rejects_invalid_run_id_and_role(self) -> None:
        with self.assertRaises(ValueError):
            request(run_id="bad run id")
        with self.assertRaises(ValueError):
            request(role="codex")

    def test_request_timeout_must_be_positive_and_bounded(self) -> None:
        self.assertEqual(
            request(timeout_seconds=MAX_AGENT_TIMEOUT_SECONDS).timeout_seconds,
            MAX_AGENT_TIMEOUT_SECONDS,
        )
        with self.assertRaises(ValueError):
            request(timeout_seconds=0)
        with self.assertRaises(ValueError):
            request(timeout_seconds=MAX_AGENT_TIMEOUT_SECONDS + 1)

    def test_request_does_not_accept_process_or_shell_configuration(self) -> None:
        with self.assertRaises(TypeError):
            request(executable_path="/usr/bin/codex")
        with self.assertRaises(TypeError):
            request(shell_command="codex exec")
        with self.assertRaises(TypeError):
            request(environment_dump={"TOKEN": "secret"})

    def test_structured_context_rejects_process_configuration_keys(self) -> None:
        for context in (
            {"environment": {"TOKEN": "secret"}},
            {"shell_command": "codex exec"},
            {"executable_path": "/usr/bin/codex"},
            {"nested": {"argv": ["codex"]}},
        ):
            with self.subTest(context=context), self.assertRaises(ValueError):
                request(structured_context=context)

    def test_mock_adapter_returns_correlated_success(self) -> None:
        result = invoke_adapter(MockAgentAdapter(), request())
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(result.run_id, "run-123")
        self.assertEqual(result.role, AgentRole.EXECUTOR)

    def test_mock_adapter_returns_configured_failure(self) -> None:
        adapter = MockAgentAdapter(
            status=AgentStatus.FAILED,
            failure_category=AgentFailureCategory.EXECUTION_FAILURE,
            diagnostic_code="mock_failure",
        )
        result = invoke_adapter(adapter, request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_category, AgentFailureCategory.EXECUTION_FAILURE)
        self.assertEqual(result.diagnostic_code, "mock_failure")

    def test_invalid_adapter_output_is_rejected(self) -> None:
        with self.assertRaises(AdapterError) as caught:
            invoke_adapter(_InvalidAdapter(), request())
        self.assertEqual(caught.exception.category, AgentFailureCategory.INVALID_OUTPUT)

    def test_adapter_result_must_correlate_with_request(self) -> None:
        class MismatchedAdapter:
            adapter_type = "mismatch-test"

            def invoke(self, agent_request):
                return AgentResult(
                    run_id="other-run",
                    role=agent_request.role,
                    status=AgentStatus.SUCCEEDED,
                )

        with self.assertRaises(AdapterError) as caught:
            invoke_adapter(MismatchedAdapter(), request())
        self.assertEqual(caught.exception.category, AgentFailureCategory.INVALID_OUTPUT)

    def test_telemetry_attributes_omit_sensitive_request_and_result_content(self) -> None:
        agent_request = request()
        result = AgentResult(
            run_id=agent_request.run_id,
            role=agent_request.role,
            status=AgentStatus.FAILED,
            structured_output={"private_result": "do not export"},
            duration_seconds=1.25,
            failure_category=AgentFailureCategory.EXECUTION_FAILURE,
            diagnostic_code="mock_failure",
        )
        attributes = safe_telemetry_attributes(
            agent_request, result, attempt=2, adapter_type="mock"
        )
        self.assertEqual(attributes["workflow.run_id"], "run-123")
        self.assertEqual(attributes["agent.role"], "executor")
        self.assertEqual(attributes["agent.attempt"], 2)
        self.assertEqual(attributes["agent.adapter_type"], "mock")
        self.assertEqual(attributes["agent.status"], "failed")
        self.assertEqual(attributes["agent.failure_category"], "execution_failure")
        self.assertEqual(attributes["agent.duration_seconds"], 1.25)
        serialized = repr(attributes)
        for secret in (
            agent_request.task,
            "private task details",
            "do not export",
            "corr-7",
        ):
            self.assertNotIn(secret, serialized)

    def test_disabled_codex_placeholder_never_runs(self) -> None:
        with self.assertRaises(AdapterError) as caught:
            DisabledCodexAdapter().invoke(request())
        self.assertEqual(caught.exception.category, AgentFailureCategory.POLICY_REJECTION)


if __name__ == "__main__":
    unittest.main()
