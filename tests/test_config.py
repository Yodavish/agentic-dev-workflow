import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_dev_workflow.config import Configuration, ConfigurationError


class ConfigurationTests(unittest.TestCase):
    def test_defaults_to_dashboard_path_and_local_jaeger_endpoint(self) -> None:
        with patch.object(Path, "resolve", side_effect=AssertionError("filesystem access")):
            config = Configuration.from_values()
        self.assertEqual(
            str(config.repository),
            "/home/dvit/projects/politician-dashboard",
        )
        self.assertEqual(
            config.otlp_endpoint,
            "http://localhost:4318/v1/traces",
        )

    def test_accepts_valid_explicit_configuration(self) -> None:
        config = Configuration.from_values(
            repository="/tmp/project",
            otlp_endpoint="http://127.0.0.1:4318/v1/traces",
        )
        self.assertEqual(config.repository, Path("/tmp/project").resolve())
        self.assertEqual(config.otlp_endpoint, "http://127.0.0.1:4318/v1/traces")
        self.assertFalse(config.codex_enabled)

    def test_codex_execution_is_disabled_by_default_and_explicitly_opt_in(self) -> None:
        disabled = Configuration.from_values(repository="/tmp/source", environ={})
        enabled = Configuration.from_values(
            repository="/tmp/source",
            environ={"AGENTIC_DEV_WORKFLOW_CODEX_ENABLED": "true"},
        )
        self.assertFalse(disabled.codex_enabled)
        self.assertTrue(enabled.codex_enabled)

    def test_invalid_codex_enable_setting_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            Configuration.from_values(
                repository="/tmp/source",
                environ={"AGENTIC_DEV_WORKFLOW_CODEX_ENABLED": "yes"},
            )

    def test_codex_executable_must_be_absolute(self) -> None:
        with self.assertRaises(ConfigurationError):
            Configuration.from_values(
                repository="/tmp/source",
                codex_executable="bin/codex",
                environ={},
            )

    def test_codex_executable_can_be_explicitly_configured(self) -> None:
        config = Configuration.from_values(
            repository="/tmp/source",
            codex_executable="/opt/codex/bin/codex",
            environ={},
        )
        self.assertEqual(config.codex_executable, Path("/opt/codex/bin/codex"))

    def test_codex_executable_environment_value_is_validated(self) -> None:
        config = Configuration.from_values(
            repository="/tmp/source",
            environ={"AGENTIC_DEV_WORKFLOW_CODEX_EXECUTABLE": "/opt/codex/bin/codex"},
        )
        self.assertEqual(config.codex_executable, Path("/opt/codex/bin/codex"))
        with self.assertRaises(ConfigurationError):
            Configuration.from_values(
                repository="/tmp/source",
                environ={"AGENTIC_DEV_WORKFLOW_CODEX_EXECUTABLE": "/opt/../bin/codex"},
            )

    def test_rejects_endpoint_with_credentials(self) -> None:
        with self.assertRaises(ConfigurationError):
            Configuration.from_values(
                repository="/tmp/project",
                otlp_endpoint="http://user:secret@localhost:4318/v1/traces",
            )

    def test_rejects_invalid_endpoint(self) -> None:
        with self.assertRaises(ConfigurationError):
            Configuration.from_values(otlp_endpoint="file:///tmp/traces")


if __name__ == "__main__":
    unittest.main()
