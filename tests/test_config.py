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
