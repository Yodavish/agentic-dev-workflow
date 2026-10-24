import copy
import os
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from agentic_dev_workflow.verification_registry import (
    CheckApplicability,
    CheckId,
    CheckPathChange,
    CheckPathChangeKind,
    CompletePathInventory,
    IncompletePathInventoryError,
    VerificationRegistryError,
    _CompleteInventoryProvenance,
    _issue_complete_path_inventory,
    create_sandbox_environment,
    get_check_definition,
    lock_check_applies,
    validate_registered_check,
)


def trusted_complete(entries: tuple[CheckPathChange, ...]) -> CompletePathInventory:
    return _issue_complete_path_inventory(entries)


class VerificationRegistryTests(unittest.TestCase):
    def test_registry_contains_fixed_runner_commands_and_limits(self) -> None:
        tests = get_check_definition("runner.tests")
        compile_check = get_check_definition(CheckId.RUNNER_COMPILE)
        diff_check = get_check_definition(CheckId.RUNNER_DIFF_CHECK)
        lock_check = get_check_definition(CheckId.RUNNER_LOCK_CHECK)

        self.assertEqual(
            tests.commands,
            (("/usr/local/bin/python", "-m", "unittest", "discover", "-s", "tests", "-v"),),
        )
        self.assertEqual((tests.timeout_seconds, tests.max_output_bytes), (300, 512 * 1024))
        self.assertEqual(
            compile_check.commands,
            (("/usr/local/bin/python", "-m", "compileall", "-q", "src", "tests"),),
        )
        self.assertEqual(
            (compile_check.timeout_seconds, compile_check.max_output_bytes),
            (120, 128 * 1024),
        )
        git_prefix = (
            "/usr/bin/git",
            "--no-pager",
            "-c",
            "core.pager=cat",
            "-c",
            "color.ui=false",
        )
        self.assertEqual(
            diff_check.commands,
            (
                git_prefix + ("diff", "--check", "--no-ext-diff", "--no-textconv"),
                git_prefix + ("diff", "--cached", "--check", "--no-ext-diff", "--no-textconv"),
            ),
        )
        self.assertEqual((diff_check.timeout_seconds, diff_check.max_output_bytes), (60, 256 * 1024))
        self.assertEqual(
            lock_check.commands,
            (("/usr/local/bin/uv", "lock", "--check", "--offline"),),
        )
        self.assertEqual((lock_check.timeout_seconds, lock_check.max_output_bytes), (180, 256 * 1024))
        for definition in (tests, compile_check, diff_check, lock_check):
            self.assertEqual(definition.working_directory, "/workspace")
            self.assertEqual(definition.runtime_profile_id, "runner-verification-v1")
            self.assertEqual(definition.applicability, CheckApplicability.ALWAYS if definition.check_id != CheckId.RUNNER_LOCK_CHECK else CheckApplicability.DEPENDENCY_FILES_CHANGED)
        self.assertEqual(lock_check.applicability, CheckApplicability.DEPENDENCY_FILES_CHANGED)

    def test_definitions_are_deeply_immutable_and_modified_copies_are_rejected(self) -> None:
        definition = get_check_definition(CheckId.RUNNER_TESTS)
        with self.assertRaises(FrozenInstanceError):
            definition.timeout_seconds = 1
        with self.assertRaises(TypeError):
            definition.commands[0][0] = "/tmp/python"
        altered = replace(definition, commands=(("/tmp/untrusted-python", "-V"),))
        with self.assertRaises(VerificationRegistryError):
            validate_registered_check(altered)

    def test_unknown_check_id_is_rejected_without_echoing_it(self) -> None:
        with self.assertRaises(VerificationRegistryError) as caught:
            get_check_definition("runner.tests;secret-value")
        self.assertNotIn("secret-value", str(caught.exception))

    def test_lock_check_applies_to_dependency_paths_for_all_change_kinds(self) -> None:
        for kind in (
            CheckPathChangeKind.ADDED,
            CheckPathChangeKind.DELETED,
            CheckPathChangeKind.CHANGED,
            CheckPathChangeKind.UNTRACKED,
        ):
            with self.subTest(kind=kind):
                inventory = trusted_complete(
                    (CheckPathChange(kind=kind, path=b"uv.lock"),)
                )
                self.assertTrue(lock_check_applies(inventory))

    def test_lock_check_checks_both_rename_endpoints(self) -> None:
        for old_path, new_path in (
            (b"pyproject.toml", b"config/project.toml"),
            (b"config/old.toml", b"uv.toml"),
        ):
            with self.subTest(old_path=old_path, new_path=new_path):
                inventory = trusted_complete(
                    (
                        CheckPathChange(
                            kind=CheckPathChangeKind.RENAMED,
                            path=new_path,
                            previous_path=old_path,
                        ),
                    )
                )
                self.assertTrue(lock_check_applies(inventory))

    def test_lock_check_checks_both_copy_endpoints(self) -> None:
        for old_path, new_path in (
            (b"pyproject.toml", b"config/project.toml"),
            (b"config/old.toml", b"uv.toml"),
        ):
            with self.subTest(old_path=old_path, new_path=new_path):
                inventory = trusted_complete(
                    (
                        CheckPathChange(
                            kind=CheckPathChangeKind.COPIED,
                            path=new_path,
                            previous_path=old_path,
                        ),
                    )
                )
                self.assertTrue(lock_check_applies(inventory))

    def test_lock_check_skips_unrelated_paths_and_rejects_incomplete_inventory(self) -> None:
        unrelated = trusted_complete(
            (CheckPathChange(kind=CheckPathChangeKind.CHANGED, path=b"README.md"),)
        )
        self.assertFalse(lock_check_applies(unrelated))
        incomplete = object.__new__(CompletePathInventory)
        object.__setattr__(incomplete, "entries", ())
        object.__setattr__(incomplete, "complete", False)
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(incomplete)

    def test_direct_complete_inventory_construction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompletePathInventory(entries=(), complete=True)

    def test_unissued_complete_inventory_cannot_suppress_lock_check(self) -> None:
        forged = object.__new__(CompletePathInventory)
        object.__setattr__(
            forged,
            "entries",
            (CheckPathChange(kind=CheckPathChangeKind.CHANGED, path=b"uv.lock"),),
        )
        object.__setattr__(forged, "complete", True)
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(forged)

    def test_unbound_inventory_rejects_a_repository_root_claim(self) -> None:
        issued = trusted_complete(())
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(issued, repository_root=".")

    def test_minted_complete_provenance_cannot_suppress_lock_check(self) -> None:
        forged = object.__new__(CompletePathInventory)
        object.__setattr__(forged, "entries", ())
        object.__setattr__(forged, "complete", True)
        object.__setattr__(forged, "_provenance", object.__new__(_CompleteInventoryProvenance))
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(forged)

    def test_copied_complete_inventory_is_not_trusted(self) -> None:
        issued = trusted_complete(
            (CheckPathChange(kind=CheckPathChangeKind.CHANGED, path=b"uv.lock"),)
        )
        self.assertTrue(lock_check_applies(issued))
        cloned = copy.copy(issued)
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(cloned)

    def test_deep_copied_complete_inventory_is_not_trusted(self) -> None:
        issued = trusted_complete(
            (CheckPathChange(kind=CheckPathChangeKind.CHANGED, path=b"uv.lock"),)
        )
        self.assertTrue(lock_check_applies(issued))
        cloned = copy.deepcopy(issued)
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(cloned)

    def test_direct_mutation_of_issued_kind_or_path_cannot_suppress_lock_check(self) -> None:
        renamed = trusted_complete(
            (
                CheckPathChange(
                    kind=CheckPathChangeKind.RENAMED,
                    path=b"notes.txt",
                    previous_path=b"uv.lock",
                ),
            )
        )
        self.assertTrue(lock_check_applies(renamed))
        object.__setattr__(renamed.entries[0], "kind", CheckPathChangeKind.CHANGED)
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(renamed)

        changed = trusted_complete(
            (CheckPathChange(kind=CheckPathChangeKind.CHANGED, path=b"pyproject.toml"),)
        )
        self.assertTrue(lock_check_applies(changed))
        object.__setattr__(changed.entries[0], "path", b"README.md")
        with self.assertRaises(IncompletePathInventoryError):
            lock_check_applies(changed)

    def test_environment_is_fixed_and_does_not_copy_host_secrets(self) -> None:
        with patch.dict(os.environ, {"API_TOKEN": "host-secret", "PATH": "/tmp/untrusted"}):
            environment = create_sandbox_environment(get_check_definition(CheckId.RUNNER_TESTS))
        self.assertEqual(environment["PATH"], "/usr/local/bin:/usr/bin:/bin")
        self.assertEqual(environment["PYTHONPYCACHEPREFIX"], "/tmp/pycache")
        self.assertNotIn("API_TOKEN", environment)
        self.assertNotIn("host-secret", repr(environment))
        self.assertNotIn("/tmp/untrusted", repr(environment))
        with self.assertRaises(TypeError):
            environment["API_TOKEN"] = "host-secret"


if __name__ == "__main__":
    unittest.main()
