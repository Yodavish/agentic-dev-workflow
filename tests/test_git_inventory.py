import copy
import gc
import hashlib
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentic_dev_workflow.git_inventory import (
    ApprovedScopeDiagnosticCode,
    ApprovedScopeError,
    GitChangeInventory,
    GitEntryType,
    GitInventoryDiagnosticCode,
    GitInventoryEntry,
    GitInventoryError,
    _snapshot_digest,
    parse_git_status_porcelain_v2,
    read_git_change_inventory,
    validate_approved_scope,
)
from agentic_dev_workflow.git_safety import git_environment
from agentic_dev_workflow.verification_registry import (
    CheckPathChange,
    CheckPathChangeKind,
    CompletePathInventory,
    IncompletePathInventoryError,
    _issue_complete_path_inventory,
    lock_check_applies,
)


def test_git_environment() -> dict[str, str]:
    """Keep repository setup hermetic from caller Git configuration."""
    environment = os.environ.copy()
    for name in tuple(environment):
        if name == "GIT_CONFIG" or name.startswith("GIT_CONFIG_"):
            environment.pop(name, None)
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CEILING_DIRECTORIES",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=test_git_environment(),
    )
    return result.stdout.strip()


def init_repository(repository: Path) -> None:
    repository.mkdir()
    git(repository.parent, "init", "-q", str(repository))
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.invalid")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    (repository / "remove-me.txt").write_text("remove\n", encoding="utf-8")
    git(repository, "add", "tracked.txt", "remove-me.txt")
    git(repository, "commit", "-qm", "base")


def by_path(inventory: GitChangeInventory, path: str) -> GitInventoryEntry:
    matches = [entry for entry in inventory.entries if entry.path_text == path]
    if len(matches) != 1:
        raise AssertionError(f"expected one entry for {path!r}, got {matches!r}")
    return matches[0]


def diagnostic(error: BaseException) -> object:
    return getattr(error, "diagnostic_code", None)


def index_snapshot(repository: Path) -> tuple[tuple[str, int, int] | None, bool]:
    index = repository / ".git" / "index"
    lock = repository / ".git" / "index.lock"
    if not index.exists():
        return None, lock.exists()
    stat = index.stat()
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    return (digest, stat.st_mtime_ns, stat.st_size), lock.exists()


class CleanAndChangedInventoryTests(unittest.TestCase):
    def test_clean_repository_returns_an_empty_complete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            inventory = read_git_change_inventory(repository)
        self.assertEqual(inventory.entries, ())
        self.assertTrue(inventory.is_clean)

    def test_staged_unstaged_and_untracked_changes_are_all_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
            (repository / "remove-me.txt").unlink()
            (repository / "staged.txt").write_text("staged\n", encoding="utf-8")
            (repository / "deleted-staged.txt").write_text("gone\n", encoding="utf-8")
            git(repository, "add", "deleted-staged.txt")
            git(repository, "commit", "-qm", "add then delete")
            (repository / "deleted-staged.txt").unlink()
            git(repository, "add", "deleted-staged.txt")
            git(repository, "add", "staged.txt")
            (repository / "untracked.txt").write_text("new\n", encoding="utf-8")

            inventory = read_git_change_inventory(repository)

        self.assertFalse(inventory.is_clean)
        self.assertEqual(by_path(inventory, "tracked.txt").kind, CheckPathChangeKind.CHANGED)
        self.assertEqual(by_path(inventory, "remove-me.txt").kind, CheckPathChangeKind.DELETED)
        self.assertEqual(by_path(inventory, "staged.txt").kind, CheckPathChangeKind.ADDED)
        self.assertEqual(
            by_path(inventory, "deleted-staged.txt").kind, CheckPathChangeKind.DELETED
        )
        self.assertEqual(by_path(inventory, "untracked.txt").kind, CheckPathChangeKind.UNTRACKED)

    def test_staged_and_unstaged_edits_to_one_file_are_one_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").write_text("staged\n", encoding="utf-8")
            git(repository, "add", "tracked.txt")
            (repository / "tracked.txt").write_text("staged plus unstaged\n", encoding="utf-8")
            inventory = read_git_change_inventory(repository)
        entries = [entry for entry in inventory.entries if entry.path_text == "tracked.txt"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].kind, CheckPathChangeKind.CHANGED)

    def test_staged_add_then_modified_stays_an_added_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "added.txt").write_text("one\n", encoding="utf-8")
            git(repository, "add", "added.txt")
            (repository / "added.txt").write_text("two\n", encoding="utf-8")
            inventory = read_git_change_inventory(repository)
        self.assertEqual(by_path(inventory, "added.txt").kind, CheckPathChangeKind.ADDED)


class WorktreeRootAndConfigurationTests(unittest.TestCase):
    def test_subdirectory_input_is_rejected_instead_of_inventorying_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            subdirectory = repository / "sub"
            subdirectory.mkdir()
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(subdirectory)
        self.assertEqual(
            diagnostic(caught.exception), GitInventoryDiagnosticCode.INVALID_REPOSITORY
        )

    def test_repository_local_worktree_redirect_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            redirected = Path(directory) / "redirected"
            init_repository(repository)
            init_repository(redirected)
            (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
            git(repository, "config", "core.worktree", str(redirected))
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository)
        self.assertEqual(
            diagnostic(caught.exception), GitInventoryDiagnosticCode.UNSAFE_CONFIGURATION
        )

    def test_case_insensitive_repository_configuration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "A.txt").write_text("new\n", encoding="utf-8")
            git(repository, "config", "core.ignorecase", "true")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository)
        self.assertEqual(
            diagnostic(caught.exception), GitInventoryDiagnosticCode.UNSAFE_CONFIGURATION
        )


class IndexFlagTests(unittest.TestCase):
    def test_index_flags_that_can_hide_worktree_changes_fail_closed(self) -> None:
        for flag in ("--assume-unchanged", "--skip-worktree"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / "repo"
                init_repository(repository)
                (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
                git(repository, "update-index", flag, "tracked.txt")
                with self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository)
                self.assertEqual(
                    diagnostic(caught.exception),
                    GitInventoryDiagnosticCode.UNSAFE_INDEX_FLAGS,
                )


class RenameInventoryTests(unittest.TestCase):
    def test_rename_retains_source_and_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").write_text("modified before rename\n", encoding="utf-8")
            git(repository, "mv", "tracked.txt", "renamed.txt")
            inventory = read_git_change_inventory(repository)

        entry = by_path(inventory, "renamed.txt")
        self.assertEqual(entry.kind, CheckPathChangeKind.RENAMED)
        self.assertEqual(entry.previous_path_text, "tracked.txt")
        self.assertEqual(entry.entry_type, GitEntryType.FILE)

    def test_rename_destination_only_inventory_does_not_include_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            git(repository, "mv", "tracked.txt", "renamed.txt")
            inventory = read_git_change_inventory(repository)
        self.assertEqual([entry.path_text for entry in inventory.entries], ["renamed.txt"])


class PathRepresentationTests(unittest.TestCase):
    def test_spaces_and_leading_dashes_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "sp ace.txt").write_text("space\n", encoding="utf-8")
            (repository / "-dash.txt").write_text("dash\n", encoding="utf-8")
            git(repository, "add", "-A")
            inventory = read_git_change_inventory(repository)
        self.assertEqual(by_path(inventory, "sp ace.txt").path, b"sp ace.txt")
        self.assertEqual(by_path(inventory, "-dash.txt").path, b"-dash.txt")

    def test_tab_and_newline_names_parse_but_fail_the_path_contract(self) -> None:
        for name in ("tab\tname.txt", "line\nbreak.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / "repo"
                init_repository(repository)
                (repository / name).write_text("x\n", encoding="utf-8")
                git(repository, "add", "-A")
                with self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository)
                self.assertEqual(
                    diagnostic(caught.exception),
                    GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH,
                )

    def test_backslash_control_character_and_oversized_paths_are_unrepresentable(self) -> None:
        names = ("back\\slash.txt", "control\x01name.txt", "a" * 200 + "/" + "b" * 200 + "/" + "c" * 120 + ".txt")
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / "repo"
                init_repository(repository)
                path = repository / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x\n", encoding="utf-8")
                git(repository, "add", "-A")
                with self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository)
                self.assertEqual(
                    diagnostic(caught.exception),
                    GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH,
                )

    def test_non_utf8_filename_is_unrepresentable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            raw_name = b"bad-\xff.txt"
            descriptor = os.open(os.path.join(os.fsencode(repository), raw_name), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, b"x\n")
            finally:
                os.close(descriptor)
            git(repository, "add", "-A")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository)
        self.assertEqual(
            diagnostic(caught.exception), GitInventoryDiagnosticCode.UNREPRESENTABLE_PATH
        )

    def test_case_and_unicode_alias_collisions_are_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "Case.txt").write_text("one\n", encoding="utf-8")
            (repository / "case.txt").write_text("two\n", encoding="utf-8")
            git(repository, "add", "-A")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.AMBIGUOUS_PATH)

    def test_ancestor_descendant_paths_are_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").unlink()
            (repository / "tracked.txt").mkdir()
            (repository / "tracked.txt" / "child.txt").write_text("child\n", encoding="utf-8")
            git(repository, "add", "-A")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.PATH_CONFLICT)


class EntryTypeTests(unittest.TestCase):
    def test_symlink_is_identified_from_git_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            os.symlink("tracked.txt", repository / "link.txt")
            git(repository, "add", "link.txt")
            inventory = read_git_change_inventory(repository)
        entry = by_path(inventory, "link.txt")
        self.assertEqual(entry.entry_type, GitEntryType.SYMLINK)
        self.assertTrue(entry.is_symlink)
        self.assertFalse(entry.is_submodule)

    def test_submodule_gitlink_is_identified_from_git_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            commit = git(repository, "rev-parse", "HEAD")
            git(repository, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor/sub")
            inventory = read_git_change_inventory(repository)
        entry = by_path(inventory, "vendor/sub")
        self.assertEqual(entry.entry_type, GitEntryType.SUBMODULE)
        self.assertTrue(entry.is_submodule)
        self.assertEqual(entry.kind, CheckPathChangeKind.DELETED)

    def test_untracked_path_has_no_git_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "new.txt").write_text("new\n", encoding="utf-8")
            inventory = read_git_change_inventory(repository)
        self.assertEqual(by_path(inventory, "new.txt").entry_type, GitEntryType.UNKNOWN)


ZERO = b"0" * 40


def ordinary(
    xy: bytes = b".M",
    sub: bytes = b"N...",
    mh: bytes = b"100644",
    mi: bytes = b"100644",
    mw: bytes = b"100644",
    path: bytes = b"a.txt",
) -> bytes:
    return b"1 " + xy + b" " + sub + b" " + mh + b" " + mi + b" " + mw + b" " + ZERO + b" " + ZERO + b" " + path + b"\x00"


def renamed(
    xy: bytes = b"R.",
    score: bytes = b"R100",
    path: bytes = b"new.txt",
    previous: bytes = b"old.txt",
) -> bytes:
    return (
        b"2 " + xy + b" N... 100644 100644 100644 " + ZERO + b" " + ZERO + b" "
        + score + b" " + path + b"\x00" + previous + b"\x00"
    )


class ParserTests(unittest.TestCase):
    def assert_malformed(self, payload: bytes) -> None:
        with self.assertRaises(GitInventoryError) as caught:
            parse_git_status_porcelain_v2(payload)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.MALFORMED_OUTPUT)

    def test_empty_output_is_a_clean_parse(self) -> None:
        self.assertEqual(parse_git_status_porcelain_v2(b""), ())

    def test_headers_are_ignored(self) -> None:
        payload = b"# branch.oid abc\x00# branch.head master\x00" + ordinary()
        entries = parse_git_status_porcelain_v2(payload)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].path, b"a.txt")

    def test_rename_and_copy_records_keep_both_ends(self) -> None:
        entries = parse_git_status_porcelain_v2(renamed() + renamed(xy=b"C.", score=b"C075"))
        self.assertEqual(entries[0].kind, CheckPathChangeKind.RENAMED)
        self.assertEqual(entries[0].path, b"new.txt")
        self.assertEqual(entries[0].previous_path, b"old.txt")
        self.assertEqual(entries[1].kind, CheckPathChangeKind.COPIED)
        self.assertEqual(entries[1].previous_path, b"old.txt")

    def test_record_shaped_rename_source_does_not_swallow_following_record(self) -> None:
        sources = (
            ordinary(path=b"swallowed.txt")[:-1],
            b"? swallowed.txt",
            b"2 R. N... 100644 100644 100644 " + ZERO + b" " + ZERO + b" R100 nested.txt",
            b"# branch.head main",
            b"! ignored.txt",
        )
        for xy, score in ((b"R.", b"R100"), (b"C.", b"C100")):
            header = (
                b"2 " + xy + b" N... 100644 100644 100644 " + ZERO + b" " + ZERO
                + b" " + score + b" new.txt\x00"
            )
            for source in sources:
                with self.subTest(xy=xy, source=source):
                    payload = header + source + b"\x00" + ordinary(path=b"following.txt")
                    self.assert_malformed(payload)

    def test_non_record_source_starting_with_a_record_prefix_is_accepted(self) -> None:
        entries = parse_git_status_porcelain_v2(renamed(previous=b"1 notes.txt"))
        self.assertEqual(entries[0].previous_path, b"1 notes.txt")

    def test_truncated_or_malformed_records_are_rejected(self) -> None:
        cases = {
            "missing final nul": ordinary()[:-1],
            "truncated ordinary": b"1 .M N...\x00",
            "unknown record": b"x y\x00",
            "empty token": b"? a\x00\x00? b\x00",
            "ignored record": b"! a\x00",
            "bad change code": ordinary(xy=b"ZZ"),
            "bad submodule field": ordinary(sub=b"Sxx."),
            "bad mode": ordinary(mw=b"099999"),
            "missing rename source": renamed()[:-len(b"old.txt\x00")],
            "empty rename source": renamed(previous=b""),
            "rename to itself": renamed(path=b"same.txt", previous=b"same.txt"),
            "score kind mismatch": renamed(xy=b"R.", score=b"C100"),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                self.assert_malformed(payload)

    def test_unmerged_record_is_rejected_as_ambiguous(self) -> None:
        payload = (
            b"u UU N... 100644 100644 100644 100644 " + ZERO + b" " + ZERO + b" "
            + ZERO + b" conflicted.txt\x00"
        )
        with self.assertRaises(GitInventoryError) as caught:
            parse_git_status_porcelain_v2(payload)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.UNMERGED_STATE)

    def test_inventory_entry_limit_is_enforced(self) -> None:
        payload = ordinary(path=b"a.txt") + ordinary(path=b"b.txt")
        with self.assertRaises(GitInventoryError) as caught:
            parse_git_status_porcelain_v2(payload, max_entries=1)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.INVENTORY_LIMIT)


class FailureAndLimitTests(unittest.TestCase):
    def _executable(self, directory: str, body: str) -> str:
        path = Path(directory) / "fake-git"
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_invalid_timeout_is_rejected_before_running_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            for value in (0, -1, True, float("nan"), float("inf"), "1"):
                with self.subTest(value=value), self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository, timeout_seconds=value)
                self.assertEqual(
                    diagnostic(caught.exception), GitInventoryDiagnosticCode.INVALID_TIMEOUT
                )

    def test_missing_or_non_repository_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GitInventoryError) as missing:
                read_git_change_inventory(Path(directory) / "absent")
            with self.assertRaises(GitInventoryError) as not_repo:
                read_git_change_inventory(Path(directory))
        self.assertEqual(
            diagnostic(missing.exception), GitInventoryDiagnosticCode.INVALID_REPOSITORY
        )
        self.assertEqual(
            diagnostic(not_repo.exception), GitInventoryDiagnosticCode.COMMAND_FAILED
        )

    def test_failed_git_command_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            executable = self._executable(directory, "echo failure >&2\nexit 9\n")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(repository, git_executable=executable)
        self.assertEqual(
            diagnostic(caught.exception), GitInventoryDiagnosticCode.COMMAND_FAILED
        )

    def test_git_timeout_is_reported_and_the_process_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            executable = self._executable(directory, "exec sleep 5\n")
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(
                    repository, timeout_seconds=0.05, git_executable=executable
                )
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.TIMEOUT)

    def test_unavailable_git_executable_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            with self.assertRaises(GitInventoryError) as caught:
                read_git_change_inventory(
                    repository, git_executable=str(Path(directory) / "no-such-git")
                )
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.GIT_UNAVAILABLE)

    def test_output_byte_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "new.txt").write_text("new\n", encoding="utf-8")
            with patch("agentic_dev_workflow.git_inventory.MAX_GIT_STATUS_OUTPUT_BYTES", 8):
                with self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.OUTPUT_LIMIT)

    def test_inventory_limit_is_enforced_through_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "one.txt").write_text("1\n", encoding="utf-8")
            (repository / "two.txt").write_text("2\n", encoding="utf-8")
            with patch("agentic_dev_workflow.git_inventory.MAX_INVENTORY_ENTRIES", 1):
                with self.assertRaises(GitInventoryError) as caught:
                    read_git_change_inventory(repository)
        self.assertEqual(diagnostic(caught.exception), GitInventoryDiagnosticCode.INVENTORY_LIMIT)

    def test_inherited_git_config_overrides_and_global_config_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "new.txt").write_text("new\n", encoding="utf-8")
            excludes = Path(directory) / "global-ignore"
            excludes.write_text("new.txt\n", encoding="utf-8")
            global_config = Path(directory) / "global.gitconfig"
            global_config.write_text(
                "[core]\n"
                "\tignorecase = true\n"
                f"\texcludesFile = {excludes}\n"
                "[status]\n"
                "\tshowUntrackedFiles = no\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.ignorecase",
                    "GIT_CONFIG_VALUE_0": "true",
                    "GIT_CONFIG_GLOBAL": str(global_config),
                    "GIT_CONFIG_SYSTEM": str(global_config),
                    "GIT_CONFIG_NOSYSTEM": "0",
                },
                clear=False,
            ):
                inventory = read_git_change_inventory(repository)
        self.assertEqual(
            by_path(inventory, "new.txt").kind, CheckPathChangeKind.UNTRACKED
        )

    def test_general_git_environment_scrubs_config_family(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG": "/tmp/inherited-config",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.ignorecase",
                "GIT_CONFIG_VALUE_0": "true",
                "GIT_CONFIG_GLOBAL": "/tmp/inherited.gitconfig",
                "GIT_CONFIG_SYSTEM": "/tmp/inherited-system.gitconfig",
                "GIT_CONFIG_NOSYSTEM": "0",
                "GIT_CONFIG_PARAMETERS": "'core.ignorecase'='true'",
            },
            clear=False,
        ):
            environment = git_environment()
        for name in (
            "GIT_CONFIG",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_CONFIG_PARAMETERS",
        ):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(environment["GIT_CONFIG_SYSTEM"], os.devnull)
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")

    def test_git_is_invoked_without_a_shell_and_with_scrubbed_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            original = subprocess.Popen
            observed = []

            def checked(*args, **kwargs):
                observed.append((args, kwargs))
                return original(*args, **kwargs)

            with patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.ignorecase",
                    "GIT_CONFIG_VALUE_0": "true",
                    "GIT_CONFIG_PARAMETERS": "'core.ignorecase'='true'",
                    "GIT_CONFIG_GLOBAL": "/tmp/inherited.gitconfig",
                    "GIT_CONFIG_SYSTEM": "/tmp/inherited-system.gitconfig",
                    "GIT_CONFIG_NOSYSTEM": "0",
                },
                clear=False,
            ), patch("agentic_dev_workflow.git_inventory.subprocess.Popen", side_effect=checked):
                read_git_change_inventory(repository)
        status_calls = [
            (args, kwargs) for args, kwargs in observed if "status" in args[0]
        ]
        self.assertEqual(len(status_calls), 1)
        argv, kwargs = status_calls[0]
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(Path(kwargs["cwd"]), repository.resolve())
        environment = kwargs["env"]
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(environment["GIT_CONFIG_SYSTEM"], os.devnull)
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(
            environment["GIT_CEILING_DIRECTORIES"], str(repository.resolve().parent)
        )
        self.assertNotIn("GIT_DIR", environment)
        self.assertNotIn("GIT_CONFIG_COUNT", environment)
        self.assertNotIn("GIT_CONFIG_KEY_0", environment)
        self.assertNotIn("GIT_CONFIG_VALUE_0", environment)
        self.assertNotIn("GIT_CONFIG_PARAMETERS", environment)
        command = argv[0]
        self.assertIn("status", command)
        self.assertIn("--porcelain=v2", command)
        self.assertIn("-z", command)
        self.assertIn("--untracked-files=all", command)


class ApprovedScopeTests(unittest.TestCase):
    def _inventory(self, directory: str, mutate) -> GitChangeInventory:
        repository = Path(directory) / "repo"
        init_repository(repository)
        mutate(repository)
        return read_git_change_inventory(repository)

    def test_clean_repository_with_empty_scope_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory = self._inventory(directory, lambda repository: None)
            canonical = validate_approved_scope(inventory, ())
        self.assertTrue(canonical.is_clean)

    def test_in_scope_and_out_of_scope_changes(self) -> None:
        def mutate(repository: Path) -> None:
            (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
            (repository / "other.txt").write_text("new\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            inventory = self._inventory(directory, mutate)
            validate_approved_scope(inventory, ("tracked.txt", "other.txt"))
            with self.assertRaises(ApprovedScopeError) as caught:
                validate_approved_scope(inventory, ("tracked.txt",))
        self.assertEqual(diagnostic(caught.exception), ApprovedScopeDiagnosticCode.OUT_OF_SCOPE)

    def test_both_rename_ends_must_be_approved(self) -> None:
        def mutate(repository: Path) -> None:
            git(repository, "mv", "tracked.txt", "renamed.txt")

        with tempfile.TemporaryDirectory() as directory:
            inventory = self._inventory(directory, mutate)
            validate_approved_scope(inventory, ("tracked.txt", "renamed.txt"))
            for approved in (("renamed.txt",), ("tracked.txt",)):
                with self.subTest(approved=approved), self.assertRaises(
                    ApprovedScopeError
                ) as caught:
                    validate_approved_scope(inventory, approved)
                self.assertEqual(
                    diagnostic(caught.exception), ApprovedScopeDiagnosticCode.OUT_OF_SCOPE
                )

    def test_submodule_gitlink_cannot_be_approved_as_a_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submodule = root / "submodule"
            repository = root / "repo"
            init_repository(submodule)
            init_repository(repository)
            git(
                repository,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                str(submodule),
                "vendor/sub",
            )
            git(repository, "commit", "-qm", "add local submodule")
            (repository / "vendor" / "sub" / "tracked.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            inventory = read_git_change_inventory(repository)
            entry = by_path(inventory, "vendor/sub")
            self.assertEqual(entry.entry_type, GitEntryType.SUBMODULE)
            with self.assertRaises(ApprovedScopeError) as caught:
                validate_approved_scope(inventory, ("vendor/sub",))
        self.assertEqual(
            diagnostic(caught.exception),
            ApprovedScopeDiagnosticCode.SUBMODULE_SCOPE_UNSUPPORTED,
        )

    def test_approval_is_exact_and_case_sensitive(self) -> None:
        def mutate(repository: Path) -> None:
            (repository / "Case.txt").write_text("new\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            inventory = self._inventory(directory, mutate)
            with self.assertRaises(ApprovedScopeError) as caught:
                validate_approved_scope(inventory, ("case.txt",))
        self.assertEqual(diagnostic(caught.exception), ApprovedScopeDiagnosticCode.OUT_OF_SCOPE)

    def test_invalid_ambiguous_and_conflicting_approved_paths_are_rejected(self) -> None:
        cases = {
            ApprovedScopeDiagnosticCode.INVALID_APPROVED_PATH: ("../outside",),
            ApprovedScopeDiagnosticCode.AMBIGUOUS_SCOPE: ("a.txt", "A.txt"),
            ApprovedScopeDiagnosticCode.AMBIGUOUS_SCOPE: ("a", "a/b.txt"),
        }
        with tempfile.TemporaryDirectory() as directory:
            inventory = self._inventory(directory, lambda repository: None)
            for expected, approved in cases.items():
                with self.subTest(approved=approved), self.assertRaises(
                    ApprovedScopeError
                ) as caught:
                    validate_approved_scope(inventory, approved)
                self.assertEqual(diagnostic(caught.exception), expected)

    def test_forged_or_conflicting_inventory_is_rejected(self) -> None:
        forged = object.__new__(GitChangeInventory)
        object.__setattr__(forged, "entries", "not-a-tuple")
        with self.assertRaises(ApprovedScopeError) as caught:
            validate_approved_scope(forged, ())
        self.assertEqual(
            diagnostic(caught.exception), ApprovedScopeDiagnosticCode.INVALID_INVENTORY
        )

    def test_duplicate_and_prefix_conflicting_inventories_cannot_be_constructed(self) -> None:
        changed = GitInventoryEntry(
            kind=CheckPathChangeKind.CHANGED, path=b"a", entry_type=GitEntryType.FILE
        )
        child = GitInventoryEntry(
            kind=CheckPathChangeKind.ADDED, path=b"a/b", entry_type=GitEntryType.FILE
        )
        with self.assertRaises(GitInventoryError) as duplicate:
            GitChangeInventory(entries=(changed, changed))
        with self.assertRaises(GitInventoryError) as conflict:
            GitChangeInventory(entries=(changed, child))
        self.assertEqual(
            diagnostic(duplicate.exception), GitInventoryDiagnosticCode.AMBIGUOUS_PATH
        )
        self.assertEqual(diagnostic(conflict.exception), GitInventoryDiagnosticCode.PATH_CONFLICT)


class RegistryBridgeTests(unittest.TestCase):
    def _reader_inventory_with_uv_lock_change(self, directory: str) -> GitChangeInventory:
        repository = Path(directory) / "repo"
        init_repository(repository)
        (repository / "uv.lock").write_text("changed\n", encoding="utf-8")
        git(repository, "add", "uv.lock")
        return read_git_change_inventory(repository)

    def test_inventory_bridges_to_complete_path_inventory_for_lock_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "uv.lock").write_text("changed\n", encoding="utf-8")
            git(repository, "add", "uv.lock")
            inventory = read_git_change_inventory(repository)
            canonical = validate_approved_scope(inventory, ("uv.lock",))
            complete = canonical.to_complete_path_inventory(repository_root=repository)
            self.assertTrue(complete.complete)
            self.assertTrue(lock_check_applies(complete, repository_root=repository))

    def test_copy_entries_count_both_ends_for_lock_check(self) -> None:
        inventory = _issue_complete_path_inventory(
            (
                CheckPathChange(
                    kind=CheckPathChangeKind.COPIED,
                    path=b"config/project.toml",
                    previous_path=b"pyproject.toml",
                ),
            )
        )
        self.assertTrue(lock_check_applies(inventory))

    def test_caller_built_inventory_cannot_claim_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
            reader_inventory = read_git_change_inventory(repository)
            self.assertFalse(reader_inventory.is_clean)
            hand_built = GitChangeInventory(entries=())
            with self.assertRaises(GitInventoryError) as caught:
                hand_built.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_minted_digest_cannot_make_empty_inventory_complete(self) -> None:
        forged = GitChangeInventory(entries=())
        object.__setattr__(forged, "_provenance", _snapshot_digest(forged.entries))
        with self.assertRaises(GitInventoryError) as caught:
            forged.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_minted_digest_cannot_make_fabricated_entries_complete(self) -> None:
        forged = GitChangeInventory(
            entries=(
                GitInventoryEntry(
                    kind=CheckPathChangeKind.CHANGED,
                    path=b"README.md",
                    entry_type=GitEntryType.FILE,
                ),
            )
        )
        object.__setattr__(forged, "_provenance", _snapshot_digest(forged.entries))
        with self.assertRaises(GitInventoryError) as caught:
            forged.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_clean_reader_provenance_cannot_transfer_to_forged_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            clean_reader = read_git_change_inventory(repository)

            forged = GitChangeInventory(entries=())
            object.__setattr__(forged, "_provenance", clean_reader._provenance)
            with self.assertRaises(GitInventoryError) as caught:
                forged.to_complete_path_inventory()
            self.assertEqual(
                diagnostic(caught.exception),
                GitInventoryDiagnosticCode.INVALID_INVENTORY,
            )
            with self.assertRaises(ApprovedScopeError) as scope_error:
                validate_approved_scope(forged, ())
            self.assertEqual(
                diagnostic(scope_error.exception),
                ApprovedScopeDiagnosticCode.INVALID_INVENTORY,
            )

    def test_reader_provenance_cannot_move_between_different_entry_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            init_repository(first)
            init_repository(second)
            (first / "tracked.txt").write_text("first change\n", encoding="utf-8")
            git(second, "rm", "-q", "remove-me.txt")
            first_reader = read_git_change_inventory(first)
            second_reader = read_git_change_inventory(second)
            self.assertNotEqual(first_reader.entries, second_reader.entries)

            forged = GitChangeInventory(entries=second_reader.entries)
            object.__setattr__(forged, "_provenance", first_reader._provenance)
            with self.assertRaises(GitInventoryError) as caught:
                forged.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_reader_provenance_fails_after_entry_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            git(repository, "mv", "tracked.txt", "renamed.txt")
            reader_inventory = read_git_change_inventory(repository)
            renamed = by_path(reader_inventory, "renamed.txt")
            mutated = replace(renamed, previous_path=b"forged-old.txt")
            object.__setattr__(reader_inventory, "entries", (mutated,))
            with self.assertRaises(GitInventoryError) as caught:
                reader_inventory.to_complete_path_inventory(repository_root=repository)
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_minted_digest_cannot_bridge_after_approved_scope_validation(self) -> None:
        forged = GitChangeInventory(entries=())
        object.__setattr__(forged, "_provenance", _snapshot_digest(forged.entries))
        with self.assertRaises(ApprovedScopeError) as caught:
            validate_approved_scope(forged, ())
        self.assertEqual(
            diagnostic(caught.exception),
            ApprovedScopeDiagnosticCode.INVALID_INVENTORY,
        )

    def test_reader_inventory_bridges_before_and_after_approved_scope_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "uv.lock").write_text("changed\n", encoding="utf-8")
            git(repository, "add", "uv.lock")
            reader_inventory = read_git_change_inventory(repository)

            self.assertTrue(
                reader_inventory.to_complete_path_inventory(
                    repository_root=repository
                ).complete
            )
            canonical = validate_approved_scope(reader_inventory, ("uv.lock",))
            complete = canonical.to_complete_path_inventory(repository_root=repository)
            self.assertTrue(complete.complete)
            self.assertTrue(lock_check_applies(complete, repository_root=repository))

    def test_minted_and_transferred_provenance_cannot_suppress_lock_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader_inventory = self._reader_inventory_with_uv_lock_change(directory)
            repository = root / "repo"
            self.assertTrue(
                lock_check_applies(
                    reader_inventory.to_complete_path_inventory(
                        repository_root=repository
                    ),
                    repository_root=repository,
                )
            )

            minted = GitChangeInventory(entries=())
            object.__setattr__(minted, "_provenance", _snapshot_digest(minted.entries))

            clean_repository = root / "clean"
            init_repository(clean_repository)
            clean_reader = read_git_change_inventory(clean_repository)
            transferred = GitChangeInventory(entries=())
            object.__setattr__(
                transferred, "_provenance", clean_reader._provenance
            )

            for forged in (minted, transferred):
                with self.subTest(provenance=type(forged._provenance)):
                    with self.assertRaises(GitInventoryError):
                        forged.to_complete_path_inventory()

    def test_replace_cannot_erase_reader_entries_and_bridge_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader_inventory = self._reader_inventory_with_uv_lock_change(directory)
            forged = replace(reader_inventory, entries=())
            with self.assertRaises(GitInventoryError) as caught:
                forged.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_replace_cannot_substitute_forged_entries_and_bridge_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader_inventory = self._reader_inventory_with_uv_lock_change(directory)
            forged = replace(
                reader_inventory,
                entries=(
                    GitInventoryEntry(
                        kind=CheckPathChangeKind.CHANGED,
                        path=b"README.md",
                        entry_type=GitEntryType.FILE,
                    ),
                ),
            )
            with self.assertRaises(GitInventoryError) as caught:
                forged.to_complete_path_inventory()
        self.assertEqual(
            diagnostic(caught.exception),
            GitInventoryDiagnosticCode.INVALID_INVENTORY,
        )

    def test_forged_inventory_cannot_bridge_after_approved_scope_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader_inventory = self._reader_inventory_with_uv_lock_change(directory)
            forged = replace(
                reader_inventory,
                entries=(
                    GitInventoryEntry(
                        kind=CheckPathChangeKind.CHANGED,
                        path=b"README.md",
                        entry_type=GitEntryType.FILE,
                    ),
                ),
            )
            try:
                canonical = validate_approved_scope(forged, ("README.md",))
            except ApprovedScopeError as error:
                self.assertEqual(
                    diagnostic(error), ApprovedScopeDiagnosticCode.INVALID_INVENTORY
                )
            else:
                with self.assertRaises(GitInventoryError) as caught:
                    canonical.to_complete_path_inventory()
                self.assertEqual(
                    diagnostic(caught.exception),
                    GitInventoryDiagnosticCode.INVALID_INVENTORY,
                )

    def test_deep_copied_reader_inventory_cannot_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "uv.lock").write_text("lock\n", encoding="utf-8")
            inventory = read_git_change_inventory(repository)
            bridged = inventory.to_complete_path_inventory(repository_root=repository)
            self.assertTrue(lock_check_applies(bridged, repository_root=repository))
            cloned = copy.deepcopy(inventory)
            with self.assertRaises(GitInventoryError) as caught:
                cloned.to_complete_path_inventory(repository_root=repository)
            self.assertEqual(
                diagnostic(caught.exception),
                GitInventoryDiagnosticCode.INVALID_INVENTORY,
            )

    def test_copied_reader_provenance_is_bound_to_exact_entry_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            git(repository, "mv", "tracked.txt", "renamed.txt")
            reader_inventory = read_git_change_inventory(repository)
            original = by_path(reader_inventory, "renamed.txt")
            mutations = {
                "kind": replace(original, kind=CheckPathChangeKind.COPIED),
                "path": replace(original, path=b"other.txt"),
                "entry_type": replace(original, entry_type=GitEntryType.SYMLINK),
                "previous_path": replace(original, previous_path=b"forged-old.txt"),
            }
            for name, mutated in mutations.items():
                with self.subTest(field=name):
                    forged = replace(reader_inventory, entries=(mutated,))
                    object.__setattr__(forged, "_provenance", reader_inventory._provenance)
                    with self.assertRaises(GitInventoryError) as caught:
                        forged.to_complete_path_inventory()
                    self.assertEqual(
                        diagnostic(caught.exception),
                        GitInventoryDiagnosticCode.INVALID_INVENTORY,
                    )

    def test_forged_reader_copies_cannot_suppress_lock_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader_inventory = self._reader_inventory_with_uv_lock_change(directory)
            repository = Path(directory) / "repo"
            canonical = validate_approved_scope(reader_inventory, ("uv.lock",))
            self.assertTrue(
                lock_check_applies(
                    canonical.to_complete_path_inventory(repository_root=repository),
                    repository_root=repository,
                )
            )

            forged_empty = replace(reader_inventory, entries=())
            object.__setattr__(
                forged_empty, "_provenance", reader_inventory._provenance
            )
            forged_unrelated = replace(
                reader_inventory,
                entries=(
                    GitInventoryEntry(
                        kind=CheckPathChangeKind.CHANGED,
                        path=b"README.md",
                        entry_type=GitEntryType.FILE,
                    ),
                ),
            )
            object.__setattr__(
                forged_unrelated, "_provenance", reader_inventory._provenance
            )

            cases = (
                (forged_empty, ()),
                (forged_unrelated, ("README.md",)),
            )
            for forged, approved in cases:
                with self.subTest(entries=forged.entries):
                    try:
                        forged_canonical = validate_approved_scope(forged, approved)
                    except ApprovedScopeError as error:
                        self.assertEqual(
                            diagnostic(error),
                            ApprovedScopeDiagnosticCode.INVALID_INVENTORY,
                        )
                    else:
                        with self.assertRaises(GitInventoryError):
                            forged_canonical.to_complete_path_inventory()

    def test_direct_mutation_of_kind_path_and_entry_type_invalidates_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "uv.lock").write_text("changed\n", encoding="utf-8")
            git(repository, "add", "uv.lock")
            mutations = {
                "kind": ("kind", CheckPathChangeKind.DELETED),
                "path": ("path", b"README.md"),
                "entry_type": ("entry_type", GitEntryType.SYMLINK),
            }
            for name, (field_name, value) in mutations.items():
                with self.subTest(field=name):
                    reader_inventory = read_git_change_inventory(repository)
                    entry = by_path(reader_inventory, "uv.lock")
                    object.__setattr__(entry, field_name, value)
                    with self.assertRaises(GitInventoryError) as caught:
                        reader_inventory.to_complete_path_inventory(
                            repository_root=repository
                        )
                    self.assertEqual(
                        diagnostic(caught.exception),
                        GitInventoryDiagnosticCode.INVALID_INVENTORY,
                    )

    def test_stale_reader_capability_cannot_bridge_after_owner_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "uv.lock").write_text("changed\n", encoding="utf-8")
            git(repository, "add", "uv.lock")

            def take_capability(root: Path):
                reader = read_git_change_inventory(root)
                return reader._provenance, reader.entries

            capability, entries = take_capability(repository)
            gc.collect()
            self.assertIsNone(capability.owner_ref())
            forged = GitChangeInventory(entries=entries)
            object.__setattr__(forged, "_provenance", capability)
            with self.assertRaises(GitInventoryError) as caught:
                forged.to_complete_path_inventory(repository_root=repository)
            self.assertEqual(
                diagnostic(caught.exception),
                GitInventoryDiagnosticCode.INVALID_INVENTORY,
            )

    def test_bridged_inventory_cannot_satisfy_lock_check_for_another_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_repo = root / "clean"
            dirty_repo = root / "dirty"
            init_repository(clean_repo)
            init_repository(dirty_repo)
            (dirty_repo / "uv.lock").write_text("changed\n", encoding="utf-8")
            git(dirty_repo, "add", "uv.lock")
            clean = read_git_change_inventory(clean_repo)
            dirty = read_git_change_inventory(dirty_repo)

            with self.assertRaises(GitInventoryError) as caught:
                clean.to_complete_path_inventory(repository_root=dirty_repo)
            self.assertEqual(
                diagnostic(caught.exception),
                GitInventoryDiagnosticCode.INVALID_INVENTORY,
            )

            clean_complete = clean.to_complete_path_inventory(repository_root=clean_repo)
            dirty_complete = dirty.to_complete_path_inventory(repository_root=dirty_repo)
            with self.assertRaises(IncompletePathInventoryError):
                lock_check_applies(clean_complete, repository_root=dirty_repo)
            with self.assertRaises(IncompletePathInventoryError):
                lock_check_applies(clean_complete)
            self.assertFalse(
                lock_check_applies(clean_complete, repository_root=clean_repo)
            )
            link = root / "clean-link"
            link.symlink_to(clean_repo)
            self.assertFalse(lock_check_applies(clean_complete, repository_root=link))
            self.assertTrue(
                lock_check_applies(dirty_complete, repository_root=dirty_repo)
            )


class RepositoryPreservationTests(unittest.TestCase):
    def test_reading_the_actual_repository_does_not_change_it(self) -> None:
        repository = Path(__file__).resolve().parents[1]

        def snapshot() -> tuple[str, bytes]:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                timeout=20,
                env=test_git_environment(),
            ).stdout
            status = subprocess.run(
                ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
                cwd=repository,
                check=True,
                capture_output=True,
                timeout=20,
                env=test_git_environment(),
            ).stdout
            return head, status

        index_before, lock_before = index_snapshot(repository)
        self.assertFalse(lock_before)
        before = snapshot()
        inventory = read_git_change_inventory(repository)
        after = snapshot()
        index_after, lock_after = index_snapshot(repository)
        self.assertEqual(before, after)
        self.assertEqual(index_before, index_after)
        self.assertFalse(lock_after)
        self.assertFalse(inventory.is_clean)
        self.assertIsInstance(inventory, GitChangeInventory)

    def test_reading_a_temporary_repository_does_not_change_its_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            init_repository(repository)
            (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
            index_before, lock_before = index_snapshot(repository)
            self.assertFalse(lock_before)
            inventory = read_git_change_inventory(repository)
            index_after, lock_after = index_snapshot(repository)
        self.assertFalse(inventory.is_clean)
        self.assertEqual(index_before, index_after)
        self.assertFalse(lock_after)


if __name__ == "__main__":
    unittest.main()
