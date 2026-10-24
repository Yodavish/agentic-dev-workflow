# Stage 2C design decisions — Git change inventory and approved scope

Status: design decisions recorded before implementation.

Current implementation status: Git inventory and approved-scope validation are
wired into the orchestration graph. A real OS-enforced sandbox backend and live
agent/model integration remain unimplemented.

## Boundary

`git_inventory` acquires a **complete, read-only snapshot** of worktree changes
and validates it against an explicitly approved path set. Acquisition returns a
`GitChangeInventory` or raises `GitInventoryError`; it never returns a partial
or empty inventory to signal failure. A clean repository is exactly
`GitChangeInventory(entries=())`.

Scope validation is a pure path comparison. It is **not an OS sandbox**, does
not isolate processes, does not prevent concurrent repository mutation, and is
not a substitute for the future runner sandbox.

## Acquisition

- Bounded, read-only Git commands validate the canonical root and effective
  configuration, reject unsafe index flags, then request one
  `git status --porcelain=v2 -z --untracked-files=all` snapshot with
  `-c status.renames=true`, `-c status.relativePaths=false`,
  `-c core.fsmonitor=false`, `--ignore-submodules=none`.
- `subprocess` argument vector, never a shell. Paths are only ever parsed from
  output, never passed back to Git as arguments.
- The inventory uses a minimal environment; inherited `GIT_DIR`,
  `GIT_WORK_TREE`, index/object overrides, and the `GIT_CONFIG_*` family are
  scrubbed. Host global/system configuration is disabled. `GIT_OPTIONAL_LOCKS=0`
  prevents index refreshes.
- Porcelain v2 covers staged and unstaged state in one `XY` record. Renames
  (`2` records) retain the destination path and the NUL-delimited source path.
- Untracked paths are expanded with `--untracked-files=all`; ignored paths are
  outside Git's change definition and are not requested. Rename detection is
  forced on and copy detection is not requested; `2` copy records are still
  parsed defensively.
- Output is read incrementally through a pipe with a hard byte cap, a wall
  clock timeout, and stderr read only to be discarded. The process is killed
  on limit or timeout. Post-hoc truncation is not trusted.
- The parser requires every record to be NUL-terminated, rejects empty tokens,
  unknown record types, malformed field counts, inconsistent rename scores,
  unknown file modes, and unmerged (`u`) records. A truncated final record is
  malformed, not an empty change set.
- Limits: at most `MAX_INVENTORY_ENTRIES` (100,000) entries and
  `MAX_GIT_STATUS_OUTPUT_BYTES` (64 MiB) of stdout.

## Path representation (fail-closed)

Git emits raw bytes. Each path (both rename ends) must decode as strict UTF-8
and satisfy the existing Stage 1 contract (`_relative_repository_path`):
repository-relative POSIX, NFC, no control/format/private-use/surrogate
characters, no backslash or Windows aliases, no `.`/`..`/`.git` components,
length limits, and no Windows reserved names. A path that fails is
`UNREPRESENTABLE_PATH`; it is never rewritten, dropped, or lossily decoded.

Consequences: spaces and leading dashes are accepted; tabs, newlines, and
non-UTF-8 names are represented correctly by the parser but rejected by the
path contract. That is intentional: the validated contract, not the parser,
defines representability.

Ambiguity is rejected, not resolved: two distinct path spellings with the same
NFKC/casefold key anywhere in the inventory (including both rename ends) fail
`AMBIGUOUS_PATH`; so do duplicate current paths. Current paths that are
ancestor/descendant of one another fail `PATH_CONFLICT`.

## Entry model

`GitInventoryEntry` reuses `CheckPathChangeKind` and adds
`GitEntryType` (`file`, `symlink`, `submodule`, `unknown`) derived from
porcelain modes `100644`, `100755`, `120000`, `160000` (worktree mode first,
then index, then HEAD). Untracked entries have no Git mode and are `unknown`.
`CheckPathChangeKind` gains `COPIED`; `CheckPathChange` and `lock_check_applies`
accept both rename ends for `RENAMED` and `COPIED`.

`GitChangeInventory.to_complete_path_inventory()` bridges to the Stage 2A
`CompletePathInventory` only for the reader's exact snapshot.
`CompletePathInventory(...)` is not a public constructor, and
`lock_check_applies` does not trust a `complete=True` flag on its own. The
bridge verifies reader provenance, including the canonical repository root, then
asks the registry issuer for a capability bound to that live object, to every
path-change field, and to that same root. `lock_check_applies` verifies the
capability and requires the caller to name that root again. It does not perform
a second Git read.

## Approved scope

`validate_approved_scope(inventory, approved_paths)` accepts exact
repository-relative `str` file paths validated by the Stage 1 contract. It:

1. rebuilds the inventory through its dataclass invariants (rejecting forged or
   conflicting inventories) and returns the canonical snapshot;
2. rejects duplicate, case-fold-colliding, or ancestor/descendant approved
   paths (`AMBIGUOUS_SCOPE`);
3. rejects any inventory end (current path, and both rename/copy ends) that is
   not exactly one of the approved paths (`OUT_OF_SCOPE`).

Approval is exact and case-sensitive. Directory prefixes are not approved
scopes in this stage; a caller must approve each file path. Rename source and
destination must both be approved.

## Known limitations

- Ignored paths are excluded by Git's own definition of a change and are not
  requested.
- Untracked paths have no Git mode, so their entry type is `unknown`; staged
  symlinks and submodule gitlinks are identified from modes.
- Rename/copy records are two NUL-delimited tokens. A record-shaped source token
  is rejected fail-closed; a legitimate filename with exactly that shape is
  also rejected. Truncation caused by this module's own byte cap or timeout is
  detected before parsing.
- The snapshot is read-only but not atomic against a concurrent same-user
  process changing the worktree during the read.
- Recursive submodule inventory is not supported; a submodule gitlink cannot be
  approved as scope.

## Deferred

Orchestration integration, a real sandbox backend, agent wiring, directory
scopes, ignored-file inventories, worktree/OS isolation, and concurrent-repo
race protection are out of scope for Stage 2C.

## Corrective pass decisions (recorded before implementation)

The first independent review found paths where Git configuration, index flags,
subdirectory inputs, submodule gitlinks, caller-built inventories, and a
record-shaped rename source could produce a false-clean result or an
over-broad approval. The following decisions close those paths without
changing the parser and path-contract work that was sound.

### Canonical worktree root and configuration

- The supplied repository path is resolved strictly and must be a directory.
  It is then compared with Git's canonical worktree root from
  `git rev-parse --path-format=absolute --is-inside-work-tree --show-toplevel`.
  The command must report `true` and the same canonical path. A subdirectory
  input and a bare/non-worktree input are rejected with `INVALID_REPOSITORY` or
  `COMMAND_FAILED`; a redirected `core.worktree` is rejected as
  `UNSAFE_CONFIGURATION` before the snapshot. The parent repository is never
  inventoried silently.
- The effective repository configuration is checked before the snapshot. A
  set `core.worktree` (even to the same path) and `core.ignorecase=true` fail
  closed with `UNSAFE_CONFIGURATION`. A case-insensitive worktree cannot be
  proven to report all case-distinct paths, so it is not treated as a complete
  inventory.
- Inventory commands use a minimal environment built by the inventory module.
  Inherited `GIT_DIR`, `GIT_WORK_TREE`, index/object overrides, and the complete
  `GIT_CONFIG_*` family are not passed through. Host global and system
  configuration is disabled; `GIT_CEILING_DIRECTORIES` is set to the validated
  root's parent for snapshot commands so a missing `.git` cannot cause
  discovery of an ancestor repository. `GIT_OPTIONAL_LOCKS=0` and
  `GIT_TERMINAL_PROMPT=0` are set. The existing `git_safety` environment helper
  likewise scrubs the complete `GIT_CONFIG_*` family and disables host
  global/system config.

### Index flags

- After the root and configuration checks, `git ls-files -v -z --` is read with
  the same bounded subprocess and controlled environment. Any tracked entry
  whose tag is lowercase (`assume-unchanged`) or `S` (`skip-worktree`) fails the
  inventory with `UNSAFE_INDEX_FLAGS`. These flags can hide a staged or
  unstaged worktree change from `git status`, so the snapshot is not complete.
- The command is read-only and runs with optional index locks disabled. The
  inventory path must not create `index.lock` or change the index bytes or
  mtime.

### Submodules

- The reader may still represent a submodule gitlink (with `entry_type ==
  submodule`), but `validate_approved_scope` rejects every submodule gitlink
  with `SUBMODULE_SCOPE_UNSUPPORTED`. Approving `vendor/sub` never approves
  arbitrary files inside the submodule. Recursive submodule inventory is not
  implemented in Stage 2C.

### Inventory provenance

- `GitChangeInventory` carries a private, non-initializable, non-comparable
  reader capability. It is not a bare digest. The reader issues an opaque
  capability object bound to that exact live inventory via a weak reference and
  a process-local HMAC signature over the inventory identity, the canonical
  repository root, and a SHA-256 snapshot digest. The digest covers the entry
  count, every entry's change kind, path, entry type, and both rename/copy
  endpoints. The stored root is `Path.resolve(strict=True).as_posix()` for the
  worktree that was read. Dataclass copies such as `dataclasses.replace()` do
  not carry the capability. Direct mutation of `kind`, `path`, or `entry_type`
  on the live entry changes the digest and fails closed. A capability whose
  owner has been collected is stale: its weak reference no longer identifies a
  live inventory, so attaching it to another object fails closed.
- `to_complete_path_inventory(repository_root=...)` verifies the capability
  type and signature, the live owner identity, the current snapshot digest, and
  the canonical root before bridging. The caller must pass the repository that
  was read. A different repository, a missing root, a caller-created inventory,
  a bare digest assigned to the private field, a capability copied from another
  inventory, or altered entries fail with `INVALID_INVENTORY`. A symlink is
  accepted only when it resolves to that same canonical root.
- `validate_approved_scope` verifies any supplied capability before rebuilding
  the canonical snapshot. It issues a fresh capability for the new canonical
  object only when the input capability was valid. A caller-built inventory
  without a capability remains untrusted and cannot bridge. A future filtered
  approved view must not be represented as a complete raw Git inventory.
- This design deliberately avoids a global mutable registry. There is no entry
  to leak, grow without bound, or survive owner collection; the weak owner
  reference makes copied or stale capabilities fail closed.
- Threat model: unrestricted malicious code running in the same Python process
  can import private module state or call private issuance helpers and can
  therefore forge or tamper with provenance. The capability prevents the public
  constructor, digest-minting, and cross-inventory-copy paths from acquiring
  trust; it is an in-process integrity mechanism under this documented threat
  model, not an OS security boundary.

### Rename/copy source-token hardening

- When a `2` record's following token is structurally a porcelain record
  (`1`, `2`, `u`, `?`, `#`, or `!` shape), the parser fails with
  `MALFORMED_OUTPUT` instead of consuming that token as a source path. The
  following record is therefore not silently swallowed.
- A legitimate source path that exactly has a record shape is rejected
  fail-closed. That residual false-positive is acceptable; it is not a
  false-clean path.

### Remaining accepted limitations

- Ignored paths are still excluded by Git's definition of a change; ignored
  files are not part of this inventory.
- With `core.fileMode=false`, Git ignores executable-bit-only changes. A
  chmod-only change can therefore be absent from an otherwise clean inventory.
- The snapshot is read-only but not atomic against a concurrent same-user
  process changing the worktree during the read.
- The parser cannot distinguish the text of a record from a deliberately
  record-shaped filename; it rejects both rather than accepting an ambiguous
  stream.
- Reader provenance and the complete-inventory capability are process-local
  integrity checks under the documented Python threat model, not an OS security
  boundary. They do not protect against unrestricted malicious code in the same
  Python process. That code can call `_issue_reader_provenance` or
  `_issue_complete_path_inventory`, or read the process-local HMAC keys.
  `_issue_complete_path_inventory` without a repository root exists so registry
  contract tests can exercise path applicability. It is not a Git snapshot, and
  `lock_check_applies` rejects a repository-root argument for that unbound
  object. A Git bridge always binds the root it verified, and `lock_check_applies`
  then rejects a missing or different root instead of treating the snapshot as
  applicable to another repository.
- `lock_check_applies` also rejects a copied complete inventory, a caller-built
  provenance marker, and in-place mutation of an issued entry's kind or path.
  Those failures use `IncompletePathInventoryError` and do not report a
  suppressed lock check.
- Recursive submodule inventory, directory scopes, ignored-file inventories,
  worktree/OS isolation, and concurrent-repository race protection remain
  deferred. Scope validation remains a pure path comparison and is not an OS
  sandbox.
- Stage 2C is not wired into orchestration or any agent/sandbox backend, and no
  real sandbox backend exists.
