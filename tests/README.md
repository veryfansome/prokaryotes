# Test Suite Overview

Tests should make intended behavior easy to find, easy to extend, and safe to refactor. This document describes the organization we want to converge toward.

## Test organization

**Default to `test_<module>.py`**: use this when the file primarily validates one production module's contract, invariants, or public helpers. This should be the default because it keeps the mapping from code to tests obvious.

**Use `test_<feature>.py` for cross-cutting behavior**: use this when the meaningful unit is a feature or workflow that intentionally spans multiple production modules. Good candidates are flows like compaction, auth/session behavior, or provider-plus-web streaming behavior.

**Pick one primary axis per file**: a test file should answer either "does this module behave correctly?" or "does this feature behave correctly across modules?" If it starts answering both, split it.

**Let feature tests own integrated scenarios**: if a workflow has a natural feature-based home, keep the full integrated scenario there. Module-based test files should focus on local contracts and edge cases instead of replaying the same multi-module story in several places.

**Centralize reused test utilities**: if a fake, builder, fixture, or assertion helper is used in more than one test file, move it to a shared location instead of copying it again.

**Use `tests/conftest.py` for shared pytest fixtures**: put shared fixture wiring and setup/teardown behavior there when multiple test modules need it.

**Use shared helper modules under `tests/` for reusable fakes, builders, and assertions**: keep non-fixture helpers out of `conftest.py` unless they are part of fixture setup.

**Keep helpers local only when they are truly local**: if a fake exists to support one file's very specific assertions, keeping it in that file is fine.

**Apply the same idea to JavaScript tests**: browser-side tests can keep the Vitest-style `*.test.js` naming, but the base name should still reflect whether the file is module-oriented or feature-oriented.

## Within-file organization

**Classes**: introduce a class only when a fake or helper needs to share mutable state or implement a multi-method protocol. Otherwise prefer module-level helper functions and plain data builders.

**Method type**: use the least powerful form that satisfies the need: instance method if it needs `self`, `@classmethod` if it needs `cls` but not an instance, `@staticmethod` if it belongs on the class but needs neither, and module-level if it has no meaningful connection to any class.

**Ordering**: imports first, then constants and fixtures, then helper classes, then helper functions, then tests. Within a class, put `__init__` first and sort the remaining methods alphabetically. For module-level helpers and tests, keep alphabetical order within a logical group unless a simple happy-path-to-edge-cases progression is clearer.

**Parameter ordering**: required parameters before optional ones, and alphabetical within each group.

## Test design

**Behavior over implementation**: assert externally visible behavior, emitted records, state transitions, and persisted shapes rather than internal call structure unless the call structure is itself the contract.

**One concern per test**: each test should fail for one clear reason. If a test would be hard to diagnose when it fails, split it into smaller behavior-focused tests.

**Behavior matrices via parametrization**: when one contract needs to hold across multiple branches, use `pytest.mark.parametrize(...)` to keep the cases compact and comparable.

**Async tests should cross real coroutine boundaries**: prefer `pytest.mark.asyncio` and await the production coroutine directly instead of wrapping async logic in sync helpers.

**Isolation**: tests should stay hermetic. Fake or mock network, storage, and provider boundaries so the suites can run without Docker or live external services.
