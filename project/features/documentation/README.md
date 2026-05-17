# Documentation Strategy

This doc defines how `README.md` / `CLAUDE.md` / `AGENTS.md` files are organized across the repo. The audience is humans editing code and agentic coding tools loading context. The goal is to put the right working context in the right directory so it is loaded only when it is needed, and to make it obvious where new documentation should land as the project evolves.

## Goals

1. **Maximize relevance per token.** A doc that gets loaded into agent context should describe something the agent needs to know to do work in that directory. Tutorial-style content that re-explains how other directories behave is dead weight, as is content that duplicates what a parent README already provides.
2. **Localize maintenance burden.** When code in directory `X` changes, only docs in `X` (and `project/features/<feature>/` if a feature is involved) should need updating. A doc that gets stale every time something elsewhere changes is misplaced.
3. **Predictable discovery.** A new contributor or fresh agent session should be able to figure out where any piece of information lives from a one-paragraph rule.

## Two axes of documentation

The repo's docs split cleanly on two axes:

- **Code-tree docs** describe the directory they sit in: what's here, what its public surface is, what non-obvious contracts callers and editors need to respect.
- **Project docs** under `project/` describe work — features being built, issues being tracked, ideas being explored. These describe *behavior* and *process*, not directory structure.

A given concept usually has exactly one home along each axis. For example, the compaction feature lives at `project/features/compaction/README.md` (cross-module behavioral spec); the code that implements it lives across `prokaryotes/web_v1/`, `prokaryotes/api_v1/`, `prokaryotes/search_v1/`, and each of those code-tree docs only needs to mention compaction at the level of "this module participates in compaction; see the feature doc."

## Code-tree docs

### Top-level `README.md`

The repo root README is the entry point: what the project is, how to run it, how to run tests, how to invoke the harnesses. It also links out to the next layer.

Stays stable. Updates only when the build/run surface changes.

### Tree-root READMEs

Each top-level tree gets a README that orients the reader to that subtree:

- `prokaryotes/README.md` — codebase map: one line per `*_v1/` module describing its role, plus cross-cutting conventions (the `ContextPartition` lingua franca, code-organization rules) and the dependency table.
- `scripts/README.md` — entry-point scripts (`cli.py`, `eval.py`, `web.py`, `search_init.py`) and their roles.
- `tests/README.md` — testing policy: organization, design principles, hermeticity rules.
- `database/README.md` — migration policy and layout.
- `project/README.md` — what each `project/<bucket>/` is for.

These are the most-loaded READMEs and the ones that most pay off being kept tight.

### Module READMEs (sparingly)

A subdirectory under `prokaryotes/` gets its own README **only when** it has non-obvious contracts that the parent `prokaryotes/README.md` cannot fairly summarize. Triggers:

- An abstract base class composed via multiple inheritance across sibling files, where ordering or method resolution matters (`web_v1/`).
- A shared protocol that other modules implement, with surprising failure modes (`tools_v1/`'s `FunctionToolCallback` contract).
- A versioned data layer with invariants that callers must preserve (`api_v1/`, possibly).

A module README documents:

- The directory's module layout (files and what each contains).
- The public surface this directory exposes to its callers.
- Non-obvious contracts and traps for someone editing files here (e.g. "call `super().init()` before adding routes").
- Outbound links to feature docs in `project/features/` for any cross-cutting feature the module participates in.

A module README does **not** document:

- Step-by-step lifecycles that walk through behavior in other directories.
- Recipes for hypothetical future subclasses when only one subclass exists.
- Feature-level behavior (that's the feature doc's job).
- Historical context preserved for its own sake.

### When to add a module README

Default to "no." Add one only when reading the module's code cold would leave a competent contributor or agent confused about a contract that isn't visible at any single call site. If the answer is "just read the file," skip it.

## Project docs

`project/` is the home for everything that isn't code-tree documentation. Each bucket has a different lifecycle and a different shape.

### `project/features/<feature>/README.md`

The authoritative behavioral spec for a feature that spans more than one code module. Use this for anything cross-cutting: data model, protocol contract, invariants, reconciliation behavior, end-to-end flow.

Created when a feature in `project/wip/` is implemented and merged. Survives as the canonical reference long after the wip doc is archived.

Code-tree READMEs link **to** these, never duplicate them.

### `project/wip/<feature-name>/README.md`

In-progress design. Lifecycle and structure are defined in `project/wip/README.md`. When the feature ships, its content is distilled into `project/features/<feature>/README.md` and the wip doc can be deleted or kept as historical record.

### `project/issues/<issue-name>/README.md`

Single-issue tracking with adversarial review. Lifecycle defined in `project/issues/README.md`. Lives until the issue is fixed or rejected.

### `project/ideas/<topic>/README.md`

Exploratory ideation. No expectation of becoming implemented. Lifecycle defined in each idea directory.

### `project/bugs/<bug-name>/README.md`

Reserved for bug-specific tracking when a bug warrants its own doc. (Empty today.)

## CLAUDE.md and AGENTS.md

These are always symlinks to the sibling `README.md`. Per the policy in the top-level README, every `README.md` outside `project/features/*/`, `project/issues/*/`, and `project/wip/*/` must have both symlinks. The feature/issue/wip exception exists because their READMEs are loaded explicitly by sessions working on a specific feature/issue/wip, not picked up as ambient directory context.

There is exactly one piece of content in any given directory; the three names exist so that whichever convention an agent or tool reads, it finds the same file.

## What goes where: decision rule

When writing new documentation, walk this list top to bottom and stop at the first match.

1. **Is this about how to run, build, or test the repo?** → top-level `README.md`.
2. **Is this an in-progress design — a feature still being planned or built?** → `project/wip/<feature-name>/README.md`.
3. **Is this the canonical behavioral spec for a *shipped* feature that spans multiple modules?** → `project/features/<feature>/README.md`.
4. **Is this a non-obvious contract for editing files in a specific directory?** → that directory's `README.md` (create one if it doesn't exist and the trigger justifies it).
5. **Is this an actionable problem to be fixed?** → `project/issues/<issue-name>/README.md`.
6. **Is this exploratory thinking that isn't ready to be a plan?** → `project/ideas/<topic>/README.md`.
7. **Is this a "why we chose X" that future work needs to know?** → inline code comment if the trigger is local; otherwise the relevant module README's contract section.
8. **Is this a "how it used to work" historical note?** → don't write it. Commit messages and `git log` already serve this purpose.

## Maintenance discipline

Each doc has a defined "source of truth" — the thing it can become stale relative to. Tying staleness to a concrete trigger makes maintenance discoverable:

| Doc | Source of truth | Update when |
|---|---|---|
| Top-level `README.md` | Build/run/test commands; top-level tree | Commands change; new top-level dir |
| `prokaryotes/README.md` | Module list under `prokaryotes/` | Module added, renamed, split, or its role meaningfully changes |
| `tests/README.md`, `database/README.md`, `scripts/README.md` | Their own subtree's conventions | Policy or entry-point set changes |
| Module README (e.g. `web_v1/`) | The directory's internal structure and contracts | Sibling file added/removed, base-class composition changes, contract changes |
| `project/features/<feature>/README.md` | The feature's behavior and data model | Feature behavior, schema, or invariants change |
| `project/wip/*` | The plan being executed | The plan changes |
| `project/issues/*` | The issue's status | Adversarial review completes; fix lands |

When making a non-trivial change, ask "which row's trigger does this hit?" and update the corresponding doc as part of the same commit.
