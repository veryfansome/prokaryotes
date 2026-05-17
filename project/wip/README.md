# Feature Planning and Implementation Process

This document formalizes how new features are planned, reviewed, and built in this repo. Include it in the context of any new session that will be working on a significant feature.

Work-in-progress material lives under `project/wip/<feature-name>/`.

---

## Phase 1 — Discovery

Before writing any plan, ask enough questions to scope the work accurately and avoid wasted planning cycles.

### Questions to ask the user

1. **Is the app currently in production?** If not, a single coherent commit is almost always preferable to a phased rollout. Phases exist to protect a live production system between deploys; without that constraint they only generate unnecessary intermediate states to analyze and debug.
2. **What is the user-visible goal?** Describe the feature from the user's perspective, separate from implementation details.
3. **What are the hard constraints?** Data compatibility, provider API limits, latency budgets, infrastructure that cannot change, etc.
4. **Are there areas of the codebase you know are off-limits or frozen?** Avoids surprises late in planning.
5. **Is there an existing design you want to start from, or should the plan be generated fresh?**

### Code review to run before planning

Read the files most likely to be touched. For most features this means:
- The relevant `*_v1/` modules for the LLM client and harness being changed
- `prokaryotes/api_v1/models.py` — shared models affected by almost everything
- `prokaryotes/web_v1/` — base harness logic
- `scripts/static/ui.js` — if any UI protocol changes are needed
- Any existing tests in `tests/` that cover the area

Goal: compile a factual understanding of the current code state before any design work begins, so the plan is grounded in what actually exists.

---

## Phase 2 — Planning

### Design document

Create `project/wip/<feature-name>/README.md`. This document should cover:

- Goals
- Core concepts and new abstractions
- Data model changes (schemas, new fields, index mappings)
- Protocol changes (new API fields, new stream event types)
- Redesigned or new functions with pseudocode
- Infrastructure changes
- Open questions

If the work is phased, create `project/wip/<feature-name>/phase<N>/README.md` for each phase. Each phase doc must specify the exact files to change, the exact code changes (not pseudocode), and the tests required. It must also describe what correct behavior looks like at the end of that phase while the phases after it are still incomplete — this is the hardest part of phased planning and should be done carefully.

If the work is a single commit, skip the per-phase docs.

---

## Phase 3 — Overlay

The overlay is a directory that mirrors the repo root and contains only the files that will change. It represents the intended end state of the current unit of work — the full feature for a single-commit implementation, or the current phase for a phased rollout.

### Directory layout

```
project/wip/<feature-name>/overlay/
  prokaryotes/         # mirrors prokaryotes/ at repo root
  scripts/             # mirrors scripts/ at repo root
  tests/               # overlay-specific tests
  README.md            # verification commands (see template below)
```

Only copy a file into the overlay when it represents the intended final version. Do not copy files that are unchanged; the overlay is a diff, not a full copy.

### README.md template

Every overlay should include a `README.md` with the exact commands needed to verify it:

```markdown
## Verification Commands

Overlay note:

`tests/conftest.py` should prepend `project/wip/<feature-name>/overlay/prokaryotes` to `prokaryotes.__path__` during pytest startup so overlay modules take precedence while unchanged modules still fall back to the real repo package.

Run Ruff against the proposed Python files:

    uv run ruff check \
      project/wip/<feature-name>/overlay/prokaryotes \
      project/wip/<feature-name>/overlay/tests

Run the overlay Python tests:

    PYTHONPATH=project/wip/<feature-name>/overlay:. \
      uv run --extra test pytest project/wip/<feature-name>/overlay/tests -q

Run the overlay JS tests:

    npx vitest run --root project/wip/<feature-name>/overlay
```

### Why this structure

- The original source files are never touched, so there is always a pristine state to diff against: `diff -ruN prokaryotes/ project/wip/<feature-name>/overlay/prokaryotes/`
- Overlay tests can be run alongside normal tests without affecting `uv run pytest`, which only collects from the real `tests/` directory (configured in `pyproject.toml`).
- A generic `overlay/tests/conftest.py` bootstrap applies equally to overlay unit tests and overlay integration tests, so both tiers can exercise the overlaid package layout without changing the real repo package.

---

## Directory conventions

```
project/
  README.md                           ← this document
  wip/
    <feature-name>/
      README.md (or AGENTS.md)        ← initial design doc
      phase1/README.md                ← per-phase docs (if phased)
      phase2/README.md
      ...
      overlay/                        ← intended end state of current unit of work
        prokaryotes/
        scripts/
        tests/
        README.md
```
