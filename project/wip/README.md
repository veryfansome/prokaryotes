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

If the work is phased, run the multi-pass review on the design doc first, then create `project/wip/<feature-name>/phase<N>/README.md` for each phase and run another review cycle across all phase docs together. Each phase doc must specify the exact files to change, the exact code changes (not pseudocode), and the tests required. It must also describe what correct behavior looks like at the end of that phase while the phases after it are still incomplete — this is the hardest part of phased planning and should be done carefully.

If the work is a single commit, skip the per-phase docs. The multi-pass review still applies to the design doc before any code is written.

### Multi-pass review

After the initial design is written, run iterative review passes using subagents until the plan is stable. A typical cycle:

**Pass 1** — Spawn a subagent. Give it the planning documents written so far plus a brief on what to check: correctness of the approach, missing cases, consistency between phases, alignment with the existing code. Ask for findings in a structured document. Write the result to `project/wip/<feature-name>/review/README.md`, overwriting any prior review.

**Pass 2** — Spawn a second, independent subagent with the same planning documents and overlay material, but **without** the Pass 1 assessment. Ask it to form its own verdict first, then read `project/wip/<feature-name>/review/README.md` and explicitly note where it agrees or disagrees with each Pass 1 finding. Write the combined result back to `project/wip/<feature-name>/review/README.md`. Withholding the prior review until after the independent pass prevents anchoring — a reviewer who sees the first assessment first will tend to validate it rather than catch what it missed.

**Pass 3 (optional)** — If Pass 1 and Pass 2 disagree on a significant point, a third subagent can break the tie. For clear consensus, skip this.

**Update** — For every finding that both passes agree on, update the plan. For findings where passes disagreed, use your own judgment and note the resolution.

**Repeat** — Delete `project/wip/<feature-name>/review/` and run the cycle again on the updated plan. Repeat until a full review cycle produces no new findings worth acting on. Stop early if two consecutive cycles raise the same unresolved finding — that is a signal the question requires a design decision that should be escalated to the user rather than iterated on further.

When any review pass uncovers a non-trivial decision — a trade-off with no clear right answer, a constraint that contradicts the design, or a scope question — stop and bring it to the user before continuing.

### Multi-pass review tips

- Give each subagent the relevant file paths rather than pasting content into the prompt. Let it read whatever it judges relevant so you don't inadvertently bias it by pre-selecting what it sees.
- Never give a reviewer a prior review before they have formed their own verdict. The value of a second pass comes from independent judgment, not from confirming the first.
- Keep the review document short and structured. Use three sections: **Bugs** (incorrect approach, broken invariant, missing case that would cause failure), **Non-blocking** (suboptimal but workable, worth noting), **Looks correct** (things explicitly verified, not just not mentioned). A long review document is harder to act on.
- When a phase doc was the source of a finding, note which phase. When the design doc was the source, note the section.
- Do not ask a subagent to also implement fixes. Analysis and implementation are separate steps.
- Do not soften findings when writing review output. If something is wrong with the design, say so plainly — the purpose is to give the next reader the most accurate picture, not to protect the original write-up.
- **Never accept a reviewer's finding without independently verifying it.** Before acting on any finding that references a specific file path, symbol name, line number, or code behavior, verify it directly — grep for the symbol, read the file, or run the code. Reviewers can and do make mistakes (hallucinated paths, wrong attribute names, incorrect behavior claims), and accepting an incorrect finding can introduce errors worse than the original issue.

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

Run Ruff against the proposed Python files:

    uv run ruff check \
      project/wip/<feature-name>/overlay/prokaryotes \
      project/wip/<feature-name>/overlay/tests

Run the overlay Python tests (overlay package takes precedence; unchanged modules fall back to the real repo package):

    PYTHONPATH=project/wip/<feature-name>/overlay:. \
      uv run --extra test pytest project/wip/<feature-name>/overlay/tests -q

Run the overlay JS tests:

    npx vitest run --root project/wip/<feature-name>/overlay
```

### Why this structure

- The original source files are never touched, so there is always a pristine state to diff against: `diff -ruN prokaryotes/ project/wip/<feature-name>/overlay/prokaryotes/`
- Overlay tests can be run alongside normal tests without affecting `uv run pytest`, which only collects from the real `tests/` directory (configured in `pyproject.toml`).

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
      review/
        README.md                     ← latest review cycle output
```
