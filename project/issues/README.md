# Issue Tracking and Review Process

This document describes how code and documentation issues are written, adversarially reviewed, and updated in this repo. Include it in the context of any session that will be writing or reviewing issues.

Issues live under `project/issues/<issue-name>/README.md`.

---

## Issue Document Structure

Every issue README.md follows this template:

```markdown
# Issue: <Short title>

## Location

File path(s) and line number(s) where the problem lives.

## Problem

What is wrong and why it matters. Do not assume the reader has seen the code or document — quote the relevant lines exactly.

## Proposed Fix

The specific change recommended. Include before/after snippets. If there are meaningful alternatives, list them and state which is preferred and why.

---

## Second Opinion

(Added after adversarial review — see below.)
```

Keep issues focused on one problem. If a review session uncovers multiple independent issues, write one document per issue.

---

## Adversarial Review Process

After issues are written, spawn one subagent per issue in parallel. Each subagent's job is to challenge the issue — prove it wrong, show it is overstated, or find flaws in the proposed fix. This catches errors in reasoning, wrong fix direction, false enforcement claims, and incomplete scope before any code is changed.

### Subagent prompt template

```
You are an adversarial reviewer. Your job is to read an issue document and then challenge it — prove it wrong, show it is overstated, or identify flaws in the proposed fix. Be rigorous and skeptical. Do not be diplomatic.

Read the issue document at:
  project/issues/<issue-name>/README.md

Then read the actual source files and documentation it references:
  <file1>
  <file2>

Your task:
1. Verify whether the factual claims in the issue are accurate.
2. Determine whether the proposed fix is correct, complete, and free of its own problems.
3. Check whether the fix has a wider blast radius than stated — for code issues: other call sites or files; for doc issues: other places in the documentation where the same misconception appears.
4. Identify any case where the current behaviour or wording might actually be intentional.
5. For code issues: check whether any stated enforcement mechanism (e.g. a linter rule) actually applies. For doc issues: verify that the proposed replacement text accurately matches what the code actually does.

Report your findings in under 300 words. Be specific: cite line numbers and exact text. State clearly whether this issue is valid, overstated, or wrong.
```

**Spawn all agents in parallel.** They are independent and should not see each other's output — a reviewer who sees another's assessment first will tend to validate it rather than catch what it missed.

### What adversarial review catches

From experience, reviewers most commonly find:

- **Wrong mechanism cited** — the issue reaches a correct conclusion via incorrect reasoning (e.g., citing Python `__doc__` semantics when the real mechanism is Pydantic field-doc extraction).
- **False enforcement claim** — citing a linter rule that the project's configured ruleset does not actually enable.
- **Incomplete fix** — the proposed change solves only part of the duplication, leaving an identical pattern untouched elsewhere.
- **Wrong fix direction** — the fix solves a cosmetic problem by introducing a worse one (e.g., embedding a policy concept into a transport-layer interface).
- **Over-engineering** — proposing a new abstraction (class, protocol, dataclass) for a problem that a one-line comment would address proportionately.
- **Missed call sites** *(code)* — the fix touches the obvious location but doesn't enumerate all call sites that would need updating.
- **Missed doc locations** *(documentation)* — the fix updates one document but the same inaccuracy or gap exists in others.
- **Proposed text contradicts the code** *(documentation)* — the replacement wording describes intended or aspirational behaviour rather than what the code actually does.
- **Overstated severity** — framing a one-line redundant computation as an "inefficiency" or "risk" when it is cosmetic duplication only; or framing a minor doc imprecision as "actively misleading" when no reader would act on it incorrectly.
- **Valid but structurally forced** — an asymmetry that is real but reflects differing external API shapes or doc constraints, not a fixable design flaw.

---

## Updating Issues After Review

After all subagents report, append a `## Second Opinion` section to each issue document. The section should cover:

- **Verdict** — one of: *Fully valid*, *Valid — [specific caveat]*, *Valid but [fix is flawed]*, *Overstated*, *Wrong*.
- **What the reviewer confirmed** — which factual claims checked out.
- **What the reviewer corrected** — errors in reasoning, mechanism, or scope.
- **What the reviewer missed or where it disagreed** — note if the adversarial case was weak.
- **Recommendation adjustment** — if the proposed fix should change as a result, state the revised recommendation explicitly.

Do not soften findings. If the reviewer caught a real error in the issue, say so plainly. The purpose of the section is to give the next reader — the one implementing the fix — the most accurate picture, not to protect the original write-up.

**Never accept a reviewer's finding without independently verifying it.** Before updating an issue or implementing a fix based on review output, verify the claim directly — grep for the symbol, read the file, or run the code. Reviewers can and do make mistakes (hallucinated paths, wrong attribute names, incorrect behavior claims), and acting on a wrong finding can introduce errors worse than the original issue.

### When the reviewer changes the fix

If the reviewer shows the proposed fix is the wrong direction, update the Proposed Fix section in the issue itself. The Second Opinion section explains the original proposal and why it was revised; the Proposed Fix section always reflects the current best recommendation.

---

## Severity guidance

When writing issues, pick the level that honestly reflects impact:

**Code issues:**

| Level | Meaning |
|---|---|
| **High** | Semantically incorrect (wrong decorator, broken type contract, silent data loss) |
| **Medium** | Maintenance hazard (schema knowledge split, dead API, deferred imports) |
| **Low** | Cosmetic (duplicate docstring, redundant computation, opaque idiom with a comment fix) |

**Documentation issues:**

| Level | Meaning |
|---|---|
| **High** | Actively misleading — causes agents or developers to make wrong implementation decisions or violate invariants |
| **Medium** | Missing context — important information an agent would need to look up elsewhere to work correctly |
| **Low** | Minor imprecision — wording is technically inaccurate but no reader would act on it incorrectly |

Do not inflate severity to make an issue seem more urgent. Adversarial review will catch it.

---

## Directory conventions

```
project/
  issues/
    README.md                    ← this document
    <issue-name>/
      README.md                  ← issue description, fix, and second opinion
```

Issue folder names use kebab-case and describe the problem, not the fix: `dead-put-partition-kwargs`, not `remove-put-partition-kwargs`.
