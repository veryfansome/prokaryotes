# Proactive Compaction

The current compaction trigger is fixed: token threshold crossed → summarize pre-tail → swap into a child snapshot. Two axes worth exploring beyond that.

## Possible triggers

- **Token threshold** (current).
- **File change.** Reconciliation detects a tracked file has been wholesale rewritten, making prior history about that file substantially less useful.
- **Task completion.** A recognizable pattern where a task ended and the next is starting fresh.
- **Explicit user request.** "Forget what we discussed about X" or similar.
- **Failed-exploration.** A long stretch of dead-end tool calls that didn't produce useful state.

Each trigger needs criteria that don't produce false positives.

## Possible methods

- **Summary-based** (current). Feed the pre-tail to an LLM, get a textual summary, replace the pre-tail with the summary in a child snapshot. Brute force and lossy: verbatim phrasing, exact constraints, specific tool outputs that turn out to matter — all dissolved into prose.
- **Surgical removal.** Identify specific messages, tool calls, or tool-call outputs that are genuinely useless (e.g., stale file snapshots already superseded by refreshed live windows, dead-end exploration branches, edit records for files no longer relevant) and remove just those, keeping everything else verbatim. Higher precision, lower context loss, but harder to identify what's safe to remove.
- **Hybrid.** Surgical for the obviously-removable, summary for the rest.

## Origins

This idea surfaced during file-tool design (now shipped — see [project/features/file_tool/README.md](../../features/file_tool/README.md)), which introduced *live windows* that refresh in-place when on-disk content changes. The specific scenario that prompted it: a tracked file is wholesale rewritten externally; reconciliation refreshes earlier reads to show the new content; but prior edit records and the model's pre-tail reasoning still reference the old contents and line numbers. The token-threshold trigger may not fire for many more turns. A proactive trigger would compact closer to the moment the history actually became outdated.

We decided to punt on implementing it for three reasons:

- The current token-threshold + summary mechanism works.
- Each new trigger condition needs a definition that doesn't produce false positives, and we lack examples to tune against. A formatter run that touches every line is not a "wholesale rewrite" in the relevant sense, even though many naive metrics would say it is. Picking thresholds without observed cases is guessing.
- Re-orientation against a wholesale-rewritten file produces a lot of new content (new reads, reasoning, edits) that pushes context toward the token threshold quickly anyway. The proactive trigger may turn out to be a marginal optimization rather than essential.

Now that the file tool is live, watch how the model actually handles fresh-content scenarios: does it re-orient gracefully? Does the existing trigger fire soon enough? Does the model exhibit confusion or wasted tool calls that a proactive trigger would have alleviated? Are there other recurring scenarios (task transitions, explicit user resets, dead-end exploration) that surface the same need? Use those observations to decide whether and how to implement.
