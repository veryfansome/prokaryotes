"""Pure interval consolidation for `WorkingFileWindow` deduplication.

No I/O, no model imports — operates on `Interval`s (inclusive `[start, end]`, matching
`WorkingFileWindow.view_start_line` / `view_end_line`). `consolidate_intervals` computes the post-read
non-overlapping cover for one resolved path; the caller reconstructs window provenance and renders content
from freshly-read file text. See `project/features/file_tool/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Interval", "ConsolidationResult", "consolidate_intervals"]


@dataclass(frozen=True, slots=True)
class Interval:
    """Inclusive line range. Half-open is tempting but inclusive matches `WorkingFileWindow`'s
    `view_start_line` / `view_end_line`."""

    start: int
    end: int

    def contains(self, other: Interval) -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: Interval) -> bool:
        return self.start <= other.end and other.start <= self.end

    def touches_or_overlaps(self, other: Interval) -> bool:
        """True when `self` and `other` overlap or are immediately adjacent (no gap between them)."""
        return self.start <= other.end + 1 and other.start <= self.end + 1

    def union(self, other: Interval) -> Interval | None:
        """Return the merged interval if `self` and `other` overlap or are contiguous, else None."""
        if not self.touches_or_overlaps(other):
            return None
        return Interval(min(self.start, other.start), max(self.end, other.end))


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    primary: Interval  # interval owned by the new call (always contains `new`)
    secondaries: list[Interval] = field(default_factory=list)  # extra new intervals from a max_size split
    retired: list[Interval] = field(default_factory=list)  # pre-existing intervals to delete
    unreached: list[Interval] = field(default_factory=list)  # pre-existing intervals left untouched


def _split_region(start: int, end: int, max_size: int) -> list[Interval]:
    """Split `[start, end]` greedily from the left into chunks of at most `max_size`."""
    chunks: list[Interval] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + max_size - 1)
        chunks.append(Interval(cursor, chunk_end))
        cursor = chunk_end + 1
    return chunks


def consolidate_intervals(
    existing: list[Interval],
    new: Interval,
    max_size: int,
) -> ConsolidationResult:
    """Compute the post-read interval cover for one resolved path.

    Precondition: `|new| <= max_size` (the caller pre-clamps to max_lines and line_count).

    `existing` is non-overlapping at turn start (reconcile's fold guarantees it) but the algorithm does not
    require it — everything reachable from `new` via overlap-or-touch is connected through `new`, so the reach
    `M` is one contiguous interval and the result is non-overlapping among `new + reached`. Only `unreached`
    intervals can retain a pre-existing overlap, which the next reconcile fold collapses.

    Anchoring: when the merged region exceeds `max_size`, the **new** interval's `start..start+max_size-1`
    boundary is preferred, keeping the model's requested range in one primary window; older content that no
    longer fits is pushed into secondaries.
    """
    reach = new
    reached: list[Interval] = []
    remaining = list(existing)
    changed = True
    while changed:
        changed = False
        still: list[Interval] = []
        for interval in remaining:
            if reach.touches_or_overlaps(interval):
                reach = Interval(min(reach.start, interval.start), max(reach.end, interval.end))
                reached.append(interval)
                changed = True
            else:
                still.append(interval)
        remaining = still
    unreached = remaining

    if reach.end - reach.start + 1 <= max_size:
        return ConsolidationResult(primary=reach, secondaries=[], retired=reached, unreached=unreached)

    primary = Interval(new.start, min(reach.end, new.start + max_size - 1))
    secondaries: list[Interval] = []
    if reach.start <= new.start - 1:
        secondaries.extend(_split_region(reach.start, new.start - 1, max_size))
    if primary.end + 1 <= reach.end:
        secondaries.extend(_split_region(primary.end + 1, reach.end, max_size))
    return ConsolidationResult(primary=primary, secondaries=secondaries, retired=reached, unreached=unreached)
