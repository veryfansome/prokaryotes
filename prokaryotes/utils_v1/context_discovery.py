"""Per-turn discovery of local context files (`CLAUDE.md` / `AGENTS.md` / `README.md`).

Recomputed every turn from the current `Conversation` state. No caching, no persistence. Surfaces only
file *locations* — content reads still go through `file_tool.read_lines`.

Pipeline: candidate sources → resolved-path dedupe → upward directory walks bounded by `workspace_root` →
`(matched_path, real_path)` matches → grouping by `real_path` → lexicographic ranking → prompt rendering.

Ranking priority is lexicographic by (max source strength, breadth via `len(discovered_from_paths)`,
min directory distance, `real_path` string). Per-path mention frequency and recency are intentionally
omitted in v1 — see `project/features/context_loader/README.md` Design Notes section.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)

SourceStrength = Literal["live_window", "annotation", "user_mention"]

SOURCE_RANK: dict[SourceStrength, int] = {"live_window": 2, "annotation": 1, "user_mention": 0}

_CONTEXT_FILENAMES = ("CLAUDE.md", "AGENTS.md", "README.md")

_LINE_REF_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
_TRAILING_PUNCT = ".,;:?!)>]}"


@dataclass(frozen=True, slots=True)
class PathCandidate:
    """One unique candidate path with aggregated signal.

    `path` is resolved absolute. `source_strength` is the max strength across contributing signals.
    """

    path: Path
    source_strength: SourceStrength


@dataclass(frozen=True, slots=True)
class DiscoveryMatch:
    """One context-file hit from an upward walk.

    `matched_path` is the literal path the walk found (may be a symlink); `real_path` is its resolved target.
    `distance` is the number of `parent` steps from the origin's start_dir to `matched_path.parent`
    (0 = same directory as the origin).
    """

    matched_path: Path
    real_path: Path
    kind: Literal["regular", "symlink"]
    distance: int
    origin: PathCandidate


@dataclass(frozen=True, slots=True)
class DiscoveryGroup:
    """Matches collapsed by `real_path`.

    `matches` is deduped on `matched_path` so a single ancestor's README only renders once even when several
    candidates walked through it. `discovered_from_paths` is the distinct set of origin candidate paths whose
    walks reached at least one match in this group; its length is the breadth signal `rank_groups` reads.
    """

    real_path: Path
    matches: list[DiscoveryMatch]
    discovered_from_paths: list[Path]
    max_source_strength: SourceStrength
    min_distance: int


def collect_candidate_paths(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution],
    workspace_root: Path,
) -> list[PathCandidate]:
    raw: list[tuple[Path, SourceStrength]] = []
    resolved_root = workspace_root.resolve()

    for window in conversation.working_file_windows:
        if window.status != "live" or window.source_kind == "tombstone":
            continue
        raw.extend(paths_from_working_file_window(window, resolved_root))

    for turn in historical_turns.values():
        for item in turn.items:
            raw.extend(paths_from_file_tool_annotations(item, resolved_root))

    for message in conversation.messages:
        if message.deleted or message.author_id == conversation.bot_author_id:
            continue
        raw.extend(paths_from_user_message_text(message, resolved_root))

    return merge_candidates(raw)


def paths_from_working_file_window(
    window: WorkingFileWindow,
    workspace_root: Path,
) -> list[tuple[Path, SourceStrength]]:
    # Live windows are post-reconcile; their `path` is authoritative. We don't require_exists here so
    # that a file deleted between reconcile and discovery still seeds ancestor-README lookups.
    resolved = _safe_resolve_under(window.path, workspace_root, require_exists=False)
    return [(resolved, "live_window")] if resolved else []


def paths_from_file_tool_annotations(
    item: TurnItem,
    workspace_root: Path,
) -> list[tuple[Path, SourceStrength]]:
    if item.type != "function_call_output":
        return []
    annotated = (item.prokaryotes_annotations or {}).get("file_tool.path")
    if not annotated:
        return []
    resolved = _safe_resolve_under(annotated, workspace_root, require_exists=True)
    return [(resolved, "annotation")] if resolved else []


def paths_from_user_message_text(
    message: ConversationMessage,
    workspace_root: Path,
) -> list[tuple[Path, SourceStrength]]:
    out: list[tuple[Path, SourceStrength]] = []
    seen: set[Path] = set()
    for token in message.content.split():
        if "://" in token:
            continue
        cleaned = _clean_mention_token(token)
        if not cleaned or "/" not in cleaned:
            continue
        resolved = _safe_resolve_under(workspace_root / cleaned, workspace_root, require_exists=True)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        out.append((resolved, "user_mention"))
    return out


def merge_candidates(raw: list[tuple[Path, SourceStrength]]) -> list[PathCandidate]:
    by_path: dict[Path, SourceStrength] = {}
    for path, source in raw:
        existing = by_path.get(path)
        if existing is None or SOURCE_RANK[source] > SOURCE_RANK[existing]:
            by_path[path] = source
    return [PathCandidate(path=p, source_strength=s) for p, s in by_path.items()]


def discover_relevant_context_files(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution],
    workspace_root: Path,
) -> list[DiscoveryGroup]:
    candidates = collect_candidate_paths(conversation, historical_turns, workspace_root)
    resolved_root = workspace_root.resolve()

    matches: list[DiscoveryMatch] = []
    for candidate in candidates:
        start_dir = candidate.path if candidate.path.is_dir() else candidate.path.parent
        for distance, directory in enumerate(iter_dirs_upward(start_dir, stop_at=resolved_root)):
            for filename in _CONTEXT_FILENAMES:
                matched_path = directory / filename
                if not matched_path.exists():
                    continue
                real_path = matched_path.resolve()
                if not _is_inside(real_path, resolved_root) or not real_path.is_file():
                    continue
                matches.append(build_match(candidate, matched_path, real_path=real_path, distance=distance))

    return rank_groups(group_matches_by_real_path(matches))


def build_match(
    candidate: PathCandidate,
    matched_path: Path,
    *,
    real_path: Path,
    distance: int,
) -> DiscoveryMatch:
    kind: Literal["regular", "symlink"] = "symlink" if matched_path.is_symlink() else "regular"
    return DiscoveryMatch(
        matched_path=matched_path,
        real_path=real_path,
        kind=kind,
        distance=distance,
        origin=candidate,
    )


def group_matches_by_real_path(matches: list[DiscoveryMatch]) -> list[DiscoveryGroup]:
    """Aggregates are computed over the FULL match list before per-`matched_path` dedupe; deduping first
    would drop origins from collapsed duplicates and could replace the smallest-distance match with a
    farther one.
    """
    grouped: dict[Path, list[DiscoveryMatch]] = {}
    for match in matches:
        grouped.setdefault(match.real_path, []).append(match)

    out: list[DiscoveryGroup] = []
    for real_path, group_matches in grouped.items():
        max_strength = max(
            (m.origin.source_strength for m in group_matches),
            key=SOURCE_RANK.__getitem__,
        )
        min_distance = min(m.distance for m in group_matches)
        discovered_from = list({m.origin.path for m in group_matches})

        deduped_by_matched: dict[Path, DiscoveryMatch] = {}
        for match in group_matches:
            deduped_by_matched.setdefault(match.matched_path, match)

        out.append(
            DiscoveryGroup(
                real_path=real_path,
                matches=list(deduped_by_matched.values()),
                discovered_from_paths=discovered_from,
                max_source_strength=max_strength,
                min_distance=min_distance,
            )
        )
    return out


def iter_dirs_upward(start_dir: Path, stop_at: Path) -> list[Path]:
    """Walk parent directories from `start_dir` up to and including `stop_at`.

    Both arguments must be resolved absolute paths. The termination check is path equality, which only
    fires when both ends share the same canonical form.
    """
    dirs: list[Path] = []
    current = start_dir
    while True:
        dirs.append(current)
        if current == stop_at:
            break
        if current.parent == current:
            break
        current = current.parent
    return dirs


def rank_groups(groups: list[DiscoveryGroup]) -> list[DiscoveryGroup]:
    """Lexicographic ranking — see module docstring."""
    return sorted(
        groups,
        key=lambda g: (
            -SOURCE_RANK[g.max_source_strength],
            -len(g.discovered_from_paths),
            g.min_distance,
            str(g.real_path),
        ),
    )


def render_context_discovery_prompt(
    groups: list[DiscoveryGroup],
    max_groups: int = 10,
    max_provenance_per_group: int = 3,
) -> list[str]:
    if not groups:
        return []

    lines = [
        "# Local context files detected",
        "",
        "The following files were found near paths relevant to this turn. "
        "They may contain local guidance. Use `file_tool.read_lines` to inspect them if useful. "
        "Do not assume their contents without reading them.",
    ]

    for idx, group in enumerate(groups[:max_groups], start=1):
        lines.append("")
        lines.append(f"{idx}. Context group: {json.dumps({'real_path': str(group.real_path)})}")
        ordered_matches = sorted(group.matches, key=lambda m: (m.kind != "regular", str(m.matched_path)))
        for match in ordered_matches:
            lines.append(render_match_line(match))
        ordered_provenance = sorted(group.discovered_from_paths, key=str)
        for origin_path in ordered_provenance[:max_provenance_per_group]:
            lines.append(f"   - discovered_from: {json.dumps({'path': str(origin_path)})}")
        provenance_omitted = len(ordered_provenance) - max_provenance_per_group
        if provenance_omitted > 0:
            lines.append(f"   - discovered_from: {json.dumps({'omitted': provenance_omitted})}")

    omitted = len(groups) - max_groups
    if omitted > 0:
        lines.append("")
        lines.append(f"Additional lower-ranked context groups omitted: {omitted}")

    return lines


def render_match_line(match: DiscoveryMatch) -> str:
    payload: dict[str, str] = {"path": str(match.matched_path), "kind": match.kind}
    if match.kind == "symlink":
        payload["target"] = str(match.real_path)
    return f"   - {json.dumps(payload)}"


def _clean_mention_token(token: str) -> str:
    cleaned = token.strip("`")
    while cleaned and cleaned[0] in "([{<":
        cleaned = cleaned[1:]
    # Alternate trailing-punct and line-ref strips until stable. Either step can hide the other —
    # `path/file:42,` → punct strips `,`, then line-ref matches `:42`; `path/file,:42` → line-ref
    # matches first, then punct strips the trailing `,`. Four passes is comfortably more than the
    # ~2 needed by any realistic mention.
    for _ in range(4):
        prev = cleaned
        while cleaned and cleaned[-1] in _TRAILING_PUNCT:
            cleaned = cleaned[:-1]
        cleaned = _LINE_REF_SUFFIX.sub("", cleaned)
        if cleaned == prev:
            break
    if cleaned.endswith("/") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_resolve_under(path: str | Path, root: Path, *, require_exists: bool) -> Path | None:
    """Resolve `path` and confirm it lives under `root`. Returns None if `resolve()` raises, the
    resolved path is outside `root`, or (when `require_exists`) the resolved path is not on disk."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    if require_exists and not resolved.exists():
        return None
    if not _is_inside(resolved, root):
        return None
    return resolved
