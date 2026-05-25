"""Tests for `prokaryotes.utils_v1.context_discovery`.

The discovery pipeline runs on real filesystem state, so most tests build a small workspace in `tmp_path`.
The conversation builders come from the repo's shared `tests/unit_tests/_builders.py` so the overlay does
not maintain a parallel fixture surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import (
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)
from prokaryotes.utils_v1.context_discovery import (
    DiscoveryGroup,
    DiscoveryMatch,
    PathCandidate,
    collect_candidate_paths,
    discover_relevant_context_files,
    group_matches_by_real_path,
    iter_dirs_upward,
    paths_from_file_tool_annotations,
    paths_from_user_message_text,
    paths_from_working_file_window,
    rank_groups,
    render_context_discovery_prompt,
    render_match_line,
)
from tests.unit_tests._builders import bot_msg, conversation, msg

# --- helpers -----------------------------------------------------------------


def _live_window(path: Path, *, window_id: str = "w1") -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=str(path),
        status="live",
        revision=None,
        rendered_output="",
        view_start_line=1,
        view_end_line=1,
        requested_end_line=1,
        line_count=1,
        origin_call_ids=[window_id],
        source_kind="read_lines",
    )


def _tombstoned_window(path: Path, *, window_id: str = "wt") -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=str(path),
        status="stale",
        revision=None,
        rendered_output="",
        view_start_line=1,
        view_end_line=1,
        requested_end_line=1,
        line_count=1,
        origin_call_ids=[window_id],
        source_kind="tombstone",
    )


def _annotation_output(call_id: str, path: Path, *, persistence: str = "working_file") -> TurnItem:
    return TurnItem(
        type="function_call_output",
        call_id=call_id,
        output="ok",
        prokaryotes_annotations={"file_tool.path": str(path), "file_tool.persistence": persistence},
    )


def _turn(*items: TurnItem, bot_id: str = "b1") -> TurnExecution:
    return TurnExecution(
        conversation_uuid="c-1",
        bot_message_source_id=bot_id,
        items=list(items),
        completed=True,
    )


def _seed_workspace(root: Path, files: dict[str, str | None]) -> dict[str, Path]:
    """Create files (and parent dirs) under `root`. Value None creates an empty file. Returns the absolute
    path for each declared filename so tests can reference them without re-deriving."""
    created: dict[str, Path] = {}
    for relpath, body in files.items():
        full = root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(body or "", encoding="utf-8")
        created[relpath] = full
    return created


# --- collect_candidate_paths -------------------------------------------------


class TestCollectCandidatePaths:
    def test_live_window_contributes(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(
            msg("1", "hello"),
            working_file_windows=[_live_window(f)],
        )
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]
        assert out[0].source_strength == "live_window"

    def test_tombstoned_window_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x")
        conv = conversation(
            msg("1", "hello"),
            working_file_windows=[_tombstoned_window(f)],
        )
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_stale_non_tombstone_window_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x")
        stale = _live_window(f)
        stale = stale.model_copy(update={"status": "stale"})
        conv = conversation(msg("1", "hi"), working_file_windows=[stale])
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_annotation_contributes(self, tmp_path: Path) -> None:
        f = tmp_path / "b.py"
        f.write_text("y")
        turn = _turn(_annotation_output("c1", f))
        conv = conversation(msg("1", "hello"))
        out = collect_candidate_paths(conv, {"b1": turn}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]
        assert out[0].source_strength == "annotation"

    def test_annotation_for_deleted_path_dropped(self, tmp_path: Path) -> None:
        gone = tmp_path / "gone.py"
        turn = _turn(_annotation_output("c1", gone))
        conv = conversation(msg("1", "hi"))
        assert collect_candidate_paths(conv, {"b1": turn}, tmp_path) == []

    def test_annotation_outside_workspace_rejected(
        self,
        tmp_path: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        outside_root = tmp_path_factory.mktemp("outside")
        outside = outside_root / "x.py"
        outside.write_text("o")
        turn = _turn(_annotation_output("c1", outside))
        conv = conversation(msg("1", "hi"))
        assert collect_candidate_paths(conv, {"b1": turn}, tmp_path) == []

    def test_dedupe_window_and_annotation_keeps_live_window(self, tmp_path: Path) -> None:
        """One read_lines call mints both a live window and a file_tool.path annotation."""
        f = tmp_path / "shared.py"
        f.write_text("z")
        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(f)])
        turn = _turn(_annotation_output("c1", f))
        out = collect_candidate_paths(conv, {"b1": turn}, tmp_path)
        assert len(out) == 1
        assert out[0].path == f.resolve()
        assert out[0].source_strength == "live_window"

    def test_user_mention_contributes_relative_path(self, tmp_path: Path) -> None:
        f = tmp_path / "docs" / "guide.md"
        f.parent.mkdir()
        f.write_text("md")
        conv = conversation(msg("1", "see docs/guide.md for details"))
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]
        assert out[0].source_strength == "user_mention"

    def test_user_mention_line_ref_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "lib.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(msg("1", "bug at src/lib.py:42 maybe"))
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]

    def test_user_mention_line_ref_with_trailing_comma_stripped(self, tmp_path: Path) -> None:
        """`see src/lib.py:42, also check ...` — trailing `,` must be stripped *before* (or alternating
        with) the line-ref regex so the `:42` is exposed and removed."""
        f = tmp_path / "src" / "lib.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(msg("1", "see src/lib.py:42, also check the test"))
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]

    def test_user_mention_bracketed_line_ref_with_punctuation(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "lib.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(msg("1", "look (src/lib.py:42), it's wrong"))
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert [c.path for c in out] == [f.resolve()]

    def test_user_mention_url_rejected(self, tmp_path: Path) -> None:
        conv = conversation(msg("1", "see http://example.com/x/y"))
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_user_mention_bare_filename_rejected_even_if_exists(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("readme")
        conv = conversation(msg("1", "look at README.md"))
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_user_mention_slash_prose_with_nonexistent_path_rejected(self, tmp_path: Path) -> None:
        conv = conversation(msg("1", "the A/B test ran 2026/05/25"))
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_bot_authored_messages_ignored_even_with_slash_paths(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(
            msg("1", "hi"),
            bot_msg("2", "I read src/a.py"),
        )
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_deleted_user_messages_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(msg("1", "see src/a.py", deleted=True))
        assert collect_candidate_paths(conv, {}, tmp_path) == []

    def test_multiple_mentions_of_same_path_dedupe_to_one_candidate(self, tmp_path: Path) -> None:
        f = tmp_path / "x" / "y.py"
        f.parent.mkdir()
        f.write_text("x")
        conv = conversation(msg("1", "x/y.py and again x/y.py"))
        out = collect_candidate_paths(conv, {}, tmp_path)
        assert len(out) == 1

    def test_max_source_strength_across_sources(self, tmp_path: Path) -> None:
        f = tmp_path / "shared.py"
        f.write_text("z")
        conv = conversation(
            msg("1", "shared.py and also see shared.py"),  # bare filename — rejected
            msg("2", "actually it's ./shared.py"),  # accepted user mention
            working_file_windows=[_live_window(f)],
        )
        turn = _turn(_annotation_output("c1", f))
        out = collect_candidate_paths(conv, {"b1": turn}, tmp_path)
        assert len(out) == 1
        assert out[0].source_strength == "live_window"


# --- per-source leaves -------------------------------------------------------


class TestPathLeaves:
    def test_window_leaf_returns_empty_for_path_outside_workspace(self, tmp_path: Path) -> None:
        outside = tmp_path.parent
        window = _live_window(outside / "x.py")
        assert paths_from_working_file_window(window, tmp_path.resolve()) == []

    def test_annotation_leaf_requires_function_call_output(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text("x")
        wrong_kind = TurnItem(
            type="function_call",
            call_id="c1",
            name="file_tool",
            arguments="{}",
            prokaryotes_annotations={"file_tool.path": str(f)},
        )
        assert paths_from_file_tool_annotations(wrong_kind, tmp_path.resolve()) == []

    def test_annotation_leaf_returns_empty_when_missing(self, tmp_path: Path) -> None:
        item = TurnItem(type="function_call_output", call_id="c1", output="ok")
        assert paths_from_file_tool_annotations(item, tmp_path.resolve()) == []

    def test_user_message_leaf_handles_absolute_paths(self, tmp_path: Path) -> None:
        f = tmp_path / "abs.py"
        f.write_text("x")
        out = paths_from_user_message_text(msg("1", f"check {f}"), tmp_path.resolve())
        assert [p for p, _ in out] == [f.resolve()]


# --- upward walk -------------------------------------------------------------


class TestIterDirsUpward:
    def test_walk_returns_self_then_parents_up_to_stop_at_inclusive(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        result = iter_dirs_upward(nested.resolve(), stop_at=tmp_path.resolve())
        assert result == [
            (tmp_path / "a" / "b" / "c").resolve(),
            (tmp_path / "a" / "b").resolve(),
            (tmp_path / "a").resolve(),
            tmp_path.resolve(),
        ]

    def test_walk_terminates_at_filesystem_root_when_stop_at_unreachable(self, tmp_path: Path) -> None:
        result = iter_dirs_upward(tmp_path.resolve(), stop_at=Path("/__definitely_not_a_real_dir__"))
        assert result[0] == tmp_path.resolve()
        assert result[-1].parent == result[-1]  # filesystem root reached


# --- discover_relevant_context_files ----------------------------------------


class TestDiscover:
    def test_finds_readmes_walking_upward(self, tmp_path: Path) -> None:
        files = _seed_workspace(
            tmp_path,
            {
                "README.md": "root",
                "src/README.md": "src",
                "src/sub/x.py": "x",
            },
        )
        conv = conversation(
            msg("1", "hi"),
            working_file_windows=[_live_window(files["src/sub/x.py"])],
        )
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        real_paths = {g.real_path for g in groups}
        assert files["src/README.md"].resolve() in real_paths
        assert files["README.md"].resolve() in real_paths

    def test_starts_from_directory_when_candidate_is_a_directory(self, tmp_path: Path) -> None:
        files = _seed_workspace(
            tmp_path,
            {
                "pkg/README.md": "pkg",
                "pkg/inner/AGENTS.md": "inner",
            },
        )
        inner_dir = tmp_path / "pkg" / "inner"
        conv = conversation(msg("1", f"see {inner_dir}"))
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        real_paths = {g.real_path for g in groups}
        assert files["pkg/inner/AGENTS.md"].resolve() in real_paths
        assert files["pkg/README.md"].resolve() in real_paths

    def test_recognizes_claude_agents_readme_filenames(self, tmp_path: Path) -> None:
        files = _seed_workspace(
            tmp_path,
            {
                "src/CLAUDE.md": "c",
                "src/AGENTS.md": "a",
                "src/README.md": "r",
                "src/x.py": "x",
            },
        )
        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(files["src/x.py"])])
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        all_matched = {str(m.matched_path) for g in groups for m in g.matches}
        assert str(files["src/CLAUDE.md"]) in all_matched
        assert str(files["src/AGENTS.md"]) in all_matched
        assert str(files["src/README.md"]) in all_matched

    def test_symlinks_inside_workspace_resolved_and_grouped(self, tmp_path: Path) -> None:
        _seed_workspace(tmp_path, {"src/README.md": "r", "src/x.py": "x"})
        claude = tmp_path / "src" / "CLAUDE.md"
        claude.symlink_to(tmp_path / "src" / "README.md")
        agents = tmp_path / "src" / "AGENTS.md"
        agents.symlink_to(tmp_path / "src" / "README.md")

        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(tmp_path / "src" / "x.py")])
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        src_group = next(g for g in groups if g.real_path == (tmp_path / "src" / "README.md").resolve())
        kinds = {m.matched_path.name: m.kind for m in src_group.matches}
        assert kinds == {"README.md": "regular", "CLAUDE.md": "symlink", "AGENTS.md": "symlink"}

    def test_symlinks_escaping_workspace_rejected(
        self,
        tmp_path: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        outside_root = tmp_path_factory.mktemp("outside")
        outside_readme = outside_root / "README.md"
        outside_readme.write_text("escape")
        _seed_workspace(tmp_path, {"src/x.py": "x"})
        bad = tmp_path / "src" / "README.md"
        bad.symlink_to(outside_readme)

        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(tmp_path / "src" / "x.py")])
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        for g in groups:
            assert g.real_path != outside_readme.resolve()
            for m in g.matches:
                assert m.matched_path != bad

    def test_cross_subtree_symlink_inside_workspace_accepted(self, tmp_path: Path) -> None:
        _seed_workspace(
            tmp_path,
            {
                "features/think_tool/README.md": "ft",
                "ideas/think_tool/x.py": "x",
            },
        )
        alias = tmp_path / "ideas" / "think_tool" / "CLAUDE.md"
        alias.symlink_to(tmp_path / "features" / "think_tool" / "README.md")

        conv = conversation(
            msg("1", "hi"),
            working_file_windows=[_live_window(tmp_path / "ideas" / "think_tool" / "x.py")],
        )
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        cross = next(g for g in groups if g.real_path == (tmp_path / "features" / "think_tool" / "README.md").resolve())
        assert any(m.matched_path == alias and m.kind == "symlink" for m in cross.matches)


# --- grouping ---------------------------------------------------------------


class TestGrouping:
    def test_collapse_by_real_path_with_intra_group_match_dedupe(self, tmp_path: Path) -> None:
        files = _seed_workspace(
            tmp_path,
            {
                "pkg/README.md": "pkg",
                "pkg/a.py": "a",
                "pkg/b.py": "b",
            },
        )
        # Two distinct candidates that share an ancestor — both reach pkg/README.md.
        conv = conversation(
            msg("1", "hi"),
            working_file_windows=[
                _live_window(files["pkg/a.py"], window_id="w1"),
                _live_window(files["pkg/b.py"], window_id="w2"),
            ],
        )
        groups = discover_relevant_context_files(conv, {}, tmp_path)
        pkg_group = next(g for g in groups if g.real_path == files["pkg/README.md"].resolve())
        # Single matched_path → single match entry, but two origins.
        assert len(pkg_group.matches) == 1
        assert len(pkg_group.discovered_from_paths) == 2
        # Both candidates live inside `pkg/`, so the start_dir is `pkg/` and `pkg/README.md` is at
        # distance 0 from the start_dir (same directory as the originating file).
        assert pkg_group.min_distance == 0


# --- ranking ----------------------------------------------------------------


def _candidate(path: str, source: str = "live_window") -> PathCandidate:
    return PathCandidate(path=Path(path), source_strength=source)  # type: ignore[arg-type]


def _match(
    matched: str,
    real: str,
    *,
    distance: int,
    source: str = "live_window",
    kind: str = "regular",
) -> DiscoveryMatch:
    return DiscoveryMatch(
        matched_path=Path(matched),
        real_path=Path(real),
        kind=kind,  # type: ignore[arg-type]
        distance=distance,
        origin=_candidate(matched, source=source),
    )


class TestRanking:
    def test_stronger_source_outranks_broader_weaker(self) -> None:
        # One live_window group with breadth 1, one user_mention group with breadth 5.
        strong = group_matches_by_real_path(
            [_match("/w/a/README.md", "/w/a/README.md", distance=0, source="live_window")]
        )
        broad_weak = group_matches_by_real_path(
            [
                DiscoveryMatch(
                    matched_path=Path("/w/b/README.md"),
                    real_path=Path("/w/b/README.md"),
                    kind="regular",
                    distance=0,
                    origin=PathCandidate(path=Path(f"/w/b/x{i}.py"), source_strength="user_mention"),
                )
                for i in range(5)
            ]
        )
        ranked = rank_groups(strong + broad_weak)
        assert ranked[0].real_path == Path("/w/a/README.md")

    def test_broader_outranks_nearer_when_source_equal(self) -> None:
        near = group_matches_by_real_path(
            [
                _match("/w/a/README.md", "/w/a/README.md", distance=0, source="annotation"),
            ]
        )
        broader_farther = group_matches_by_real_path(
            [
                DiscoveryMatch(
                    matched_path=Path("/w/b/README.md"),
                    real_path=Path("/w/b/README.md"),
                    kind="regular",
                    distance=3,
                    origin=PathCandidate(path=Path(f"/w/b/sub/x{i}.py"), source_strength="annotation"),
                )
                for i in range(2)
            ]
        )
        ranked = rank_groups(near + broader_farther)
        assert ranked[0].real_path == Path("/w/b/README.md")

    def test_nearer_outranks_farther_when_source_and_breadth_equal(self) -> None:
        a = group_matches_by_real_path([_match("/w/a/README.md", "/w/a/README.md", distance=0, source="annotation")])
        b = group_matches_by_real_path([_match("/w/b/README.md", "/w/b/README.md", distance=2, source="annotation")])
        ranked = rank_groups(a + b)
        assert ranked[0].real_path == Path("/w/a/README.md")

    def test_real_path_ascending_breaks_remaining_ties(self) -> None:
        a = group_matches_by_real_path([_match("/w/b/README.md", "/w/b/README.md", distance=0, source="annotation")])
        b = group_matches_by_real_path([_match("/w/a/README.md", "/w/a/README.md", distance=0, source="annotation")])
        ranked = rank_groups(a + b)
        assert [str(g.real_path) for g in ranked] == ["/w/a/README.md", "/w/b/README.md"]


# --- rendering --------------------------------------------------------------


def _single_group(real_path: str, *, matches: list[DiscoveryMatch], origins: list[str]) -> DiscoveryGroup:
    return DiscoveryGroup(
        real_path=Path(real_path),
        matches=matches,
        discovered_from_paths=[Path(p) for p in origins],
        max_source_strength="live_window",
        min_distance=0,
    )


class TestRendering:
    def test_no_groups_returns_empty(self) -> None:
        assert render_context_discovery_prompt([]) == []

    def test_renders_heading_and_one_group(self) -> None:
        group = _single_group(
            "/w/a/README.md",
            matches=[_match("/w/a/README.md", "/w/a/README.md", distance=0)],
            origins=["/w/a/x.py"],
        )
        lines = render_context_discovery_prompt([group])
        assert lines[0] == "# Local context files detected"
        assert any("Use `file_tool.read_lines`" in line for line in lines)
        assert any('"real_path": "/w/a/README.md"' in line for line in lines)
        assert any('"path": "/w/a/x.py"' in line for line in lines)

    def test_intra_group_match_ordering_regular_before_symlinks(self) -> None:
        regular = _match("/w/a/README.md", "/w/a/README.md", distance=0)
        sym_a = _match("/w/a/AGENTS.md", "/w/a/README.md", distance=0, kind="symlink")
        sym_c = _match("/w/a/CLAUDE.md", "/w/a/README.md", distance=0, kind="symlink")
        group = _single_group("/w/a/README.md", matches=[sym_a, regular, sym_c], origins=["/w/a/x.py"])
        lines = render_context_discovery_prompt([group])
        match_lines = [line for line in lines if line.lstrip().startswith("- {")]
        assert '"path": "/w/a/README.md"' in match_lines[0]
        # Symlink lines follow, sorted by matched_path string.
        assert '"path": "/w/a/AGENTS.md"' in match_lines[1]
        assert '"path": "/w/a/CLAUDE.md"' in match_lines[2]

    def test_top_n_truncation_with_omitted_count(self) -> None:
        groups = [
            _single_group(
                f"/w/{i:02d}/README.md",
                matches=[_match(f"/w/{i:02d}/README.md", f"/w/{i:02d}/README.md", distance=0)],
                origins=[f"/w/{i:02d}/x.py"],
            )
            for i in range(12)
        ]
        lines = render_context_discovery_prompt(groups, max_groups=10)
        assert any("Additional lower-ranked context groups omitted: 2" in line for line in lines)
        rendered_real_paths = [line for line in lines if "Context group:" in line]
        assert len(rendered_real_paths) == 10

    def test_provenance_cap_with_omitted_trailer(self) -> None:
        group = _single_group(
            "/w/a/README.md",
            matches=[_match("/w/a/README.md", "/w/a/README.md", distance=0)],
            origins=[f"/w/a/x{i}.py" for i in range(5)],
        )
        lines = render_context_discovery_prompt([group], max_provenance_per_group=3)
        prov_lines = [line for line in lines if "discovered_from" in line]
        assert len(prov_lines) == 4  # 3 paths + 1 omitted trailer
        assert any('"omitted": 2' in line for line in prov_lines)

    def test_provenance_sorted_ascending(self) -> None:
        group = _single_group(
            "/w/a/README.md",
            matches=[_match("/w/a/README.md", "/w/a/README.md", distance=0)],
            origins=["/w/a/zeta.py", "/w/a/alpha.py", "/w/a/mid.py"],
        )
        lines = render_context_discovery_prompt([group])
        prov_lines = [line for line in lines if "discovered_from" in line and "omitted" not in line]
        assert "alpha.py" in prov_lines[0]
        assert "mid.py" in prov_lines[1]
        assert "zeta.py" in prov_lines[2]

    def test_malicious_filename_with_newline_and_quotes_escaped(self) -> None:
        # json.dumps escapes newlines and quotes; the rendered line must remain single-line valid JSON.
        match = _match('/w/a/weird\n"name.md', '/w/a/weird\n"name.md', distance=0)
        out = render_match_line(match)
        body = out.split("- ", 1)[1]
        payload = json.loads(body)
        assert payload["path"] == '/w/a/weird\n"name.md'
        assert "\n" not in out
