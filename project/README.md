# project

Project documentation buckets — each describes work rather than code.

- `features/<feature>/` — canonical spec for shipped features (e.g. `compaction/`, `file_tool/`, `think_tool/`).
- `wip/<feature-name>/` — in-progress designs. Process: [wip/README.md](wip/README.md).
- `issues/<issue-name>/` — actionable problems with adversarial review. Process: [issues/README.md](issues/README.md).
- `ideas/<topic>/` — exploratory ideation.
- `bugs/<bug-name>/` — bug-specific docs.

`features/`, `wip/`, and `issues/` skip the `CLAUDE.md` / `AGENTS.md` symlinks (loaded explicitly per task).
