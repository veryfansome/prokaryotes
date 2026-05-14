# prokaryotes

This project is about exploring agentic harnesses. It is set up as a FastAPI-based application backed by multiple data stores. Currently, it has a web-harness that provides a chat UI for human users, a script-harness for running tasks non-interactively, and an eval-harness that can run a small curated evaluation set.

## Running the app

```bash
# Run the full web stack
docker compose up --build
```

## Commands

### Python unit tests

```bash
# Install dependencies (including dev/test extras)
uv sync --extra dev --extra test

# Run the default unit test suite
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

### Python integration tests

The `tests/integration_tests/` tier lives outside the default `testpaths` (`tests/unit_tests/`), so `uv run pytest` does not collect it. Tier B requires the docker-compose data stores; Tier A also requires real LLM API keys.

```bash
# Bring up the data stores Tier A and B both depend on
docker compose up -d elasticsearch elasticsearch-init postgres postgres-migrate redis

# Tier B — fake LLM, real Redis/Postgres/Elasticsearch
uv run --extra test pytest tests/integration_tests/tier_b

# Tier A — live LLM smoke
# Structural tests skip per-provider when that provider's API key is absent.
# Judged tests require OPENAI_API_KEY.
uv run --extra test pytest tests/integration_tests/tier_a
```

### JavaScript unit tests

```bash
# Run JS tests
npm run test:js
```

### CLI and harness evals

```bash
# Run script-harness via CLI (Anthropic by default)
docker run --env-file .env prokaryotes:latest -- python -m scripts.cli "What's in the working directory?"

# Select provider, model, reasoning effort, and cap tool-call rounds
docker run --env-file .env prokaryotes:latest -- python -m scripts.cli "Summarise the repo" \
  --impl openai \
  --model gpt-5.4 \
  --reasoning-effort low \
  --max-tool-call-rounds 10 \
  --cwd /app
```

```bash
# List available tasks (combinable with --tier)
python -m scripts.eval --list
python -m scripts.eval --list --tier 1

# Run the full eval suite
docker run --env-file .env prokaryotes:latest -- python -m scripts.eval

# Run a single tier or task
docker run --env-file .env prokaryotes:latest -- python -m scripts.eval --tier 1
docker run --env-file .env prokaryotes:latest -- python -m scripts.eval --task-id t1_implement_function
```

## Documentation

Be concise, not didactic. Maximize relevance per token.

**Directory READMEs** sit at the root of each top-level tree (`prokaryotes/`, `scripts/`, `tests/`, `database/`, `project/`) and at deeper levels only when a subdirectory has contracts the parent README cannot summarize. Common triggers: multi-inheritance where ordering matters (e.g. `web_v1/`), a shared protocol with surprising failure modes (e.g. `tools_v1/`'s `FunctionToolCallback`), or a versioned data layer with invariants. Each has `CLAUDE.md` and `AGENTS.md` symlinked to it.

**Project docs** under `project/` describe specific features, designs, issues, or topics.

Place each doc where its source of truth lives so it only goes stale when that source changes. Directory READMEs link to feature docs but never duplicate them. They should also not duplicate context already provided by parent READMEs. Full strategy: [project/features/context_architecture/README.md](project/features/context_architecture/README.md).

## Navigation

- [database/](database/) — Postgres and Neo4j migration scripts
- [project/](project/) — features, in-progress designs, issues, and ideas
- [prokaryotes/](prokaryotes/README.md) — codebase overview: module layout, key design patterns, and dependencies
- [scripts/](scripts/) — entry-point scripts (`cli.py`, `eval.py`, `web.py`, `search_init.py`)
- [tests/](tests/README.md) — test organization and design principles
