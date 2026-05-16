# scripts

Entry-point scripts and web UI assets.

## Python entry points

Thin wrappers around the harnesses in `prokaryotes/harness_v1/`:

- `cli.py` — `ScriptHarness` for a one-off task (Anthropic by default; `--impl openai` switches provider).
- `eval.py` — `EvalHarness` over the curated tasks in `prokaryotes/eval_v1/tasks.py`. Use `--list` to enumerate tasks, `--tier`/`--task-id` to filter.
- `web.py` — `WebHarness` FastAPI app for the `docker compose up` deployment. Provider via `WEB_HARNESS_IMPL` (default `anthropic`).
- `search_init.py` — bootstraps the Elasticsearch `context-partitions` and `topics` indices with a custom stop-word analyzer. Idempotent.

## Web UI assets

- `html/` — `login.html`, `register.html`, `ui.html`. Served by `WebHarness` via routes defined in `prokaryotes/web_v1/auth.py`.
- `static/` — CSS and third-party JS (`markdown-it`, `highlight.js`, `DOMPurify`). Mounted at `/static` by `WebHarness.init()`.
