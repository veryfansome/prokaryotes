# Eval task fixtures

Each subdirectory is one `EvalTask` consumed by `EvalHarness` (see `prokaryotes/eval_v1/tasks.py` for the loader).

## Layout

```
evals/
  <task_id>/
    meta.toml      # description (required); timeout_seconds, setup_command (optional)
    prompt.md      # the prompt sent to the model
    check.sh       # the check command; exit 0 = pass
    check.py       # optional helper invoked by check.sh; copied into workspace at check time only (model never sees it)
    setup/         # optional; files copied into the workspace before the model runs
      <files...>
```

The task `id` is the directory name. The `tier` is parsed from the `t<N>_` prefix.

## Conventions

- `check.sh` is invoked from the workspace via `bash -c`. For Python checks, write the assertions in `check.py` and have `check.sh` contain `python3 check.py`.
- Files at the task root other than `meta.toml`, `prompt.md`, `check.sh`, and `setup/` (e.g. `check.py`) are treated as **check helpers** — written into the workspace at check time only, so the model cannot read or tamper with them during the task.
- Files under `setup/` are written into the workspace before the model runs and are visible to it. Put fixture data and any scripts the prompt asks the model to run there.
