import tomllib
from pathlib import Path

from prokaryotes.eval_v1.models import EvalTask

EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"


def _load_task(task_dir: Path) -> EvalTask:
    meta = tomllib.loads((task_dir / "meta.toml").read_text())
    tier = int(task_dir.name.split("_", 1)[0].lstrip("t"))

    setup_files: dict[str, str] = {}
    setup_dir = task_dir / "setup"
    if setup_dir.is_dir():
        for path in sorted(setup_dir.rglob("*")):
            if path.is_file():
                setup_files[str(path.relative_to(setup_dir))] = path.read_text()

    check_files: dict[str, str] = {}
    for path in sorted(task_dir.iterdir()):
        if path.name in {"meta.toml", "prompt.md", "check.sh", "setup"}:
            continue
        if path.is_file():
            check_files[path.name] = path.read_text()

    return EvalTask(
        check_command=(task_dir / "check.sh").read_text(),
        check_files=check_files,
        description=meta["description"],
        id=task_dir.name,
        prompt=(task_dir / "prompt.md").read_text(),
        setup_command=meta.get("setup_command"),
        setup_files=setup_files,
        tier=tier,
        timeout_seconds=meta.get("timeout_seconds", 180),
    )


def _load_tasks() -> list[EvalTask]:
    return [_load_task(d) for d in sorted(EVALS_ROOT.iterdir()) if d.is_dir()]


TASKS: list[EvalTask] = _load_tasks()
