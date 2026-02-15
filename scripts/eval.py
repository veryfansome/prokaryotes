import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from prokaryotes.eval_v1.harness import EvalHarness
from prokaryotes.eval_v1.tasks import TASKS


async def main(args: argparse.Namespace):
    tasks = TASKS
    if args.tier is not None:
        tasks = [t for t in tasks if t.tier == args.tier]
    if args.task_id is not None:
        tasks = [t for t in tasks if t.id == args.task_id]

    if not tasks:
        print("No tasks matched the given filters.")
        return

    if args.list:
        current_tier = None
        for t in tasks:
            if t.tier != current_tier:
                current_tier = t.tier
                print(f"\nTier {t.tier}")
            print(f"  {t.id:<30} {t.description}")
        return

    harness = EvalHarness(
        impl=args.impl,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tool_call_rounds=args.max_tool_call_rounds,
    )

    output_path = Path(args.output) if args.output else None
    run = await harness.run(tasks, output_path=output_path)

    print()
    print(run.summary())


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run eval tasks via EvalHarness.")
    parser.add_argument("--impl", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--list", action="store_true", help="List available tasks and exit")
    parser.add_argument("--max-tool-call-rounds", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    parser.add_argument("--task-id", default=None, help="Run a single task by ID")
    parser.add_argument(
        "--tier", type=int, default=None, choices=[1, 2], help="Run only tasks of this tier",
    )
    asyncio.run(main(parser.parse_args()))
