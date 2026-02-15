import argparse
import asyncio

from dotenv import load_dotenv


async def main(args: argparse.Namespace):
    if args.impl == "anthropic":
        from prokaryotes.anthropic_v1.script_harness import ScriptHarness
        model = args.model or "claude-opus-4-7"
    else:
        from prokaryotes.openai_v1.script_harness import ScriptHarness
        model = args.model or "gpt-5.4"

    harness = ScriptHarness(model=model, reasoning_effort=args.reasoning_effort)
    try:
        await harness.run(
            task=args.task,
            cwd=args.cwd,
            max_tool_call_rounds=args.max_tool_call_rounds,
        )
    finally:
        await harness.close()


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a one-off task via ScriptHarness.")
    parser.add_argument("task", help="The task to run.")
    parser.add_argument("--cwd", default=None, help="Working directory to run the task in.")
    parser.add_argument("--impl", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--max-tool-call-rounds", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    asyncio.run(main(parser.parse_args()))
