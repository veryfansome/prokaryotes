# Shell command background-timeout integration notes

This document describes what is needed to integrate the updated `scratch/shell_command.py` into the existing harness.

## What the updated version does

The updated tool changes shell-command execution behavior in these ways:

1. New commands are run as before, but the tool only waits in the foreground for 5 seconds by default.
2. If the command finishes within that window, the tool returns the final exit code and captured stdout/stderr.
3. If the command is still running after 5 seconds, the tool returns immediately with:
   - `Status: running`
   - a generated `Job ID`
   - current stdout/stderr snapshots
4. The subprocess keeps running in the background.
5. The tool continues draining stdout/stderr in background tasks so long-running, chatty commands do not block on full pipes.
6. Later tool calls can use `job_id` plus optional `action`:
   - `status` to poll
   - `terminate` to stop the job
7. Captured stdout and stderr are each truncated to the last 400 lines in memory.

## Important integration requirement

## 1. Persist the `ShellCommandTool` instance across related tool calls

The updated implementation stores background jobs in memory on the `ShellCommandTool` instance.

That means background-job polling only works if the same Python object handles:
- the original `command` call, and
- later `job_id` status/terminate calls.

### Current state observed in repository

The current harnesses create a fresh `ShellCommandTool()` inside each request/run method:
- `prokaryotes/openai_v1/web_harness.py`
- `prokaryotes/openai_v1/script_harness.py`
- `prokaryotes/anthropic_v1/web_harness.py`
- `prokaryotes/anthropic_v1/script_harness.py`

Because of that, background jobs would be lost after the current request/run ends.

## Required code changes

### A. Web harnesses

Move `ShellCommandTool()` construction out of `post_chat()` and onto the harness instance.

#### Current pattern

```python
async def post_chat(...):
    ...
    shell_command_tool = ShellCommandTool()
    think_tool = ThinkTool(...)
    tool_callbacks = {
        shell_command_tool.name: shell_command_tool,
        think_tool.name: think_tool,
    }
```

#### Required pattern

Create it once, for example in `__init__`:

```python
class WebHarness(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.llm_client = OpenAIClient()
        self.shell_command_tool = ShellCommandTool()
```

Then use the persistent instance in `post_chat()`:

```python
shell_command_tool = self.shell_command_tool
```

Apply the same change to:
- `prokaryotes/openai_v1/web_harness.py`
- `prokaryotes/anthropic_v1/web_harness.py`

### B. Script harnesses

If you want background jobs to remain available across multiple tool-call rounds within a single `run()` call only, the current per-run instantiation is acceptable.

If you want jobs to remain available across multiple separate `run()` invocations on the same harness object, move construction to `__init__` there as well:

```python
class ScriptHarness:
    def __init__(...):
        ...
        self.shell_command_tool = ShellCommandTool()
```

and in `run()`:

```python
shell_command_tool = self.shell_command_tool
```

Apply as desired to:
- `prokaryotes/openai_v1/script_harness.py`
- `prokaryotes/anthropic_v1/script_harness.py`

## 2. Replace the current implementation file

Copy `scratch/shell_command.py` over:

- `prokaryotes/tools_v1/shell_command.py`

## 3. Update tool-facing documentation

The tool contract exposed to the model has changed. Update documentation that currently says only that shell commands run and output is truncated.

Files observed that should be updated:
- `prokaryotes/tools_v1/README.md`
- `prokaryotes/tools_v1/AGENTS.md`
- `prokaryotes/tools_v1/CLAUDE.md`
- `prokaryotes/README.md`
- `prokaryotes/AGENTS.md`
- `prokaryotes/CLAUDE.md`

### Suggested wording changes

Document that:
- shell commands are waited on for up to 5 seconds,
- long-running commands continue in the background,
- a `job_id` is returned for follow-up,
- later calls may use `job_id` with `status` or `terminate`,
- stdout and stderr are each stored as bounded rolling buffers.

## 4. Optional cleanup/shutdown improvement

The updated tool includes a periodic cleanup loop and retention window for completed jobs, but it does not currently terminate still-running jobs on application shutdown.

If desired, add a shutdown hook method to the tool, such as `async close()` that:
- cancels the cleanup task,
- optionally terminates or kills still-running child processes,
- awaits remaining wait/stream tasks.

Then call it from harness shutdown paths such as:
- `WebHarness.on_stop()`
- any script-harness teardown path you control

This is optional for first integration, but recommended for operational cleanliness.

## 5. Backward-compatibility note

The response shape has changed.

Previously every successful call returned a final result with `Exit code: ...`.
Now some successful calls will instead return a running-status block and require a later follow-up by `job_id`.

This is intentional, but any tests or prompt assumptions expecting immediate final completion should be updated.

## Example tool calls after integration

### Start a command

```json
{
  "command": "python -m http.server 9000",
  "reason": "Start a local web server"
}
```

Possible response excerpt:

```text
Status: running (foreground timeout after 5.0s)
Job ID: 1234...
PID: 999
Command: python -m http.server 9000
```

### Poll it later

```json
{
  "job_id": "1234...",
  "action": "status",
  "reason": "Check whether the web server is still running"
}
```

### Terminate it

```json
{
  "job_id": "1234...",
  "action": "terminate",
  "reason": "Stop the web server"
}
```

## Known limitations of this scratch implementation

1. Job state is in-memory only.
   - It is lost on process restart.
   - It is not shared across multiple server workers.
2. Because commands are launched with `create_subprocess_shell()`, process-tree handling is shell-based.
   - `terminate` targets the shell process directly.
   - Commands that fork descendants may need a process-group-based implementation later.
3. Output buffers keep only the last 400 lines of each stream.
4. Commands waiting for stdin are still not a good fit for this tool.
