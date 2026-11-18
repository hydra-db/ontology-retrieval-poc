# Claude Code upload hook

`claude_code_upload.py` uploads new lines of the current Claude Code session transcript
(and of every subagent transcript under `<session-id>/subagents/`) to the ORP API.
Stdlib only, runs with the system `python3`. It never prints to stdout and always exits 0;
errors go to `~/.orp/hook.log`.

## 1. Set the two env vars

Claude Code must be started from a shell that has them (e.g. in `~/.zshrc`):

```bash
export ORP_API_URL="http://localhost:8765"
export ORP_API_KEY="<your key>"
```

If either is missing the hook does nothing.

## 2. Register the hook

Add to `~/.claude/settings.json` (all projects) or `<repo>/.claude/settings.json`.
Replace `/ABS/PATH` with the absolute path of this repo.

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "python3 /ABS/PATH/hooks/claude_code_upload.py", "async": true } ] }
    ],
    "SubagentStop": [
      { "hooks": [ { "type": "command", "command": "python3 /ABS/PATH/hooks/claude_code_upload.py", "async": true } ] }
    ],
    "PreCompact": [
      { "hooks": [ { "type": "command", "command": "python3 /ABS/PATH/hooks/claude_code_upload.py", "async": true } ] }
    ],
    "SessionEnd": [
      { "hooks": [ { "type": "command", "command": "ORP_TIME_BUDGET=8 python3 /ABS/PATH/hooks/claude_code_upload.py", "timeout": 10 } ] }
    ]
  }
}
```

Why each event:

- **Stop** — fires after every finished turn; this is the main upload. `async` so the turn never waits on the network.
- **SubagentStop** — its input carries `agent_transcript_path`, so a subagent's own file is uploaded the moment it finishes (session id `<session_id>:<agent_id>`).
- **PreCompact** — not needed for correctness (the transcript file is append-only, compaction does not remove lines); it is a mid-turn flush for long autonomous turns where Stop has not fired yet. Must stay silent because PreCompact stdout becomes compaction instructions, and the script never writes to stdout.
- **SessionEnd** — final flush; also the only one that fires after a user interrupt (Stop does not). Not `async`, because the process is exiting; its default budget is 1.5 s, so `timeout: 10` raises it and `ORP_TIME_BUDGET=8` keeps the script inside it.

## Manual backfill

```bash
python3 hooks/claude_code_upload.py --file ~/.claude/projects/<project-dir>/<session-id>.jsonl [--session-id X]
```

Uploads that file and its `subagents/` folder. Safe to re-run: only lines the server does not have are sent.

## Optional env

| Variable | Default | Meaning |
| :- | :- | :- |
| `ORP_STATE_DIR` | `~/.orp/state` | per-session upload position (`next_offset`, byte position) |
| `ORP_HOOK_LOG` | `~/.orp/hook.log` | error log, trimmed at 256 KB |
| `ORP_TIME_BUDGET` | `20` (hook), `600` (backfill) | total seconds one run may spend; the rest is picked up by the next run |
