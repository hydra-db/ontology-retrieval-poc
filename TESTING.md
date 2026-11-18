# Testing the dreaming harness

What you can test today: your Claude Code sessions are uploaded as you work, and `orp dream` turns them into typed snippets. Search does not use the snippets yet; that part is not built.

All commands run from the repo root and read settings from `.env` (copy `.env.example`).

## 1. Check the setup

```bash
.venv/bin/orp doctor --user <name>
```

It checks, in order: the database, the API, your key, the Claude Code hook, the model endpoint, and that the model can call tools. Each failed line says what to do.

- API not running: `.venv/bin/orp serve` (leave it running in its own terminal).
- No user yet: `.venv/bin/orp user add <name>`, then save the printed key to `~/.orp/key`.
- Hook not registered: see `hooks/README.md`.
- No model endpoint: start a vLLM server for `Qwen/Qwen3.6-35B-A3B` and put its address and key in `.env`. The flags that work are in the README under "Trial run".

## 2. Get transcripts in

New sessions upload by themselves after each turn once the hook is registered. To pull in sessions from before that:

```bash
.venv/bin/orp backfill --user <name> --days 7 --dry-run   # list what would be uploaded
.venv/bin/orp backfill --user <name> --days 7
.venv/bin/orp status --user <name>
```

`status` lists the sessions waiting to be dreamed over. To read one the way the stored transcript looks:

```bash
.venv/bin/orp render --user <name> --session <session-id> | less
```

## 3. Dream

Start small, on one session, so you can read everything it did:

```bash
.venv/bin/orp dream --user <name> --quiet-minutes 0 --session <session-id>
```

Then everything that is waiting:

```bash
.venv/bin/orp dream --user <name> --quiet-minutes 0
```

- `--quiet-minutes 0` takes sessions that are still in use. The default of 30 skips anything with activity in the last half hour.
- `--limit N` takes only the N oldest waiting sessions.
- A session whose analyzer fails is not marked done and is picked up by the next run.

The output lists, per session, the analyzer's summary and every write or rejected call, then what the organizer changed. `.venv/bin/orp runs --user <name>` shows the last run again.

## 4. Read the result

```bash
.venv/bin/orp snippets --user <name> --evidence            # in force, with the quotes behind each
.venv/bin/orp snippets --user <name> --status all          # including superseded and conflicts
```

A line reads `"turbolay", "turbo lay" = HydraDB` for an alias, and `"shipped" -> state eq "Released"  [require]` for a constraint.

Things worth checking by eye:

- Is each snippet something you actually said or do, and is the quote the right evidence for it?
- Is the kind right: a name on the right-hand side for an alias, a field condition for a constraint?
- Is `require` used only where you said only, means or never?
- Would the trigger terms fire on queries where they should not?
- What did it miss that you know you said?

## 5. Run it again

To change a prompt (`src/orp/harness/prompts/`) and compare, forget what was learned and dream again. Transcripts are kept.

```bash
.venv/bin/orp reset --user <name> --yes
.venv/bin/orp dream --user <name> --quiet-minutes 0
```

## A known-answer check

`examples/make_sessions.py` writes five made-up sessions and prints what a good run should produce from them. Use a separate user so it does not mix with yours:

```bash
python3 examples/make_sessions.py
.venv/bin/orp user add demo        # note the key
for f in examples/sessions/*.jsonl; do
  ORP_API_URL=http://localhost:8765 ORP_API_KEY=<demo key> ORP_STATE_DIR=/tmp/orp-demo-state \
    python3 hooks/claude_code_upload.py --file "$f"
done
.venv/bin/orp dream --user demo --quiet-minutes 0
.venv/bin/orp snippets --user demo --status all
```

## What the analyzer is shown

Everything is stored. The analyzer is shown your messages in full, the agent's replies shortened to 500 characters, and each run of tool calls as one line (`TOOLS USED: Bash x12, Read x5`); the agent's private reasoning is left out. This keeps a day-long session to a few analyzer runs. To show it everything, set `ORP_ANALYZER_FULL=1`; expect long sessions to take several times longer.

Subagent transcripts are stored but not dreamed over, because their "user" is the parent agent.

## Limits to keep in mind

- No evals yet: nothing measures whether the snippets are right or whether they help search.
- Alias targets carry a name and a type but no ontology node id, because the ontology store does not exist yet.
- The fields a constraint can use are the provisional list in `src/orp/schema.py`.
- Only Claude Code sessions are captured.
