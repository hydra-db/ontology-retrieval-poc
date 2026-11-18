# ontology-retrieval-poc

POC for ENG-4426. This branch holds the first part of the design: collecting agent-user transcripts and turning them, in the background, into typed per-user snippets that a retrieval engine can apply without a model in the query path.

Not built yet: the ontology store, the vector store, the retrieval engine, the MCP server, and evals for the snippets.

## What is here

```
Claude Code session ──hook──▶ POST /v1/sessions/{id}/events ──▶ raw events (SQLite)
                                                                     │
                                              orp dream  (lead: plain code)
                                                                     │
                              one analyzer per session, in groups of 20, parallel inside a group
                                                                     │
                                                     organizer, once per user
                                                                     │
                                                           typed snippets (SQLite)
```

- `hooks/claude_code_upload.py`: Claude Code hook. Uploads the lines of the session file that the server does not have yet. Everything in the session file is sent: messages, thinking, tool calls, tool results, skill loads.
- `src/orp/api.py`: the API. The bearer key identifies the user.
- `src/orp/transcripts.py`: turns raw events into the labelled text the analyzer reads.
- `src/orp/schema.py`: the snippet schema and the shared field list.
- `src/orp/harness/`: the dreaming harness, following Anthropic's lead / analyzer / organizer design. The lead is code. The analyzer and organizer are a model behind an OpenAI-compatible endpoint, calling the tools in `tools.py`.

## Snippets

Two kinds, split by what the snippet can do to search results.

| Kind | Right-hand side | Effect |
|---|---|---|
| `alias` | a name: a term, or an entity | adds ways to match; never removes a record |
| `constraint` | a condition on a field | `require` drops records that do not match; `prefer` only ranks matching ones higher |

Common fields: `statement`, `trigger_terms`, `evidence` (exact quotes with session id), `basis` (`stated` or `inferred`), `status` (`active`, `superseded`, `conflict`), `first_seen`, `last_seen`, `use_count`.

Checks done in code, not left to the model:

- every quote must appear word for word in the transcript;
- a constraint's field, op and value must fit the shared field list in `schema.py`;
- a constraint with no trigger terms is always on and may only be `prefer`;
- an inferred snippet cannot supersede a stated one; the two are marked as in conflict;
- nothing is deleted: a replaced or merged snippet is kept as `superseded`.

A durable preference that fits neither kind goes to a separate `unmapped` list, which search does not use.

## Run it

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest

.venv/bin/orp user add <name>        # prints the API key once
.venv/bin/orp serve                  # http://127.0.0.1:8765
```

Send transcripts: see `hooks/README.md`.

Dream (needs a model endpoint):

```bash
export ORP_LLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
export ORP_LLM_API_KEY=...
export ORP_ANALYZER_MODEL=Qwen/Qwen3.6-35B-A3B
.venv/bin/orp dream --user <name> --quiet-minutes 30
.venv/bin/orp snippets --user <name>
```

`orp dream` picks sessions that have new events and have been quiet for the given minutes. A session whose analyzer fails is picked up again on the next run.

## Open

- Evals for snippet quality and for whether snippets help retrieval.
- The entity resolver that fills `entity_id` on alias targets (needs the ontology store).
- The shared field list is provisional until the ontology schema is fixed.
- Sign-in for the MCP server; per-user keys for now.
