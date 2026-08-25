# researcher

A local deep-research agent. Give it a question; it plans sub-questions,
researches them in parallel, checks its own coverage, and writes a cited report
you can verify.

Runs entirely against Ollama plus a Tavily key.

```bash
pip install -r requirements.txt
cp .env.example .env        # add TAVILY_API_KEY
ollama pull qwen3:8b
ollama pull nomic-embed-text   # optional: enables semantic note search

python main.py                              # interactive
python main.py "how has AI training compute scaled?"   # one-shot
pytest                                      # 79 tests, no network needed
```

## How it works

```
plan ──► research (×N, parallel) ──► gaps ──┬──► research (follow-ups) ...
                                            └──► report
```

**plan** turns the question into a brief and 3–6 independently researchable
sub-questions. This is the step that makes it a researcher rather than a search
wrapper: one query gets one perspective, and decomposing first gives the gap
check something concrete to measure coverage against.

**research** runs once per sub-question, fanned out with LangGraph's `Send` so
each executes concurrently in its own context. Each branch generates search
queries, retrieves *full page text* (not snippets), synthesises findings, and
compresses before returning. Raw page text never reaches the parent state —
that is how multi-agent research systems accumulate their way into
context-window failure.

**gaps** is a barrier. It sees every branch together and decides which
sub-questions are still unsupported and *what query would close each one*. That
last part is the difference between iterating and re-rolling: a pass/fail critic
can only tell you to try again, so a loop built on one repeats the same search.

**report** is a single synthesis pass. Research parallelises; synthesis does
not — writing sections concurrently produces output that repeats itself and
cites the same source under different numbers.

The loop stops on whichever comes first: no gaps left, or the round, search, or
source budget. All four are configurable in `.env`.

## Verifiable citations

This is the part worth understanding, because it is the difference between "the
model wrote some URLs" and "these claims are tied to documents we retrieved."

Every retrieved page enters a **source registry** as a record with a stable id
derived from its normalised URL — so two branches that independently land on the
same page produce the same id and deduplicate without coordinating.

Citations then move through four stages, all in code:

1. A sub-researcher sees *its own* sources numbered `1..N` and cites `[1]`.
   Small integers are the one part of this a local model does reliably.
2. `citations.remap_local_citations` rewrites those to global ids. Out-of-range
   numbers are dropped and reported.
3. `citations.validate` flags any marker not backed by the registry.
4. `citations.renumber` produces reader-facing `[1]`, `[2]` and a bibliography
   containing only sources actually cited.

No model is asked to check any of this. A fabricated citation is stripped and
counted, and the report says how many were dropped.

## Configuration

Everything lives in `config.py`, overridable from `.env`. The two that matter
most:

**Model tiering.** `FAST_MODEL` handles high-volume mechanical work — query
generation, extraction, routing. `SMART_MODEL` handles judgment — planning, gap
analysis, synthesis. They default to the same model so this runs unchanged on a
single-model install. Pointing `SMART_MODEL` at something larger (`qwen3:32b`,
or an API model) is the biggest quality lever available after retrieval, because
those three steps are exactly where an 8B model is weakest.

**Budgets.** `MAX_ROUNDS`, `MAX_SEARCHES`, `MAX_SOURCES`. A research run without
an explicit ceiling is a research run that sometimes doesn't end.

## Notes and QA

Reports save as Markdown with YAML frontmatter recording the query,
sub-questions, source count, rounds, and confidence — a note whose
sub-questions went unresolved is marked `confidence: low` rather than being
indistinguishable from one that didn't.

Asking about something already researched routes to a QA agent that searches
note *contents* semantically (local embeddings via `nomic-embed-text`, cached in
a JSON sidecar, cosine similarity in numpy — no vector database to run). If the
embedding model isn't pulled it falls back to keyword search rather than
failing.

## Layout

```
main.py                    CLI
config.py                  settings and budgets
state.py                   graph state, concurrency-safe reducers, source ids
llm.py                     model tiering, structured output with fallbacks
citations.py               citation validation and renumbering (pure functions)
agents/
  base.py                  tool-calling loop, capped and error-contained
  planner.py               question -> brief + sub-questions
  sub_researcher.py        one sub-question, isolated context
  gap_analyzer.py          coverage audit -> targeted follow-up queries
  reporter.py              single-pass synthesis
  qa.py                    question answering over saved notes
graph/
  research_graph.py        the graph
  router.py                research vs. notes, validated in code
tools/
  search.py                retrieval -> Source records
  file_tools.py            safe note storage with frontmatter
  notes_index.py           semantic search over notes
evals/                     mechanical evaluation harness
tests/                     79 tests, all offline
```

## Testing

The suite runs the whole graph — fan-out, reducers, the gap loop, budgets,
citation integrity — against fake models and fake search. No Ollama, no API key,
under a second.

```bash
pytest -q
```

That matters more than the count: a research agent whose control flow can only
be tested by making live API calls is one that never actually gets tested. If
you change a prompt or a budget, the tests tell you whether the machinery still
holds.

## Evaluation

```bash
python evals/run_eval.py --limit 3
```

Runs a fixed question set and scores citation validity, sub-question coverage,
and keyword recall — all mechanically, so numbers are comparable across runs.
`citation_validity` should be 1.0; anything less is a pipeline bug, not a model
quality issue.

The obvious next addition is an LLM judge for factual accuracy, using a model
strictly stronger than the one being evaluated. Put it behind a flag so the
mechanical numbers stay comparable.

## Notes on what changed

This replaces an earlier single-pass version (supervisor → researcher → critic →
note writer → QA). The substantive changes, beyond the architecture above:

- The tool-calling loop has an iteration cap and per-tool error containment. It
  previously ran `while True:` with an unguarded `tool.invoke`, so one search
  timeout ended the process.
- Routing decisions are constrained to an enum and validated against the allowed
  set before use. A raw model string never addresses a graph node.
- The CLI uses a fresh thread id per query, and prints the report *before*
  asking whether to save it. Persistence is no longer a gate on output.
- Note filenames go through a character allowlist, so a model-generated topic
  can't walk out of `notes/` with `../`.
- `SqliteSaver` replaces `MemorySaver`, so checkpoints survive the process.

Human-in-the-loop before saving moved from a graph interrupt to a plain CLI
prompt. The graph's job is research; persistence is the caller's decision, and
an `interrupt_before` that can leave a thread parked indefinitely is a bad trade
for a confirmation dialog.

### Superseded files

The v1 modules have been deleted. They map onto the current code as:

```
agents/researcher.py      -> agents/sub_researcher.py
agents/critic.py          -> agents/gap_analyzer.py
agents/notes_writer.py    -> agents/reporter.py + tools/file_tools.py
agents/qa_agent.py        -> agents/qa.py
graph/supervisor.py       -> graph/research_graph.py + graph/router.py
```

Existing `notes/*.txt` files are not read by the new code, which expects `.md`
with frontmatter. Renaming them to `.md` makes them visible again — they'll have
no frontmatter, which the loader handles.

To revert everything: `git checkout .` — the previous implementation is intact
in your history.
