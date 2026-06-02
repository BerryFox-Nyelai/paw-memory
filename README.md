# The Paw Memory

A self-contained, card-based long-term memory engine for chat backends.

Card-based memory with a co-occurrence "dictionary map", auto-generated node
summaries, and a 5-step retrieval pipeline (summary → drill-down → one-hop
association → vector fallback). Drop it into any backend: provide two async LLM
callables and call four entry points.

```python
from thepaw_memory import MemoryEngine

engine = MemoryEngine(
    "data/memory",            # per-persona stores land under here
    subbrain=my_subbrain,     # async (messages, *, max_tokens, temperature, persona_id) -> str
    main_brain=my_main_brain, # async (persona_id, messages, *, max_tokens, temperature) -> str
)

# register a persona once before using it (idempotent; creates data/memory/alice/)
engine.register_persona("alice", display_name="Alice", user_display_name="Bob")

# read (per turn, no LLM cost)
r = engine.retrieve("alice", recent_dialogue_text, session_id="sess-1")
print(r["prompt_text"])       # ready to inject into the model prompt

# write (post-response, background)
await engine.ingest("alice", evicted_messages, session_id="sess-1")

# nightly maintenance
await engine.review("alice", today_cards)
```

> `ingest()` and `review()` require the LLM seams — `main_brain` harvests cards,
> `subbrain` writes summaries during review. Without them, only the manual
> `store()` + `retrieve()` paths work. `review()` does **not** self-schedule: the
> host invokes it (e.g. a nightly cron), once per persona per day — the engine
> ships no timer or background loop of its own.

> `retrieve()` needs the `[vectors]` extra installed — without an embedding
> backend it degrades to an empty result (no cards) rather than raising.

Install with vector retrieval:

```bash
pip install -e ".[vectors]"
```

> Status: early extraction (v0.1). The 14 core modules are lifted from a
> production memory engine with their logic unchanged (only import paths and a
> few comment examples were adjusted); the LLM/provider plumbing is replaced by
> injected seams. Config defaults (including tuned prompts) ship in
> `thepaw_memory/config/`.
>
> The bundled example prompts are tuned for a first-person intimate-companion
> persona (the assistant says "I" and addresses the user as a close companion).
> These are defaults, not requirements — edit `thepaw_memory/config/prompts.yml`
> to match your own tone and relationship model.
>
> Provided as-is, with **no commitment to maintenance, support, or responding to
> issues**. MIT licensed — fork freely.
