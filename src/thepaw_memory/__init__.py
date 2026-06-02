"""thepaw_memory — a self-contained, card-based memory engine.

Card-based long-term memory with a co-occurrence "dictionary map", node
summaries, and a 5-step retrieval pipeline. Drop it into any chat backend:
provide two LLM callables (a main brain to harvest memories, a subbrain for
summaries / nightly review) and call the four entry points.

    from thepaw_memory import MemoryEngine

    engine = MemoryEngine("data/memory", subbrain=my_sb, main_brain=my_mb)
    r = engine.retrieve(persona_id, recent_text)        # read
    await engine.ingest(persona_id, evicted_messages)   # write
    await engine.review(persona_id, today_cards)        # nightly
"""
from thepaw_memory._engine import MemoryEngine

__all__ = ["MemoryEngine"]
__version__ = "0.1.1"
