# Changelog

All notable changes to **The Paw Memory**. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.1] — 2026-06-06

Bug-fix release. No data migration — `pip install --force-reinstall "the_paw_memory-0.2.1-py3-none-any.whl[vectors]"` and you're done.

### Fixed
- **Retention decay no longer silently halts.** `_decay_retention` mixed a tz-aware
  `base` with a `now` whose tz-awareness depended on the runtime; if `now` resolved
  to naive, the subtraction threw and was swallowed by the caller — stopping *all*
  decay with no error. `now` is now forced tz-aware, so decay is timezone-independent.
- **Nightly-review portrait buffer is idempotent by date.** The "reviewed today" flag
  is only persisted at end-of-tick, so a mid-tick crash could re-enter and append a
  second same-day entry to each portrait's `recent_buffer`. Same-day entries are now
  replaced, not stacked.
- **Portrait-split failure skips instead of cross-contaminating.** When the sub-brain
  split step errored, the whole undifferentiated daily summary was fanned into all
  three portrait buffers (self / other / relationship). It now skips, matching the
  empty-parse path; the next clean run catches up.
- **Short corrections are no longer dropped.** The correction-match length floor went
  from `>10` to `>=4` (the parser accepts any non-empty original; the old floor
  silently discarded valid short facts). Corrections only append to `edit_history`
  (non-destructive), so the looser match is low-risk.
- **Region retrieval None-guards.** No more `NoneType` errors when the summary-name
  list contains `None` entries or the query is empty.

### Changed
- `main_brain` seam now accepts a `session_id` parameter (for parity with host apps
  that use it for provider-side session caching) and exposes a `call_raw` alias.
  `session_id` is **not** forwarded to your `main_brain` callback — its contract
  `(persona_id, messages, *, max_tokens, temperature)` is unchanged.

## [0.2.0] — 2026-06-03

### Added
- **Literal region retrieval**: exact name + tail-variant matching + rare-name
  substring → summaries; keywords → FTS episodic cards; plus an **association path**
  (mentioning A surfaces strongly co-occurring B). More precise recall, fewer
  irrelevant cards.

### Changed
- **Summary gate is now "rare-enough = specific name"**, dropping the unstable
  cohesion axis — common-word topics no longer get vague summaries (they rely on
  episodic cards). The gate **no longer needs an embedding model**.
- **Portraits get a 5000-token cap** (they're inherently long; the old 500 cap made
  every nightly portrait update fail and freeze). Existing frozen portraits revive
  on the next nightly run.

### Removed
- Harvesting no longer requires the main brain to emit a `subject` field.

### Migration
Run once per persona after upgrading (idempotent, touches only that persona's data):
```python
from thepaw_memory import MemoryEngine
engine = MemoryEngine("data/memory")
for pid in ["alice", "bob"]:
    print(pid, engine.migrate(pid))   # -> {'pruned': N, 'kept': M}
```
