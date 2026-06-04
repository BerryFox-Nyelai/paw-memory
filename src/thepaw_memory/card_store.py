# ============================================================
# Yelin Card Store — unified memory card storage
#
# SQLite (WAL) per persona + in-memory index + FTS5 search.
# All memory types (event/self/understanding) share one schema.
# ============================================================
from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

import jieba

from thepaw_memory.config_registry import cfg
from thepaw_memory.utils import estimate_tokens
from thepaw_memory.core.errors import report_error
from thepaw_memory.feature_log import feature_log
from thepaw_memory.utils import gen_id, now_iso, safe_filename

_log = lambda event, detail="": feature_log("card_store", event, detail)

from thepaw_memory.paths import MEMORY_BASE

_jieba_ready = False

def _ensure_jieba() -> None:
    global _jieba_ready
    if not _jieba_ready:
        jieba.initialize()
        _jieba_ready = True


class CardTooLargeError(ValueError):
    pass


def _check_token_cap(content: str, is_portrait: bool = False) -> None:
    # 画像卡天生长(长期+中期理解),用单独的高上限；普通经历卡仍受 350字级的 cap 约束。
    cap = cfg("card_store.max_portrait_tokens") if is_portrait else cfg("card_store.max_card_tokens")
    tokens = estimate_tokens(content)
    if tokens > cap:
        raise CardTooLargeError(
            f"card content {tokens} tokens exceeds cap {cap}"
        )


def normalize_level(level) -> int:
    if isinstance(level, int):
        return max(level, 0)
    if isinstance(level, (float,)):
        return max(int(level), 0)
    if isinstance(level, str):
        s = level.strip().lower()
        if s.startswith("m") and s[1:].isdigit():
            return int(s[1:])
        if s.isdigit():
            return int(s)
    return 0


_HAS_WORD_CHAR = re.compile(r'[\w]', re.UNICODE)

def _fts_tokenize(text: str) -> str:
    if not text:
        return ""
    _ensure_jieba()
    return " ".join(t for t in jieba.cut(text) if _HAS_WORD_CHAR.search(t))


# ============================================================
# Card schema
# ============================================================

def new_card(
    *,
    content: str,
    level: Any = 0,
    keywords: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    links: Optional[List[Dict[str, Any]]] = None,
    card_id: Optional[str] = None,
    source: str = "conversation",
    session_id: Optional[str] = None,
    source_time: Optional[str] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    edit_history: Optional[List[Dict[str, Any]]] = None,
    card_type: Optional[str] = None,
    retention: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a new card dict with all required fields."""
    now = now_iso()
    lvl = normalize_level(level)
    if card_type is None and lvl == 0:
        card_type = "experience"
    if retention is None and lvl == 0 and card_type == "experience":
        retention = 8.0
    return {
        "id": card_id or gen_id("c_", length=8),
        "level": lvl,
        "content": content,
        "keywords": keywords or [],
        "tags": tags or [],
        "links": links or [],
        "created_at": now,
        "last_updated": now,
        "last_referenced": None,
        "reference_count": 0,
        "source": source,
        "session_id": session_id,
        "source_time": source_time,
        "subject": subject,
        "topic": topic,
        "annotations": annotations or [],
        "edit_history": edit_history or [{"timestamp": now, "old_content": None, "reason": "created"}],
        "card_type": card_type,
        "retention": retention,
    }


# ============================================================
# Store: per-persona SQLite + in-memory index
# ============================================================

class CardStore:
    """
    Per-persona card storage backed by SQLite (WAL mode).
    All cards kept in memory for fast reads; SQLite is the
    durable persistence layer.  FTS5 index for full-text search.
    """

    def __init__(self, persona_id: str):
        self.persona_id = safe_filename(persona_id)
        from thepaw_memory.core.registry import entity_registry
        if not entity_registry.get(self.persona_id):
            raise ValueError(f"unknown persona: {self.persona_id}")
        self._dir = MEMORY_BASE / self.persona_id
        self._db_path = self._dir / "cards.db"
        self._lock = threading.Lock()
        self._cards: Dict[str, Dict[str, Any]] = {}
        self._snapshot: Dict[str, Dict[str, Any]] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._generation: int = 0
        self._db_mtime: float = 0
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id   TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
                card_id UNINDEXED,
                content,
                tags,
                keywords,
                tokenize='unicode61'
            )
        """)
        self._conn.commit()

    def _load(self) -> None:
        self._cards.clear()
        cursor = self._conn.execute("SELECT id, data FROM cards")
        skipped = 0
        migrated = 0
        purged = 0
        for row in cursor:
            try:
                card = json.loads(row[1])
                if not isinstance(card, dict) or not card.get("id"):
                    skipped += 1
                    continue
                old_status = card.pop("status", None)
                if old_status in ("deleted", "superseded", "merged"):
                    purged += 1
                    continue
                if old_status == "legacy" or old_status == "archived":
                    card["legacy"] = True
                    migrated += 1
                elif old_status == "active":
                    migrated += 1
                self._cards[card["id"]] = card
            except Exception as e:
                skipped += 1
                report_error("warning", "card_store.load", e, f"card_id={row[0]}")
                continue
        if migrated or purged:
            self._conn.execute("DELETE FROM cards")
            self._conn.execute("DELETE FROM cards_fts")
            for card in self._cards.values():
                self._db_upsert(card)
            self._conn.commit()
            _log("migrated_status", f"persona={self.persona_id} migrated={migrated} purged={purged}")

        v2_migrated = self._migrate_v2_fields()
        if v2_migrated:
            self._rewrite()
            _log("migrated_v2", f"persona={self.persona_id} cards={v2_migrated}")

        stripped = self._strip_dead_fields()
        if stripped:
            self._rewrite()
            _log("stripped_dead", f"persona={self.persona_id} cards={stripped}")

        if not migrated and not purged and not v2_migrated and not stripped:
            orphans = self._conn.execute(
                "SELECT COUNT(*) FROM cards_fts WHERE card_id IS NULL OR card_id NOT IN (SELECT id FROM cards)"
            ).fetchone()[0]
            if orphans:
                self._rewrite()
                _log("fts_orphans_cleaned", f"persona={self.persona_id} orphans={orphans}")

        self._snapshot = copy.deepcopy(self._cards)
        try:
            self._db_mtime = self._db_path.stat().st_mtime
        except OSError:
            pass
        _log("loaded", f"persona={self.persona_id} cards={len(self._cards)} skipped={skipped}")

    def _check_external_changes(self) -> None:
        """Reload if the db file was modified externally (e.g. sqlite3 CLI)."""
        try:
            current_mtime = self._db_path.stat().st_mtime
        except OSError:
            return
        if current_mtime > self._db_mtime:
            with self._lock:
                if self._db_path.stat().st_mtime > self._db_mtime:
                    _log("external_change_detected", f"persona={self.persona_id}")
                    self._load()
                    self._generation += 1

    @property
    def generation(self) -> int:
        self._check_external_changes()
        return self._generation

    def reload(self) -> None:
        with self._lock:
            self._load()
            self._generation += 1

    # --- DB helpers (call within lock) ---

    def _db_upsert(self, card: Dict[str, Any]) -> None:
        card_id = card["id"]
        data = json.dumps(card, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO cards (id, data) VALUES (?, ?)",
            (card_id, data),
        )
        self._conn.execute("DELETE FROM cards_fts WHERE card_id = ?", (card_id,))
        self._conn.execute(
            "INSERT INTO cards_fts (card_id, content, tags, keywords) VALUES (?, ?, ?, ?)",
            (
                card_id,
                _fts_tokenize(card.get("content", "")),
                _fts_tokenize(" ".join(card.get("tags") or [])),
                _fts_tokenize(" ".join(card.get("keywords") or [])),
            ),
        )

    def _db_delete(self, card_id: str) -> None:
        self._conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        self._conn.execute("DELETE FROM cards_fts WHERE card_id = ?", (card_id,))

    # --- Read (unchanged — all from _cards dict) ---

    def get(self, card_id: str) -> Optional[Dict[str, Any]]:
        return self._snapshot.get(card_id)

    def _is_active(self, card: Dict[str, Any]) -> bool:
        return not card.get("legacy")

    def all_active(self) -> List[Dict[str, Any]]:
        return [c for c in self._snapshot.values() if self._is_active(c)]

    def all_searchable(self) -> List[Dict[str, Any]]:
        """Active cards + legacy cards with substantial content.
        Excludes portrait cards — they have a dedicated injection slot."""
        active = [c for c in self.all_active() if not c.get("is_portrait")]
        legacy = [c for c in self._snapshot.values()
                  if c.get("legacy") and not c.get("is_portrait")
                  and len(c.get("content") or "") > 50]
        return active + legacy

    def by_level(self, level) -> List[Dict[str, Any]]:
        target = normalize_level(level)
        return [
            c for c in self._snapshot.values()
            if self._is_active(c) and normalize_level(c.get("level", 0)) == target
        ]

    def count(self) -> int:
        return len(self.all_active())

    # --- Write ---

    def append(self, card: Dict[str, Any]) -> Dict[str, Any]:
        _check_token_cap(card.get("content", ""), bool(card.get("is_portrait")))

        card_id = card.get("id")
        if not card_id:
            card["id"] = gen_id("c_", length=8)
            card_id = card["id"]

        with self._lock:
            self._cards[card_id] = card
            self._db_upsert(card)
            self._conn.commit()
            self._generation += 1
            self._snapshot = copy.deepcopy(self._cards)

        _log("append", f"id={card_id} level={card.get('level')} source={card.get('source')}")
        return card

    def update(self, card_id: str, **fields) -> Optional[Dict[str, Any]]:
        if "content" in fields:
            _is_p = bool(fields.get("is_portrait") or (self._cards.get(card_id) or {}).get("is_portrait"))
            _check_token_cap(fields["content"], _is_p)

        with self._lock:
            card = self._cards.get(card_id)
            if not card:
                return None
            for k, v in fields.items():
                card[k] = v
            if "last_updated" not in fields:
                card["last_updated"] = now_iso()
            self._cards[card_id] = card
            self._db_upsert(card)
            self._conn.commit()
            self._generation += 1
            self._snapshot = copy.deepcopy(self._cards)
        return card

    def remove(self, card_id: str) -> bool:
        with self._lock:
            if card_id not in self._cards:
                return False
            del self._cards[card_id]
            self._db_delete(card_id)
            self._conn.commit()
            self._generation += 1
            self._snapshot = copy.deepcopy(self._cards)
        _log("remove", f"id={card_id} persona={self.persona_id}")
        return True

    def remove_links_involving(self, card_id: str) -> int:
        removed = 0
        with self._lock:
            for card in self._cards.values():
                links = card.get("links") or []
                before = len(links)
                card["links"] = [l for l in links if l.get("target_id") != card_id]
                removed += before - len(card["links"])
            own = self._cards.get(card_id)
            if own and own.get("links"):
                removed += len(own["links"])
                own["links"] = []
            if removed:
                self._rewrite()
        return removed

    def supersede(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Supersede a card: remove links + delete in one atomic lock."""
        with self._lock:
            for card in self._cards.values():
                links = card.get("links") or []
                card["links"] = [l for l in links if l.get("target_id") != card_id]
            snapshot = copy.deepcopy(self._cards.get(card_id))
            if card_id in self._cards:
                del self._cards[card_id]
            self._rewrite()
        _log("supersede", f"id={card_id} persona={self.persona_id}")
        return snapshot

    def _rewrite(self) -> None:
        """Sync all in-memory cards to SQLite. Call within lock."""
        self._conn.execute("DELETE FROM cards")
        self._conn.execute("DELETE FROM cards_fts")
        for card in self._cards.values():
            self._db_upsert(card)
        self._conn.commit()
        self._snapshot = copy.deepcopy(self._cards)
        self._generation += 1

    # --- Links ---

    def add_link(
        self,
        from_id: str,
        to_id: str,
        relation: str = "",
        weight: float = 1.0,
    ) -> bool:
        with self._lock:
            from_card = self._cards.get(from_id)
            to_card = self._cards.get(to_id)
            if not from_card or not to_card:
                return False

            links = from_card.get("links") or []
            for link in links:
                if link.get("target_id") == to_id:
                    link["weight"] = weight
                    link["relation"] = relation or link.get("relation", "")
                    self._db_upsert(from_card)
                    self._conn.commit()
                    self._generation += 1
                    self._snapshot = copy.deepcopy(self._cards)
                    return True

            links.append({
                "target_id": to_id,
                "relation": relation,
                "weight": weight,
            })
            from_card["links"] = links
            self._db_upsert(from_card)
            self._conn.commit()
            self._generation += 1
            self._snapshot = copy.deepcopy(self._cards)
        return True

    def get_linked(self, card_id: str, min_weight: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        card = self._snapshot.get(card_id)
        if not card:
            return []
        results = []
        for link in card.get("links") or []:
            w = link.get("weight", 0.0)
            if w < min_weight:
                continue
            target = self._snapshot.get(link.get("target_id", ""))
            if target and self._is_active(target):
                results.append((target, w))
        return results

    def get_linked_from(self, card_id: str, min_weight: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        results = []
        for card in self._snapshot.values():
            if not self._is_active(card):
                continue
            for link in card.get("links") or []:
                if link.get("target_id") == card_id:
                    w = link.get("weight", 0.0)
                    if w >= min_weight:
                        results.append((card, w))
        return results

    def get_portrait(self, portrait_type: str) -> Optional[Dict[str, Any]]:
        """Find non-legacy portrait card by type (user/persona/relationship)."""
        for card in self._snapshot.values():
            if (card.get("is_portrait")
                    and card.get("portrait_type") == portrait_type
                    and self._is_active(card)):
                return card
        return None

    def save_portrait(self, portrait_card: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a portrait card (upsert)."""
        cid = portrait_card.get("id", "")
        existing = self.get(cid) if cid else None
        if existing:
            fields = {k: v for k, v in portrait_card.items() if k != "id"}
            return self.update(cid, **fields)
        return self.append(portrait_card)

    # --- V2 migration ---

    def _migrate_v2_fields(self) -> int:
        """Backfill retention + card_type for L0 cards missing them."""
        count = 0
        for card in list(self._cards.values()):
            changed = False
            level = card.get("level", 0)
            if "card_type" not in card and level == 0:
                card["card_type"] = "experience"
                changed = True
            if "retention" not in card and level == 0 and card.get("card_type") == "experience":
                card["retention"] = 8.0
                changed = True
            if "last_updated" not in card:
                card["last_updated"] = card.get("created_at", now_iso())
                changed = True
            if changed:
                count += 1

        return count

    # Fields no longer part of the card schema, stripped on load (_strip_dead_fields).
    # MAINTENANCE CONTRACT: cards are loose dicts with no central schema, so retiring
    # a field means adding it here by hand — otherwise stale values silently survive
    # across versions. Split by provenance so the list stays self-explaining.

    # (A) Once carried real data; stripping is a one-time cleanup of old cards.
    _RETIRED_FIELDS = frozenset({
        "clarity", "significance",   # V1 strength dims → merged into `retention`
        "status",                    # V1 lifecycle flag → migrated to `legacy` (see _load)
        "draft_id",                  # draft layer (DraftStore) → V2 ingests directly
        "fatigue", "_wake_signal", "_collision_report",  # V1 retrieval/activation transients
        "confidence", "context",     # V1 fields, unused in V2
    })
    # (B) Planned in design but never written by any code — stripping is a pure no-op,
    # kept as a guard so a half-built field can't be resurrected without a deliberate
    # decision. Do NOT read these as "we used to write them."
    _UNBUILT_FIELDS = frozenset({
        "continues_from",  # event-arc traceback — evaluated 2026-05-30, deferred
        "source_seq",      # original-text traceback (session_id+source_seq) — never built
    })
    _DEAD_FIELDS = _RETIRED_FIELDS | _UNBUILT_FIELDS

    # V1 LLM-built link relations, replaced by the 3 auto-derived edges
    # (instance_of / time co-occurrence / entity). Dangling links are pruned alongside.
    _DEAD_LINK_RELATIONS = frozenset({
        "related", "extends", "extended_by", "contradicts",
    })

    def _strip_dead_fields(self) -> int:
        """Remove dead fields and dead/dangling links from all cards."""
        existing_ids = set(self._cards.keys())
        count = 0
        for card in self._cards.values():
            changed = False
            for f in self._DEAD_FIELDS:
                if f in card:
                    del card[f]
                    changed = True
            links = card.get("links")
            if links:
                clean = [
                    l for l in links
                    if l.get("relation") not in self._DEAD_LINK_RELATIONS
                    and l.get("target_id") in existing_ids
                ]
                if len(clean) != len(links):
                    card["links"] = clean
                    changed = True
            if changed:
                count += 1
        return count

    # --- Lifecycle helpers ---

    def batch_update(self, updates: List[Tuple[str, Dict[str, Any]]]) -> int:
        """Update multiple cards in one lock + one deepcopy.

        ``updates`` is a list of (card_id, {field: value, ...}).
        Returns number of cards actually updated.
        """
        if not updates:
            return 0
        now = now_iso()
        count = 0
        with self._lock:
            for card_id, fields in updates:
                if "content" in fields:
                    _is_p = bool(fields.get("is_portrait") or (self._cards.get(card_id) or {}).get("is_portrait"))
                    _check_token_cap(fields["content"], _is_p)
                card = self._cards.get(card_id)
                if not card:
                    continue
                for k, v in fields.items():
                    card[k] = v
                if "last_updated" not in fields:
                    card["last_updated"] = now
                self._db_upsert(card)
                count += 1
            if count:
                self._conn.commit()
                self._generation += 1
                self._snapshot = copy.deepcopy(self._cards)
        return count

    def update_reference(self, card_ids: List[str]) -> None:
        if not card_ids:
            return
        now = now_iso()
        with self._lock:
            changed_cards = []
            for cid in card_ids:
                card = self._cards.get(cid)
                if not card or not self._is_active(card):
                    continue
                card["last_referenced"] = now
                card["reference_count"] = card.get("reference_count", 0) + 1
                changed_cards.append(card)
            if changed_cards:
                for card in changed_cards:
                    self._db_upsert(card)
                self._conn.commit()
                self._generation += 1
                self._snapshot = copy.deepcopy(self._cards)

    def high_level_cards(self, min_level=1) -> List[Dict[str, Any]]:
        min_rank = normalize_level(min_level)
        return [
            c for c in self._snapshot.values()
            if self._is_active(c)
            and normalize_level(c.get("level", 0)) >= min_rank
        ]

    # --- Maintenance ---

    # --- Search ---

    def search_fts(self, query: str, *, limit: int = 0, ranked: bool = False) -> List[str]:
        """FTS5 search → matching card IDs. ``ranked=True`` orders by BM25 relevance
        (best first); ``limit > 0`` caps the rows returned."""
        if not query:
            return []
        tok = _fts_tokenize(query)
        if not tok.strip():
            return []
        sql = "SELECT card_id FROM cards_fts WHERE cards_fts MATCH ? AND card_id IS NOT NULL"
        params: tuple = (tok,)
        if ranked:
            sql += " ORDER BY rank"
        if limit > 0:
            sql += " LIMIT ?"
            params = (tok, limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            report_error("warning", "card_store.search_fts", e, f"query={query[:50]}")
            return []

    # --- Stats ---

    def stats(self) -> Dict[str, Any]:
        active = self.all_active()
        levels = {}
        for c in active:
            l = c.get("level", 0)
            levels[l] = levels.get(l, 0) + 1
        total_links = sum(len(c.get("links") or []) for c in active)
        return {
            "total": len(self._snapshot),
            "active": len(active),
            "levels": levels,
            "total_links": total_links,
        }


# ============================================================
# Singleton cache
# ============================================================

def card_text(card: Dict[str, Any]) -> str:
    """Build searchable text from a card."""
    parts = []
    v = card.get("content", "")
    if v:
        parts.append(v)
    for kw in card.get("keywords") or []:
        if kw:
            parts.append(str(kw))
    for tag in card.get("tags") or []:
        if tag:
            parts.append(str(tag))
    return " ".join(parts)


_stores: Dict[str, CardStore] = {}
_stores_lock = threading.Lock()


def get_store(persona_id: str, auto_create: bool = True) -> CardStore:
    pid = safe_filename(persona_id)
    if not auto_create and not (MEMORY_BASE / pid).is_dir():
        raise KeyError(f"persona '{pid}' not found")
    if pid not in _stores:
        with _stores_lock:
            if pid not in _stores:
                _stores[pid] = CardStore(pid)
    return _stores[pid]


def checkpoint_all() -> None:
    """Flush all open card-store WALs to disk (called on shutdown)."""
    for store in list(_stores.values()):
        try:
            with store._lock:
                store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
