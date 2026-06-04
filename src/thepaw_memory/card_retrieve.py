# ============================================================
# Yelin Card Retrieve — vector search + link spreading activation
#
# Replaces memory_retrieve.py for the card-based memory system.
# Flow: vector search → spread one hop → self-layer priority → co-activation boost
# ============================================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log
from thepaw_memory.card_store import CardStore, get_store, card_text as _card_text

_log = lambda event, detail="": feature_log("card_retrieve", event, detail)


# ============================================================
# Card-aware FAISS index
# ============================================================

_card_indices: Dict[str, "_CardIndex"] = {}


class _CardIndex:
    """FAISS index built from a CardStore's active cards."""

    def __init__(self, store: CardStore):
        self.store = store
        self._items: List[Dict[str, Any]] = []
        self._index = None
        self._built = False
        self._last_generation: int = -1

    def build(self, force: bool = False) -> None:
        """Build/rebuild index from store's searchable cards (active + archived)."""
        gen = self.store.generation
        if self._built and gen == self._last_generation and not force:
            return
        searchable = self.store.all_searchable()

        self._items = searchable

        if not self._items:
            self._index = None
            self._built = True
            self._last_generation = gen
            return

        try:
            import faiss
            from thepaw_memory.embeddings import encode

            texts = [_card_text(c) for c in self._items]
            embeddings = encode(texts)

            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)
            self._built = True
            self._last_generation = gen
        except Exception as e:
            _log("index_build_error", str(e)[:200])
            self._index = None
            self._built = True
            self._last_generation = gen

    def search(self, query: str, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar cards. Returns [(card, score), ...]."""
        if not self._built:
            self.build()
        if self._index is None or not self._items:
            return []

        try:
            from thepaw_memory.embeddings import encode_one
            q_emb = encode_one(query).reshape(1, -1)

            search_k = min(k * 2, self._index.ntotal)
            if search_k == 0:
                return []
            scores, indices = self._index.search(q_emb, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._items):
                    continue
                results.append((self._items[idx], float(score)))
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            _log("search_error", str(e)[:200])
            return []

    def search_by_embedding(self, query_emb, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """Search using a pre-computed embedding vector."""
        if not self._built:
            self.build()
        if self._index is None or not self._items:
            return []
        try:
            import numpy as np
            q = np.asarray(query_emb).reshape(1, -1)
            search_k = min(k * 2, self._index.ntotal)
            if search_k == 0:
                return []
            scores, indices = self._index.search(q, search_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._items):
                    continue
                results.append((self._items[idx], float(score)))
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            _log("search_emb_error", str(e)[:200])
            return []


def _get_card_index(persona_id: str) -> _CardIndex:
    """Get or create card index for a persona."""
    if persona_id not in _card_indices:
        store = get_store(persona_id)
        _card_indices[persona_id] = _CardIndex(store)
    idx = _card_indices[persona_id]
    idx.build()
    return idx


def search_by_embedding(
    persona_id: str,
    query_emb,
    k: int = 20,
    filter_ids: Optional[Set[str]] = None,
    min_score: float = 0.0,
) -> List[Tuple[Dict[str, Any], float]]:
    """Search the FAISS index with a pre-computed embedding.
    If filter_ids is set, only return cards whose id is in the set."""
    idx = _get_card_index(persona_id)
    hits = idx.search_by_embedding(query_emb, k=k * 3 if filter_ids else k)
    results = []
    for card, score in hits:
        if score < min_score:
            continue
        if filter_ids and card.get("id") not in filter_ids:
            continue
        results.append((card, score))
        if len(results) >= k:
            break
    return results


# ============================================================
# Main retrieval function
# ============================================================

def retrieve(
    query: str = "",
    persona_id: str = "",
    memory_count: int = 0,
    exclude_ids: set | None = None,
    rerank_context: List[str] | None = None,
    queries: List[str] | None = None,
) -> Dict[str, Any]:
    """Vector+keyword search → spreading activation → reranker.

    Seeds from FAISS + FTS5, energy propagates along temporal (same-session) /
    entity (same-name) co-occurrence edges, then dynamic cutoff. Re-ranks the
    seed pool only; no new cards are pulled in.
    When `queries` is provided, seeds are collected from all queries and merged.
    """
    if not memory_count:
        memory_count = cfg("card_retrieve.default_memory_count")

    store = get_store(persona_id)

    if memory_count <= 0:
        return _empty_result("disabled")

    _exclude = set(exclude_ids or ())

    _queries = queries if queries else ([query] if query else [])
    if not _queries:
        return _empty_result("no_query")

    raw_hits: List[Tuple[Dict[str, Any], float]] = []
    _seen_raw: Dict[str, int] = {}
    search_mode = "vector"

    try:
        idx = _get_card_index(persona_id)
        _use_faiss = True
    except Exception as e:
        _log("faiss_init_fail", f"persona={persona_id} error={e}")
        idx = None
        _use_faiss = False
        search_mode = "keyword"

    for q in _queries:
        # Step 1: Vector search
        if _use_faiss:
            try:
                q_hits = idx.search(q, k=memory_count * 3)
            except Exception as e:
                _log("faiss_fallback", f"persona={persona_id} error={e}")
                q_hits = _keyword_fallback(q, store, k=memory_count * 3)
                search_mode = "keyword"
        else:
            q_hits = _keyword_fallback(q, store, k=memory_count * 3)

        # Step 2: Hybrid keyword matching
        if cfg("card_retrieve.keyword_match_enabled"):
            kw_hits = _keyword_card_search(q, store)
            if kw_hits:
                existing_ids = {c.get("id", "") for c, _ in q_hits}
                for card, kw_score in kw_hits:
                    cid = card.get("id", "")
                    if cid in existing_ids:
                        q_hits = [(c, max(s, kw_score) if c.get("id") == cid else s)
                                    for c, s in q_hits]
                    else:
                        q_hits.append((card, kw_score))
                if search_mode == "vector":
                    search_mode = "hybrid"

        for card, score in q_hits:
            cid = card.get("id", "")
            if cid in _seen_raw:
                idx_existing = _seen_raw[cid]
                if score > raw_hits[idx_existing][1]:
                    raw_hits[idx_existing] = (card, score)
            else:
                _seen_raw[cid] = len(raw_hits)
                raw_hits.append((card, score))

    if len(_queries) > 1:
        search_mode = f"multi_{search_mode}"

    if not raw_hits:
        return _empty_result(search_mode)

    # Step 3: Threshold filter (cosine gate, not scoring)
    threshold = cfg("card_retrieve.score_threshold")
    legacy_threshold = cfg("card_retrieve.legacy_threshold")
    seeds: List[Tuple[Dict[str, Any], float]] = []
    seen_ids: Set[str] = set()

    for card, vector_sim in raw_hits:
        cid = card.get("id", "")
        if cid in _exclude or cid in seen_ids:
            continue
        seen_ids.add(cid)
        t = legacy_threshold if card.get("legacy") else threshold
        if vector_sim < t:
            continue
        seeds.append((card, vector_sim))

    if not seeds:
        return _empty_result(search_mode)

    # Step 4: Spreading activation — replaces parent chains + lateral neighbors + scoring
    activated = _spreading_activation(seeds, store, _exclude, memory_count * 2)

    # Step 5: Reranker (optional)
    if cfg("reranker.enabled") and activated:
        from thepaw_memory.reranker import rerank as _rerank
        rerank_query = "\n".join(_queries) if len(_queries) > 1 else (_queries[0] if _queries else query)
        if rerank_context:
            rerank_query = "\n".join(rerank_context + [rerank_query])
        result_cards = _rerank(rerank_query, activated, top_k=memory_count)
    else:
        result_cards = activated[:memory_count]

    hits_debug = [{
        "id": c.get("id", ""), "level": c.get("level", 0),
        "score": round(c.get("_energy", 0), 4),
        "content_preview": (c.get("content") or "")[:100],
        "source": c.get("_source", "spread"),
    } for c in activated]

    feature_log("card_retrieve", "retrieve_ok",
                f"persona={persona_id} seeds={len(seeds)} "
                f"activated={len(activated)} final={len(result_cards)} mode={search_mode}",
                data={"hits": hits_debug, "total_raw": len(raw_hits)})

    return {
        "cards": result_cards,
        "activated": activated,
        "search_mode": search_mode,
        "total_raw": len(raw_hits),
        "hits_debug": hits_debug,
    }


# ============================================================
# Entity name registry — derived from all cards' subject fields
# ============================================================

_entity_cache: Dict[str, Tuple[int, Set[str]]] = {}


def _get_entity_names(store: CardStore) -> Set[str]:
    """Collect known entity names from all active cards' subject fields.
    Cached per store generation to avoid full scan on every retrieval.
    TEMPORARY: subject-based extraction is a placeholder (#48).
    Should be replaced by a proper entity registry once the data source is decided."""
    pid = store.persona_id
    gen = store.generation
    cached = _entity_cache.get(pid)
    if cached and cached[0] == gen:
        return cached[1]
    names: Set[str] = set()
    for card in store.all_active():
        subj = (card.get("subject") or "").strip()
        if subj and len(subj) >= 2:
            names.add(subj)
    _entity_cache[pid] = (gen, names)
    return names


# ============================================================
# Spreading activation (replaces scoring + parent chains + lateral neighbors)
# ============================================================

def _is_dormant(card: Dict[str, Any]) -> bool:
    """衰减到 0 的卡是沉寂卡，不参与扩散传播。
    retention<0 是 core 永久锚点的哨兵值（永不衰减），不算沉寂。"""
    ret = card.get("retention")
    return ret is not None and ret == 0


def _spreading_activation(
    seeds: List[Tuple[Dict[str, Any], float]],
    store: CardStore,
    exclude_ids: Set[str],
    max_results: int,
    decay: float = 0.5,
    iterations: int = 2,
    energy_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Spreading activation over flat cards. Seeds get initial energy from vector_sim,
    energy propagates along temporal co-occurrence (same session) + entity co-reference
    (same entity name) edges. Dormant cards (retention<=0) are excluded from propagation.
    Re-ranks the seed pool only; brings in no new cards."""
    import numpy as np
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    pool: Dict[str, Dict[str, Any]] = {}
    for card, vsim in seeds:
        cid = card["id"]
        if cid in exclude_ids:
            continue
        if _is_dormant(card) and vsim < 0.6:
            continue
        days_old = 0
        created = card.get("created_at", "")
        if created:
            try:
                days_old = max(0, (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days)
            except Exception:
                pass
        recency_boost = math.exp(-days_old * 0.01)
        pool[cid] = {**card, "_energy": vsim * (0.7 + 0.3 * recency_boost), "_source": "seed"}

    ids = list(pool.keys())
    n = len(ids)
    if n <= 1:
        return sorted(pool.values(), key=lambda c: c["_energy"], reverse=True)[:max_results]

    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    adj = np.zeros((n, n), dtype=np.float32)
    dormant_mask = np.ones(n, dtype=np.float32)

    session_groups: Dict[str, List[int]] = {}
    entity_names = _get_entity_names(store)

    entity_groups: Dict[str, List[int]] = {}
    for cid, idx in id_to_idx.items():
        card = pool[cid]
        if _is_dormant(card):
            dormant_mask[idx] = 0.0
        sid = card.get("session_id") or (card.get("source_time") or "")[:10]
        if sid:
            session_groups.setdefault(sid, []).append(idx)
        content = card.get("content") or ""
        for name in entity_names:
            if name in content:
                entity_groups.setdefault(name, []).append(idx)

    for group in session_groups.values():
        if len(group) > 1:
            for a in group:
                for b in group:
                    if a != b:
                        adj[a, b] = max(adj[a, b], 0.3)

    for group in entity_groups.values():
        if len(group) > 1:
            for a in group:
                for b in group:
                    if a != b:
                        adj[a, b] = max(adj[a, b], 0.5)

    adj *= dormant_mask.reshape(-1, 1)
    adj *= dormant_mask.reshape(1, -1)

    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    energy = np.array([pool[cid]["_energy"] for cid in ids], dtype=np.float32)
    initial = energy.copy()

    for _ in range(iterations):
        spread = decay * (adj_norm.T @ energy)
        energy = initial + spread

    for i, cid in enumerate(ids):
        pool[cid]["_energy"] = float(energy[i])

    above = [c for c in pool.values() if c["_energy"] >= energy_threshold]
    above.sort(key=lambda c: c["_energy"], reverse=True)

    if len(above) > 2:
        energies = [c["_energy"] for c in above]
        for i in range(1, len(energies)):
            if energies[i - 1] > 0 and energies[i] / energies[i - 1] < 0.3:
                above = above[:i]
                break

    return above[:max_results]


# Region retrieval: name match (exact + suffix-variant) → entity summary 封面;
# FTS card content (BM25) → concrete cards. No reranker, no persona/model-tuned
# numbers — generality is structural, not calibrated.
_REGION_FTS_PER_WORD = 3          # cards per query word from FTS, BM25-ranked
_REGION_RARE_NAME_GF = 100        # below this general-corpus word frequency an entity
                                  # name counts as "rare/specific" and may be matched
                                  # by raw substring (segmentation-free). 遛狗=3 sits
                                  # here; the nearest generic junk name 记得=2545 is
                                  # ~25x above — the cut is a gap, not a tuned knob.
_REGION_ASSOC_MAX = 2             # 联想路径每轮最多带几个相关词条的摘要
_REGION_ASSOC_MIN_PMI = 1.0       # 联想 PMI 门槛：只跳够强的共现(语料内部信号，不绑模型)

# Search-side tokenizer — keep only nouns/proper-nouns (jieba POS), minus a standard
# Chinese stopword list + digits. This is what makes "你好 / 帮我写python / 我今天好累"
# resolve to empty: the chit-chat words aren't nouns. A general resource (746-word
# stopword list + POS), not tuned to any persona.
_STOPWORDS: Optional[Set[str]] = None


def _stopwords() -> Set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "cn_stopwords.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _STOPWORDS = {ln.strip() for ln in f if ln.strip()}
        except Exception as e:
            _log("stopwords_load_error", str(e)[:120])
            _STOPWORDS = set()
    return _STOPWORDS


def _search_tokens(query: str) -> List[str]:
    """Pull the locatable words out of a turn: nouns / proper nouns / English /
    abbreviations only (jieba POS), minus stopwords and digits. Verbs, particles
    and interjections (你好 / 帮 / 写 / 累) drop out, so chit-chat tokenizes to
    nothing and the region channel injects nothing."""
    import jieba.posseg as pseg
    sw = _stopwords()
    out: List[str] = []
    seen: Set[str] = set()
    for w, flag in pseg.cut(query or ""):
        w = w.strip().lower()
        if not w or w.isdigit() or w in sw or w in seen:
            continue
        # 名词 + 动名词(vn:训练/设计/研究)+ 形名词(an)+ 英文/简称。动名词是名词性
        # 内容词,必须留——否则"训练/分析"这类话题词被当动词丢掉、连经历卡都搜不到。
        # 纯动词/形容词/语气词(帮/写/累/吧)仍被剔,闲聊照样抠不出词。
        if flag[0] == "n" or flag in ("vn", "an", "eng", "j"):
            seen.add(w)
            out.append(w)
    return out


def region_activation(
    query: str,
    persona_id: str,
    session_id: str = "",
) -> Dict[str, Any]:
    """Always-on topic-background activation — public entry for the region channel.

    Wraps ``_select_region_cards`` in a retrieve()-shaped dict so callers route
    the output through the existing memory-cards channel (format_cards_for_prompt
    -> cache-safe msg_type=memory injection). Zero LLM cost beyond the reranker.

    ``session_id`` enables draw-without-replacement: cards already injected this
    conversation are dropped from the candidate pools.
    """
    cards = _select_region_cards(query, persona_id, session_id)
    if not cards:
        return _empty_result("region")
    return {
        "cards": cards,
        "activated": cards,
        "search_mode": "region",
        "total_raw": len(cards),
        "hits_debug": [],
    }


def _region_card(store: CardStore, cid: str, exclude: Set[str]) -> Optional[Dict[str, Any]]:
    """Resolve a card id to an active, non-empty, non-excluded card dict."""
    if cid in exclude:
        return None
    c = store.get(cid)
    if not c or c.get("legacy"):
        return None
    if not (c.get("content") or "").strip():
        return None
    return c


def _select_region_cards(
    query: str,
    persona_id: str,
    session_id: str = "",
) -> List[Dict[str, Any]]:
    """Always-on region retrieval — literal, zero-LLM, fully general.

    Work out which entity/topic the turn names, then hand back its summary (封面)
    plus the most on-point concrete cards. Names are matched two ways, because
    jieba segmentation alone is a single point of failure:

      (a) noun token == entity name — the precise path (handles common-word topics
          strictly so dirty generic "names" 记得/而是/方式 aren't amplified).
      (b) a *rare* entity name appearing literally in the raw turn (substring,
          segmentation-free). Rarity (general-corpus gf < _REGION_RARE_NAME_GF)
          restricts this to specific/coined entities (遛狗/Alice/Sam/lora…), so a
          literal hit is a real reference and jieba can't hide it — "今天遛狗了吗"
          merges 遛狗 into a verb and the noun filter drops it, yet the substring
          still matches. English/abbr names need a whole-token hit so short ones
          (ai/web/api) don't fire inside unrelated words.
      (c) suffix-variant: a token is the tail of a compound name (蛇 → Boa). Only
          this direction; the reverse surfaced junk (聊聊天 → 聊天) so it's dropped.

    Cards come from FTS over card text (BM25), per matched name and per leftover
    token. One budget (region_memory_count), summaries first, already-injected
    cards dropped (draw-without-replacement). Nothing matched → inject nothing
    (宁缺毋滥). Which of two same-named entities / literal-vs-metaphor it is — that
    is reading comprehension, left to the main brain.
    """
    from thepaw_memory.memory_graph import get_graph, general_word_freq
    from thepaw_memory.datastore import injected_card_ids_for_session

    toks = _search_tokens(query)
    graph = get_graph(persona_id)
    names = graph.summary_names()                    # names only — content fetched later
    name_set = set(names)

    # — which entity names does the turn mention? (ordered, de-duplicated) —
    matched: List[str] = []
    seen: Set[str] = set()

    def _mark(n: str) -> None:
        if n not in seen:
            seen.add(n)
            matched.append(n)

    tset = set(toks)
    for t in toks:                                  # (a) exact: noun token == name
        if t in name_set:
            _mark(t)
    ql = query.lower()                              # (b) rare name, raw substring
    for n in names:
        if n in seen or len(n) < 2 or general_word_freq(n) >= _REGION_RARE_NAME_GF:
            continue
        if n.isascii():
            if n in tset:        # english/abbr: whole-token only
                _mark(n)
        elif n in ql:
            _mark(n)
    for t in toks:                                  # (c) token is tail of a name
        for n in names:
            if n not in seen and n != t and n.endswith(t):
                _mark(n)

    # 联想路径：从命中的词条/话题词顺词典地图跳到强共现的*别的有摘要词条*。让"训练"这种
    # 没自己摘要的话题词，也能把 lora/语料/rlhf 这些相关词条的摘要带出来。PMI 卡门槛(语料
    # 内部信号、不绑模型)，只跳有摘要的真词条(prune 后干净)；残留少量低频垃圾词条(think)。
    assoc: List[str] = []
    assoc_cand: List[Tuple[str, float]] = []
    for w in toks:   # 只用原始抠词做种子，别用变体/子串派生的名字(否则伙伴被污染)
        for p, pmi in graph.strong_partners(w, top_k=5, min_pmi=_REGION_ASSOC_MIN_PMI):
            if p in name_set and p not in seen:
                assoc_cand.append((p, pmi))
    assoc_cand.sort(key=lambda x: x[1], reverse=True)
    for p, _pmi in assoc_cand:
        if len(assoc) >= _REGION_ASSOC_MAX:
            break
        if p in seen:
            continue
        seen.add(p)
        assoc.append(p)

    if not matched and not assoc and not toks:
        return []

    budget = cfg("card_retrieve.region_memory_count", persona_id) or 5
    store = get_store(persona_id)
    exclude: Set[str] = injected_card_ids_for_session(session_id) if session_id else set()

    result: List[Dict[str, Any]] = []
    chosen: Set[str] = set()

    # 摘要在前：先直接命中的词条，再联想带出的相关词条；经历卡在后。
    contents = graph.summary_contents(matched + assoc)
    for n in matched + assoc:
        if len(result) >= budget:
            break
        content = contents.get(n)
        if not content:
            continue
        sid = f"summary_{n}"
        if sid in exclude or sid in chosen:
            continue
        chosen.add(sid)
        result.append({
            "id": sid,
            "content": f"[{n}] {content}",
            "level": -1,
            "source": "summary_assoc" if n in assoc else "summary",
            "is_summary": True,
        })

    # concrete cards — FTS each matched name + each leftover token, BM25-ranked via
    # the store's public search_fts. cards_fts also indexes superseded/archived cards,
    # so over-fetch a window and keep the top few that still resolve to a live card
    # (_region_card drops the rest).
    fts_terms = matched + [t for t in toks if t not in seen]
    fts_window = max(_REGION_FTS_PER_WORD * 6, 20)
    for term in fts_terms:
        if len(result) >= budget:
            break
        kept = 0
        for cid in store.search_fts(term, limit=fts_window, ranked=True):
            if len(result) >= budget or kept >= _REGION_FTS_PER_WORD:
                break
            if cid in chosen:
                continue
            card = _region_card(store, cid, exclude)
            if card:
                chosen.add(cid)
                result.append(card)
                kept += 1

    return result[:budget]


# ============================================================
# Format for prompt injection
# ============================================================

def _fuzzy_time_prefix(card: Dict[str, Any]) -> str:
    from datetime import datetime, timezone
    created = card.get("created_at", "")
    if not created:
        return ""
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = (now - dt).days
        hours = (now - dt).total_seconds() / 3600
    except Exception:
        return ""

    cid = card.get("id", "")
    seed = sum(ord(ch) for ch in cid)
    pick = lambda opts: opts[seed % len(opts)]
    st = card.get("source_time", "")

    if days == 0:
        if hours < 2:
            return pick(["刚才，", "刚刚，"])
        if "早" in st or "上午" in st:
            return pick(["今天早上，", "今天上午，"])
        if "下午" in st:
            return pick(["今天下午，", "今天，"])
        if "晚" in st or "夜" in st:
            return pick(["今晚，", "今天晚上，"])
        return "今天，"

    if days == 1:
        if "晚" in st or "夜" in st or "凌晨" in st:
            return pick(["昨晚，", "昨天晚上，"])
        if "下午" in st:
            return pick(["昨天下午，", "昨天，"])
        return "昨天，"

    if days == 2:
        return pick(["前天，", "前两天，"])

    if days <= 4:
        return pick(["前几天，", "好几天前，"])

    if days <= 7:
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[dt.weekday()]
        return pick([f"上{wd}，", "好几天前，", "前几天，"])

    if days <= 14:
        return pick(["上周，", "一两周前，"])

    if days <= 30:
        return pick(["几周前，", "大半个月前，", "半个多月前，"])

    if days <= 60:
        return pick(["上个月，", "一个多月前，"])

    if days <= 120:
        return pick(["几个月前，", "两三个月前，"])

    return pick(["好几个月前，", "很久以前，", "很早之前，"])


def format_cards_for_prompt(
    result: Dict[str, Any],
) -> str:
    """Format selected cards for main-brain prompt injection.

    Node summaries (the "framework page") render first, verbatim; experience
    cards follow as bullet lines. Annotations follow their card.
    """
    cards = result.get("cards", [])
    if not cards:
        return ""

    # Node summaries (the "framework page") render first, verbatim — their
    # content is already "[node] …". They carry level=-1, so split them out
    # explicitly or they'd be mixed in with the experience cards.
    summaries = [c for c in cards if c.get("is_summary")]
    rest = [c for c in cards if not c.get("is_summary")]

    parts: list[str] = []
    for c in summaries:
        if c.get("content"):
            parts.append(c["content"])
    for c in rest:
        time_hint = _fuzzy_time_prefix(c)
        line = f"- {time_hint}{c['content']}"
        if c.get("needs_review"):
            line += "（⚠ 这条记忆可能不完整，需要确认）"
        parts.append(line)
        for anno in (c.get("annotations") or []):
            obs = (anno.get("text") or anno.get("observation") or "") if isinstance(anno, dict) else str(anno)
            if obs:
                parts.append(f"  ※ {obs}")
    return "\n".join(parts)


# ============================================================
# Hybrid keyword search — substring matching on keywords/tags/content
# ============================================================

def _keyword_card_search(
    query: str,
    store: CardStore,
) -> List[Tuple[Dict[str, Any], float]]:
    """FTS5-based keyword search. Returns empty if no FTS5 matches."""
    base_score = cfg("card_retrieve.keyword_base_score")

    hit_ids = store.search_fts(query)
    results = []
    for cid in hit_ids:
        card = store.get(cid)
        if card and not card.get("legacy"):
            results.append((card, base_score))
    return results


# ============================================================
# Keyword fallback (when vector search unavailable)
# ============================================================

def _keyword_fallback(
    query: str,
    store: CardStore,
    k: int = 15,
) -> List[Tuple[Dict[str, Any], float]]:
    """Simple keyword overlap fallback when FAISS unavailable."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for card in store.all_active():
        card_tokens = _tokenize(_card_text(card))
        if not card_tokens:
            continue
        overlap = len(query_tokens & card_tokens)
        if overlap == 0:
            continue
        score = overlap / (len(query_tokens) ** 0.5)
        scored.append((card, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _tokenize(text: str) -> Set[str]:
    """Simple CJK bigram + word tokenizer."""
    import re
    tokens: Set[str] = set()
    # English/number words
    for m in re.finditer(r'[a-zA-Z0-9]+', text):
        w = m.group().lower()
        if len(w) >= 2:
            tokens.add(w)
    # CJK: individual chars + bigrams
    cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', text)
    for run in cjk:
        for i in range(len(run)):
            if len(run[i:]) >= 2:
                tokens.add(run[i:i+2])
            tokens.add(run[i])
    return tokens


def _empty_result(search_mode: str) -> Dict[str, Any]:
    return {
        "cards": [],
        "activated": [],
        "search_mode": search_mode,
        "total_raw": 0,
        "hits_debug": [],
    }


# ============================================================
# Invalidation
# ============================================================

def invalidate(persona_id: str = "") -> None:
    """Force rebuild card index on next search."""
    if persona_id:
        if persona_id in _card_indices:
            _card_indices[persona_id]._built = False
    else:
        for idx in _card_indices.values():
            idx._built = False


# ============================================================
# Portrait injection
# ============================================================

def get_portrait_injection(persona_id: str, user_id: str | None = None) -> Dict[str, Any]:
    store = get_store(persona_id)
    portraits: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def _add(card):
        if not card or card.get("legacy") or card["id"] in seen_ids:
            return
        seen_ids.add(card["id"])
        portraits.append(card)

    _add(store.get_portrait("persona"))
    _add(store.get_portrait("user"))
    _add(store.get_portrait("relationship"))

    token_count = sum(len(c.get("content") or "") // 2 for c in portraits)

    return {"portraits": portraits, "token_count": token_count}
