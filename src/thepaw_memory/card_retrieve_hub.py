# ============================================================
# Hub-first Retrieve — 顺着个人联想地图翻记忆 (Phase 2, 并行/实验)
#
# 老路 (card_retrieve.retrieve): 拿单句去撒网捞"字面/意思像"的卡。
# 新路 (这里):
#   query → 点亮地图上沾到的点(seed hubs) → 顺边联想点亮邻居
#          → 顺"点→卡"账本捞回挂在点上的卡 → 并入老相似法保底 → 排序
#
# 与 retrieve() 并存、同形返回，便于用 Phase 0 两把尺子 A/B 量。
# **线上未启用**：只有 bench / 显式调用走这里，生产检索仍走 retrieve()。
# ============================================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log
from thepaw_memory.card_store import CardStore, get_store
from thepaw_memory.memory_graph import get_graph, seed_vocab, _norm

_log = lambda event, detail="": feature_log("card_retrieve_hub", event, detail)

# query 分词时认得出实体名，靠把全库 subject/keyword 喂进 jieba。按 generation 缓存。
_seeded_gen: Dict[str, int] = {}


def _ensure_vocab(store: CardStore) -> None:
    pid = store.persona_id
    gen = store.generation
    if _seeded_gen.get(pid) == gen:
        return
    vocab: Set[str] = set()
    for c in store.all_active():
        s = (c.get("subject") or "").strip()
        if s:
            vocab.add(s)
        for kw in (c.get("keywords") or []):
            kw = (kw or "").strip()
            if kw:
                vocab.add(kw)
    seed_vocab(vocab)
    _seeded_gen[pid] = gen


def _query_seed_nodes(query: str, graph) -> Dict[str, float]:
    """query 里直接点名的点 → 满能量种子。jieba 切词后跟图节点对齐，
    只留确实在图里的点(没在图里=没卡提过，给不出联想)。"""
    import jieba
    toks = {_norm(t) for t in jieba.lcut(query or "")}
    toks = {t for t in toks if len(t) >= 2}
    if not toks:
        return {}
    dfs = graph.node_dfs(toks)
    return {n: 1.0 for n in dfs}


def _spread(
    seeds: Dict[str, float],
    graph,
    hops: int = 1,
    decay: float = 0.5,
    fanout: int = 12,
) -> Dict[str, float]:
    """从种子点顺着边联想：邻居能量 += 来源能量 × decay × (边权/该点总边权)。
    归一化(/总边权)= 通称节点连得多、每条分得少，自动稀释万能节点。"""
    lit: Dict[str, float] = dict(seeds)
    frontier: Dict[str, float] = dict(seeds)
    for _ in range(max(1, hops)):
        nxt: Dict[str, float] = {}
        for node, energy in frontier.items():
            nbrs = graph.neighbors(node, fanout)
            total = sum(w for _, w in nbrs) or 1
            for other, w in nbrs:
                add = energy * decay * (w / total)
                if add > 0:
                    nxt[other] = nxt.get(other, 0.0) + add
        for n, e in nxt.items():
            lit[n] = lit.get(n, 0.0) + e
        frontier = nxt
    return lit


def _idf_reweight(lit: Dict[str, float], graph, ncards: int) -> Dict[str, float]:
    """通称降权：跟 query 共享一个"小美"(到处都是)远不如共享一个稀有点有信息量。
    点的能量 × idf(df 越大权越小)。"""
    dfs = graph.node_dfs(lit.keys())
    out: Dict[str, float] = {}
    for n, e in lit.items():
        df = dfs.get(n, 1)
        idf = math.log((ncards + 1) / (df + 1)) + 1.0
        out[n] = e * idf
    return out


def retrieve_hub(
    query: str = "",
    persona_id: str = "",
    memory_count: int = 0,
    exclude_ids: set | None = None,
    *,
    alpha: float = 0.6,
    hops: int = 1,
) -> Dict[str, Any]:
    """从枢纽进 + 保底。alpha=hub 权重(0=纯相似≈老味道, 1=纯联想)。同形返回。"""
    if not memory_count:
        memory_count = cfg("card_retrieve.default_memory_count")
    if memory_count <= 0 or not (query or "").strip():
        return _empty("hub")

    store = get_store(persona_id)
    graph = get_graph(persona_id)
    _exclude = set(exclude_ids or ())
    _ensure_vocab(store)

    # 1. 相似命中：既当保底，又当"借相似卡找该亮的点"的引子
    sim_by_id: Dict[str, float] = {}
    sim_hits: List[Tuple[Dict[str, Any], float]] = []
    try:
        from thepaw_memory.card_retrieve import _get_card_index
        idx = _get_card_index(persona_id)
        sim_hits = idx.search(query, k=memory_count * 3)
        for card, s in sim_hits:
            cid = card.get("id", "")
            if cid:
                sim_by_id[cid] = max(sim_by_id.get(cid, 0.0), float(s))
    except Exception as e:
        _log("sim_fail", str(e)[:200])

    # 2. 种子点 = query 直接点名的 + 相似卡里的点(按卡相似度给能量)
    seeds = _query_seed_nodes(query, graph)
    seed_card_ids = [c.get("id", "") for c, _ in sim_hits[:8]]
    for cid, nodes in graph.nodes_for_cards(seed_card_ids).items():
        s = sim_by_id.get(cid, 0.0)
        for n in nodes:
            seeds[n] = max(seeds.get(n, 0.0), s)

    if not seeds and not sim_by_id:
        return _empty("hub")

    # 3. 联想 → 4. 通称降权 → 5. 顺账本捞卡
    lit = _spread(seeds, graph, hops=hops)
    ncards = graph.stats().get("carded", 1) or 1
    lit = _idf_reweight(lit, graph, ncards)
    hub_scores = graph.cards_for_nodes(lit)

    # 6. 归一 + 混合(保底)：两边各归到[0,1]再按 alpha 加权
    hub_max = max(hub_scores.values(), default=0.0) or 1.0
    sim_max = max(sim_by_id.values(), default=0.0) or 1.0
    blended: Dict[str, float] = {}
    for cid, hs in hub_scores.items():
        blended[cid] = alpha * (hs / hub_max)
    for cid, ss in sim_by_id.items():
        blended[cid] = blended.get(cid, 0.0) + (1.0 - alpha) * (ss / sim_max)

    # 7. 排序 + 取整卡
    ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
    cards: List[Dict[str, Any]] = []
    for cid, score in ranked:
        if cid in _exclude:
            continue
        card = store.get(cid)
        if not card:
            continue
        cards.append({**card, "_energy": round(float(score), 4),
                      "_source": "hub" if cid in hub_scores else "backstop"})
        if len(cards) >= memory_count:
            break

    feature_log("card_retrieve_hub", "retrieve_ok",
                f"persona={persona_id} seeds={len(seeds)} lit={len(lit)} "
                f"hub_cards={len(hub_scores)} sim={len(sim_by_id)} "
                f"final={len(cards)} alpha={alpha}")

    return {
        "cards": cards,
        "activated": cards,
        "search_mode": "hub",
        "total_raw": len(blended),
        "hits_debug": [],
    }


def _empty(mode: str) -> Dict[str, Any]:
    return {"cards": [], "activated": [], "search_mode": mode,
            "total_raw": 0, "hits_debug": []}
