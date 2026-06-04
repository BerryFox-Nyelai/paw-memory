# ============================================================
# Yelin Memory Query — 自己抠词 + 地图扩写 (query construction layer)
#
# 不靠主脑吐 [recall] 关键词：从用户的话里**自己**抠出检索词，再顺着个人联想
# 地图把词补全，交给现有的 card_retrieve.retrieve() 排序。
#
# 这是"查询怎么搭"这一层，跟"用哪个检索器"解耦：
#   text → extract_keywords(自己抠) → expand_keywords(地图扩) → retrieve(queries=...)
# 以后 region_activation 那条常驻线路直接用这层即可(目前先供 bench A/B)。
# ============================================================
from __future__ import annotations

import math
from typing import Dict, List

from thepaw_memory.memory_graph import get_graph, seed_vocab, _norm, _STOP, _MIN_LEN

# 全库实体名喂进 jieba(认得出名字才抠得准)，按 generation 缓存。
_seeded_gen: Dict[str, int] = {}


def _ensure_vocab(persona_id: str) -> None:
    from thepaw_memory.card_store import get_store
    store = get_store(persona_id)
    gen = store.generation
    if _seeded_gen.get(persona_id) == gen:
        return
    vocab: set = set()
    for c in store.all_active():
        s = (c.get("subject") or "").strip()
        if s:
            vocab.add(s)
        for kw in (c.get("keywords") or []):
            kw = (kw or "").strip()
            if kw:
                vocab.add(kw)
    seed_vocab(vocab)
    _seeded_gen[persona_id] = gen


def extract_keywords(text: str, persona_id: str, top_k: int = 5) -> List[str]:
    """从一段话里自己抠检索词。优先认地图上已知的实体(Alice/Rex…)，
    且越稀有(df 越小)越靠前——稀有词更能定位，通称("模型""对话")往后排。"""
    text = (text or "").strip()
    if not text:
        return []
    _ensure_vocab(persona_id)
    import jieba
    import jieba.analyse

    graph = get_graph(persona_id)

    # TF-IDF 显著词 + 全切词，合成候选池(去停用、去单字、去重)
    pool: List[str] = []
    seen: set = set()
    for t in jieba.analyse.extract_tags(text, topK=top_k * 3):
        t = _norm(t)
        if len(t) >= _MIN_LEN and t not in _STOP and t not in seen:
            seen.add(t)
            pool.append(t)
    for t in jieba.lcut(text):
        t = _norm(t)
        if len(t) >= _MIN_LEN and t not in _STOP and t not in seen:
            seen.add(t)
            pool.append(t)
    if not pool:
        return []

    dfs = graph.node_dfs(pool)               # 哪些候选是地图认得的实体
    entities = sorted((t for t in pool if t in dfs), key=lambda t: dfs[t])  # 稀有优先
    non_entities = [t for t in pool if t not in dfs]
    return (entities + non_entities)[:top_k]


def expand_keywords(
    keywords: List[str],
    persona_id: str,
    max_total: int = 5,
    min_weight: int = 4,
    min_df_b: int = 8,
    max_df_a_frac: float = 0.15,
    pool: int = 80,
) -> List[str]:
    """把抠出的词顺着地图补全。三道闸夹出"特异且扎实"的关联词：
      - 源头闸：只从"有定位力"的词往外联想——Alice/Cleo这种霸占半库的通称
        (df≥max_df_a_frac·全库)谁都沾，从它出发只能得噪声，直接不联想。
      - 频率闸：共现≥min_weight 且 邻居本身 df≥min_df_b(排掉一次性碎词)。
      - 关联闸：按 PMI×log(共现) 排——PMI 衡量"比偶然多撞上多少"(通称
        Cleo/Bob跟谁都≈0、出局)，×log(共现) 偏向证据多的(别被稀有碎词刷分)。
    只回**新增**的扩写词(不含原词)。"""
    graph = get_graph(persona_id)
    N = max(1, graph.stats().get("carded", 1))
    max_df_a = max(min_df_b, int(N * max_df_a_frac))
    have = {_norm(k) for k in keywords if _norm(k)}
    kw_df = graph.node_dfs(list(have))
    cand: Dict[str, float] = {}
    for kw in keywords:
        a = _norm(kw)
        df_a = kw_df.get(a, 0)
        if df_a <= 0 or df_a > max_df_a:
            continue                      # 不认得 / 是霸库通称，都给不出有用联想
        nbrs = graph.neighbors(a, pool)
        df_bs = graph.node_dfs([_norm(b) for b, _ in nbrs])
        for b_raw, w in nbrs:
            b = _norm(b_raw)
            if not b or b in have or b in _STOP or len(b) < _MIN_LEN:
                continue
            df_b = df_bs.get(b, 1)
            if w < min_weight or df_b < min_df_b:
                continue
            pmi = math.log((w * N) / (df_a * df_b))
            if pmi <= 0:                  # 共现没超过偶然 = 没特异关联(俩都通称)
                continue
            score = pmi * math.log(w)
            cand[b] = max(cand.get(b, -9.0), score)
    ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    return [b for b, _ in ranked[:max_total]]


def build_queries(
    text: str,
    persona_id: str,
    *,
    expand: bool = True,
    top_k: int = 5,
    max_expand: int = 5,
) -> List[str]:
    """一步到位：自己抠词 (+地图扩写) → 给 retrieve(queries=...) 用的词表。"""
    kws = extract_keywords(text, persona_id, top_k=top_k)
    if not kws:
        return []
    if expand:
        kws = kws + expand_keywords(kws, persona_id, max_total=max_expand)
    return kws
