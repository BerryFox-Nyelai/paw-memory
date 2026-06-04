# ============================================================
# Yelin Co-occurrence Graph — 个人联想地图 (shadow, Phase 1)
#
# 每张记忆卡落库时，抽出它的节点(实体/关键词/内容显著词)，把同卡内的节点两两
# 连边、权重累加。这就是"谁老跟谁一起出现"的关联地图 —— 记忆里那张静态的网。
# (扩散激活 = 这张网上跑的光，同一个东西、一静一动。)
#
# Phase 1 = shadow：只建图、只写独立 graph.db，绝不参与检索。Phase 2 才让检索
# 顺着这张网联想。存储 per-persona，mirror CardStore 的 SQLite(WAL) 范式。
# ============================================================
from __future__ import annotations

import json
import re
import sqlite3
import threading
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import jieba
import jieba.analyse

from thepaw_memory.paths import MEMORY_BASE
from thepaw_memory.feature_log import feature_log
from thepaw_memory.utils import now_iso, safe_filename

_log = lambda event, detail="": feature_log("memory_graph", event, detail)

# 通称/代词/高频虚词：当节点会连爆成"万能节点"(可视化里也最丑)，直接不收。
# = 设计里"通称降权"的最硬版本(不降权、直接丢)。
_STOP: Set[str] = {
    "她", "他", "它", "你", "我", "您", "咱", "俺", "谁", "自己", "大家",
    "他们", "她们", "它们", "我们", "你们", "咱们",
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里",
    "什么", "怎么", "为什么", "怎样", "哪", "哪个", "哪些", "多少",
    "因为", "所以", "但是", "然后", "就是", "还是", "如果", "虽然", "而且", "或者",
    "现在", "今天", "昨天", "明天", "时候", "一下", "一直", "已经", "可能", "应该",
    "一个", "没有", "可以", "知道", "觉得", "东西", "事情", "其实", "这种", "一种",
    "不是", "就是", "一样", "还有", "比如", "这么", "那么", "之类", "等等",
}

_MIN_LEN = 2                # 节点最短长度(单字噪音多，丢)
_CONTENT_TOPK = 12         # 每张卡从 content 抽几个 TF-IDF 显著词
_MAX_NODES_PER_CARD = 16   # 单卡节点封顶，防 all-pairs 边数爆炸

# 复合 subject/keyword 拆成多个实体：「Alice、Bob」→ Alice + Bob。
# 只切标点/空白(安全)，不切"和/与/跟"(怕劈开"和解""参与"这类词)。
_SPLIT_RE = re.compile(r"[、，,；;/／|｜&＆+＋\s]+")


def _norm(term: str) -> str:
    # 统一小写：Claude/claude、AI/ai、LoRA/lora 归一(中文不受影响)
    return (term or "").strip().lower()


def _split(raw: str) -> List[str]:
    return [p for p in _SPLIT_RE.split(raw or "") if p]


# 英文虚词：jieba 是中文词典，me/she/her/so 在通用词频里查无(gf=0)，会被名字轴误当
# "生造名字"。单列一张表直接剔。
_EN_STOP: Set[str] = {
    "the", "a", "an", "i", "me", "my", "mine", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "we", "us", "our", "ours", "they", "them", "their", "theirs",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "and", "or", "but", "so", "as", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "this", "that", "these", "those",
    "not", "no", "yes", "if", "then", "than", "too", "very", "just",
}

# 词条配不配摘要：通用词频低于此 = 够生僻 = 特定名字的词条(人/宠物/地点/自造词)。
# 阈值落在真词条(男朋友 326 封顶)和通用废词(记得 2545 起跳)之间那道宽沟里，不敏感；
# gf 来自通用词典，对所有 persona 一样、不看本地经历卡多少。
_SUMMARY_NAME_MAX_GF = 500

# 通用语料词频参照(判"是不是生造名字"用)。必须独立 pristine 分词器：全局 jieba 被
# seed_vocab/add_word 喂过实体名，FREQ 已污染，拿它查会把实体误当常用词。
_gen_tok = None


def _gen_tokenizer():
    global _gen_tok
    if _gen_tok is None:
        t = jieba.Tokenizer()
        t.initialize()
        _gen_tok = t
    return _gen_tok


def general_word_freq(word: str) -> int:
    """通用语料词频(0 = 查无 = 生造/极生僻)。公开入口，给"这是常用词还是特定名字"
    这个判断复用——别去 import 私有的 _gen_tokenizer 抓 FREQ。"""
    return _gen_tokenizer().FREQ.get(word, 0)


def seed_vocab(names: Iterable[str]) -> None:
    """把已知实体名喂进 jieba 词典 —— content 分词时名字(Alice/Bob…)不被切碎，
    才连得出"谁和谁"的边(认人地基)。idempotent，live 路径也可随时补。"""
    for n in names:
        n = _norm(n)
        if len(n) >= _MIN_LEN:
            jieba.add_word(n)


def extract_nodes(card: Dict[str, Any]) -> List[str]:
    """从一张卡切出节点：subject(最干净的实体名) + keywords + content 显著词。
    去重、去停用、单卡封顶。"""
    nodes: List[str] = []
    seen: Set[str] = set()

    def _add(term: str) -> None:
        t = _norm(term)
        if len(t) < _MIN_LEN or t in _STOP or t in seen:
            return
        seen.add(t)
        nodes.append(t)

    for piece in _split(card.get("subject") or ""):
        _add(piece)
    for kw in (card.get("keywords") or []):
        for piece in _split(kw):
            _add(piece)
    content = card.get("content") or ""
    if content:
        for term in jieba.analyse.extract_tags(content, topK=_CONTENT_TOPK):
            _add(term)
    return nodes[:_MAX_NODES_PER_CARD]


# ============================================================
# 存储：per-persona graph.db (nodes / edges)
# ============================================================

class CooccurrenceGraph:
    def __init__(self, persona_id: str):
        self.persona_id = safe_filename(persona_id)
        self._dir = MEMORY_BASE / self.persona_id
        self._db_path = self._dir / "graph.db"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                name      TEXT PRIMARY KEY,
                df        INTEGER NOT NULL DEFAULT 0,
                last_seen TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                a         TEXT NOT NULL,
                b         TEXT NOT NULL,
                weight    INTEGER NOT NULL DEFAULT 0,
                last_seen TEXT,
                PRIMARY KEY (a, b)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(a)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(b)")
        # 点→卡账本(倒排)：顺着点亮的节点找回挂在上面的卡。Phase 2 检索靠它。
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS card_nodes (
                card_id TEXT NOT NULL,
                node    TEXT NOT NULL,
                PRIMARY KEY (card_id, node)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_card_nodes_node ON card_nodes(node)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_card_nodes_card ON card_nodes(card_id)")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                node_name       TEXT PRIMARY KEY,
                content         TEXT NOT NULL DEFAULT '',
                card_count      INTEGER NOT NULL DEFAULT 0,
                new_since_write INTEGER NOT NULL DEFAULT 0,
                last_rewrite    TEXT,
                edit_history    TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self._conn.commit()

    # -- 写 --------------------------------------------------

    def _bump(self, nodes: List[str], when: str, card_id: Optional[str] = None) -> None:
        """假定已持锁、不 commit。df 按卡去重计数(=出现在几张卡里)；
        边权=两节点共同出现的卡数。给了 card_id 就同时记"点→卡"账本。"""
        uniq = sorted({n for n in nodes if n})
        c = self._conn
        for n in uniq:
            c.execute(
                "INSERT INTO nodes(name, df, last_seen) VALUES(?,1,?) "
                "ON CONFLICT(name) DO UPDATE SET df=df+1, last_seen=excluded.last_seen",
                (n, when),
            )
        # Increment new_since_write for any node that already has a summary
        for n in uniq:
            c.execute(
                "UPDATE summaries SET new_since_write = new_since_write + 1 "
                "WHERE node_name = ?", (n,)
            )
        for a, b in combinations(uniq, 2):
            c.execute(
                "INSERT INTO edges(a, b, weight, last_seen) VALUES(?,?,1,?) "
                "ON CONFLICT(a, b) DO UPDATE SET weight=weight+1, last_seen=excluded.last_seen",
                (a, b, when),
            )
        if card_id:
            for n in uniq:
                c.execute(
                    "INSERT OR IGNORE INTO card_nodes(card_id, node) VALUES(?,?)",
                    (card_id, n),
                )

    def add_card(self, nodes: List[str], when: Optional[str] = None,
                 card_id: Optional[str] = None) -> None:
        """回填用：写但不 commit，由 caller 控批 commit()。"""
        with self._lock:
            self._bump(nodes, when or now_iso(), card_id)

    def record_card(self, nodes: List[str], when: Optional[str] = None,
                    card_id: Optional[str] = None) -> None:
        """live 用：写 + 立即 commit。"""
        with self._lock:
            self._bump(nodes, when or now_iso(), card_id)
            self._conn.commit()

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()

    def clear(self) -> None:
        """重建式回填前清空。"""
        with self._lock:
            self._conn.execute("DELETE FROM nodes")
            self._conn.execute("DELETE FROM edges")
            self._conn.execute("DELETE FROM card_nodes")
            self._conn.commit()

    # -- 读(检视/后续 Phase 2 用) -----------------------------

    def top_nodes(self, n: int = 30) -> List[Tuple[str, int]]:
        return self._conn.execute(
            "SELECT name, df FROM nodes ORDER BY df DESC, name LIMIT ?", (n,)
        ).fetchall()

    def top_edges(self, n: int = 30) -> List[Tuple[str, str, int]]:
        return self._conn.execute(
            "SELECT a, b, weight FROM edges ORDER BY weight DESC, a, b LIMIT ?", (n,)
        ).fetchall()

    def neighbors(self, node: str, n: int = 15) -> List[Tuple[str, int]]:
        node = _norm(node)
        return self._conn.execute(
            "SELECT other, weight FROM ("
            "  SELECT b AS other, weight FROM edges WHERE a=? "
            "  UNION ALL "
            "  SELECT a AS other, weight FROM edges WHERE b=?"
            ") ORDER BY weight DESC, other LIMIT ?",
            (node, node, n),
        ).fetchall()

    def node_dfs(self, names: Iterable[str]) -> Dict[str, int]:
        """批量取若干节点的 df(出现在几张卡里)。只回存在的节点。
        Phase 2 用 df 做"通称降权"：越普遍的点(df 大)点亮后给的能量越小。"""
        names = [_norm(n) for n in names if _norm(n)]
        if not names:
            return {}
        qmarks = ",".join("?" * len(names))
        rows = self._conn.execute(
            f"SELECT name, df FROM nodes WHERE name IN ({qmarks})", tuple(names)
        ).fetchall()
        return {name: df for name, df in rows}

    def nodes_for_cards(self, card_ids: Iterable[str]) -> Dict[str, List[str]]:
        """反查若干卡各自的节点(从账本读，跟图里的点一致)。
        Phase 2 用相似命中的几张卡作种子时，取它们的点来起头联想。"""
        ids = [c for c in card_ids if c]
        if not ids:
            return {}
        qmarks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT card_id, node FROM card_nodes WHERE card_id IN ({qmarks})",
            tuple(ids),
        ).fetchall()
        out: Dict[str, List[str]] = {}
        for cid, node in rows:
            out.setdefault(cid, []).append(node)
        return out

    def cards_for_nodes(self, node_weights: Dict[str, float]) -> Dict[str, float]:
        """给一组点亮的节点(node→能量)，顺着"点→卡"账本捞回挂在上面的卡，
        返回 card_id→累计分(=该卡命中的点的能量之和)。Phase 2 的核心查询。"""
        weights = {_norm(n): w for n, w in node_weights.items() if _norm(n)}
        if not weights:
            return {}
        qmarks = ",".join("?" * len(weights))
        rows = self._conn.execute(
            f"SELECT card_id, node FROM card_nodes WHERE node IN ({qmarks})",
            tuple(weights.keys()),
        ).fetchall()
        scores: Dict[str, float] = {}
        for card_id, node in rows:
            scores[card_id] = scores.get(card_id, 0.0) + weights.get(node, 0.0)
        return scores

    def topic_sharpness(self, node: str, top_k: int = 10) -> float:
        """Average PMI of a node's top-K edges. High = real topic with its own
        vocabulary; low/negative = meta-word that borrows everyone else's words.

        PMI(a,b) = log( (weight_ab * N) / (df_a * df_b) )
        where N = total carded documents.
        """
        import math
        node = _norm(node)
        N = max(1, self._conn.execute(
            "SELECT COUNT(DISTINCT card_id) FROM card_nodes"
        ).fetchone()[0])
        df_a = (self._conn.execute(
            "SELECT df FROM nodes WHERE name = ?", (node,)
        ).fetchone() or (0,))[0]
        if df_a <= 0:
            return 0.0
        nbrs = self.neighbors(node, top_k)
        if not nbrs:
            return 0.0
        dfs_b = self.node_dfs([b for b, _ in nbrs])
        pmis = []
        for b, w in nbrs:
            df_b = dfs_b.get(b, 1)
            if df_b <= 0 or w <= 0:
                continue
            pmi = math.log((w * N) / (df_a * df_b))
            pmis.append(pmi)
        return sum(pmis) / len(pmis) if pmis else 0.0

    def cards_for_node(self, node: str, limit: int = 20) -> List[str]:
        """Card ids attached to a node, most-recent first (rowid desc)."""
        node = _norm(node)
        rows = self._conn.execute(
            "SELECT card_id FROM card_nodes WHERE node = ? ORDER BY rowid DESC LIMIT ?",
            (node, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def strong_partners(
        self,
        node: str,
        top_k: int = 5,
        min_pmi: float = 0.0,
        max_df_ratio: float = 0.5,
        scan: int = 15,
    ) -> List[Tuple[str, float]]:
        """Genuine co-occurrence partners of a node, for word-graph spreading.

        Filters the node's neighbors two ways so the spread stays meaningful:
        - per-edge PMI >= min_pmi: drops "both words are common so they collide
          by chance" pairs, keeps "these two specifically belong together";
        - df(partner) <= max_df_ratio * N: skips hub / 万金油 words that link to
          everything and carry no signal.
        Returns [(partner_node, pmi)] sorted by PMI desc.
        """
        import math
        node = _norm(node)
        N = max(1, self._conn.execute(
            "SELECT COUNT(DISTINCT card_id) FROM card_nodes"
        ).fetchone()[0])
        df_a = (self._conn.execute(
            "SELECT df FROM nodes WHERE name = ?", (node,)
        ).fetchone() or (0,))[0]
        if df_a <= 0:
            return []
        nbrs = self.neighbors(node, scan)
        if not nbrs:
            return []
        dfs_b = self.node_dfs([b for b, _ in nbrs])
        df_cap = max_df_ratio * N
        out: List[Tuple[str, float]] = []
        for b, w in nbrs:
            df_b = dfs_b.get(b, 0)
            if df_b <= 0 or w <= 0 or df_b > df_cap:
                continue
            pmi = math.log((w * N) / (df_a * df_b))
            if pmi < min_pmi:
                continue
            out.append((b, pmi))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    def stats(self) -> Dict[str, int]:
        nn = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        ne = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        np = self._conn.execute("SELECT COUNT(*) FROM card_nodes").fetchone()[0]
        nc = self._conn.execute("SELECT COUNT(DISTINCT card_id) FROM card_nodes").fetchone()[0]
        return {"nodes": nn, "edges": ne, "postings": np, "carded": nc}

    def get_summary(self, node: str) -> Optional[Dict[str, Any]]:
        """Get summary for a node, or None."""
        row = self._conn.execute(
            "SELECT content, card_count, new_since_write, last_rewrite, edit_history "
            "FROM summaries WHERE node_name = ?", (_norm(node),)
        ).fetchone()
        if not row:
            return None
        return {
            "node_name": _norm(node),
            "content": row[0],
            "card_count": row[1],
            "new_since_write": row[2],
            "last_rewrite": row[3],
            "edit_history": json.loads(row[4] or "[]"),
        }

    def save_summary(self, node: str, content: str, card_count: int) -> None:
        """Write or rewrite a summary. Old version goes to edit_history."""
        node = _norm(node)
        with self._lock:
            old = self._conn.execute(
                "SELECT content, edit_history FROM summaries WHERE node_name = ?",
                (node,)
            ).fetchone()
            history = json.loads(old[1] or "[]") if old else []
            if old and old[0]:
                history.append({"content": old[0], "replaced_at": now_iso()})
            self._conn.execute(
                "INSERT INTO summaries(node_name, content, card_count, new_since_write, last_rewrite, edit_history) "
                "VALUES(?, ?, ?, 0, ?, ?) "
                "ON CONFLICT(node_name) DO UPDATE SET "
                "content=excluded.content, card_count=excluded.card_count, "
                "new_since_write=0, last_rewrite=excluded.last_rewrite, "
                "edit_history=excluded.edit_history",
                (node, content, card_count, now_iso(), json.dumps(history, ensure_ascii=False)),
            )
            self._conn.commit()

    def nodes_needing_summary(
        self, min_df: int = 5, min_new: int = 3,
    ) -> List[Tuple[str, int, int]]:
        """Find nodes that deserve a summary and have new cards since last write.

        够厚(df>=min_df) + 自上次写后有新卡 + 过摘要门控(见 _summary_axis)：
        词条够生僻(通用词频低)=特定名字才写摘要。通用常用词(方式/训练/对话)不写——
        它的经历卡照样能被搜到。冒号碎片/英文虚词/纯数字直接剔。
        """
        self._prune_orphan_summaries()
        rows = self._conn.execute(
            "SELECT n.name, n.df, COALESCE(s.new_since_write, n.df) as new_count "
            "FROM nodes n LEFT JOIN summaries s ON n.name = s.node_name "
            "WHERE n.df >= ? AND (s.node_name IS NULL OR s.new_since_write >= ?) "
            "ORDER BY n.df DESC",
            (min_df, min_new),
        ).fetchall()
        result = []
        for name, df, new_count in rows:
            if self._summary_axis(name):
                result.append((name, df, new_count))
            else:
                cur = self._conn.execute(
                    "SELECT content FROM summaries WHERE node_name = ?", (name,)
                ).fetchone()
                if cur and (cur[0] or "").strip():
                    # 已有真摘要却判跳(门控以后变严才会出现)：绝不静默抹掉它——
                    # 只清 new_since_write 止住每晚重选，内容留给显式 prune 决定。
                    with self._lock:
                        self._conn.execute(
                            "UPDATE summaries SET new_since_write = 0 WHERE node_name = ?",
                            (name,),
                        )
                        self._conn.commit()
                    _log("summary_kept_despite_skip", f"node={name} df={df}")
                else:
                    # 通用词/碎片且本就没内容：写空壳标记，避免每晚重判。
                    self.save_summary(name, "", df)
                    _log("summary_skip", f"node={name} df={df}")
        return result

    def _summary_axis(self, name: str) -> str:
        """词条配不配摘要：返回 '名字' = 该写，'' = 跳过。

        判据只剩一条——通用词频够低 = 够生僻 = 特定名字的词条(人/宠物/地点/自造词)。
        砍掉了原来的"聚拢轴"(看经历卡聚不聚)：它既放通用词进来(方式碰巧聚拢)、又误杀
        横跨多场景的真词条(遛狗)，还绑 embedder。常用词当话题的不写摘要，靠经历卡。"""
        if "：" in name or ":" in name:
            return ""                       # 字段:值 元数据碎片
        if name.lower() in _EN_STOP:
            return ""                       # 英文虚词
        if re.fullmatch(r"[0-9.]+", name):
            return ""                       # 纯数字
        if general_word_freq(name) < _SUMMARY_NAME_MAX_GF:
            return "名字"                   # 通用语料里够生僻 = 特定名字的词条
        return ""

    def _prune_orphan_summaries(self) -> None:
        """清掉孤儿总结行：卡被超写/删除后图会全量重建(clear→回填 nodes)，但 summaries 不在
        重建范围内，于是留下"节点已不存在、总结还在"的孤儿。不清的话，一提到该词检索就会注入
        一条来源已不存在的旧总结。"""
        with self._lock:
            if self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0:
                return  # 图空(可能正在重建)，不动 summaries，避免误删全部
            n = self._conn.execute(
                "DELETE FROM summaries WHERE node_name NOT IN (SELECT name FROM nodes)"
            ).rowcount
            self._conn.commit()
        if n:
            _log("summary_orphan_pruned", f"n={n}")

    def summary_names(self) -> List[str]:
        """Node names that have a written summary — names only, cheap. The retriever
        matches a turn against these, then pulls content for the few that hit
        (summary_contents), instead of dragging every summary's text in each turn."""
        rows = self._conn.execute(
            "SELECT node_name FROM summaries WHERE content != ''"
        ).fetchall()
        return [r[0] for r in rows]

    def summary_contents(self, names: List[str]) -> Dict[str, str]:
        """Summary text for the given node names: {node_name: content}."""
        names = [n for n in names if n]
        if not names:
            return {}
        qmarks = ",".join("?" * len(names))
        rows = self._conn.execute(
            f"SELECT node_name, content FROM summaries WHERE node_name IN ({qmarks}) AND content != ''",
            tuple(names),
        ).fetchall()
        return {r[0]: r[1] for r in rows}


# ============================================================
# Factory + shadow hook
# ============================================================

_graphs: Dict[str, CooccurrenceGraph] = {}
_graphs_lock = threading.Lock()


def get_graph(persona_id: str) -> CooccurrenceGraph:
    key = safe_filename(persona_id)
    with _graphs_lock:
        g = _graphs.get(key)
        if g is None:
            g = CooccurrenceGraph(persona_id)
            _graphs[key] = g
        return g


def on_card_created(persona_id: str, card: Dict[str, Any]) -> None:
    """Shadow hook：落卡后顺手把它撒进关联地图。
    绝不抛异常打断写路径，只写独立 graph.db，检索侧零依赖。"""
    try:
        subj = _norm(card.get("subject") or "")
        if len(subj) >= _MIN_LEN:
            jieba.add_word(subj)  # 至少让它自己的名字成词
        nodes = extract_nodes(card)
        if not nodes:
            return
        get_graph(persona_id).record_card(
            nodes, card.get("source_time") or card.get("created_at"),
            card_id=card.get("id"),
        )
    except Exception as e:  # noqa: BLE001 — shadow 绝不影响写路径
        _log("shadow_build_error", str(e)[:200])
