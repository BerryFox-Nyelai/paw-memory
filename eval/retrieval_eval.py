"""Synthetic retrieval eval for The Paw Memory.

Seeds a labelled corpus of cards into a throwaway store, then fires paraphrased
queries (worded differently from the cards on purpose, so this measures semantic
recall rather than keyword overlap) and reports hit@k / MRR.

SCOPE — read this before trusting the numbers:
Cards are seeded directly via the store, *without* running the ingest/review
write pipeline. So the dictionary-map graph and node summaries are empty, and
this eval exercises only the **vector-fallback path** (region retrieval step 5),
gated by the cosine ``score_threshold`` (0.35). It does NOT test the system's
primary summary-first / dictionary-map retrieval, which only exists after the
write pipeline has built it. Treat this as a vector-net sanity/regression guard,
not a benchmark of the full engine. A full-pipeline eval needs the LLM seams.

The corpus includes near-miss distractor pairs (coffee/tea, cat/dog,
guitar/piano) so a top-1 hit means the retriever discriminated, not just landed
in the right neighbourhood.

Run:  PYTHONPATH=src python eval/retrieval_eval.py
No network, no LLM, no private data — fully reproducible.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_BASE = Path(tempfile.mkdtemp(prefix="thepaw_eval_"))
os.environ["THEPAW_MEMORY_BASE"] = str(_BASE)

from thepaw_memory import MemoryEngine
from thepaw_memory.card_store import new_card

PERSONA = "eval_test"

# (key, card content, topic). "U" is the fictional user.
CORPUS = [
    ("coffee",  "U 每天早上都要喝一杯手冲咖啡，最爱耶加雪菲那种果酸味。", "习惯"),
    ("tea",     "U 周末偶尔泡乌龙茶，但平时提神还是靠咖啡因。",           "习惯"),
    ("cat",     "U 养了一只橘猫，名字叫团子，特别黏人。",                 "宠物"),
    ("dog",     "U 小时候家里养过一只土狗叫旺财，后来送人了。",           "宠物"),
    ("guitar",  "U 最近在自学吉他，已经能弹下来一整首《晴天》。",         "爱好"),
    ("piano",   "U 小学学过几年钢琴，不过早就荒废了。",                   "爱好"),
    ("hiking",  "U 上个月去爬黄山，凌晨四点爬起来看日出。",               "出游"),
    ("job",     "U 是个后端工程师，平时主要写 Go 和 Python。",            "工作"),
    ("allergy", "U 对花生过敏，外食会特意问酱料里有没有坚果。",           "健康"),
    ("sister",  "U 有个妹妹在念高三，明年就要高考了。",                   "家人"),
    ("movie",   "U 很喜欢宫崎骏的动画，最爱的是《千与千寻》。",           "爱好"),
    ("travel",  "U 一直想去京都看红叶，但还没排上假期。",                 "心愿"),
    ("cooking", "U 最近迷上自己做意面，尤其是青酱口味的。",               "日常"),
    ("fitness", "U 办了健身卡，每周三、五去撸铁。",                       "习惯"),
    ("book",    "U 在读《三体》，已经看到第二部黑暗森林了。",             "阅读"),
    ("weather", "U 说今天下雨，懒得出门，窝在家里。",                     "闲聊"),
]

# (paraphrased query, gold card key). Wording intentionally avoids the card's words.
QUERIES = [
    ("U 每天早上离不开什么饮料？",  "coffee"),
    ("U 现在养着什么宠物？",      "cat"),
    ("U 会演奏哪种乐器？",        "guitar"),
    ("U 爬过哪座山？",            "hiking"),
    ("U 是做什么职业的？",        "job"),
    ("U 在饮食上有什么忌口？",    "allergy"),
    ("U 家里还有谁要参加考试？",  "sister"),
    ("U 偏爱哪位导演的作品？",    "movie"),
    ("U 想去哪里旅游？",          "travel"),
    ("U 最近在读什么书？",        "book"),
]


def main() -> int:
    engine = MemoryEngine(_BASE)
    engine.register_persona(PERSONA, display_name="EvalUser", user_display_name="U")

    store = engine.store(PERSONA)
    key_by_id = {}
    for key, content, topic in CORPUS:
        card = new_card(content=content, subject="U", topic=topic, session_id="seed")
        store.append(card)
        key_by_id[card["id"]] = key
    id_by_key = {v: k for k, v in key_by_id.items()}

    print(f"corpus: {len(CORPUS)} cards | queries: {len(QUERIES)} | "
          f"embed: bge-small-zh-v1.5 | store: {_BASE}\n")

    ranks = []
    header = f"{'query':<26}{'gold':<9}{'rank':<6}{'top-1 returned':<16}{'#ret'}"
    print(header)
    print("-" * len(header))
    for q, gold_key in QUERIES:
        gold_id = id_by_key[gold_key]
        res = engine.retrieve(PERSONA, q)
        ids = res.get("card_ids", [])
        rank = ids.index(gold_id) + 1 if gold_id in ids else 0
        ranks.append(rank)
        top1 = key_by_id.get(ids[0], "-") if ids else "-"
        rank_disp = str(rank) if rank else "miss"
        flag = "" if rank == 1 else ("  <" if rank else "  XX")
        print(f"{q:<26}{gold_key:<9}{rank_disp:<6}{top1:<16}{len(ids)}{flag}")

    n = len(ranks)
    def hit_at(k): return sum(1 for r in ranks if 1 <= r <= k) / n
    mrr = sum((1.0 / r if r else 0.0) for r in ranks) / n

    print("\n--- metrics ---")
    print(f"hit@1 = {hit_at(1):.2f}   hit@3 = {hit_at(3):.2f}   "
          f"hit@5 = {hit_at(5):.2f}   MRR = {mrr:.3f}")

    # Soft regression bar: most golds should land in the top 3.
    bar = 0.8
    ok = hit_at(3) >= bar
    print(f"\n{'PASS' if ok else 'FAIL'}: hit@3 {hit_at(3):.2f} "
          f"{'>=' if ok else '<'} {bar:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
