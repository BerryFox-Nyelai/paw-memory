"""memory_review — Nightly review pipeline (V2).

Triggered when user is idle 5h+.
Steps: daily summary → dedup → retention decay → portrait update → node summaries.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from thepaw_memory.config_registry import cfg
from thepaw_memory.core.entity import get_persona_name, get_user_name
from thepaw_memory.feature_log import feature_log
from thepaw_memory.utils import now_iso

_log = lambda event, detail="": feature_log("memory_review", event, detail)

PORTRAIT_TYPES = ["user", "persona", "relationship"]

PORTRAIT_DESCRIPTIONS = {
    "user": "你对她",
    "persona": "你对自己",
    "relationship": "你对你们的关系",
}


# ------------------------------------------------------------------
# Module-level topology + dedup (extracted from MemoryEngine)
# ------------------------------------------------------------------


async def _dedup_cards(store, persona_id: str, report: Dict[str, Any]):
    try:
        from thepaw_memory.embeddings import encode
        import numpy as np
    except ImportError:
        return

    from thepaw_memory.card_store import card_text as _card_text

    active = [c for c in store.all_active()
              if not c.get("legacy") and not c.get("is_portrait")]
    if len(active) < 2:
        return

    dedup_threshold = float(cfg("heartbeat.dedup_cosine_threshold", persona_id=persona_id))

    try:
        texts = [_card_text(c) for c in active]
        vecs = encode(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vecs / norms
    except Exception as e:
        _log("dedup_encode_error", str(e)[:200])
        return

    superseded: set = set()

    for i in range(len(active)):
        if active[i]["id"] in superseded:
            continue
        for j in range(i + 1, len(active)):
            if active[j]["id"] in superseded:
                continue
            sim = float(np.dot(normed[i], normed[j]))
            if sim >= dedup_threshold:
                ci, cj = active[i], active[j]
                older = ci if ci.get("created_at", "") <= cj.get("created_at", "") else cj
                newer = cj if older is ci else ci
                store.supersede(older["id"])
                superseded.add(older["id"])
                report["dedup"] += 1
                feature_log("memory_review", "dedup",
                            f"superseded={older['id']} cosine={sim:.3f}",
                            data={"superseded": {"id": older["id"], "content": (older.get("content") or "")[:80]},
                                  "kept": {"id": newer["id"], "content": (newer.get("content") or "")[:80]}})
                if older is ci:
                    break


class NightlyReview:
    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self._persona_name: Optional[str] = None
        self._user_name: Optional[str] = None
        self._subbrain_name: Optional[str] = None

    @property
    def persona_name(self) -> str:
        if not self._persona_name:
            self._persona_name = get_persona_name(self.persona_id)
        return self._persona_name

    @property
    def user_name(self) -> str:
        if not self._user_name:
            self._user_name = get_user_name(self.persona_id)
        return self._user_name

    @property
    def subbrain_name(self) -> str:
        if not self._subbrain_name:
            self._subbrain_name = cfg(
                "memory_review.subbrain_display_name",
                persona_id=self.persona_id,
            ) or "副脑"
        return self._subbrain_name

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        today_cards: List[Dict[str, Any]],
        last_review_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the V2 nightly pipeline: summary → dedup → topology → retention decay → portrait."""
        import time as _time
        t0 = _time.monotonic()

        report: Dict[str, Any] = {
            "dedup": 0, "archived": 0,
            "retention_decayed": 0,
        }

        from thepaw_memory.card_store import get_store
        store = get_store(self.persona_id)
        total_active = len(store.all_active())

        _log("start", f"persona={self.persona_id} new_cards={len(today_cards)} "
             f"total_active={total_active} last_review={last_review_at or 'never'}")

        self._session_id = today_cards[0].get("session_id", "default") if today_cards else "default"

        # Step 1: daily summary
        t1 = _time.monotonic()
        daily_summary = await self._daily_summary(today_cards)
        _log("step_summary", f"ok={'yes' if daily_summary else 'no'} "
             f"elapsed={_time.monotonic()-t1:.1f}s")

        # Step 2: dedup
        t2 = _time.monotonic()
        try:
            await _dedup_cards(store, self.persona_id, report)
            _log("step_dedup", f"removed={report.get('dedup',0)} "
                 f"elapsed={_time.monotonic()-t2:.1f}s")
        except Exception as e:
            _log("step_dedup_error", str(e)[:200])

        # Step 3: refresh retrieval index (dedup may have superseded cards)
        try:
            from thepaw_memory.card_retrieve import invalidate
            invalidate(self.persona_id)
        except Exception as e:
            _log("step_invalidate_error", str(e)[:200])

        # Step 4: retention decay
        t4 = _time.monotonic()
        try:
            decay_result = self._decay_retention(store)
            report["retention_decayed"] = decay_result["count"]
            report["retention_zeroed"] = decay_result["zeroed"]
            _log("step_retention", f"decayed={decay_result['count']} "
                 f"zeroed={decay_result['zeroed']} "
                 f"elapsed={_time.monotonic()-t4:.1f}s")
        except Exception as e:
            _log("step_retention_error", str(e)[:200])

        # Step 5: portrait updates
        t5 = _time.monotonic()
        try:
            if daily_summary:
                await self._portrait_update(daily_summary, store)
                _log("step_portrait", f"ok elapsed={_time.monotonic()-t5:.1f}s")
        except Exception as e:
            _log("step_portrait_error", str(e)[:200])

        # Step 6: write/rewrite node summaries on the dictionary map
        t6 = _time.monotonic()
        try:
            from thepaw_memory.summary_writer import write_summaries
            summaries_written = await write_summaries(self.persona_id)
            report["summaries_written"] = summaries_written
            _log("step_summaries", f"written={summaries_written} elapsed={_time.monotonic()-t6:.1f}s")
        except Exception as e:
            _log("step_summaries_error", str(e)[:200])

        elapsed = _time.monotonic() - t0
        report["elapsed_seconds"] = round(elapsed, 1)
        feature_log("memory_review", "done",
                    f"persona={self.persona_id} elapsed={elapsed:.1f}s "
                    f"dedup={report['dedup']} "
                    f"decayed={report.get('retention_decayed',0)} "
                    f"zeroed={report.get('retention_zeroed',0)}",
                    data=report)
        return report

    async def _main_brain_call(self, review_prompt: str, **kwargs) -> Optional[str]:
        """Call the main brain through the same pipeline as normal chat."""
        from thepaw_memory.chat_common import build_context_messages
        messages, _, _ = await build_context_messages(
            persona_id=self.persona_id,
            session_id=self._session_id,
            user_text=review_prompt,
        )
        from thepaw_memory.main_brain import call as main_brain_call
        max_tok = kwargs.pop("max_tokens", 1500)
        temp = kwargs.pop("temperature", 0.7)
        return await main_brain_call(
            self.persona_id, messages,
            max_tokens=max_tok, temperature=temp,
        )

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def _daily_summary(
        self, today_cards: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Produce daily summary from today's new cards."""
        if not today_cards:
            return None

        drafts_text = "\n".join(
            f"- [{c.get('subject', '?')}/{c.get('topic', '?')}] {(c.get('content') or '')[:200]}"
            for c in today_cards
        )

        template = cfg("memory_review.prompt_daily_summary", persona_id=self.persona_id)
        prompt = template.format(
            persona_name=self.persona_name,
            drafts_text=drafts_text,
        )

        from thepaw_memory.subbrain_client import call_safe
        cr = await call_safe(
            [{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.2, persona_id=self.persona_id,
        )
        if not cr.content:
            return None
        text = re.sub(r"<think>.*?</think>", "", cr.content, flags=re.DOTALL).strip()
        return text or None

    # ------------------------------------------------------------------
    # Retention decay
    # ------------------------------------------------------------------

    def _decay_retention(self, store) -> Dict[str, int]:
        """Decay retention by elapsed calendar time (≈1 point per 30 days).

        Each card carries ``last_decay_at``; decay = days_since_that / 30.
        Driving off the calendar (not per-run) means a skipped night catches
        up on the next run and a double run in one day barely moves the needle.
        """
        from datetime import datetime, timezone
        now_str = now_iso()
        now_dt = datetime.fromisoformat(now_str)
        updates = []
        zeroed = 0
        for card in store.all_active():
            if card.get("level", 0) != 0:
                continue
            ret = card.get("retention")
            if ret is None or ret <= 0:
                continue
            base_raw = card.get("last_decay_at") or card.get("created_at")
            if not base_raw:
                continue
            try:
                base = datetime.fromisoformat(base_raw)
            except ValueError:
                continue
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            elapsed_days = (now_dt - base).total_seconds() / 86400.0
            if elapsed_days <= 0:
                continue
            new_ret = round(ret - elapsed_days / 30.0, 4)
            if new_ret <= 0:
                new_ret = 0.0
                zeroed += 1
            updates.append((card["id"], {"retention": new_ret, "last_decay_at": now_str}))
        count = store.batch_update(updates)
        return {"count": count, "zeroed": zeroed}

    # ------------------------------------------------------------------
    # Portrait time-depth updates
    # ------------------------------------------------------------------

    async def _portrait_update(
        self, daily_summary: str, store,
    ) -> None:
        from thepaw_memory.card_store import new_card
        buffer_max = int(cfg("memory_review.buffer_max", persona_id=self.persona_id))
        staging_trigger = int(cfg("memory_review.staging_pool_trigger", persona_id=self.persona_id))
        today = now_iso()[:10]

        typed_summaries = await self._split_daily_by_portrait(daily_summary)

        for ptype in PORTRAIT_TYPES:
            portrait = store.get_portrait(ptype)
            if not portrait:
                portrait = new_card(
                    content=f"[{ptype} portrait — initializing]",
                    level=2,
                    source="nightly_review",
                    tags=["portrait", ptype],
                )
                portrait["is_portrait"] = True
                portrait["portrait_type"] = ptype
                portrait["stable"] = ""
                portrait["mid_term"] = ""
                portrait["recent_buffer"] = []
                portrait["staging_pool"] = []
                store.append(portrait)
                _log("portrait_created", f"type={ptype} id={portrait['id']}")

            # First run: migrate existing content to stable
            if not isinstance(portrait.get("stable"), str) or not portrait["stable"].strip():
                old_content = (portrait.get("content") or "").strip()
                if old_content and "initializing" not in old_content:
                    portrait["stable"] = old_content
                    _log("portrait_migrate", f"type={ptype} len={len(old_content)}")

            summary_for_type = typed_summaries.get(ptype, "").strip()
            if not summary_for_type:
                _log("portrait_skip", f"type={ptype} no content from daily summary")
                store.save_portrait(portrait)
                continue

            recent_buffer = list(portrait.get("recent_buffer") or [])
            recent_buffer.append({"date": today, "summary": summary_for_type})

            staging_pool = list(portrait.get("staging_pool") or [])
            if len(recent_buffer) > buffer_max:
                overflow = recent_buffer[:-buffer_max]
                recent_buffer = recent_buffer[-buffer_max:]
                staging_pool.extend(overflow)

            staging_pool_max = staging_trigger * 3
            if len(staging_pool) > staging_pool_max:
                staging_pool = staging_pool[-staging_pool_max:]

            portrait["recent_buffer"] = recent_buffer
            portrait["staging_pool"] = staging_pool

            should_rewrite = len(staging_pool) >= staging_trigger
            if not should_rewrite and staging_pool:
                should_rewrite = await self._judge_staging_significance(
                    staging_pool, portrait.get("mid_term", ""))

            if should_rewrite:
                new_midterm = await self._rewrite_midterm(
                    ptype, portrait.get("mid_term", ""), staging_pool)
                if new_midterm:
                    old_mid = portrait.get("mid_term", "")
                    portrait["mid_term"] = new_midterm
                    portrait["staging_pool"] = []
                    eh = list(portrait.get("edit_history") or [])
                    eh.append({
                        "timestamp": now_iso(), "old_content": old_mid, "reason": "midterm_rewrite",
                    })
                    portrait["edit_history"] = eh
                    _log("midterm_rewritten", f"type={ptype}")

                    stable_update = await self._review_stable(
                        ptype, portrait.get("stable", ""), new_midterm)
                    if stable_update.get("should_update") and stable_update.get("new_stable"):
                        old_stable = portrait.get("stable", "")
                        portrait["stable"] = stable_update["new_stable"]
                        eh = list(portrait.get("edit_history") or [])
                        eh.append({
                            "timestamp": now_iso(), "old_content": old_stable,
                            "reason": f"stable_update: {stable_update.get('reasoning', '')[:50]}",
                        })
                        portrait["edit_history"] = eh
                        _log("stable_updated", f"type={ptype} reason={stable_update.get('reasoning', '')[:80]}")

            parts = []
            if portrait.get("stable"):
                parts.append(portrait["stable"])
            if portrait.get("mid_term"):
                parts.append(portrait["mid_term"])
            portrait["content"] = "\n".join(parts) if parts else portrait.get("content", "")

            store.save_portrait(portrait)

    async def _split_daily_by_portrait(
        self, daily_summary: str,
    ) -> Dict[str, str]:
        """Split one daily summary into three portrait-typed perspectives."""
        from thepaw_memory.subbrain_client import call_safe
        template = cfg("memory_review.prompt_portrait_split", persona_id=self.persona_id)
        prompt = template.format(
            daily_summary=daily_summary,
        )
        try:
            cr = await call_safe(
                [{"role": "user", "content": prompt}],
                max_tokens=1500, temperature=0.3, persona_id=self.persona_id,
            )
            text = cr.content if cr and cr.content else ""
            return self._parse_portrait_split(text)
        except Exception as e:
            _log("portrait_split_error", str(e)[:200])
            return {pt: daily_summary for pt in PORTRAIT_TYPES}

    @staticmethod
    def _parse_portrait_split(text: str) -> Dict[str, str]:
        import re
        result: Dict[str, str] = {}
        for ptype, marker in [
            ("persona", "===persona==="),
            ("user", "===user==="),
            ("relationship", "===relationship==="),
        ]:
            pattern = re.escape(marker) + r"\s*\n(.*?)(?=\n===|$)"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                content = m.group(1).strip()
                if content and content != "无" and content != "（无）":
                    result[ptype] = content
        return result

    async def _judge_staging_significance(
        self, staging_pool: List[Dict], current_midterm: str,
    ) -> bool:
        from thepaw_memory.subbrain_client import call_safe
        from thepaw_memory.review_parser import parse_staging_judge
        template = cfg("memory_review.prompt_staging_significance", persona_id=self.persona_id)
        entries_text = "\n".join(
            f"- [{e.get('date', '?')}] {e.get('summary', '')}" for e in staging_pool
        )
        prompt = template.format(
            staging_entries_text=entries_text,
            current_midterm=current_midterm,
        )
        cr = await call_safe(
            [{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.1, persona_id=self.persona_id,
        )
        result = parse_staging_judge(cr.content) if cr.content else {}
        return bool(result.get("should_rewrite"))

    async def _rewrite_midterm(
        self, portrait_type: str, current_midterm: str,
        staging_pool: List[Dict],
    ) -> Optional[str]:
        template = cfg("memory_review.prompt_midterm_rewrite", persona_id=self.persona_id)
        entries_text = "\n".join(
            f"- [{e.get('date', '?')}] {e.get('summary', '')}" for e in staging_pool
        )
        prompt = template.format(
            subbrain_name=self.subbrain_name,
            portrait_description=PORTRAIT_DESCRIPTIONS.get(portrait_type, "你"),
            current_midterm=current_midterm or "（空）",
            staging_entries_text=entries_text,
        )
        try:
            return await self._main_brain_call(prompt, max_tokens=800, temperature=0.4)
        except Exception as e:
            _log("midterm_rewrite_error", str(e)[:200])
            return None

    async def _review_stable(
        self, portrait_type: str, current_stable: str, new_midterm: str,
    ) -> Dict[str, Any]:
        template = cfg("memory_review.prompt_stable_review", persona_id=self.persona_id)
        prompt = template.format(
            subbrain_name=self.subbrain_name,
            portrait_description=PORTRAIT_DESCRIPTIONS.get(portrait_type, "你"),
            current_stable=current_stable or "（空）",
            new_midterm=new_midterm,
        )
        try:
            text = await self._main_brain_call(prompt, max_tokens=3000, temperature=0.3)
            from thepaw_memory.review_parser import parse_stable_review
            return parse_stable_review(text) if text else {"should_update": False}
        except Exception as e:
            _log("stable_review_error", str(e)[:200])
            return {"should_update": False}





