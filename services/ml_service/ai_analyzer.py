"""
ai_analyzer.py - Calls 3 AIs in parallel to analyse fixtures.
Bulletproof JSON parsing with 4 fallback strategies.
"""
import httpx
import asyncio
import json
import re
import os
from typing import Dict, List, Tuple


class AIAnalyzer:
    """Sends fixtures to Claude, Gemini, and Groq for parallel analysis."""

    def __init__(self):
        self.claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")

    def _build_prompt(self, fixture_text: str, date: str) -> str:
        return (
            f"You are a professional football value betting analyst. Date: {date}.\n\n"
            f"REAL MATCHES TODAY:\n{fixture_text}\n\n"
            "ANALYSIS FRAMEWORK - check ALL for every tip:\n"
            "1. Form: last 5-10 results, goals scored/conceded, streak\n"
            "2. Home/Away: separate records this season\n"
            "3. Injuries/Suspensions: missing key players\n"
            "4. Motivation: title, relegation, cup knockout, dead rubber\n"
            "5. H2H: last 5-6 meetings goals and results\n"
            "6. Tactics: style matchup, pressing vs counter-attack\n"
            "7. xG: expected goals for/against\n"
            "8. Stats: shots/game, corners/game, cards/game, clean sheets\n"
            "9. Coaching: rotation, tactical flexibility\n"
            "10. Conditions: fatigue, travel, weather, referee tendencies\n\n"
            "MARKETS ALLOWED:\n"
            "Over/Under 1.5/2.5/3.5/4.5 Goals | BTTS Yes/No | "
            "1st Half Over 0.5/1.5 | 2nd Half Over 0.5/1.5 | "
            "Over/Under 8.5/9.5/10.5/11.5 Corners | "
            "Over/Under 3.5/4.5/5.5 Cards | Both Halves Over 0.5 | Clean Sheet\n\n"
            "FORBIDDEN: match winner, double chance, correct score, goalscorer, handicap.\n\n"
            "Generate EXACTLY 20 tips. Variety of markets AND leagues.\n"
            "reasoning: 3-4 sentences with specific stats.\n"
            "key_stats: 4-5 data points.\n"
            "confidence: 65-92. risk: LOW(80+), MEDIUM(65-79), HIGH(<65).\n\n"
            "Return ONLY a JSON array. Start with [ end with ]. No markdown.\n\n"
            '[{"match":"Team A vs Team B","league":"League (Country)","time":"HH:MM GMT",'
            '"market":"Over/Under 2.5 Goals","pick":"Over 2.5 Goals",'
            '"odds_range":"1.80-2.00","confidence":84,'
            '"reasoning":"Stats here.","key_stats":["s1","s2","s3","s4"],"risk":"LOW"}]'
        )

    # ── Individual AI callers ────────────────────────────────────
    async def _call_claude(self, prompt: str) -> str:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return ""
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 4000,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                d = r.json()
                return "".join(
                    b["text"] for b in d.get("content", []) if b.get("type") == "text"
                ).strip()
            except Exception as e:
                return f"ERR:{e}"

    async def _call_gemini(self, prompt: str) -> str:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return ""
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 4000, "temperature": 0.1},
                    },
                )
                d = r.json()
                return d.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except Exception as e:
                return f"ERR:{e}"

    async def _call_groq(self, prompt: str) -> str:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return ""
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "Football analyst. Return ONLY a JSON array. Start [ end ]. No markdown."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 4000,
                        "temperature": 0.1,
                    },
                )
                d = r.json()
                return d.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                return f"ERR:{e}"

    # ── Bulletproof JSON parser ──────────────────────────────────
    def _extract_tips(self, raw: str, ai_name: str) -> Tuple[List[Dict], str]:
        if not raw or len(raw) < 10 or raw.startswith("ERR:"):
            return [], f"{ai_name}: {(raw or 'empty')[:80]}"

        text = re.sub(r"```json\s*", "", raw)
        text = re.sub(r"```\s*", "", text).strip()
        arr = []

        # Strategy 1: direct parse
        if text.startswith("["):
            try:
                arr = json.loads(text)
            except json.JSONDecodeError:
                pass

        # Strategy 2: regex extract
        if not arr:
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                try:
                    arr = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        # Strategy 3: individual objects
        if not arr:
            objs = re.findall(r"\{[^{}]{20,}\}", text)
            for obj_str in objs:
                try:
                    arr.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass

        # Strategy 4: fix trailing commas
        if not arr:
            fixed = re.sub(r",\s*]", "]", text)
            fixed = re.sub(r",\s*}", "}", fixed)
            m = re.search(r"\[[\s\S]*\]", fixed)
            if m:
                try:
                    arr = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not arr:
            return [], f"{ai_name}: no JSON found ({text[:80]})"

        tips = []
        for t in arr:
            if not isinstance(t, dict):
                continue
            match = t.get("match") or t.get("teams") or t.get("game") or ""
            pick = t.get("pick") or t.get("bet") or t.get("selection") or ""
            if len(match) < 4 or len(pick) < 3:
                continue
            tips.append({
                "match": match,
                "league": t.get("league") or t.get("competition") or "Unknown",
                "time": t.get("time") or t.get("kickoff") or "TBD",
                "market": t.get("market") or t.get("type") or "",
                "pick": pick,
                "odds_range": t.get("odds_range") or t.get("odds") or "",
                "confidence": min(max(int(t.get("confidence", 72)), 50), 98),
                "reasoning": t.get("reasoning") or t.get("analysis") or "",
                "key_stats": t.get("key_stats") or t.get("stats") or [],
                "risk": t.get("risk", "MEDIUM"),
                "ai_source": ai_name,
            })

        return tips, f"{ai_name}: {len(tips)} tips"

    # ── Merge tips across AIs ────────────────────────────────────
    def _merge(self, all_tips: List[List[Dict]]) -> List[Dict]:
        merged = {}
        names = ["Claude", "Gemini", "Groq"]

        for i, tips in enumerate(all_tips):
            name = names[i]
            for t in tips:
                norm_key = re.sub(r"[^a-z0-9]", "", t["match"].lower())[:20] + \
                           re.sub(r"[^a-z0-9]", "", t["pick"].lower())[:15]
                if norm_key not in merged:
                    merged[norm_key] = {**t, "ais": [], "votes": 0, "confs": []}
                m = merged[norm_key]
                if name not in m["ais"]:
                    m["ais"].append(name)
                m["votes"] += 1
                m["confs"].append(t["confidence"])
                if len(t.get("reasoning", "")) > len(m.get("reasoning", "")):
                    m["reasoning"] = t["reasoning"]
                if len(t.get("key_stats", [])) > len(m.get("key_stats", [])):
                    m["key_stats"] = t["key_stats"]

        result = []
        for item in merged.values():
            avg_conf = sum(item["confs"]) / len(item["confs"])
            boost = 5 if item["votes"] == 2 else (10 if item["votes"] >= 3 else 0)
            item["confidence"] = min(98, round(avg_conf + boost))
            item["aiCount"] = item["votes"]
            item["multiAI"] = item["votes"] >= 2
            item["confirmed"] = item["votes"] >= 3
            item["id"] = os.urandom(4).hex()
            result.append(item)

        return sorted(result, key=lambda x: (
            -int(x.get("confirmed", False)),
            -int(x.get("multiAI", False)),
            -x.get("confidence", 0),
        ))

    # ── Main analysis function ───────────────────────────────────
    async def analyse(self, fixture_text: str, date: str) -> Dict:
        """
        Run all 3 AIs in parallel, parse results, merge.
        Returns {"tips": [...], "activeAIs": [...], "debug": "..."}
        """
        prompt = self._build_prompt(fixture_text, date)

        claude_raw, gemini_raw, groq_raw = await asyncio.gather(
            self._call_claude(prompt),
            self._call_gemini(prompt),
            self._call_groq(prompt),
        )

        c_tips, c_dbg = self._extract_tips(claude_raw, "Claude")
        g_tips, g_dbg = self._extract_tips(gemini_raw, "Gemini")
        q_tips, q_dbg = self._extract_tips(groq_raw, "Groq")

        tips = self._merge([c_tips, g_tips, q_tips])
        active = []
        if c_tips: active.append("Claude")
        if g_tips: active.append("Gemini")
        if q_tips: active.append("Groq")

        return {
            "tips": tips,
            "activeAIs": active,
            "debug": f"{c_dbg} | {g_dbg} | {q_dbg}",
        }
