"""
ai_analyzer.py - Multi-AI football analysis
Uses requests library + ThreadPoolExecutor for reliable parallel execution.
"""
import requests
import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple


def _call_claude_sync(prompt: str) -> str:
    """Synchronous Claude call - more reliable on Railway."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("[Claude] No key in env")
        return "ERR:no_key"
    try:
        print(f"[Claude] Calling with key starting {key[:12]}...")
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        print(f"[Claude] HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"[Claude] Body: {r.text[:300]}")
            return f"ERR:HTTP{r.status_code}:{r.text[:200]}"
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
        print(f"[Claude] Got {len(text)} chars")
        return text
    except Exception as e:
        print(f"[Claude] Exception: {type(e).__name__}: {e}")
        return f"ERR:{type(e).__name__}:{e}"


def _call_gemini_sync(prompt: str) -> str:
    """Synchronous Gemini call."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("[Gemini] No key in env")
        return "ERR:no_key"
    try:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # 2.0 retired June 2026
        print(f"[Gemini] Calling {model} with key starting {key[:12]}...")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 4000, "temperature": 0.1},
            },
            timeout=90,
        )
        print(f"[Gemini] HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"[Gemini] Body: {r.text[:300]}")
            return f"ERR:HTTP{r.status_code}:{r.text[:200]}"
        d = r.json()
        candidates = d.get("candidates", [])
        if not candidates:
            print(f"[Gemini] No candidates: {str(d)[:300]}")
            return f"ERR:no_candidates:{str(d)[:200]}"
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"[Gemini] Got {len(text)} chars")
        return text
    except Exception as e:
        print(f"[Gemini] Exception: {type(e).__name__}: {e}")
        return f"ERR:{type(e).__name__}:{e}"


def _call_groq_sync(prompt: str) -> str:
    """Synchronous Groq call."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print("[Groq] No key in env")
        return "ERR:no_key"
    try:
        print(f"[Groq] Calling with key starting {key[:12]}...")
        r = requests.post(
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
                "max_tokens": 8000,
                "temperature": 0.1,
            },
            timeout=90,
        )
        print(f"[Groq] HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"[Groq] Body: {r.text[:300]}")
            return f"ERR:HTTP{r.status_code}:{r.text[:200]}"
        d = r.json()
        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"[Groq] Got {len(text)} chars")
        return text
    except Exception as e:
        print(f"[Groq] Exception: {type(e).__name__}: {e}")
        return f"ERR:{type(e).__name__}:{e}"


def _call_openai_compatible(prompt: str, *, url: str, key_env: str,
                            model: str, label: str) -> str:
    """Generic caller for ANY OpenAI-compatible chat API (DeepSeek, OpenRouter, Together, etc.).
    To add a new model you usually only need a one-line wrapper around this."""
    key = os.getenv(key_env, "").strip()
    if not key:
        print(f"[{label}] No key in env ({key_env})")
        return "ERR:no_key"
    try:
        print(f"[{label}] Calling {model} ...")
        r = requests.post(
            url,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Football analyst. Return ONLY a JSON array. Start [ end ]. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 8000,
                "temperature": 0.1,
            },
            timeout=120,
        )
        print(f"[{label}] HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"[{label}] Body: {r.text[:300]}")
            return f"ERR:HTTP{r.status_code}:{r.text[:200]}"
        d = r.json()
        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"[{label}] Got {len(text)} chars")
        return text
    except Exception as e:
        print(f"[{label}] Exception: {type(e).__name__}: {e}")
        return f"ERR:{type(e).__name__}:{e}"


def _call_deepseek_sync(prompt: str) -> str:
    """DeepSeek, OpenAI-compatible. Needs DEEPSEEK_API_KEY in Railway variables."""
    return _call_openai_compatible(
        prompt,
        url="https://api.deepseek.com/chat/completions",
        key_env="DEEPSEEK_API_KEY",
        model="deepseek-v4-flash",   # current V4 (cheap+fast). legacy 'deepseek-chat' retires 2026-07-24
        label="DeepSeek",
    )



def _call_openai_sync(prompt: str) -> str:
    """OpenAI direct. Needs OPENAI_API_KEY. Model via OPENAI_MODEL (default gpt-4o-mini).
    For newer GPT-5.x models you may need to set OPENAI_MODEL and note some reject
    custom temperature; gpt-4o-mini works with these defaults."""
    return _call_openai_compatible(
        prompt,
        url="https://api.openai.com/v1/chat/completions",
        key_env="OPENAI_API_KEY",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        label="OpenAI",
    )


def _call_openrouter_sync(prompt: str) -> str:
    """OpenRouter: ONE key, hundreds of models. Needs OPENROUTER_API_KEY.
    Choose any model via OPENROUTER_MODEL env var, e.g. 'openai/gpt-4o-mini',
    'google/gemini-2.5-flash', 'anthropic/claude-sonnet-4', 'deepseek/deepseek-chat',
    'meta-llama/llama-3.1-70b-instruct', or 'openrouter/auto'."""
    return _call_openai_compatible(
        prompt,
        url="https://openrouter.ai/api/v1/chat/completions",
        key_env="OPENROUTER_API_KEY",
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        label="OpenRouter",
    )


def _call_cerebras_sync(prompt: str) -> str:
    """Cerebras: FREE, no credit card, extremely fast. Needs CEREBRAS_API_KEY.
    Model via CEREBRAS_MODEL (default llama-3.3-70b). Get a key at cloud.cerebras.ai."""
    return _call_openai_compatible(
        prompt,
        url="https://api.cerebras.ai/v1/chat/completions",
        key_env="CEREBRAS_API_KEY",
        model=os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
        label="Cerebras",
    )


def _call_github_sync(prompt: str) -> str:
    """GitHub Models: FREE for any GitHub user. Needs GITHUB_MODELS_TOKEN (a GitHub
    fine-grained Personal Access Token with the 'models:read' permission).
    Model via GITHUB_MODELS_MODEL (default openai/gpt-4o-mini)."""
    return _call_openai_compatible(
        prompt,
        url="https://models.github.ai/inference/chat/completions",
        key_env="GITHUB_MODELS_TOKEN",
        model=os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini"),
        label="GitHub",
    )


def _call_mistral_sync(prompt: str) -> str:
    """Mistral: free tier (~1B tokens/month). Needs MISTRAL_API_KEY (console.mistral.ai).
    Model via MISTRAL_MODEL (default mistral-small-latest)."""
    return _call_openai_compatible(
        prompt,
        url="https://api.mistral.ai/v1/chat/completions",
        key_env="MISTRAL_API_KEY",
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        label="Mistral",
    )


def _call_nvidia_sync(prompt: str) -> str:
    """NVIDIA NIM: free (40 req/min). Needs NVIDIA_API_KEY (build.nvidia.com).
    Model via NVIDIA_MODEL (default meta/llama-3.3-70b-instruct)."""
    return _call_openai_compatible(
        prompt,
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        key_env="NVIDIA_API_KEY",
        model=os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
        label="NVIDIA",
    )


RETRY_WAIT_SECONDS = 2.0  # pause before one retry when a provider is rate-limited (HTTP 429)


def _retry_429(fn):
    """Wrap a provider call so a single 429 (rate-limit) triggers ONE retry after a
    short wait. Turns most intermittent throttles into successful responses."""
    def wrapped(prompt: str) -> str:
        out = fn(prompt)
        if isinstance(out, str) and out.startswith("ERR:HTTP429"):
            time.sleep(RETRY_WAIT_SECONDS)
            out = fn(prompt)
        return out
    wrapped.__name__ = getattr(fn, "__name__", "provider")
    return wrapped


# ── AI PROVIDER REGISTRY ──────────────────────────────────────────────────────
# This is the ONLY place to edit to add/remove an AI. Each entry is
# (DisplayName, function(prompt)->str). A provider with no API key set simply
# returns no tips and is skipped — safe to list before you add its key.
PROVIDERS = [
    ("Claude",     _retry_429(_call_claude_sync)),
    ("Gemini",     _retry_429(_call_gemini_sync)),
    ("Groq",       _retry_429(_call_groq_sync)),
    ("DeepSeek",   _retry_429(_call_deepseek_sync)),
    ("OpenAI",     _retry_429(_call_openai_sync)),
    ("OpenRouter", _retry_429(_call_openrouter_sync)),
    ("Cerebras",   _retry_429(_call_cerebras_sync)),
    ("GitHub",     _retry_429(_call_github_sync)),
    ("Mistral",    _retry_429(_call_mistral_sync)),
    ("NVIDIA",     _retry_429(_call_nvidia_sync)),
]


class AIAnalyzer:
    """Multi-AI football analyzer."""

    def _build_prompt(self, fixture_text: str, date: str) -> str:
        return (
            f"You are a disciplined football betting analyst. Today is {date}.\n\n"
            "Below is REAL recent-form data for each fixture, computed from actual match "
            "results in our database. Base every judgement ONLY on this data.\n\n"
            "================ HARD RULES (violating any = invalid output) ================\n"
            "1. Use ONLY the numbers shown in each match's DATA block. Never invent, guess, "
            "or recall statistics from memory. Every stat you write MUST appear in that "
            "match's DATA. If it is not in the data, you may not say it.\n"
            "2. We ONLY have goals and results data. We do NOT have xG, injuries, suspensions, "
            "lineups, referees, weather, possession, shots, corners or cards. NEVER mention or "
            "invent any of these.\n"
            "3. Your pick MUST be logically supported by the data. Example: if both teams average "
            "around 2+ goals and most recent games went Over 2.5, you may NOT pick Under 2.5.\n"
            "4. Confidence must reflect the evidence. Real betting edges are SMALL. Do not inflate "
            "confidence. If the data is mixed, thin, or contradictory, lower the confidence or skip "
            "the match entirely.\n"
            "5. Only tip matches that appear in the DATA below. Never tip a match that is not listed.\n\n"
            "================ ALLOWED MARKETS (all derivable from goals) ================\n"
            "Match Result (Home/Draw/Away) | Double Chance (1X/12/X2) | Draw No Bet | "
            "Over/Under 1.5/2.5/3.5 Goals | BTTS Yes/No | Win to Nil | Clean Sheet (Home/Away) | "
            "Odd/Even Total Goals | Team Over/Under 1.5 Goals.\n"
            "FORBIDDEN - never tip these (we have no data, tipping them = fabrication): corners, "
            "cards, shots, fouls, throw-ins, offsides, possession, 1st/2nd half markets, HT/FT, "
            "player props, goalscorers, Asian handicaps.\n"
            "ALSO FORBIDDEN - Correct Score and any exact-scoreline market. These are lottery bets "
            "(typical odds 5.0+ means under a 20% chance). They cannot be priced honestly by our "
            "model and have no place in a rollover strategy that needs high-probability legs.\n\n"
            "================ FIXTURE DATA ================\n"
            f"{fixture_text}\n\n"
            "================ OUTPUT ================\n"
            "Return the best grounded tips only (up to 20). Quality over quantity - it is far better "
            "to return a few well-supported tips than many weak ones. If nothing is well supported, "
            "return an empty array [].\n"
            "Each tip:\n"
            "  reasoning: 2-3 sentences citing the SPECIFIC numbers from that match's DATA.\n"
            "  key_stats: 3-5 stats, each copied directly from the DATA "
            "(e.g. \"France scored 2.67/game in last 6\", \"France 6/6 games Over 2.5\").\n"
            "  confidence: integer 60-88. risk: LOW(78-88), MEDIUM(66-77), HIGH(60-65).\n"
            "  home_team, away_team: exactly as written in the fixture line.\n"
            "  settle_type: ONE of goals_ou, btts, result, double_chance, dnb, win_to_nil, "
            "odd_even, clean_sheet, correct_score, team_goals.\n"
            "  line: numeric line if the market has one (e.g. 2.5), else null.\n"
            "  side: short code - over/under, yes/no, home/draw/away, 1x/12/x2, odd/even - else null.\n\n"
            "Return ONLY a JSON array. Start with [ end with ]. No markdown.\n\n"
            '[{"match":"Team A vs Team B","home_team":"Team A","away_team":"Team B",'
            '"league":"League (Country)","time":"HH:MM GMT",'
            '"market":"Over/Under 2.5 Goals","pick":"Over 2.5 Goals",'
            '"settle_type":"goals_ou","line":2.5,"side":"over",'
            '"odds_range":"1.80-2.00","confidence":78,'
            '"reasoning":"Cite the real numbers from the data block here.",'
            '"key_stats":["stat copied from data","stat copied from data","stat copied from data"],'
            '"risk":"LOW"}]'
        )

    # Async wrappers that just call the sync versions
    async def _call_claude(self, prompt: str) -> str:
        return _call_claude_sync(prompt)

    async def _call_gemini(self, prompt: str) -> str:
        return _call_gemini_sync(prompt)

    async def _call_groq(self, prompt: str) -> str:
        return _call_groq_sync(prompt)

    # ── Bulletproof JSON parser ──────────────────────────────────
    def _extract_tips(self, raw: str, ai_name: str) -> Tuple[List[Dict], str]:
        if not raw or len(raw) < 10 or raw.startswith("ERR:"):
            return [], f"{ai_name}: {(raw or 'empty')[:120]}"

        text = re.sub(r"```json\s*", "", raw)
        text = re.sub(r"```\s*", "", text).strip()
        arr = []

        if text.startswith("["):
            try:
                arr = json.loads(text)
            except json.JSONDecodeError:
                pass

        if not arr:
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                try:
                    arr = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not arr:
            objs = re.findall(r"\{[^{}]{20,}\}", text)
            for obj_str in objs:
                try:
                    arr.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass

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
            return [], f"{ai_name}: no JSON ({text[:80]})"

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
                "home_team": t.get("home_team") or "",
                "away_team": t.get("away_team") or "",
                "settle_type": t.get("settle_type") or "manual",
                "line": t.get("line"),
                "side": t.get("side") or None,
            })

        return tips, f"{ai_name}: {len(tips)} tips"

    # ── Merge tips across AIs ────────────────────────────────────
    def _merge(self, named_tips: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        merged = {}

        for name, tips in named_tips:
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

    # ── Main analysis - runs all 3 AIs in parallel via threads ───
    @staticmethod
    def validate_tips(tips, fixtures):
        """Drop AI tips that don't belong to a real fixture, and strip stats the
        database cannot support.

        Two real failures this prevents:
          * a pick naming a team that isn't in that match (a 'Draw/Huracan' tip
            attached to Talleres vs Velez),
          * fabricated advanced metrics ('Rangers away xG: 2.1') — the xG,
            shots, possession, corners and cards columns are all NULL in our
            database, so any such number is invented.
        """
        import re as _re
        UNSUPPORTED = _re.compile(
            r"\b(xg|xga|expected goals|possession|shots?|corners?|cards?|"
            r"fouls?|offsides?|injur|lineup|line-up|referee|weather)\b", _re.I)

        # Markets we can price, settle and calibrate. Anything else (correct
        # score, HT/FT, goalscorer) is a lottery market: our model gives it no
        # honest probability and the calibration tracker cannot grade it, so it
        # has no place in a rollover strategy that needs high-probability legs.
        ALLOWED_SETTLE = {"goals_ou", "btts", "result", "double_chance", "dnb",
                          "win_to_nil", "clean_sheet", "odd_even", "team_goals"}
        LOTTERY = _re.compile(
            r"(correct\s*score|ht/?ft|half\s*time/?full\s*time|score\s*cast|"
            r"scorer|exact\s*goals|score\s*in\s*both)", _re.I)

        by_match = {}
        for f in fixtures or []:
            h, a = (f.get("home") or "").strip(), (f.get("away") or "").strip()
            if h and a:
                by_match[f"{h} vs {a}".lower()] = (h, a)

        out = []
        for t in tips or []:
            # reject lottery markets outright
            st = (t.get("settle_type") or "").strip().lower()
            blob = f"{t.get('market','')} {t.get('pick','')}"
            if LOTTERY.search(blob) or (st and st not in ALLOWED_SETTLE):
                continue

            match = (t.get("match") or "").strip()
            teams = by_match.get(match.lower())
            if by_match and not teams:
                continue                      # tip references no real fixture
            if teams:
                h, a = teams
                pick = (t.get("pick") or "")
                # if the pick names a club, it must be one of THIS match's teams
                for other in by_match.values():
                    for nm in other:
                        if nm in (h, a):
                            continue
                        if len(nm) > 4 and nm.lower() in pick.lower():
                            pick = None
                            break
                    if pick is None:
                        break
                if pick is None:
                    continue
                t["home_team"], t["away_team"] = h, a

            stats = [s for s in (t.get("key_stats") or [])
                     if not (UNSUPPORTED.search(str(s)) and _re.search(r"\d", str(s)))]
            t["key_stats"] = stats
            t["ai_unverified"] = True         # AI prose is commentary, not data
            out.append(t)
        return out

    async def analyse(self, fixture_text: str, date: str) -> Dict:
        prompt = self._build_prompt(fixture_text, date)

        # Run every configured AI in parallel. To add a model, edit PROVIDERS
        # at the top of this file — nothing in this method needs to change.
        with ThreadPoolExecutor(max_workers=max(2, len(PROVIDERS))) as executor:
            futures = {name: executor.submit(fn, prompt) for name, fn in PROVIDERS}
            raws = {name: fut.result() for name, fut in futures.items()}

        named_tips: List[Tuple[str, List[Dict]]] = []
        active: List[str] = []
        debugs: List[str] = []
        for name, _fn in PROVIDERS:
            tips_n, dbg_n = self._extract_tips(raws[name], name)
            named_tips.append((name, tips_n))
            debugs.append(dbg_n)
            if tips_n:
                active.append(name)

        tips = self._merge(named_tips)
        return {
            "tips": tips,
            "activeAIs": active,
            "debug": " | ".join(debugs),
        }
