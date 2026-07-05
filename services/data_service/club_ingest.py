"""
club_ingest.py — Keep club-league history fresh, automatically.

The DB's club data (Brazil, Argentina, MLS, Liga MX, J1, China CSL) comes from
the openfootball GitHub archives. Those repos keep filling in results as
seasons progress, so this module re-downloads the season files that can still
change (current + previous year) and delta-inserts any newly finished matches.
It is wired into /api/admin/weekly and /api/admin/refresh-db.

Free (raw.githubusercontent.com), no API keys, no quota. Every fetch and every
file is best-effort: a missing file (404) or a bad line never breaks the run.

IMPORTANT: RENAME below must stay in sync with the names already in the DB —
it maps openfootball spellings to the canonical names used at first ingestion,
so re-ingested rows dedupe correctly instead of duplicating under new names.
"""
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Tuple

DB_PATH = os.getenv("FOOTBALL_DB", os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "football.db"))

RAW = "https://raw.githubusercontent.com"

# league_id, league_name, country, repo, path-template ({y}=year, {y2}=next-yy)
LEAGUES = [
    (71,  "Serie A", "Brazil", "openfootball/south-america", "brazil/{y}_br1.txt"),
    (72,  "Serie B", "Brazil", "openfootball/south-america", "brazil/{y}_br2.txt"),
    (128, "Liga Profesional Argentina", "Argentina",
          "openfootball/south-america", "argentina/{y}_ar1.txt"),
    (253, "Major League Soccer", "USA",
          "openfootball/world", "north-america/major-league-soccer/{y}_mls.txt"),
    (262, "Liga MX", "Mexico",
          "openfootball/world", "north-america/mexico/{y}-{y2}_mx1.txt"),
    (98,  "J1 League", "Japan", "openfootball/world", "asia/japan/{y}_jp1.txt"),
    (169, "Super League", "China", "openfootball/world", "asia/china/{y}_cn1.txt"),
]

# openfootball name -> canonical DB name (only where token matching fails)
RENAME = {
    "CA Mineiro": "Atletico-MG",
    "CA Paranaense": "Athletico-PR",
    "Atletico Mineiro": "Atletico-MG",
    "Athletico Paranaense": "Athletico-PR",
    "EC Juventude": "Juventude",
    "Criciuma EC": "Criciuma",
    "Cuiaba EC": "Cuiaba",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
                     r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
                     r"(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s*(?:\d{1,2}[.:]\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)")

MAX_BYTES = 2_000_000   # never read more than ~2MB per file


def _clean(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\s*\(\*\)$", "", name)
    return RENAME.get(name, name)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "rollover-app/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read(MAX_BYTES).decode("utf-8", errors="replace")


def _parse(text: str, default_year: int) -> List[Tuple[str, str, str, int, int]]:
    year, month, day = default_year, None, None
    out = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        dm = DATE_RE.match(line)
        if dm:
            mon, d, y = dm.group(1), int(dm.group(2)), dm.group(3)
            if y:
                year = int(y)
            if month == 12 and MONTHS[mon] == 1:   # season crossing new year
                year += 1
            month, day = MONTHS[mon], d
            continue
        mm = MATCH_RE.match(line)
        if mm and month:
            home, away = _clean(mm.group(1)), _clean(mm.group(2))
            if not home or not away or "Matchday" in home:
                continue
            out.append((f"{year:04d}-{month:02d}-{day:02d}",
                        home, away, int(mm.group(3)), int(mm.group(4))))
    return out


def _files_for(template: str, this_year: int) -> List[Tuple[str, int]]:
    """(path, season_year) candidates: current + previous year, both formats."""
    cands = []
    for y in (this_year, this_year - 1):
        if "{y2}" in template:   # cross-year seasons like 2025-26_mx1.txt
            cands.append((template.format(y=y, y2=str(y + 1)[2:]), y))
        else:
            cands.append((template.format(y=y), y))
    return cands


def refresh_club_results(db_path: str = DB_PATH) -> Dict:
    """Delta-ingest newly finished club matches. Returns per-league counts."""
    this_year = datetime.now(timezone.utc).year
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    per: Dict[str, int] = {}
    fetched = skipped = 0
    for lid, lname, country, repo, template in LEAGUES:
        added = 0
        for path, season in _files_for(template, this_year):
            url = f"{RAW}/{repo}/master/{path}"
            try:
                text = _fetch(url)
                fetched += 1
            except Exception:
                skipped += 1          # 404 for not-yet-existing seasons is normal
                continue
            try:
                for date, home, away, hg, ag in _parse(text, season):
                    if cur.execute(
                        "SELECT 1 FROM fixtures WHERE league_id=? AND date=? "
                        "AND home_team=? AND away_team=? LIMIT 1",
                        (lid, date, home, away)).fetchone():
                        continue
                    cur.execute(
                        "INSERT INTO fixtures (league_id, league_name, country, "
                        "season, date, home_team, away_team, home_goals, "
                        "away_goals, total_goals, status, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,'FT',?)",
                        (lid, lname, country, season, date, home, away,
                         hg, ag, hg + ag, now))
                    added += 1
            except Exception:
                continue
        if added:
            per[lname] = added
    con.commit()
    con.close()
    return {"added": per, "total_added": sum(per.values()),
            "files_fetched": fetched, "files_missing": skipped}
