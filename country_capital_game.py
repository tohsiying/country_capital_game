# app.py
import streamlit as st
import streamlit.components.v1 as components
import random
import json
import pathlib
from typing import List, Dict, Optional
import requests
import plotly.graph_objects as go
import time
import datetime

# ---------- Config ----------
# Anchor persistence to the script's directory to survive working directory changes
DATA_FILE = pathlib.Path(__file__).resolve().with_name("progress.json")

SAMPLE_COUNTRIES = [
    {"country": "France", "capital": "Paris", "continent": "Europe", "avatar": "🗼", "political_fact": "President Emmanuel Macron leads this republic with a semi-presidential system", "cca2": "FR", "flag": "https://flagcdn.com/w80/fr.png"},
    {"country": "Kenya", "capital": "Nairobi", "continent": "Africa", "avatar": "🦁", "political_fact": "President William Ruto leads this presidential republic", "cca2": "KE", "flag": "https://flagcdn.com/w80/ke.png"},
    {"country": "Japan", "capital": "Tokyo", "continent": "Asia", "avatar": "🗾", "political_fact": "Prime Minister Fumio Kishida leads this constitutional monarchy", "cca2": "JP", "flag": "https://flagcdn.com/w80/jp.png"},
    {"country": "Brazil", "capital": "Brasília", "continent": "South America", "avatar": "🌴", "political_fact": "President Luiz Inácio Lula da Silva leads this federal republic", "cca2": "BR", "flag": "https://flagcdn.com/w80/br.png"},
    {"country": "Australia", "capital": "Canberra", "continent": "Oceania", "avatar": "🦘", "political_fact": "Prime Minister Anthony Albanese leads this parliamentary democracy", "cca2": "AU", "flag": "https://flagcdn.com/w80/au.png"},
    {"country": "Canada", "capital": "Ottawa", "continent": "North America", "avatar": "🍁", "political_fact": "Prime Minister Justin Trudeau leads this parliamentary democracy", "cca2": "CA", "flag": "https://flagcdn.com/w80/ca.png"},
    {"country": "Egypt", "capital": "Cairo", "continent": "Africa", "avatar": "🐫", "political_fact": "President Abdel Fattah el-Sisi leads this presidential republic", "cca2": "EG", "flag": "https://flagcdn.com/w80/eg.png"},
    {"country": "India", "capital": "New Delhi", "continent": "Asia", "avatar": "🕌", "political_fact": "Prime Minister Narendra Modi leads this federal parliamentary republic", "cca2": "IN", "flag": "https://flagcdn.com/w80/in.png"},
    {"country": "Germany", "capital": "Berlin", "continent": "Europe", "avatar": "🍺", "political_fact": "Chancellor Olaf Scholz leads this federal parliamentary republic", "cca2": "DE", "flag": "https://flagcdn.com/w80/de.png"},
    {"country": "Argentina", "capital": "Buenos Aires", "continent": "South America", "avatar": "⚽", "political_fact": "President Javier Milei leads this federal presidential republic", "cca2": "AR", "flag": "https://flagcdn.com/w80/ar.png"},
]

# ---------- Data source (expanded to ~197 countries) ----------
# We build a comprehensive list using the Rest Countries API, then augment to reach
# the widely referenced "197 countries" (193 UN members + 2 observers + Kosovo & Taiwan).

OBSERVERS = {"Holy See", "Palestine"}
EXTRA_STATES = {"Kosovo", "Taiwan"}

def _continent_from_region(region: str, subregion: str) -> str:
    region = (region or "").strip()
    subregion = (subregion or "").strip()
    if region != "Americas":
        return region or "Other"
    # Split Americas into North/South using subregion
    sr = subregion.lower()
    if "south" in sr:
        return "South America"
    # Central America and Caribbean grouped under North America for this app
    return "North America"

def _normalize_continent_name(name: str) -> str:
    """Return a canonical continent name used for comparisons and UI.

    Normalizes by trimming whitespace. Case-insensitive comparisons are
    performed at call sites as needed.
    """
    return (name or "").strip()

@st.cache_data(ttl=60 * 60 * 24)
def load_full_country_dataset() -> List[Dict]:
    try:
        resp = requests.get(
            "https://restcountries.com/v3.1/all",
            params={
                # Request extra facts to power varied clues
                "fields": (
                    "name,capital,region,subregion,unMember,cca2,flags,"
                    "population,area,landlocked,borders,languages,currencies,"
                    "fifa,timezones,tld"
                ),
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        countries: List[Dict] = []
        for item in data:
            name = (item.get("name", {}).get("common") or "").strip()
            capitals = item.get("capital") or []
            if not name or not capitals:
                continue
            capital = (capitals[0] or "").strip()
            if not capital:
                continue

            un_member = bool(item.get("unMember", False))
            include = un_member or (name in OBSERVERS) or (name in EXTRA_STATES)
            if not include:
                continue

            continent = _continent_from_region(item.get("region", ""), item.get("subregion", ""))

            cca2 = (item.get("cca2") or "").strip().upper()
            flags = item.get("flags") or {}
            flag_png = (flags.get("png") or "").strip()
            flag_svg = (flags.get("svg") or "").strip()

            # Assign a simple continent-based avatar
            avatar_by_continent = {
                "Africa": "🦁",
                "Asia": "🗾",
                "Europe": "🏰",
                "North America": "🦅",
                "South America": "🌴",
                "Oceania": "🦘",
            }
            avatar = avatar_by_continent.get(continent, "👑")

            if un_member:
                political_fact = "UN member state"
            elif name in OBSERVERS:
                political_fact = "UN observer state"
            else:
                political_fact = "Widely recognized; not a UN member"

            countries.append({
                "country": name,
                "capital": capital,
                "continent": continent,
                "avatar": avatar,
                "political_fact": political_fact,
                "cca2": cca2,
                "flag": flag_png or flag_svg,
                # Optional enrichment for clues (best-effort)
                "population": int(item.get("population") or 0),
                "area_km2": float(item.get("area") or 0.0),
                "landlocked": bool(item.get("landlocked", False)),
                "borders": list(item.get("borders") or []),
                "languages": list((item.get("languages") or {}).values()),
                "currencies": [
                    c.get("name")
                    for c in (item.get("currencies") or {}).values()
                    if isinstance(c, dict) and c.get("name")
                ],
                "fifa": (item.get("fifa") or "").strip(),
                "timezones": list(item.get("timezones") or []),
                "tld": list(item.get("tld") or []),
            })

        # Deduplicate by country name and sort
        unique: Dict[str, Dict] = {}
        for c in countries:
            unique[c["country"]] = c
        result = sorted(unique.values(), key=lambda x: x["country"])

        # Sanity filter: return only if we reached a plausible size (e.g., > 190)
        if len(result) >= 190:
            return result
        # Fallback to sample if something went wrong
        return SAMPLE_COUNTRIES
    except Exception:
        return SAMPLE_COUNTRIES

# Primary dataset used by the app
COUNTRIES: List[Dict] = load_full_country_dataset()

# ---------- Persistence helpers ----------
def _normalize_progress_dict(raw_progress: Dict) -> Dict[str, Dict[str, int]]:
    """Ensure progress is stored as {country: {right: int, wrong: int}}.

    Backward compatible with older format {country: int_correct}.
    """
    normalized: Dict[str, Dict[str, int]] = {}
    for country, value in (raw_progress or {}).items():
        if isinstance(value, dict):
            right = int(value.get("right", 0))
            wrong = int(value.get("wrong", 0))
            normalized[country] = {"right": max(0, right), "wrong": max(0, wrong)}
        else:
            # legacy integer means correct/right count only
            try:
                right = int(value)
            except Exception:
                right = 0
            normalized[country] = {"right": max(0, right), "wrong": 0}
    return normalized

def load_users_state() -> Dict:
    """Load multi-user state.

    Schema:
    {
      "users": {"Alice": {"progress": {...}}, "Bob": {"progress": {...}}},
      "active_user": "Alice"
    }

    Backward compatible with legacy schema {"progress": {...}}.
    """
    try:
        if DATA_FILE.exists():
            raw = json.loads(DATA_FILE.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict) and "users" in raw:
                users_raw = raw.get("users", {}) or {}
                users: Dict[str, Dict[str, Dict[str, int]]] = {}
                for name, payload in users_raw.items():
                    progress_raw = (payload or {}).get("progress", {})
                    users[name] = {"progress": _normalize_progress_dict(progress_raw)}
                active_user: Optional[str] = raw.get("active_user")
                if not active_user or active_user not in users:
                    active_user = next(iter(users), "Player 1")
                if not users:
                    users = {"Player 1": {"progress": {}}}
                    active_user = "Player 1"
                return {"users": users, "active_user": active_user}
            else:
                # Legacy single-user format
                progress_raw = (raw or {}).get("progress", {})
                users = {"Player 1": {"progress": _normalize_progress_dict(progress_raw)}}
                return {"users": users, "active_user": "Player 1"}
    except Exception:
        pass
    return {"users": {"Player 1": {"progress": {}}}, "active_user": "Player 1"}


def save_users_state(state: Dict) -> None:
    users_clean: Dict[str, Dict[str, Dict[str, int]]] = {}
    for name, payload in (state.get("users", {}) or {}).items():
        users_clean[name] = {
            "progress": _normalize_progress_dict((payload or {}).get("progress", {}))
        }
    data = {
        "users": users_clean,
        "active_user": state.get("active_user"),
    }
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_progress() -> Dict:
    """Return the active user's progress in the legacy wrapper shape {"progress": {...}}."""
    users_state = load_users_state()
    active = users_state.get("active_user")
    progress = (users_state.get("users", {}) or {}).get(active, {}).get("progress", {})
    return {"progress": _normalize_progress_dict(progress)}


def save_progress(state: Dict) -> None:
    """Persist progress for the current user when multi-user state is present.

    Falls back to legacy single-user save otherwise.
    """
    try:
        users_state = st.session_state.get("users_state")
        current_user = st.session_state.get("current_user")
        if users_state and current_user:
            users_state = dict(users_state)
            users = dict(users_state.get("users", {}))
            user_payload = dict(users.get(current_user, {}))
            user_payload["progress"] = _normalize_progress_dict(state.get("progress", {}))
            users[current_user] = user_payload
            users_state["users"] = users
            users_state["active_user"] = current_user
            save_users_state(users_state)
            st.session_state["users_state"] = users_state
            return
    except Exception:
        # On any unexpected error, fallback to legacy write
        pass
    # Legacy single-user save
    DATA_FILE.write_text(
        json.dumps({"progress": _normalize_progress_dict(state.get("progress", {}))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ---------- Utility ----------
def get_available_continents() -> List[str]:
    return sorted({
        _normalize_continent_name(c.get("continent", ""))
        for c in COUNTRIES
        if _normalize_continent_name(c.get("continent", ""))
    })


def get_current_pool() -> List[Dict]:
    """Return the list of countries to play from based on current scope selection."""
    scope = st.session_state.get("play_scope", "World")
    if scope == "By continent":
        selected = st.session_state.get("selected_continent")
        if not selected:
            continents = get_available_continents()
            selected = continents[0] if continents else None
            st.session_state["selected_continent"] = selected
        if selected:
            sel_norm = _normalize_continent_name(selected).lower()
            pool = [
                c for c in COUNTRIES
                if _normalize_continent_name(c.get("continent", "")).lower() == sel_norm
            ]
            # Always return the filtered pool to avoid falling back to World unexpectedly
            return pool
    return COUNTRIES

def pick_question(pool: List[Dict], exclude_recent=3) -> Dict:
    # choose a random country; avoid last few asked if possible
    recent = st.session_state.get("recent_questions", [])
    choices = [c for c in pool if c["country"] not in recent]
    if not choices:
        choices = pool
    q = random.choice(choices)
    # update recent list
    recent.insert(0, q["country"])
    st.session_state["recent_questions"] = recent[:exclude_recent]
    return q

def generate_clue(capital: str, attempt: int, country_data: Dict) -> str:
    """Generate a clue focused on country facts (not capital spelling).

    Facts include population, area, continent, landlocked/borders, languages, currencies,
    timezones, top-level domains, and occasional political context.
    """
    country = (country_data.get("country") or "").strip()
    continent = (country_data.get("continent") or "").strip()
    political_fact = (country_data.get("political_fact") or "").strip()

    population = int(country_data.get("population") or 0)
    area_km2 = float(country_data.get("area_km2") or 0.0)
    landlocked = bool(country_data.get("landlocked", False))
    borders = list(country_data.get("borders") or [])
    languages = [l for l in (country_data.get("languages") or []) if l]
    currencies = [c for c in (country_data.get("currencies") or []) if c]
    fifa = (country_data.get("fifa") or "").strip()
    cca2 = (country_data.get("cca2") or "").strip().upper()
    timezones = list(country_data.get("timezones") or [])
    tlds = list(country_data.get("tld") or [])

    facts: List[str] = []
    # Geography and region
    if continent:
        facts.append(f"It's in {continent}.")
    if population:
        # Round population to nearest million for readability
        if population >= 1_000_000:
            millions = int(round(population / 1_000_000))
            facts.append(f"Population is about {millions} million.")
        else:
            facts.append(f"Population is roughly {population:,}.")
    if area_km2:
        if area_km2 >= 1_000_000:
            facts.append(f"Area is about {area_km2/1_000_000:.1f} million km².")
        else:
            facts.append(f"Area is about {int(round(area_km2)):,} km².")
    # Geography details
    coastland_text = ("It is landlocked." if landlocked else "It has a coastline.")
    # We'll add this later with lower priority so more informative facts show first
    if borders:
        facts.append(f"It borders {len(borders)} countries.")
    if languages:
        facts.append("Language example: " + languages[0] + ".")
    if currencies:
        facts.append("Currency example: " + currencies[0] + ".")
    if fifa:
        facts.append(f"FIFA code: {fifa}.")
    if cca2:
        facts.append(f"ISO code: {cca2}.")
    if timezones:
        facts.append(f"Timezone example: {timezones[0]}.")
    if tlds:
        facts.append(f"Internet domain ends with '{tlds[0]}'.")
    if political_fact and random.random() < 0.35:
        facts.append(political_fact + ".")

    # Add coastline info last to avoid dominating
    facts.append(coastland_text)

    # Prefer non-continent facts if any exist
    non_continent_facts = [f for f in facts if not f.startswith("It's in ")]
    pool = non_continent_facts if non_continent_facts else facts

    if not pool:
        pool = ["Think about its region, size, and neighbors."]

    # Deterministic shuffle so attempts rotate different clues
    seed_str = f"{country}|{capital}|facts|v2"
    rnd = random.Random(seed_str)
    pool_shuffled = list(pool)
    rnd.shuffle(pool_shuffled)

    # Return up to two different facts per attempt for richness
    if len(pool_shuffled) == 1:
        return pool_shuffled[0]
    start = ((attempt - 1) * 2) % len(pool_shuffled)
    first = pool_shuffled[start]
    second = pool_shuffled[(start + 1) % len(pool_shuffled)]
    if first == second:
        return first
    return f"{first} {second}"

def continent_progress_summary(progress: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    # Count countries with at least one correct ("right") answer
    summary: Dict[str, int] = {}
    for c in COUNTRIES:
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        has_any_right = int(entry.get("right", 0)) > 0
        summary[c["continent"]] = summary.get(c["continent"], 0) + (1 if has_any_right else 0)
    return summary


def compute_top_capitals(progress: Dict[str, Dict[str, int]], top_n: int = 3):
    """Return two lists of up to top_n items each:
    - top_right: capitals with highest right counts
    - top_wrong: capitals with highest wrong counts

    Each item is a dict: {country, capital, right, wrong}.
    Countries with zero attempts are ignored.
    """
    items = []
    for c in COUNTRIES:
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        right = int(entry.get("right", 0))
        wrong = int(entry.get("wrong", 0))
        attempts = right + wrong
        if attempts == 0:
            continue
        items.append({
            "country": c["country"],
            "capital": c["capital"],
            "right": right,
            "wrong": wrong,
            "flag": c.get("flag", ""),
        })

    top_right = sorted(
        (i for i in items if i["right"] > 0),
        key=lambda i: (-i["right"], i["wrong"], i["country"]),
    )[:top_n]

    top_wrong = sorted(
        (i for i in items if i["wrong"] > 0),
        key=lambda i: (-i["wrong"], i["right"], i["country"]),
    )[:top_n]

    return top_right, top_wrong


def compute_users_leaderboard(users_state: Dict, total_target: int = 197):
    """Compute leaderboard rows from multi-user state.

    Returns a sorted list of dicts with keys:
    {name, recognized, right, wrong, attempts, accuracy, mastery_pct}.
    Sorted by recognized desc, then right desc, then accuracy desc, then name.
    """
    users = (users_state or {}).get("users", {}) or {}
    rows = []
    for name, payload in users.items():
        prog = _normalize_progress_dict((payload or {}).get("progress", {}))
        total_right = sum(int(v.get("right", 0)) for v in prog.values())
        total_wrong = sum(int(v.get("wrong", 0)) for v in prog.values())
        attempts = total_right + total_wrong
        # Recognized = number of countries with at least one correct
        recognized = 0
        for c in COUNTRIES:
            entry = prog.get(c["country"], {"right": 0, "wrong": 0})
            if int(entry.get("right", 0)) > 0:
                recognized += 1
        accuracy = int(round((total_right / attempts) * 100)) if attempts else 0
        mastery_pct = int(round((recognized / total_target) * 100)) if total_target else 0
        rows.append({
            "name": name,
            "recognized": recognized,
            "right": total_right,
            "wrong": total_wrong,
            "attempts": attempts,
            "accuracy": accuracy,
            "mastery_pct": mastery_pct,
        })

    rows.sort(key=lambda r: (-r["recognized"], -r["right"], -r["accuracy"], r["name"]))
    return rows


def _compute_overall_stats(progress: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Return aggregate stats from a normalized progress dict.

    Keys: total_right, total_wrong, attempts, recognized
    """
    progress = _normalize_progress_dict(progress or {})
    total_right = sum(int(v.get("right", 0)) for v in progress.values())
    total_wrong = sum(int(v.get("wrong", 0)) for v in progress.values())
    attempts = total_right + total_wrong
    recognized = 0
    for c in COUNTRIES:
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        if int(entry.get("right", 0)) > 0:
            recognized += 1
    return {
        "total_right": total_right,
        "total_wrong": total_wrong,
        "attempts": attempts,
        "recognized": recognized,
    }

def build_world_map_wrong_rate(progress: Dict[str, Dict[str, int]]) -> go.Figure:
    # Calculate totals and percents per continent (based on countries with any right answers)
    continent_totals: Dict[str, int] = {}
    continent_completed_counts: Dict[str, int] = {}
    for c in COUNTRIES:
        cont = c["continent"]
        continent_totals[cont] = continent_totals.get(cont, 0) + 1
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        if int(entry.get("right", 0)) > 0:
            continent_completed_counts[cont] = continent_completed_counts.get(cont, 0) + 1

    continent_percent: Dict[str, int] = {}
    for cont, total in continent_totals.items():
        got = continent_completed_counts.get(cont, 0)
        pct = int(100 * got / total) if total else 0
        continent_percent[cont] = pct

    # Map each country to its wrong rate in percent (0–100).
    # Use sentinel -1 for not attempted so we can color them grey.
    country_names = [c["country"] for c in COUNTRIES]
    right_counts = [int(progress.get(c["country"], {"right": 0, "wrong": 0}).get("right", 0)) for c in COUNTRIES]
    wrong_counts = [int(progress.get(c["country"], {"right": 0, "wrong": 0}).get("wrong", 0)) for c in COUNTRIES]
    attempts_counts = [max(0, r + w) for r, w in zip(right_counts, wrong_counts)]
    # If there are no attempts OR zero wrong answers, show grey using sentinel -1
    wrong_rates_pct = [(-1 if (a == 0 or w == 0) else int(round((w / a) * 100))) for w, a in zip(wrong_counts, attempts_counts)]

    hover_text = []
    for c in COUNTRIES:
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        right = int(entry.get("right", 0))
        wrong = int(entry.get("wrong", 0))
        attempts = right + wrong
        wrong_rate = (int(round((wrong / attempts) * 100)) if (attempts > 0 and wrong > 0) else None)
        hover_text.append(
            f"<b>{c['country']}</b>"
            f"<br>Continent: {c['continent']}"
            f"<br>Right: {right} | Wrong: {wrong} | Attempts: {attempts}"
            f"<br>Wrong rate: {('-' if wrong_rate is None else str(wrong_rate) + '%')}"
            f"<br>Continent coverage: {continent_percent.get(c['continent'], 0)}%"
        )

    # Colorscale: grey for not attempted, then light red to deep red as wrong rate increases
    # We set zmin=-1 and zmax=100, so normalized position of 0 is ~0.0099
    red_only_colorscale = [
        (0.0, "#E2E8F0"),   # -1 (not attempted) -> grey
        (0.01, "#FEE2E2"),  # ~0% wrong (attempted) -> very light red
        (0.50, "#FCA5A5"),  # ~50% wrong -> medium red
        (1.0, "#B91C1C"),   # 100% wrong -> deep red
    ]

    choropleth = go.Choropleth(
        locations=country_names,
        locationmode="country names",
        z=wrong_rates_pct,
        zmin=-1,
        zmax=100,
        text=hover_text,
        colorscale=red_only_colorscale,
        autocolorscale=False,
        colorbar=dict(
            title=dict(text="Wrong rate (%)", side="top"),
            thickness=12,
            len=0.7,
            bgcolor="rgba(255,255,255,0.85)",
            outlinewidth=0,
            tickmode="array",
            tickvals=[0, 25, 50, 75, 100],
        ),
        marker_line_width=0.7,
        marker_line_color="#94A3B8",
        hovertemplate="%{text}<extra></extra>",
    )

    # Approximate label positions for continents
    continent_label_positions = {
        "Africa": {"lon": 20, "lat": 0},
        "Europe": {"lon": 15, "lat": 50},
        "Asia": {"lon": 90, "lat": 35},
        "North America": {"lon": -100, "lat": 40},
        "South America": {"lon": -60, "lat": -15},
        "Oceania": {"lon": 140, "lat": -25},
    }

    label_lons = []
    label_lats = []
    label_texts = []
    for cont, pos in continent_label_positions.items():
        pct = continent_percent.get(cont, 0)
        label_lons.append(pos["lon"]) 
        label_lats.append(pos["lat"]) 
        label_texts.append(f"{cont}: {pct}%")

    labels = go.Scattergeo(
        lon=label_lons,
        lat=label_lats,
        mode="text",
        text=label_texts,
        textfont=dict(size=13, color="#334155"),
        hoverinfo="skip",
    )

    fig = go.Figure(data=[choropleth, labels])
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif", color="#0f172a"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(color="#0f172a")),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CBD5E1",
            coastlinewidth=0.5,
            showland=True,
            landcolor="#F8FAFC",
            showocean=True,
            oceancolor="#F1F5F9",
            lakecolor="#F1F5F9",
            projection_type="natural earth",
        ),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig

def build_world_map_correct(progress: Dict[str, Dict[str, int]]) -> go.Figure:
    # Calculate totals and percents per continent (based on countries with any right answers)
    continent_totals: Dict[str, int] = {}
    continent_completed_counts: Dict[str, int] = {}
    for c in COUNTRIES:
        cont = c["continent"]
        continent_totals[cont] = continent_totals.get(cont, 0) + 1
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        if int(entry.get("right", 0)) > 0:
            continent_completed_counts[cont] = continent_completed_counts.get(cont, 0) + 1

    continent_percent: Dict[str, int] = {}
    for cont, total in continent_totals.items():
        got = continent_completed_counts.get(cont, 0)
        pct = int(100 * got / total) if total else 0
        continent_percent[cont] = pct

    # Map each country to capped right counts (0–5). Use sentinel -1 for not attempted -> grey.
    country_names = [c["country"] for c in COUNTRIES]
    right_counts = [int(progress.get(c["country"], {"right": 0, "wrong": 0}).get("right", 0)) for c in COUNTRIES]
    capped_counts = [min(v, 5) for v in right_counts]
    attempts_counts = [int(progress.get(c["country"], {"right": 0, "wrong": 0}).get("right", 0)) + int(progress.get(c["country"], {"right": 0, "wrong": 0}).get("wrong", 0)) for c in COUNTRIES]
    z_correct = [(-1 if a == 0 else c) for a, c in zip(attempts_counts, capped_counts)]

    hover_text = []
    for c in COUNTRIES:
        entry = progress.get(c["country"], {"right": 0, "wrong": 0})
        right = int(entry.get("right", 0))
        wrong = int(entry.get("wrong", 0))
        attempts = right + wrong
        hover_text.append(
            f"<b>{c['country']}</b>"
            f"<br>Continent: {c['continent']}"
            f"<br>Right: {right} | Wrong: {wrong} | Attempts: {attempts}"
            f"<br>Continent coverage: {continent_percent.get(c['continent'], 0)}%"
        )

    # Colorscale: grey for not attempted (-1), then light green to deep green as correct increases
    green_only_colorscale = [
        (0.0, "#E2E8F0"),  # -1 -> grey
        (0.01, "#DCFCE7"), # ~0 (attempted but 0 correct) -> very light green
        (0.40, "#86EFAC"),
        (0.80, "#22C55E"),
        (1.0, "#15803D"),  # max -> deep green
    ]

    choropleth = go.Choropleth(
        locations=country_names,
        locationmode="country names",
        z=z_correct,
        zmin=-1,
        zmax=5,
        text=hover_text,
        colorscale=green_only_colorscale,
        autocolorscale=False,
        colorbar=dict(
            title=dict(text="Correct (0–5)", side="top"),
            thickness=12,
            len=0.7,
            bgcolor="rgba(255,255,255,0.85)",
            outlinewidth=0,
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5],
        ),
        marker_line_width=0.7,
        marker_line_color="#94A3B8",
        hovertemplate="%{text}<extra></extra>",
    )

    # Approximate label positions for continents
    continent_label_positions = {
        "Africa": {"lon": 20, "lat": 0},
        "Europe": {"lon": 15, "lat": 50},
        "Asia": {"lon": 90, "lat": 35},
        "North America": {"lon": -100, "lat": 40},
        "South America": {"lon": -60, "lat": -15},
        "Oceania": {"lon": 140, "lat": -25},
    }

    label_lons = []
    label_lats = []
    label_texts = []
    for cont, pos in continent_label_positions.items():
        pct = continent_percent.get(cont, 0)
        label_lons.append(pos["lon"]) 
        label_lats.append(pos["lat"]) 
        label_texts.append(f"{cont}: {pct}%")

    labels = go.Scattergeo(
        lon=label_lons,
        lat=label_lats,
        mode="text",
        text=label_texts,
        textfont=dict(size=13, color="#334155"),
        hoverinfo="skip",
    )

    fig = go.Figure(data=[choropleth, labels])
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif", color="#0f172a"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(color="#0f172a")),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CBD5E1",
            coastlinewidth=0.5,
            showland=True,
            landcolor="#F8FAFC",
            showocean=True,
            oceancolor="#F1F5F9",
            lakecolor="#F1F5F9",
            projection_type="natural earth",
        ),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


# ---------- Footer ----------
def render_footer() -> None:
    """Render a small footer indicating creation date and purpose."""
    try:
        stats = pathlib.Path(__file__).resolve().stat()
        created_ts = getattr(stats, "st_birthtime", stats.st_ctime)
    except Exception:
        created_ts = time.time()
    created_dt = datetime.datetime.fromtimestamp(created_ts)
    created_str = created_dt.strftime("%B %d, %Y")

    st.write("---")
    st.markdown(
        (
            f"<div style='color:#64748b;font-size:12px;text-align:center;'>"
            f"Created {created_str}. A vibe-coded passion project — built with GPT-5 and Cursor. "
            f"Made for my love for countries and the world."
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

# ---------- App UI ----------
st.set_page_config(page_title="Capital Gains: World Edition", layout="centered")

# Introduction page (shown until user starts)
if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = True

if st.session_state["show_intro"]:
    st.title("🌍 Capital Gains: World Edition")
    st.markdown("Learn country capitals and track progress by continent.")
    st.write("---")
    st.subheader("How it works")
    st.markdown(
        "- 3 attempts per question.\n"
        "- Progress saves locally to `progress.json` and supports multiple players.\n"
        "- Maps visualize what you've mastered and where to review.\n"
        "- A leaderboard tracks recognition, accuracy, and attempts."
    )
    if st.button("Start playing", type="primary"):
        st.session_state["show_intro"] = False
        # initialize session timer at play start
        st.session_state["session_start_ts"] = time.time()
        # set baseline snapshot at session start (previous performance)
        try:
            initial_progress = load_progress().get("progress", {})
            st.session_state["baseline_progress"] = _normalize_progress_dict(initial_progress)
        except Exception:
            st.session_state["baseline_progress"] = {}
        st.rerun()
    # Footer on intro page
    render_footer()
    st.stop()

st.title("🌍 Capital Gains: World Edition")
st.write("Learn country capitals with a playful leader avatar and track progress by continent.")

# Global progress bar (toward 197 countries)
# Use session progress if initialized; otherwise fall back to saved progress
_progress_state = _normalize_progress_dict(st.session_state.get("progress", load_progress().get("progress", {})))
_recognized_count = sum(1 for c in COUNTRIES if _progress_state.get(c["country"], {"right": 0}).get("right", 0) > 0)
_TOTAL_TARGET = 197
_percent_int = int(min(100, (_recognized_count / _TOTAL_TARGET) * 100)) if _TOTAL_TARGET else 0
st.progress(_percent_int)
st.caption(f"Global mastery: {_recognized_count}/{_TOTAL_TARGET} countries")

# session state defaults
if "users_state" not in st.session_state:
    st.session_state["users_state"] = load_users_state()
if "current_user" not in st.session_state:
    users_dict = st.session_state["users_state"].get("users", {}) or {"Player 1": {"progress": {}}}
    st.session_state["current_user"] = (
        st.session_state["users_state"].get("active_user")
        or next(iter(users_dict))
    )
if "progress" not in st.session_state:
    cu = st.session_state["current_user"]
    st.session_state["progress"] = (
        st.session_state["users_state"].get("users", {}).get(cu, {}).get("progress", {})
    )  # normalized later
if "baseline_progress" not in st.session_state:
    # baseline = snapshot of user's progress at the start of the session
    st.session_state["baseline_progress"] = _normalize_progress_dict(st.session_state.get("progress", {}))
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "wrong" not in st.session_state:
    st.session_state["wrong"] = 0
if "play_scope" not in st.session_state:
    st.session_state["play_scope"] = "World"
if "selected_continent" not in st.session_state:
    st.session_state["selected_continent"] = (get_available_continents()[0] if get_available_continents() else None)
if "question" not in st.session_state:
    st.session_state["question"] = pick_question(get_current_pool())
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False
if "attempts" not in st.session_state:
    st.session_state["attempts"] = 0
if "should_clear_feedback" not in st.session_state:
    st.session_state["should_clear_feedback"] = False
if "show_next_button" not in st.session_state:
    st.session_state["show_next_button"] = False
if "new_player_input_counter" not in st.session_state:
    st.session_state["new_player_input_counter"] = 0
if "rename_input_counter" not in st.session_state:
    st.session_state["rename_input_counter"] = 0
if "session_start_ts" not in st.session_state:
    # Fallback initialization in case user bypassed intro or on first load
    st.session_state["session_start_ts"] = time.time()
if "show_summary_page" not in st.session_state:
    st.session_state["show_summary_page"] = False
if "session_end_ts" not in st.session_state:
    st.session_state["session_end_ts"] = None

# Render summary page when requested
if st.session_state.get("show_summary_page", False):
    st.title("📊 Session Summary")
    # freeze end time if not set
    if not st.session_state.get("session_end_ts"):
        st.session_state["session_end_ts"] = time.time()
    duration_s = max(0, int(st.session_state["session_end_ts"] - st.session_state.get("session_start_ts", time.time())))
    h = duration_s // 3600
    m = (duration_s % 3600) // 60
    s = duration_s % 60
    duration_str = (f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}")

    baseline = _normalize_progress_dict(st.session_state.get("baseline_progress", {}))
    current = _normalize_progress_dict(st.session_state.get("progress", {}))
    bstats = _compute_overall_stats(baseline)
    cstats = _compute_overall_stats(current)

    delta_right = cstats["total_right"] - bstats["total_right"]
    delta_wrong = cstats["total_wrong"] - bstats["total_wrong"]
    delta_attempts = cstats["attempts"] - bstats["attempts"]
    delta_recognized = cstats["recognized"] - bstats["recognized"]

    session_right = int(st.session_state.get("score", 0))
    session_wrong = int(st.session_state.get("wrong", 0))
    session_attempts = max(0, session_right + session_wrong)
    session_acc = int(round((session_right / session_attempts) * 100)) if session_attempts else 0

    acc_before = int(round((bstats["total_right"] / bstats["attempts"]) * 100)) if bstats["attempts"] else 0
    acc_after = int(round((cstats["total_right"] / cstats["attempts"]) * 100)) if cstats["attempts"] else 0
    delta_acc = acc_after - acc_before

    # Encouraging message
    if delta_recognized > 0:
        encouragement = f"Fantastic! You unlocked {delta_recognized} new {'country' if delta_recognized==1 else 'countries'} this session. Keep going!"
    elif session_acc >= 80 and session_attempts >= 5:
        encouragement = "Great accuracy this session — you’re really mastering these capitals!"
    elif session_attempts >= 10:
        encouragement = "Awesome stamina — consistency is how pros are made. Keep it up!"
    else:
        encouragement = "Nice work — every attempt builds your map mastery."

    st.caption(f"Session duration: {duration_str}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("This session")
        st.metric("Correct", session_right)
        st.metric("Wrong", session_wrong)
        st.metric("Accuracy", f"{session_acc}%")
    with c2:
        st.subheader("Overall (before → after)")
        st.metric("Recognized countries", cstats["recognized"], delta=delta_recognized)
        st.metric("Total answers", cstats["attempts"], delta=delta_attempts)
        st.metric("Overall accuracy", f"{acc_after}%", delta=delta_acc)

    st.success(encouragement)

    st.write("")
    if st.button("Back to game", type="primary"):
        # start a fresh session using current progress as new baseline
        st.session_state["show_summary_page"] = False
        st.session_state["session_start_ts"] = time.time()
        st.session_state["session_end_ts"] = None
        st.session_state["baseline_progress"] = _normalize_progress_dict(st.session_state.get("progress", {}))
        st.session_state["score"] = 0
        st.session_state["wrong"] = 0
        st.session_state["attempts"] = 0
        st.rerun()

    # Footer on summary page
    render_footer()
    st.stop()


# Sidebar controls
with st.sidebar:
    # Players management
    st.header("Players")
    users_state = st.session_state.get("users_state", load_users_state())
    user_names = list((users_state.get("users", {}) or {}).keys())
    if not user_names:
        users_state = {"users": {"Player 1": {"progress": {}}}, "active_user": "Player 1"}
        save_users_state(users_state)
        st.session_state["users_state"] = users_state
        user_names = ["Player 1"]

    current_user = st.session_state.get("current_user") or users_state.get("active_user") or user_names[0]
    # Sync radio selection with session if previously set or pending
    pending_select = st.session_state.pop("pending_select_user", None)
    if pending_select and pending_select in user_names:
        st.session_state["player_select"] = pending_select
    # If stored radio value is no longer valid (e.g., after rename), clear it so index applies
    if (stored := st.session_state.get("player_select")) and stored not in user_names:
        st.session_state.pop("player_select", None)
    try:
        selected_user = st.radio("Current player", options=user_names, index=user_names.index(current_user), key="player_select")
    except ValueError:
        selected_user = st.radio("Current player", options=user_names, index=0, key="player_select")

    if selected_user != current_user:
        st.session_state["current_user"] = selected_user
        # Swap in that user's progress
        st.session_state["progress"] = (users_state.get("users", {}).get(selected_user, {}).get("progress", {}))
        # Reset session counters on switch
        st.session_state["score"] = 0
        st.session_state["wrong"] = 0
        st.session_state["attempts"] = 0
        # restart session timer on user switch
        st.session_state["session_start_ts"] = time.time()
        # reset baseline to new user's current progress
        st.session_state["baseline_progress"] = _normalize_progress_dict(st.session_state.get("progress", {}))
        users_state["active_user"] = selected_user
        save_users_state(users_state)
        st.session_state["users_state"] = users_state
        st.rerun()

    new_player_key = f"new_player_name_{st.session_state['new_player_input_counter']}"
    new_player = st.text_input("Add player", key=new_player_key, placeholder="e.g., Alice")
    if st.button("Create player"):
        # Read from the latest dynamic key
        name = (st.session_state.get(new_player_key) or "").strip()
        if not name:
            st.warning("Enter a name.")
        elif name in (users_state.get("users", {}) or {}):
            st.warning("That name already exists.")
        else:
            users_state.setdefault("users", {})[name] = {"progress": {}}
            users_state["active_user"] = name
            save_users_state(users_state)
            st.session_state["users_state"] = users_state
            st.session_state["current_user"] = name
            st.session_state["progress"] = {}
            st.session_state["score"] = 0
            st.session_state["wrong"] = 0
            st.session_state["attempts"] = 0
            # start timer for the new player's session
            st.session_state["session_start_ts"] = time.time()
            # baseline is empty for a brand new player
            st.session_state["baseline_progress"] = {}
            # Clear input by rotating key on next render
            st.session_state["new_player_input_counter"] += 1
            # Ensure radio selects the new player on next render
            st.session_state["pending_select_user"] = name
            st.success(f"Player '{name}' created.")
            st.rerun()

    # Rename current player
    rename_key = f"rename_player_name_{st.session_state['rename_input_counter']}"
    st.text_input("Rename current player", key=rename_key, placeholder="New name")
    if st.button("Rename"):
            old_name = st.session_state.get("current_user") or users_state.get("active_user")
            new_name = (st.session_state.get(rename_key) or "").strip()
            users = users_state.get("users", {}) or {}
            if not new_name:
                st.warning("Enter a new name.")
            elif new_name == old_name:
                st.info("That is already the current name.")
            elif new_name in users:
                st.warning("Another player already has that name.")
            else:
                payload = users.get(old_name, {"progress": st.session_state.get("progress", {})})
                users[new_name] = payload
                if old_name in users:
                    del users[old_name]
                users_state["users"] = users
                users_state["active_user"] = new_name
                save_users_state(users_state)
                st.session_state["users_state"] = users_state
                st.session_state["current_user"] = new_name
                # keep current progress/counters as-is on rename
                # Clear input by rotating key on next render
                st.session_state["rename_input_counter"] += 1
                # Ensure radio selects the new player on next render
                st.session_state["pending_select_user"] = new_name
                st.success("Player renamed.")
                st.rerun()

    # Delete current player
    if st.button("Delete current player"):
        users_state = st.session_state.get("users_state", load_users_state())
        users = dict(users_state.get("users", {}) or {})
        current = st.session_state.get("current_user") or users_state.get("active_user")
        if len(users) <= 1:
            st.warning("Cannot delete the only player.")
        elif current not in users:
            st.warning("Current player not found.")
        else:
            # Remove the current user and select the next available one
            del users[current]
            if not users:
                users = {"Player 1": {"progress": {}}}
                next_user = "Player 1"
            else:
                next_user = next(iter(users))

            users_state["users"] = users
            users_state["active_user"] = next_user
            save_users_state(users_state)

            st.session_state["users_state"] = users_state
            st.session_state["current_user"] = next_user
            st.session_state["progress"] = users.get(next_user, {}).get("progress", {})
            # Reset session counters on delete
            st.session_state["score"] = 0
            st.session_state["wrong"] = 0
            st.session_state["attempts"] = 0
            # restart timer after switching to next available user
            st.session_state["session_start_ts"] = time.time()
            # reset baseline to the selected next user's progress
            st.session_state["baseline_progress"] = _normalize_progress_dict(st.session_state.get("progress", {}))
            # Ensure radio selects the new player on next render
            st.session_state["pending_select_user"] = next_user
            st.success(f"Player '{current}' removed.")
            st.rerun()

    st.write("---")
    st.header("Mode")
    prev_scope = st.session_state.get("play_scope", "World")
    prev_cont = st.session_state.get("selected_continent")
    scope_value = st.radio("Play scope", options=["World", "By continent"], index=(0 if prev_scope == "World" else 1), key="play_scope")
    if scope_value == "By continent":
        continents = get_available_continents()
        # Ensure a valid selection exists
        if (prev_cont not in continents) and continents:
            st.session_state["selected_continent"] = continents[0]
            prev_cont = continents[0]
        # Bind purely via key so the widget always reflects session state; avoid fragile index math
        st.selectbox("Continent", options=continents, key="selected_continent")

    if (prev_scope != st.session_state.get("play_scope")) or (prev_cont != st.session_state.get("selected_continent")):
        st.session_state["question"] = pick_question(get_current_pool())
        st.session_state["attempts"] = 0
        st.session_state["show_next_button"] = False
        st.session_state["feedback"] = ""
        st.session_state["input_counter"] = st.session_state.get("input_counter", 0) + 1
        st.session_state["clear_input"] = True
        # Ensure the newly scoped question renders immediately
        st.rerun()

    st.write("---")
    st.header("Session")
    if st.button("New random question"):
        st.session_state["question"] = pick_question(get_current_pool())
        st.session_state["clear_input"] = True
        st.session_state["feedback"] = ""
        st.session_state["attempts"] = 0
        st.session_state["show_next_button"] = False
    if st.button("Reset progress"):
        st.session_state["progress"] = {}
        save_progress({"progress": st.session_state["progress"]})
        st.success("Progress reset.")
        # When progress resets mid-session, also reset baseline to reflect the new starting point
        st.session_state["baseline_progress"] = {}
    if st.button("End session"):
        # show summary page comparing current session vs baseline
        st.session_state["show_summary_page"] = True
        st.session_state["session_end_ts"] = time.time()
        st.rerun()
    st.write("---")
    st.subheader("Persistence")
    st.write(f"Progress file: `{DATA_FILE.name}`")
    if DATA_FILE.exists():
        st.write("Saved locally ✅")
        # Show brief per-player stats
        users_state = st.session_state.get("users_state", load_users_state())
        users = users_state.get("users", {}) or {}
        if users:
            lines = []
            for uname, payload in users.items():
                prog = _normalize_progress_dict((payload or {}).get("progress", {}))
                recognized = sum(1 for c in COUNTRIES if prog.get(c["country"], {"right": 0}).get("right", 0) > 0)
                lines.append(f"- {uname}: {recognized} countries recognized")
            st.caption("Players: \n" + "\n".join(lines))
    else:
        st.write("No saved progress yet")

# main card
# Ensure the current question respects the current scope/continent
current_pool = get_current_pool()
q = st.session_state["question"]
if current_pool:
    pool_country_names = {c["country"] for c in current_pool}
    if q.get("country") not in pool_country_names:
        # If the stored question is out-of-scope (e.g., after a scope change), replace it
        st.session_state["question"] = pick_question(current_pool)
        q = st.session_state["question"]
avatar = q.get("avatar", "👑")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown(f"<div style='font-size:72px;text-align:center'>{avatar}</div>", unsafe_allow_html=True)
with col2:
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader(f"What is the capital of **{q['country']}**?")
    with header_right:
        _start_ts_hdr = int(st.session_state.get("session_start_ts", time.time()))
        _timer_html_inline = """
        <div style='display:flex;justify-content:flex-end;'>
          <div style='display:inline-block;padding:8px 14px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;'>
            <div style='font-size:12px;color:#64748b;line-height:1;'>Session timer</div>
            <div id='session-timer' style='font-size:28px;font-weight:700;color:#0f172a;line-height:1.1;'>--:--</div>
          </div>
        </div>
        <script>
          const __start = __START_TS__ * 1000;
          function __fmt(ms){
            let s = Math.floor(ms/1000);
            const h = Math.floor(s/3600); s %= 3600;
            const m = Math.floor(s/60); s %= 60;
            const hh = h > 0 ? String(h).padStart(2,'0') + ':' : '';
            const mm = String(m).padStart(2,'0');
            const ss = String(s).padStart(2,'0');
            return hh + mm + ':' + ss;
          }
          function __tick(){
            const el = document.getElementById('session-timer');
            if(!el) return;
            el.textContent = __fmt(Date.now() - __start);
          }
          __tick();
          setInterval(__tick, 1000);
        </script>
        """
        components.html(_timer_html_inline.replace("__START_TS__", str(_start_ts_hdr)), height=80, scrolling=False)
    
    # Show attempts remaining
    attempts_remaining = 3 - st.session_state["attempts"]
    if attempts_remaining < 3:
        st.info(f"Attempts remaining: {attempts_remaining}")
    
    # Use a form so Enter can submit when answering; show a right-aligned Next button when available
    with st.form(key=f"answer_form_{st.session_state.get('input_counter', 0)}"):
        # Determine state first so we can hide the input when showing Next
        is_next_state = st.session_state.get("show_next_button", False)

        # Only render the text input when awaiting an answer
        if not is_next_state:
            # Use a unique key for the text input that changes when we want to clear it
            input_key = f"input_{st.session_state.get('input_counter', 0)}"
            user_input = st.text_input(
                "Your answer",
                key=input_key,
                placeholder="Type the capital and press Enter",
            )
        else:
            user_input = ""

        # Single submit target: label toggles based on state so Enter always triggers it
        primary_label = "🎯 Next Question" if is_next_state else "Submit"
        trigger = st.form_submit_button(primary_label, type="primary")

        if trigger and not st.session_state.get("show_next_button", False):
            # Clear feedback if flag is set (from previous correct answer)
            if st.session_state.get("should_clear_feedback", False):
                st.session_state["feedback"] = ""
                st.session_state["should_clear_feedback"] = False

            answer = (user_input or "").strip().lower()
            correct = q["capital"].strip().lower()
            if answer == "":
                st.session_state["feedback"] = "Please enter an answer."
            elif answer == correct:
                st.session_state["score"] += 1
                # increment right count for that country
                # normalize current progress first (in case of legacy structure)
                st.session_state["progress"] = _normalize_progress_dict(st.session_state["progress"])
                entry = st.session_state["progress"].setdefault(q["country"], {"right": 0, "wrong": 0})
                entry["right"] = int(entry.get("right", 0)) + 1
                # Show success message
                st.session_state["feedback"] = f"🎉 **Correct!** The capital of {q['country']} is **{q['capital']}**. Great job!"
                # auto-save
                save_progress({"progress": st.session_state["progress"]})
                # Set flag to show next question button
                st.session_state["show_next_button"] = True
                # Increment counter to get a new input key (clears the field)
                st.session_state["input_counter"] = st.session_state.get("input_counter", 0) + 1
                # Immediately rerun to render the Next button without extra Enter
                st.rerun()
            else:
                st.session_state["attempts"] += 1
                st.session_state["wrong"] += 1
                # increment wrong count for that country
                st.session_state["progress"] = _normalize_progress_dict(st.session_state["progress"])
                entry = st.session_state["progress"].setdefault(q["country"], {"right": 0, "wrong": 0})
                entry["wrong"] = int(entry.get("wrong", 0)) + 1
                save_progress({"progress": st.session_state["progress"]})
                attempts_left = 3 - st.session_state["attempts"]

                if attempts_left > 0:
                    # No clues; just show attempts remaining
                    st.session_state["feedback"] = f"❌ Not quite right. (Attempts remaining: {attempts_left})"
                else:
                    # No more attempts: reveal correct answer and show Next button
                    st.session_state["feedback"] = f"❌ The correct answer was **{q['capital']}**."
                    st.session_state["show_next_button"] = True
                    # Increment counter to get a new input key (clears the field)
                    st.session_state["input_counter"] = st.session_state.get("input_counter", 0) + 1
                    # Rerun to render the Next button immediately
                    st.rerun()

        if trigger and st.session_state.get("show_next_button", False):
            st.session_state["question"] = pick_question(get_current_pool())
            st.session_state["attempts"] = 0
            st.session_state["show_next_button"] = False
            st.session_state["feedback"] = ""
            st.session_state["input_counter"] = st.session_state.get("input_counter", 0) + 1
            st.rerun()

    # feedback
    if "feedback" in st.session_state and st.session_state["feedback"]:
        st.write(st.session_state["feedback"])

# Handle clear input flag from sidebar
if st.session_state.get("clear_input", False):
    st.session_state["input_counter"] = st.session_state.get("input_counter", 0) + 1
    st.session_state["clear_input"] = False
    st.session_state["attempts"] = 0
    st.rerun()

# Score and progress
st.write("---")
st.subheader("Session score")
st.metric("Correct answers this session", st.session_state["score"])
st.metric("Wrong answers this session", st.session_state.get("wrong", 0))

st.subheader("Progress maps")
st.session_state["progress"] = _normalize_progress_dict(st.session_state.get("progress", {}))
summary = continent_progress_summary(st.session_state["progress"])

# Two maps in tabs: Progress rate and Error rate
tab1, tab2 = st.tabs(["Progress rate", "Error rate"])
with tab1:
    fig_correct = build_world_map_correct(st.session_state["progress"])
    st.plotly_chart(fig_correct, use_container_width=True)
with tab2:
    fig_wrong = build_world_map_wrong_rate(st.session_state["progress"])
    st.plotly_chart(fig_wrong, use_container_width=True)

# Top performers and tough ones
top_right, top_wrong = compute_top_capitals(st.session_state["progress"], top_n=3)

col_tr, col_tw = st.columns(2)
with col_tr:
    if top_right:
        rows_html = "".join([
            (
                f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0;'>"
                f"  <img src='{i.get('flag','')}' alt='{i['capital']} flag' style='width:64px;height:auto;border-radius:4px;box-shadow:0 1px 2px rgba(0,0,0,0.08);' />"
                f"  <div style='font-size:14px;color:#334155;'>{i['capital']}</div>"
                f"</div>"
                if i.get("flag") else
                f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0;'>"
                f"  <div style='width:64px;height:42px;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;'></div>"
                f"  <div style='font-size:14px;color:#334155;'>{i['capital']}</div>"
                f"</div>"
            ) for i in top_right
        ])
        container_html = (
            f"<div style='background:#ECFDF5;border:1px solid #A7F3D0;border-radius:12px;padding:12px;'>"
            f"<div style='font-weight:600;color:#065F46;margin-bottom:8px;'>Top 3 you nailed</div>"
            f"{rows_html}"
            f"</div>"
        )
        height_tr = 90 + 72 * len(top_right)
        components.html(container_html, height=height_tr, scrolling=False)
    else:
        components.html("<div style='background:#ECFDF5;border:1px solid #A7F3D0;border-radius:12px;padding:12px;'><div style='font-weight:600;color:#065F46;margin-bottom:8px;'>Top 3 you nailed</div><div style='color:#065F46;'>No correct answers yet.</div></div>", height=110, scrolling=False)

with col_tw:
    if top_wrong:
        rows_html = "".join([
            (
                f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0;'>"
                f"  <img src='{i.get('flag','')}' alt='{i['capital']} flag' style='width:64px;height:auto;border-radius:4px;box-shadow:0 1px 2px rgba(0,0,0,0.08);' />"
                f"  <div style='font-size:14px;color:#334155;'>{i['capital']}</div>"
                f"</div>"
                if i.get("flag") else
                f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0;'>"
                f"  <div style='width:64px;height:42px;background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;'></div>"
                f"  <div style='font-size:14px;color:#334155;'>{i['capital']}</div>"
                f"</div>"
            ) for i in top_wrong
        ])
        container_html = (
            f"<div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;padding:12px;'>"
            f"<div style='font-weight:600;color:#7F1D1D;margin-bottom:8px;'>Top 3 to review</div>"
            f"{rows_html}"
            f"</div>"
        )
        height_tw = 90 + 72 * len(top_wrong)
        components.html(container_html, height=height_tw, scrolling=False)
    else:
        components.html("<div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;padding:12px;'><div style='font-weight:600;color:#7F1D1D;margin-bottom:8px;'>Top 3 to review</div><div style='color:#7F1D1D;'>No wrong answers yet.</div></div>", height=110, scrolling=False)

# Leaderboard
st.write("---")
st.subheader("Leaderboard")
users_state = st.session_state.get("users_state", load_users_state())
leader_rows = compute_users_leaderboard(users_state, total_target=_TOTAL_TARGET)

if leader_rows:
    # Build compact styled rows
    html_rows = []
    for rank, r in enumerate(leader_rows, start=1):
        badge = (
            "🥇" if rank == 1 else
            "🥈" if rank == 2 else
            "🥉" if rank == 3 else
            f"#{rank}"
        )
        html_rows.append(
            """
            <div style='display:flex;align-items:center;gap:12px;justify-content:space-between;padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#ffffff;'>
              <div style='display:flex;align-items:center;gap:10px;'>
                <div style='font-size:18px;width:32px;text-align:center;'>{badge}</div>
                <div style='font-weight:600;color:#0f172a;'>{name}</div>
              </div>
              <div style='display:flex;align-items:center;gap:16px;color:#334155;font-size:13px;'>
                <div title='Countries recognized'>🌍 {recognized}</div>
                <div title='Accuracy'>🎯 {accuracy}%</div>
                <div title='Mastery of 197'>📈 {mastery}%</div>
                <div title='Total answers'>🧮 {attempts}</div>
              </div>
            </div>
            """.format(
                badge=badge,
                name=r["name"],
                recognized=r["recognized"],
                accuracy=r["accuracy"],
                mastery=r["mastery_pct"],
                attempts=r["attempts"],
            )
        )
    list_html = (
        "<div style='display:flex;flex-direction:column;gap:8px'>" + "".join(html_rows) + "</div>"
    )
    components.html(list_html, height=min(420, 72 * len(leader_rows) + 30), scrolling=True)
else:
    st.caption("No players yet. Add one in the sidebar.")

# Textual summary per continent
for cont in sorted(summary):
    total = sum(1 for c in COUNTRIES if c["continent"] == cont)
    got = summary[cont]
    pct = int(100 * got / total) if total else 0
    st.write(f"**{cont}** — {got}/{total} ({pct}%)")

st.write("---")
st.subheader("Detailed progress (but also a cheatsheet)")
visible_countries = get_current_pool()
for c in visible_countries:
    entry = st.session_state["progress"].get(c["country"], {"right": 0, "wrong": 0})
    right = int(entry.get("right", 0))
    wrong = int(entry.get("wrong", 0))
    attempts = right + wrong
    wrong_rate = int(round((wrong / attempts) * 100)) if attempts > 0 else 0
    st.write(f"{c['country']}: {c['capital']} — Right: {right} | Wrong: {wrong} | Wrong rate: {wrong_rate}%")

st.write("")

# Always show footer at end of main page
render_footer()
