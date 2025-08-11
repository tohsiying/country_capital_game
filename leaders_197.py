import json
import csv
import time
from typing import Dict, List, Set, Tuple

import requests


RESTCOUNTRIES_URL = "https://restcountries.com/v3.1/all"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "country-capital-game/1.0 (leader list)"


OBSERVERS = {"Holy See", "Palestine"}
EXTRA_STATES = {"Kosovo", "Taiwan"}


def load_197_country_set() -> List[Dict[str, str]]:
    """Return list of dicts with keys: country, cca2 using the app's 197 scope.

    Source: Rest Countries + add observers and extra states.
    """
    params = {
        "fields": "name,region,subregion,unMember,cca2",
    }
    resp = requests.get(RESTCOUNTRIES_URL, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    data = resp.json()

    def continent_from_region(region: str, subregion: str) -> str:
        region = (region or "").strip()
        subregion = (subregion or "").strip()
        if region != "Americas":
            return region or "Other"
        sr = subregion.lower()
        if "south" in sr:
            return "South America"
        return "North America"

    countries: List[Dict[str, str]] = []
    for item in data:
        name = (item.get("name", {}).get("common") or "").strip()
        if not name:
            continue
        cca2 = (item.get("cca2") or "").strip().upper()
        if not cca2:
            continue
        un_member = bool(item.get("unMember", False))
        include = un_member or (name in OBSERVERS) or (name in EXTRA_STATES)
        if not include:
            continue
        countries.append({
            "country": name,
            "cca2": cca2,
            "continent": continent_from_region(item.get("region", ""), item.get("subregion", "")),
        })

    # Deduplicate and sort by country name
    unique: Dict[str, Dict[str, str]] = {}
    for c in countries:
        unique[c["country"]] = c
    result = sorted(unique.values(), key=lambda x: x["country"])
    return result


def query_wikidata_heads_by_iso2(iso2_values: List[str]) -> List[Dict[str, str]]:
    """Return raw rows from Wikidata for a batch of ISO2 codes.

    Columns: iso2, headOfStateLabel, headOfGovernmentLabel
    """
    if not iso2_values:
        return []
    values_clause = " ".join(f'"{c}"' for c in iso2_values)
    query = f"""
    SELECT ?iso2 ?headOfStateLabel ?headOfGovernmentLabel WHERE {{
      VALUES ?iso2 {{ {values_clause} }}
      ?country wdt:P297 ?iso2.
      OPTIONAL {{ ?country wdt:P35 ?headOfState. }}
      OPTIONAL {{ ?country wdt:P6 ?headOfGovernment. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    resp = requests.post(SPARQL_ENDPOINT, data={"query": query}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rows: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        row = {
            "iso2": b.get("iso2", {}).get("value", ""),
            "headOfStateLabel": b.get("headOfStateLabel", {}).get("value", ""),
            "headOfGovernmentLabel": b.get("headOfGovernmentLabel", {}).get("value", ""),
        }
        rows.append(row)
    return rows


def fallback_query_by_label(name: str) -> Tuple[Set[str], Set[str]]:
    """Fallback for rare cases where ISO2 lookup fails (e.g., Kosovo label).

    Returns sets of heads of state and heads of government labels.
    """
    query = f"""
    SELECT ?headOfStateLabel ?headOfGovernmentLabel WHERE {{
      ?country rdfs:label "{name}"@en.
      OPTIONAL {{ ?country wdt:P35 ?headOfState. }}
      OPTIONAL {{ ?country wdt:P6 ?headOfGovernment. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    resp = requests.post(SPARQL_ENDPOINT, data={"query": query}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hos: Set[str] = set()
    hog: Set[str] = set()
    for b in data.get("results", {}).get("bindings", []):
        hos_label = b.get("headOfStateLabel", {}).get("value")
        hog_label = b.get("headOfGovernmentLabel", {}).get("value")
        if hos_label:
            hos.add(hos_label)
        if hog_label:
            hog.add(hog_label)
    return hos, hog


def build_leaders_index(countries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    iso2_list = [c["cca2"] for c in countries]
    # Batch in chunks to keep queries manageable
    batch_size = 90
    aggregated: Dict[str, Dict[str, Set[str]]] = {}

    for i in range(0, len(iso2_list), batch_size):
        batch = iso2_list[i : i + batch_size]
        rows = query_wikidata_heads_by_iso2(batch)
        # Be polite with the endpoint
        time.sleep(0.3)
        for r in rows:
            iso2 = r.get("iso2", "")
            if not iso2:
                continue
            hos = r.get("headOfStateLabel") or ""
            hog = r.get("headOfGovernmentLabel") or ""
            entry = aggregated.setdefault(iso2, {"hos": set(), "hog": set()})
            if hos:
                entry["hos"].add(hos)
            if hog:
                entry["hog"].add(hog)

    # Fallbacks for any countries that didn't return via ISO2
    for c in countries:
        iso2 = c["cca2"]
        if iso2 not in aggregated or (not aggregated[iso2]["hos"] and not aggregated[iso2]["hog"]):
            hos, hog = fallback_query_by_label(c["country"])
            if hos or hog:
                aggregated.setdefault(iso2, {"hos": set(), "hog": set()})
                aggregated[iso2]["hos"].update(hos)
                aggregated[iso2]["hog"].update(hog)
                time.sleep(0.2)

    # Build final rows
    results: List[Dict[str, str]] = []
    for c in countries:
        iso2 = c["cca2"]
        entry = aggregated.get(iso2, {"hos": set(), "hog": set()})
        hos_list = sorted(entry["hos"]) if entry["hos"] else []
        hog_list = sorted(entry["hog"]) if entry["hog"] else []
        results.append({
            "country": c["country"],
            "cca2": iso2,
            "head_of_state": "; ".join(hos_list) if hos_list else "",
            "head_of_government": "; ".join(hog_list) if hog_list else "",
        })
    return results


def save_outputs(rows: List[Dict[str, str]]) -> None:
    timestamp = int(time.time())
    # Deterministic file names for easy discovery
    json_path = "leaders_197.json"
    csv_path = "leaders_197.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": timestamp, "rows": rows}, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["country", "cca2", "head_of_state", "head_of_government"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    countries = load_197_country_set()
    rows = build_leaders_index(countries)
    save_outputs(rows)
    # Print a short summary
    completed = sum(1 for r in rows if r.get("head_of_state") or r.get("head_of_government"))
    print(f"Generated leaders for {completed}/{len(rows)} countries. Files: leaders_197.json, leaders_197.csv")


if __name__ == "__main__":
    main()


