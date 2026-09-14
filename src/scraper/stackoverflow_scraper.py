#!/usr/bin/env python3
"""
Stack Exchange API scraper.

Fetches questions (score >= MIN_SCORE, with accepted answer) and their
accepted answers for each target tag. Saves to data/raw/stackoverflow/.

Quota:
  - Without API key: 300 requests/day
  - With STACKEXCHANGE_KEY in .env: 10,000 requests/day

Usage:
    python stackoverflow_scraper.py
"""

import os
import json
import time
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAGE_SIZE = 100

# Per-tag config: (min_score)
TAGS = {
    "fastapi":  5,
    "supabase": 5,
    "pydantic": 5,
    "appwrite": 3,
}
SE_API_BASE = "https://api.stackexchange.com/2.3"
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "stackoverflow"

# Optional API key — dramatically raises quota from 300 to 10,000 req/day
API_KEY = os.getenv("STACKEXCHANGE_KEY", "")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class StackExchangeClient:
    def __init__(self, key: str = ""):
        self.key = key
        self.session = requests.Session()
        self.session.headers.update({"Accept-Encoding": "gzip"})
        self._quota_remaining = None

    def get(self, endpoint: str, params: dict = None) -> dict:
        params = dict(params or {})
        params["site"] = "stackoverflow"
        if self.key:
            params["key"] = self.key

        url = f"{SE_API_BASE}{endpoint}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        self._quota_remaining = data.get("quota_remaining")
        if self._quota_remaining is not None and self._quota_remaining % 100 == 0:
            log.info(f"SE quota remaining: {self._quota_remaining}")

        # Respect backoff directive
        if data.get("backoff"):
            wait = data["backoff"] + 1
            log.warning(f"Backoff requested: sleeping {wait}s")
            time.sleep(wait)

        return data

    def get_all_pages(self, endpoint: str, params: dict = None, max_pages: int = 100) -> list:
        """Paginate through all results for an endpoint."""
        params = dict(params or {})
        params["pagesize"] = PAGE_SIZE
        results = []

        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self.get(endpoint, params)
            items = data.get("items", [])
            results.extend(items)
            log.info(f"  page {page}: {len(items)} items (total so far: {len(results)})")

            if not data.get("has_more", False):
                break

            time.sleep(0.5)  # gentle throttle between pages

        return results


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def fetch_questions_for_tag(client: StackExchangeClient, tag: str, min_score: int = 5) -> list[dict]:
    """Fetch all questions tagged `tag` with score >= min_score that have an accepted answer."""
    log.info(f"Fetching questions for tag: [{tag}] (min_score={min_score})")
    questions = client.get_all_pages(
        "/questions",
        params={
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "min": min_score,
            "filter": "withbody",  # includes body field
        },
    )
    # Keep only those with an accepted answer
    with_answer = [q for q in questions if q.get("accepted_answer_id")]
    log.info(f"  [{tag}]: {len(questions)} questions fetched, {len(with_answer)} have accepted answers.")
    return with_answer


def fetch_answers_by_ids(client: StackExchangeClient, answer_ids: list[int]) -> dict[int, dict]:
    """Fetch answers by ID in batches of 100. Returns {answer_id: answer_obj}."""
    answers = {}
    batch_size = 100
    for i in tqdm(range(0, len(answer_ids), batch_size), desc="Fetching answers"):
        batch = answer_ids[i : i + batch_size]
        ids_str = ";".join(str(aid) for aid in batch)
        data = client.get(f"/answers/{ids_str}", params={"filter": "withbody", "pagesize": batch_size})
        for ans in data.get("items", []):
            answers[ans["answer_id"]] = ans
        time.sleep(0.3)
    return answers


def scrape_tag(client: StackExchangeClient, tag: str, min_score: int = 5) -> list[dict]:
    """Return list of {question, accepted_answer} artifact dicts for a tag."""
    questions = fetch_questions_for_tag(client, tag, min_score=min_score)
    if not questions:
        return []

    answer_ids = [q["accepted_answer_id"] for q in questions]
    answers = fetch_answers_by_ids(client, answer_ids)

    artifacts = []
    for q in questions:
        ans = answers.get(q["accepted_answer_id"])
        if ans is None:
            continue  # answer fetch missed this one

        artifacts.append({
            "source_type": "stackoverflow_qa",
            "tag": tag,
            "question_id": q["question_id"],
            "title": q.get("title", ""),
            "question_body": q.get("body", ""),
            "question_score": q.get("score", 0),
            "answer_id": ans["answer_id"],
            "answer_body": ans.get("body", ""),
            "answer_score": ans.get("score", 0),
            "is_accepted": True,
            "tags": q.get("tags", []),
            "url": q.get("link", ""),
            "created_at": q.get("creation_date", ""),
        })

    log.info(f"  [{tag}]: {len(artifacts)} Q+A pairs assembled.")
    return artifacts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = StackExchangeClient(key=API_KEY)

    if API_KEY:
        log.info("Using Stack Exchange API key (10,000 req/day quota).")
    else:
        log.warning("No STACKEXCHANGE_KEY found — running with 300 req/day quota.")

    all_artifacts = []
    tag_counts = {}

    for tag, min_score in TAGS.items():
        out_file = OUTPUT_DIR / f"{tag}_raw.json"
        if out_file.exists():
            log.info(f"[SKIP] {tag} — {out_file} already exists.")
            with open(out_file) as f:
                existing = json.load(f)
            all_artifacts.extend(existing)
            tag_counts[tag] = len(existing)
            continue

        artifacts = scrape_tag(client, tag, min_score=min_score)
        tag_counts[tag] = len(artifacts)
        all_artifacts.extend(artifacts)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(artifacts)} artifacts → {out_file}")

        time.sleep(1)  # pause between tags

    # Merged output
    merged_file = OUTPUT_DIR / "stackoverflow_all_raw.json"
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(all_artifacts, f, ensure_ascii=False, indent=2)

    log.info("\n=== Stack Overflow scraping complete ===")
    for tag, count in tag_counts.items():
        log.info(f"  [{tag}]: {count} Q+A pairs")
    log.info(f"  Total: {len(all_artifacts)}")
    log.info(f"  Merged → {merged_file}")


if __name__ == "__main__":
    main()
