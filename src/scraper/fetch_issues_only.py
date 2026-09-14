"""
One-off script: fetch issues for tiangolo/fastapi and merge into existing raw JSON.
Run from project root with tf-env active.
"""

import json
import logging
from pathlib import Path

from scraper.github_scraper import GitHubClient, fetch_issues

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = Path(__file__).parents[2] / "data" / "raw" / "github" / "tiangolo_fastapi_raw.json"
OWNER, REPO = "tiangolo", "fastapi"

def main():
    # Load existing artifacts
    with open(RAW_FILE, encoding="utf-8") as f:
        artifacts = json.load(f)

    # Drop any stale issue artifacts (in case of partial previous run)
    before = len(artifacts)
    artifacts = [a for a in artifacts if a["source_type"] != "issue"]
    log.info(f"Removed {before - len(artifacts)} stale issue artifacts.")

    # Fetch issues
    client = GitHubClient()
    issues = fetch_issues(client, OWNER, REPO, max_issues=500)
    artifacts.extend(issues)

    # Save
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    from collections import Counter
    counts = Counter(a["source_type"] for a in artifacts)
    log.info(f"Saved {len(artifacts)} total artifacts. Breakdown: {dict(counts)}")

if __name__ == "__main__":
    main()
