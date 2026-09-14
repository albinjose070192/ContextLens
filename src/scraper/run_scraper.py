#!/usr/bin/env python3
"""
Run the GitHub scraper for all target repos.
FastAPI is already scraped; this handles Supabase, Pydantic, and Appwrite.
"""

import sys
import logging
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent))
from github_scraper import scrape_repo

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "github"

# ---------------------------------------------------------------------------
# Per-repo configuration
#
# docs_prefixes: path prefixes inside the repo tree that contain documentation.
#   - Supabase: monorepo — docs live under apps/docs/content/ (MDX) and docs/
#   - Pydantic: docs/ directory with Markdown (88 files)
#   - Appwrite: docs/ directory with Markdown (545 files)
#
# fetch_docstrings: True for Python-primary repos (FastAPI, Pydantic).
#   Supabase is TypeScript; Appwrite is PHP/TypeScript.
# ---------------------------------------------------------------------------

REPOS = [
    {
        "owner": "supabase",
        "repo": "supabase",
        "docs_prefixes": ["apps/docs/content/", "docs/"],
        "fetch_docstrings": False,
        "max_issues": 500,
    },
    {
        "owner": "pydantic",
        "repo": "pydantic",
        "docs_prefixes": ["docs/"],
        "fetch_docstrings": True,
        "max_issues": 500,
    },
    {
        "owner": "appwrite",
        "repo": "appwrite",
        "docs_prefixes": ["docs/"],
        "fetch_docstrings": False,
        "max_issues": 500,
    },
]


def main():
    for cfg in REPOS:
        label = f"{cfg['owner']}/{cfg['repo']}"
        out_file = OUTPUT_DIR / f"{cfg['owner']}_{cfg['repo']}_raw.json"

        if out_file.exists():
            print(f"\n[SKIP] {label} — {out_file.name} already exists. Delete to re-scrape.")
            continue

        print(f"\n{'='*60}")
        print(f"  Scraping {label}")
        print(f"{'='*60}")

        scrape_repo(
            owner=cfg["owner"],
            repo=cfg["repo"],
            output_dir=OUTPUT_DIR,
            max_issues=cfg["max_issues"],
            docs_prefixes=cfg["docs_prefixes"],
            fetch_docstrings=cfg["fetch_docstrings"],
        )

    print("\nAll repos done.")


if __name__ == "__main__":
    main()
