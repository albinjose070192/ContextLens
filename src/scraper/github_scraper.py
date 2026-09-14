"""
GitHub repository scraper.
Collects READMEs, CHANGELOGs, issues+comments, and Python docstrings
from a target repository and saves raw artifacts as JSON.
"""

import os
import ast
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN not set — add it to .env or pass directly.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def get(self, path: str, params: dict = None) -> dict | list:
        url = f"{self.BASE}{path}"
        resp = self.session.get(url, params=params)
        self._check_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    def get_paginated(self, path: str, params: dict = None, max_pages: int = 50) -> list:
        params = dict(params or {})
        params.setdefault("per_page", 100)
        results = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            batch = self.get(path, params)
            if not batch:
                break
            results.extend(batch)
            if len(batch) < params["per_page"]:
                break
        return results

    def _check_rate_limit(self, resp: requests.Response):
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 9999))
        if remaining < 5:
            reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(0, reset_at - time.time()) + 5
            log.warning(f"Rate limit low ({remaining} remaining). Sleeping {wait:.0f}s.")
            time.sleep(wait)
        elif remaining % 200 == 0:
            log.info(f"Rate limit remaining: {remaining}")


# ---------------------------------------------------------------------------
# Artifact collectors
# ---------------------------------------------------------------------------

def fetch_readme(client: GitHubClient, owner: str, repo: str) -> dict:
    log.info("Fetching README...")
    data = client.get(f"/repos/{owner}/{repo}/readme")
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return {
        "source_type": "readme",
        "repo": f"{owner}/{repo}",
        "path": data["path"],
        "content": content,
        "url": data["html_url"],
    }


def fetch_markdown_docs(
    client: GitHubClient,
    owner: str,
    repo: str,
    docs_prefixes: list[str] = None,
) -> list[dict]:
    """
    Fetch all .md / .mdx files under the given path prefixes plus root-level .md files.

    docs_prefixes: list of path prefixes to include (e.g. ["docs/en/"] for FastAPI,
                   ["docs/", "apps/docs/content/"] for Supabase).
                   Defaults to ["docs/en/"] for backward compatibility.
    """
    if docs_prefixes is None:
        docs_prefixes = ["docs/en/"]

    log.info(f"Fetching file tree for markdown docs (prefixes: {docs_prefixes})...")
    tree = client.get(f"/repos/{owner}/{repo}/git/trees/HEAD", {"recursive": "1"})
    md_files = [
        item for item in tree.get("tree", [])
        if item["type"] == "blob"
        and (item["path"].endswith(".md") or item["path"].endswith(".mdx"))
        and item["path"].lower() not in ("readme.md", "readme.mdx")
        and (
            any(item["path"].startswith(p) for p in docs_prefixes)
            or "/" not in item["path"]  # root-level .md files (CONTRIBUTING, CHANGELOG, etc.)
        )
    ]

    artifacts = []
    for item in tqdm(md_files, desc="Markdown docs"):
        try:
            raw = client.get(f"/repos/{owner}/{repo}/contents/{item['path']}")
            content = base64.b64decode(raw["content"]).decode("utf-8", errors="replace")
            source_type = "changelog" if "change" in item["path"].lower() else "doc"
            artifacts.append({
                "source_type": source_type,
                "repo": f"{owner}/{repo}",
                "path": item["path"],
                "content": content,
                "url": raw["html_url"],
            })
        except Exception as e:
            log.warning(f"Skipping {item['path']}: {e}")
        time.sleep(0.1)  # gentle throttle

    log.info(f"Fetched {len(artifacts)} markdown docs.")
    return artifacts


def fetch_issues(client: GitHubClient, owner: str, repo: str, max_issues: int = 500) -> list[dict]:
    """
    Fetch open + closed issues (not PRs) with their comments.
    Uses the Issues REST API with state=all; filters PRs by absence of 'pull_request' key.
    """
    log.info("Fetching issues via Issues API (state=all, filtering PRs)...")

    issues = []
    per_page = 100
    page = 1

    while len(issues) < max_issues:
        params = {"state": "all", "per_page": per_page, "page": page}
        try:
            batch = client.get(f"/repos/{owner}/{repo}/issues", params)
            if not batch:
                break
            # Filter out PRs — they have a 'pull_request' key
            only_issues = [i for i in batch if "pull_request" not in i]
            issues.extend(only_issues)
            log.info(f"  page={page}: {len(batch)} items, {len(only_issues)} issues (total: {len(issues)})")
            if len(batch) < per_page:
                break
            page += 1
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"Issues API error on page {page}: {e}")
            break

    issues = issues[:max_issues]
    log.info(f"Found {len(issues)} issues.")

    artifacts = []
    for issue in tqdm(issues[:max_issues], desc="Issues + comments"):
        body = issue.get("body") or ""
        comments = []

        if issue["comments"] > 0:
            try:
                raw_comments = client.get(
                    f"/repos/{owner}/{repo}/issues/{issue['number']}/comments",
                    params={"per_page": 100},
                )
                comments = [c["body"] for c in raw_comments if c.get("body")]
            except Exception as e:
                log.warning(f"Could not fetch comments for issue #{issue['number']}: {e}")

        artifacts.append({
            "source_type": "issue",
            "repo": f"{owner}/{repo}",
            "issue_number": issue["number"],
            "title": issue["title"],
            "state": issue["state"],
            "labels": [l["name"] for l in issue.get("labels", [])],
            "body": body,
            "comments": comments,
            "url": issue["html_url"],
            "created_at": issue["created_at"],
        })
        time.sleep(0.05)

    log.info(f"Fetched {len(artifacts)} issues with comments.")
    return artifacts


def fetch_python_docstrings(client: GitHubClient, owner: str, repo: str) -> list[dict]:
    """
    Walk the file tree for .py files, download each, and extract
    module-level, class-level, and function-level docstrings via AST.
    Only files under the main package directory are targeted.
    """
    log.info("Fetching file tree for Python sources...")
    tree = client.get(f"/repos/{owner}/{repo}/git/trees/HEAD", {"recursive": "1"})

    # Target: .py files inside the main package (heuristic: first-level dir matching repo name or 'src/')
    py_files = [
        item for item in tree.get("tree", [])
        if item["type"] == "blob"
        and item["path"].endswith(".py")
        and not item["path"].startswith("tests/")
        and not item["path"].startswith("test_")
    ]

    artifacts = []
    for item in tqdm(py_files, desc="Python docstrings"):
        try:
            raw = client.get(f"/repos/{owner}/{repo}/contents/{item['path']}")
            source = base64.b64decode(raw["content"]).decode("utf-8", errors="replace")
            docstrings = _extract_docstrings(source, item["path"])
            for ds in docstrings:
                ds.update({
                    "source_type": "docstring",
                    "repo": f"{owner}/{repo}",
                    "path": item["path"],
                    "url": raw["html_url"],
                })
                artifacts.append(ds)
        except Exception as e:
            log.warning(f"Skipping {item['path']}: {e}")
        time.sleep(0.1)

    log.info(f"Extracted {len(artifacts)} docstrings from {len(py_files)} Python files.")
    return artifacts


def _extract_docstrings(source: str, path: str) -> list[dict]:
    """Parse a Python file and return all non-trivial docstrings with their context."""
    results = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if not docstring or len(docstring.strip()) < 30:
                continue

            kind = type(node).__name__.lower().replace("def", "").replace("async", "").strip()
            name = getattr(node, "name", "<module>")
            results.append({
                "node_type": kind if kind else "module",
                "name": name,
                "docstring": docstring,
                "lineno": getattr(node, "lineno", 0),
            })

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scrape_repo(
    owner: str,
    repo: str,
    output_dir: Path,
    max_issues: int = 500,
    docs_prefixes: list[str] = None,
    fetch_docstrings: bool = True,
):
    client = GitHubClient()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_artifacts = []

    # 1. README
    readme = fetch_readme(client, owner, repo)
    all_artifacts.append(readme)

    # 2. Markdown / MDX docs + CHANGELOG
    md_docs = fetch_markdown_docs(client, owner, repo, docs_prefixes=docs_prefixes)
    all_artifacts.extend(md_docs)

    # 3. Issues + comments
    issues = fetch_issues(client, owner, repo, max_issues=max_issues)
    all_artifacts.extend(issues)

    # 4. Python docstrings (skip for non-Python repos)
    if fetch_docstrings:
        docstrings = fetch_python_docstrings(client, owner, repo)
        all_artifacts.extend(docstrings)

    # Save
    out_file = output_dir / f"{owner}_{repo}_raw.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_artifacts, f, ensure_ascii=False, indent=2)

    log.info(f"Saved {len(all_artifacts)} artifacts to {out_file}")

    # Summary
    from collections import Counter
    counts = Counter(a["source_type"] for a in all_artifacts)
    log.info(f"Breakdown: {dict(counts)}")

    return all_artifacts


if __name__ == "__main__":
    scrape_repo(
        owner="tiangolo",
        repo="fastapi",
        output_dir=Path(__file__).parents[2] / "data" / "raw" / "github",
        max_issues=500,
        docs_prefixes=["docs/en/"],
        fetch_docstrings=True,
    )
