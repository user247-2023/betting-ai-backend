"""
github_commit.py — Persist files back to GitHub from the running backend.

Railway's disk is ephemeral, so anything the weekly job regenerates (the refreshed
football.db and the re-fitted dc_params.json) must be committed back to the repo to
survive the next redeploy. This module commits one OR MORE files in a SINGLE commit
using GitHub's Git Data API, so a weekly run triggers only one Railway redeploy.

Config (Railway env vars):
    GITHUB_COMMIT_TOKEN   fine-grained PAT, scoped to this repo, Contents: Read+Write
    GITHUB_REPO           "owner/repo"  (default user247-2023/betting-ai-backend)
    GITHUB_BRANCH         branch name   (default main)

Nothing here ever runs from untrusted input — it is only invoked by the key-guarded
/api/admin/weekly endpoint.
"""
import base64
import os
from typing import Dict

import requests

_API = "https://api.github.com"


def _cfg():
    token = os.getenv("GITHUB_COMMIT_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "user247-2023/betting-ai-backend").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()
    if not token:
        raise RuntimeError("GITHUB_COMMIT_TOKEN not set in Railway variables")
    if "/" not in repo:
        raise RuntimeError(f"GITHUB_REPO must be 'owner/repo', got '{repo}'")
    return token, repo, branch


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _check(r, what):
    if r.status_code >= 300:
        # surface GitHub's own message but never echo the token
        msg = ""
        try:
            msg = r.json().get("message", "")
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"GitHub {what} failed ({r.status_code}): {msg}")
    return r.json()


def commit_files(files: Dict[str, str], message: str, timeout: int = 60) -> Dict:
    """Commit multiple local files to the repo in ONE commit.

    files: { path_in_repo : local_filesystem_path }
    Returns a small summary including the new commit SHA and URL.
    """
    token, repo, branch = _cfg()
    h = _headers(token)
    base = f"{_API}/repos/{repo}/git"

    # 1) current branch head -> latest commit sha
    ref = _check(requests.get(f"{base}/ref/heads/{branch}", headers=h, timeout=timeout),
                 "get ref")
    head_sha = ref["object"]["sha"]

    # 2) the commit -> its tree sha
    head_commit = _check(requests.get(f"{base}/commits/{head_sha}", headers=h, timeout=timeout),
                         "get commit")
    base_tree = head_commit["tree"]["sha"]

    # 3) upload each file as a blob (base64), collect tree entries
    tree_entries = []
    for repo_path, local_path in files.items():
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        blob = _check(requests.post(f"{base}/blobs", headers=h, timeout=timeout,
                                    json={"content": content_b64, "encoding": "base64"}),
                      f"create blob {repo_path}")
        tree_entries.append({"path": repo_path, "mode": "100644",
                             "type": "blob", "sha": blob["sha"]})

    # 4) new tree based on the current one
    tree = _check(requests.post(f"{base}/trees", headers=h, timeout=timeout,
                                json={"base_tree": base_tree, "tree": tree_entries}),
                  "create tree")

    # 5) new commit pointing at the new tree
    commit = _check(requests.post(f"{base}/commits", headers=h, timeout=timeout,
                                  json={"message": message, "tree": tree["sha"],
                                        "parents": [head_sha]}),
                    "create commit")

    # 6) move the branch to the new commit
    _check(requests.patch(f"{base}/refs/heads/{branch}", headers=h, timeout=timeout,
                          json={"sha": commit["sha"], "force": False}),
           "update ref")

    return {"committed": list(files.keys()), "commit_sha": commit["sha"][:7],
            "commit_url": commit.get("html_url", ""), "repo": repo, "branch": branch}
