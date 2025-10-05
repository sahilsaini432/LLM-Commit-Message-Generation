from calendar import c
import datetime
import json
import requests
import time
from pathlib import Path
import argparse
from dotenv import load_dotenv
import re
import os
from tqdm import tqdm
from pprint import pprint as _print

GITHUB_TOKEN = None


class CommitData:
    def __init__(self, commit):
        self.sha = commit["sha"]
        self.Message = commit["commit"]["message"]
        self.diff = commit.get("diff", "")

    def to_dict(self):
        return {"sha": self.sha, "Author": self.Author, "Date": self.Date, "Files": self.Files}


def parse_github_url(url):
    """Parse GitHub URL to extract owner and repo."""
    patterns = [
        r"https://github\.com/([^/]+)/([^/]+)/?",
        r"git@github\.com:([^/]+)/([^/]+)\.git",
        r"([^/]+)/([^/]+)",  # Simple format: owner/repo
    ]

    for pattern in patterns:
        match = re.match(pattern, url.strip())
        if match:
            owner, repo = match.groups()
            if repo.endswith(".git"):
                repo = repo.rstrip(".git")

            return owner, repo

    raise ValueError(f"Invalid GitHub URL format: {url}")


def get_all_commits(owner, repo):
    global GITHUB_TOKEN
    """Get all commits for a GitHub repository with pagination."""
    headers = {}
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    train_commit_sha = []
    test_commit_sha = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API

    print(f"🔍 Fetching commits from {owner}/{repo}...")

    #  Get Train Commits
    while len(train_commit_sha) < 400:
        # Add pagination parameters
        url = f"{base_url}?page={page}&per_page={per_page}"

        try:
            resp = requests.get(url, headers=headers)

            if resp.status_code == 200:
                commits = resp.json()

                # If no commits returned, we've reached the end
                if not commits:
                    print("no commits found")
                    break

                for commit in commits:
                    if len(commit["parents"]) == 1:
                        train_commit_sha.append(commit["sha"])

            page += 1
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None

    print(f"Total Train Commits fetched: {len(train_commit_sha)}")

    path = Path(__file__).parent.parent
    with open(f"{path}/datasets/{repo}_train.jsonl", "w") as train_outfile:
        for sha in tqdm(train_commit_sha):
            diffs = get_commit_diffs(owner, repo, sha)
            for diff in diffs:
                train_outfile.write(json.dumps(diff) + "\n")


def get_commit_diffs(owner, repo, sha):
    global GITHUB_TOKEN
    """Get files diffs for a specific commit."""
    headers = {}
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            commit_data = resp.json()

            if "files" not in commit_data:
                print("No files changed in this commit.")
                return None

            diffs = []
            for file in commit_data["files"]:
                if "patch" in file and file["patch"] is not None:
                    diff = file["patch"]
                    hunk_header_pattern = r"@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@.*?\n"
                    mod_diff = re.sub(hunk_header_pattern, "", diff)

                    new_fileName = file["filename"]
                    old_fileName = file.get("previous_filename", None)
                    if old_fileName is None:
                        old_fileName = new_fileName

                    mod_diff = f"mmm a / {old_fileName} <nl> ppp b / {new_fileName} <nl>{mod_diff}"
                    # replace \n with <nl> in mod_diff
                    mod_diff = mod_diff.replace("\n", "<nl>")
                    diff_line = {
                        "message": commit_data["commit"]["message"],
                        "sha": commit_data["sha"],
                        "og_diff": diff,
                        "mod_diff": mod_diff,
                    }
                    diffs.append(diff_line)

            return diffs
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None


def main():
    path = Path(__file__).parent.parent
    env_path = Path(f"{path}/.env")
    load_dotenv(dotenv_path=env_path)

    global GITHUB_TOKEN
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("repository", help="GitHub repository in format 'owner/repo' or full GitHub URL")

    args = parser.parse_args()

    try:
        # Parse the repository URL
        owner, repo = parse_github_url(args.repository)
        get_all_commits(owner, repo)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
