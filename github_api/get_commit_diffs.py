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
    print(f"Base URL: {base_url}")

    total_commits = []
    train_commit_sha = []
    test_commit_sha = []
    validation_commit_sha = []

    page = 1
    per_page = 100  # Maximum allowed by GitHub API

    print(f"🔍 Fetching commits from {owner}/{repo}...")

    #  Get Commits
    while len(total_commits) < 1500:
        # Add pagination parameters
        url = f"{base_url}?page={page}&per_page={per_page}"

        try:
            resp = requests.get(url, headers=headers)

            if resp.status_code == 200:
                commits = resp.json()
                print(f"Fetched {len(commits)} commits from page {page}")

                # If no commits returned, we've reached the end
                if not commits:
                    print("no commits found")
                    break

                for commit in commits:
                    if len(commit["parents"]) == 1:
                        total_commits.append(commit["sha"])

            page += 1
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None

    # Take 1050 commits for training, 300 for testing, and 150 for validation
    train_commit_sha = total_commits[:1050]
    validation_commit_sha = total_commits[1050:1350]
    test_commit_sha = total_commits[1350:1500]

    print(f"Total Commits fetched: {len(total_commits)}")
    print(f"Total Train Commits fetched: {len(train_commit_sha)}")
    print(f"Total Validate Commits fetched: {len(validation_commit_sha)}")
    print(f"Total Test Commits fetched: {len(test_commit_sha)}")

    # Save Train Commits
    path = Path(__file__).parent.parent
    with open(f"{path}/datasets/{repo}/train.jsonl", "w") as train_outfile:
        for sha in tqdm(train_commit_sha):
            diffs = get_commit_diffs(owner, repo, sha)
            for diff in diffs:
                train_outfile.write(json.dumps(diff) + "\n")

    # Save Validate Commits
    with open(f"{path}/datasets/{repo}/validation.jsonl", "w") as val_outfile:
        for sha in tqdm(validation_commit_sha):
            diffs = get_commit_diffs(owner, repo, sha)
            for diff in diffs:
                val_outfile.write(json.dumps(diff) + "\n")

    # Save Test Commits
    with open(f"{path}/datasets/{repo}/test.jsonl", "w") as test_outfile:
        for sha in tqdm(test_commit_sha):
            diffs = get_commit_diffs(owner, repo, sha)
            for diff in diffs:
                test_outfile.write(json.dumps(diff) + "\n")


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

            diff_line = {
                "message": commit_data["commit"]["message"],
                "sha": commit_data["sha"],
            }

            diff = ""
            mod_diff = ""
            for file in commit_data["files"]:
                if "patch" in file and file["patch"] is not None:
                    diff += f" \n{file["patch"]}"

            hunk_header_pattern = r"@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@.*?\n"
            mod_diff = re.sub(hunk_header_pattern, "", diff)
            mod_diff = f"mmm a / old_file <nl> ppp b / new_file <nl>{mod_diff}"
            # replace \n with <nl> in mod_diff
            mod_diff = mod_diff.replace("\n", "<nl>")

            diff_line["og_diff"] = diff
            diff_line["mod_diff"] = mod_diff
            return diff_line
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
