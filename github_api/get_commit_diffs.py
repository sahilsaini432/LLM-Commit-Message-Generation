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

    total_commits = []

    page = 1
    per_page = 100  # Maximum allowed by GitHub API

    print(f"🔍 Fetching commits from {owner}/{repo}...")

    #  Get Commits
    while True:
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

                total_commits.extend(commits)
                print(f"📄 Fetched page {page} - {len(commits)} commits (Total: {len(total_commits)})")

                # If we got fewer commits than per_page, we're on the last page
                if len(commits) < per_page:
                    break

                page += 1
                # Small delay to be respectful to the API
            elif resp.status_code == 403:
                print("Received a 403 error.")
                # Check for specific rate limit headers
                if "retry-after" in resp.headers:
                    wait_time = int(resp.headers["retry-after"])
                    print(f"Waiting for {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                elif "X-RateLimit-Reset" in resp.headers:
                    reset_timestamp = int(resp.headers["X-RateLimit-Reset"])
                    wait_time = reset_timestamp - time.time()
                    if wait_time > 0:
                        print(f"Primary rate limit exceeded. Waiting for {wait_time} sec")
                        time.sleep(wait_time)
            else:
                print(f"❌ Failed to fetch commits: {resp.status_code} - {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None

    print(f"Total Commits fetched: {len(total_commits)}")

    # Save Train Commits
    path = Path(__file__).parent.parent

    # check if the output directory exists, if not create it
    os.makedirs(f"{path}/datasets/{repo}", exist_ok=True)

    with open(f"{path}/datasets/{repo}/un_clean_commits.jsonl", "w") as outfile:
        for commit in tqdm(total_commits):
            diff = get_commit_diffs(owner, repo, commit["sha"])
            outfile.write(json.dumps(diff) + "\n")


def format_text(message):
    diff = message
    # Add spaces around every special character/symbol
    special_chars = r'([+\-*/%=!&|^~?:;,.\[\]{}()\'"@#$`])'
    diff = re.sub(special_chars, r" \1 ", diff)

    # Normalize multiple spaces to single space
    diff = re.sub(r"\s+", " ", diff)

    # Clean up
    diff = diff.strip()

    # Replace newlines with <nl> first
    diff = diff.replace("\n", " <nl> ")
    return diff


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

            final_diff = ""
            for file in commit_data["files"]:
                if "patch" in file and file["patch"] is not None:
                    diff = file["patch"]
                    hunk_header_pattern = r"@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@.*?\n"
                    diff = re.sub(hunk_header_pattern, "", diff)

                    new_fileName = file["filename"]
                    old_fileName = file.get("previous_filename", None)
                    if old_fileName is None:
                        old_fileName = new_fileName

                    diff = f"mmm a / {old_fileName} <nl> ppp b / {new_fileName} <nl> {diff}"
                    diff = format_text(diff)

                    final_diff += diff + " "

            message = commit_data["commit"]["message"]
            message = message.split("\n\n<!--")[0]
            diff_line = {
                "msg": format_text(message),
                "sha": commit_data["sha"],
                "mod_diff": final_diff,
            }
            return diff_line

        elif resp.status_code == 403:
            print("Received a 403 error.")
            # Check for specific rate limit headers
            if "retry-after" in resp.headers:
                wait_time = int(resp.headers["retry-after"]) + 5
                print(f"Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
                return get_commit_diffs(owner, repo, sha)
            elif "X-RateLimit-Reset" in resp.headers:
                reset_timestamp = int(resp.headers["X-RateLimit-Reset"])
                wait_time = reset_timestamp - time.time()
                if wait_time > 0:
                    print(f"Primary rate limit exceeded. Waiting for {wait_time} sec")
                    time.sleep(wait_time)
                    return get_commit_diffs(owner, repo, sha)
        else:
            print(f"❌ Failed to fetch commit {sha}: {resp.status_code} - {resp.text}")
            return None
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
