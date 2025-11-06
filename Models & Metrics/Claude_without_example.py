import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import re
import numpy as np
import nltk
import traceback
from anthropic import Anthropic
import time
import tqdm
import ssl


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def to_Underline(x):
    """Convert to space-separated naming"""
    return re.sub("(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])", " \g<0>", x).lower()


def remove_between_identifiers(text, identifier_start, identifier_end):
    # Define regular expression pattern
    pattern = f"(?<={identifier_start}).*?(?={identifier_end})"

    # Use re.sub method to replace matched parts with empty string
    result = re.sub(pattern, "", text)
    if identifier_start == "mmm a":
        result = result.replace("mmm a<nl>", "")
    if identifier_start == "ppp b":
        result = result.replace("ppp b<nl>", "")
        result = result.replace("<nl>", "\n")
    result = result.replace(" . ", ".")
    result = result.replace("  ", ".")
    result = result.replace(" = ", "=")
    result = result.replace(" ; ", ";")
    result = result.replace(" (", "(")
    result = result.replace(") ", ")")
    return result


def get_tokens(text):
    tokens = nltk.word_tokenize(text)
    if len(tokens) > 1024:
        return " ".join(tokens[:1024])
    else:
        return " ".join(tokens)


def process_diff(diff):
    wordsGPT = diff.split()
    msgGPT_list = []
    for wordGPT in wordsGPT:
        if len(wordGPT) > 1:
            if is_camel_case(wordGPT):
                msgGPT_list.append(to_Underline(wordGPT))
            else:
                msgGPT_list.append(wordGPT)
        else:
            msgGPT_list.append(wordGPT)
    diff = " ".join(msgGPT_list)

    result = remove_between_identifiers(diff, "mmm a", "<nl>")
    diff = remove_between_identifiers(result, "ppp b", "<nl>")

    return get_tokens(diff)


def _estimate_tokens(text: str) -> int:
    # Rough estimate: ~4 characters per token
    return max(1, len(text) // 4)


def _sleep_after_call(diff_text: str, n: int = 30, max_tokens: int = 50, tpm: int = 40000):
    # Pace calls to stay under the Tokens-Per-Minute limit
    # Claude has higher default limits (40k TPM for most tiers)
    approx_in = _estimate_tokens(diff_text)
    approx_out = n * max_tokens  # upper bound
    sleep_s = ((approx_in + approx_out) / max(1, tpm)) * 60.0
    time.sleep(min(max(sleep_s, 0.5), 6.0))  # clamp between 0.5s and 6s


def _chat_with_retry(
    client, system_prompt: str, user_prompt: str, max_tokens: int = 50, max_retries: int = 6
):
    # Retry on rate limits with exponential backoff
    # Note: Claude API doesn't support n parameter, so we make multiple calls
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.8,
            )
            return response
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "overloaded" in msg:
                wait = min(8.0, 0.5 * (2**attempt))
                print(
                    f"Rate limited or overloaded. Waiting {wait:.2f}s then retrying ({attempt+1}/{max_retries})..."
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded max retries due to rate limits")


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=30, help="Number of commit messages to generate per diff"
    )
    args = parser.parse_args()

    # Open JSONL file and read data
    with open(args.input, "r", encoding="utf8") as f:
        json_data = f.readlines()

    results = []

    path = Path(__file__).parent.parent
    env_path = Path(f"{path}/.env")
    load_dotenv(dotenv_path=env_path)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    for item in tqdm.tqdm(json_data):
        # Parse JSON data
        data = json.loads(item)

        # Extract diff and msg
        sha = data["sha"]
        diff = data["mod_diff"]

        result = remove_between_identifiers(diff, "mmm a", "<nl>")
        diff = get_tokens(remove_between_identifiers(result, "ppp b", "<nl>"))

        # split message
        msg = data["msg"]
        words = msg.split()
        msg_list = []

        for word in words:
            if len(word) > 1:
                if is_camel_case(word):
                    msg_list.append(to_Underline(word))
                else:
                    msg_list.append(word)
            else:
                msg_list.append(word)
        msg = " ".join(msg_list)

        try:
            system_prompt = "You are a programmer who makes the above code changes."
            user_prompt = f"""{diff}\n Please write a commit message that contains only one simple sentence for the above code change.\n"""

            # Claude API doesn't support n parameter like OpenAI
            # So we need to make multiple individual calls
            msgGPTs = []
            for i in range(args.num_samples):
                response = _chat_with_retry(client, system_prompt, user_prompt, max_tokens=50)

                # Extract the text from Claude's response
                msgGPT = response.content[0].text

                # Process the message
                wordsGPT = msgGPT.split()
                msgGPT_list = []
                for wordGPT in wordsGPT:
                    if len(wordGPT) > 1:
                        if is_camel_case(wordGPT):
                            msgGPT_list.append(to_Underline(wordGPT))
                        else:
                            msgGPT_list.append(wordGPT)
                    else:
                        msgGPT_list.append(wordGPT)
                msgGPT = " ".join(msgGPT_list)
                msgGPTs.append(msgGPT)

                # Small delay between individual calls to avoid rate limits
                if i < args.num_samples - 1:  # Don't sleep after the last call
                    time.sleep(0.2)

            # Add diff and msg to results
            data = {"sha": sha, "msg": f"{msg}"}
            for i in range(args.num_samples):
                data[f"msgClaude{i}"] = f"{msgGPTs[i]}"

            results.append(data)

            # Wait to respect rate limits (TPM) before next request
            _sleep_after_call(diff, n=args.num_samples, max_tokens=50)

        except Exception as e:
            traceback.print_exc()
            print(f"{item} failed: {str(e)}")
            # Continue to next item instead of breaking
            continue

    # Write results to output file
    with open(f"{args.output}/claude_responses.jsonl", "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    print(f"\n✅ Successfully processed {len(results)} items")
    print(f"📁 Output saved to: {args.output}/claude_responses.jsonl")


if __name__ == "__main__":
    main()
