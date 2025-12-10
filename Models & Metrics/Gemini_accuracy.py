import argparse
import os
import statistics
from anyio import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import re
import numpy as np
import google.generativeai as genai
import nltk
import traceback
from openai import OpenAI
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


def _sleep_after_call(diff_text: str, n: int = 1, max_tokens: int = 50, tpm: int = 30000):
    # Pace calls to stay under the Tokens-Per-Minute limit
    approx_in = _estimate_tokens(diff_text)
    approx_out = n * max_tokens  # upper bound
    sleep_s = ((approx_in + approx_out) / max(1, tpm)) * 60.0
    time.sleep(min(max(sleep_s, 0.5), 6.0))  # clamp between 0.5s and 6s


def _chat_with_retry(client, req_kwargs, max_retries: int = 6):
    # Retry on rate limits with exponential backoff
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**req_kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                wait = min(8.0, 0.5 * (2**attempt))
                print(f"Rate limited. Waiting {wait:.2f}s then retrying ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded max retries due to rate limits")


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataInput", required=True, help="original test file")
    parser.add_argument("-r", "--result", required=True, help="File with gpt generated messages")
    args = parser.parse_args()

    dataset = []
    # Open JSONL file and read data
    with open(args.dataInput, "r", encoding="utf8") as f:
        json_data = f.readlines()

    for item in tqdm.tqdm(json_data):
        # Parse JSON data
        dataset.append(json.loads(item))

    with open(args.result, "r", encoding="utf8") as f:
        results = f.readlines()

    final_dataset = []
    for item in tqdm.tqdm(results):
        # Process each item
        line = json.loads(item)
        gptMsg = line["msgGPT0"]

        # get line with same sha in dataset
        dataItem = next((item for item in dataset if item["sha"] == line["sha"]), None)
        dataItem["msgGPT0"] = gptMsg
        final_dataset.append(dataItem)

    results = []

    path = Path(__file__).parent.parent
    env_path = Path(f"{path}/.env")
    load_dotenv(dotenv_path=env_path)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Set up Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {"temperature": 0.8, "top_p": 0.95, "max_output_tokens": 50}
    # Define safety settings to be more permissive
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    for data in tqdm.tqdm(final_dataset):
        # Extract diff and msg
        diff = data["mod_diff"]

        # if no diff skip
        if diff is None or diff == "" or diff.strip() == "":
            continue

        result = remove_between_identifiers(diff, "mmm a", "<nl>")
        diff = get_tokens(remove_between_identifiers(result, "ppp b", "<nl>"))

        # gpt message
        gptMsg = data["msgGPT0"]
        prompt = f"""Code Diff - {diff}\n Commit Message - {gptMsg}\n How accurate is the commit message for the provided diff? \n Provide a single accuracy score from 1 to 10. A score of 1 indicates that the commit message is completely inaccurate and does not reflect the changes in the code diff at all. A score of 10 indicates that the commit message is perfectly accurate and fully describes all the changes made in the code diff. Only return the score as an integer between 1 and 10."""

        try:
            response = model.generate_content([prompt])
            generated_score = response.text.strip()
            results.append(int(re.findall(r"\d+", generated_score)[0]))
        except:
            traceback.print_exc()
            print(f"{item} has been retried 3 times and still failed.")

    # Output average scores
    print(f"\nAverage Accuracy Score: {statistics.mean(results)}")
    print(f"Median Accuracy Score: {statistics.median(results)}")


if __name__ == "__main__":
    main()
