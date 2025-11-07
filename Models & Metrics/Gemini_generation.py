import argparse
import json
import os
from pathlib import Path
import re
import time
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize
import tqdm

import ssl

# Initialize NLTK components
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def to_Underline(x):
    """Convert to space-separated naming"""
    return re.sub(r"(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])", r" \g<0>", x).lower()


def get_tokens(text):
    tokens = word_tokenize(text)
    if len(tokens) > 1024:
        return " ".join(tokens[:1024])
    else:
        return " ".join(tokens)


def remove_between_identifiers(text, identifier_start, identifier_end):
    # Define regular expression pattern
    pattern = f"(?<={identifier_start}).*?(?={identifier_end})"

    # Use re.sub method to replace matched parts with an empty string
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


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

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

    # List available models
    # for m in genai.list_models():
    #     if "generateContent" in m.supported_generation_methods:
    #         print(m.name)

    # Read and process code diff data
    with open(args.input, "r") as f:
        json_data = f.readlines()

    results = []
    for item in tqdm.tqdm(json_data):
        data = json.loads(item)

        sha = data["sha"]
        diff = data["mod_diff"]

        # if no diff skip
        if diff is None or diff == "" or diff.strip() == "":
            continue

        # Apply preprocessing steps
        result = remove_between_identifiers(diff, "mmm a", "<nl>")
        diff = get_tokens(remove_between_identifiers(result, "ppp b", "<nl>"))

        # For loop that runs 30 times
        output = {}
        output["sha"] = sha
        output["msg"] = data["msg"]

        prompt = f"{diff}\nPlease write a commit message for the above code change.\n"

        generated_msg = None
        attempt = 0
        while attempt < 10 and generated_msg is None:
            try:
                response = model.generate_content([prompt])
                generated_msg = response.text.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(5)  # Simple wait mechanism to avoid retrying too quickly

        # If the message was successfully generated or max attempts reached, save the result to the corresponding file
        if generated_msg is not None:
            output[f"msgGPT0"] = generated_msg
        else:
            print(f"Could not generate a message for sha {sha}.")
        time.sleep(5)

        results.append(output)

    with open(f"{args.output}/gemini-2.5pro_response_on_test.jsonl", "w", encoding="utf8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
