import argparse
import os
from anyio import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import re
import numpy as np
import nltk
import traceback
from openai import OpenAI
import time
import tqdm
import ssl

lan = "py.jsonl"
output_filename = "pygptnoexample.jsonl"


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


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    # Open JSONL file and read data
    with open(args.input, "r", encoding="utf8") as f:
        json_data = f.readlines()

    data = {
        "diff_id": 0,
        "msg": f"0",
        "msgGPT": f"0",
        "METEOR Score": f"0",
        "BLEU Score": f"0",
        "ROUGE-L Score": f"0",
    }
    results = []

    path = Path(__file__).parent.parent
    env_path = Path(f"{path}/.env")
    load_dotenv(dotenv_path=env_path)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    for item in tqdm.tqdm(json_data):
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

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
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a programmer who makes the above code changes.",
                    },
                    {
                        "role": "user",
                        "content": f"""{diff}\n Please write a commit message that contains only one simple sentence for the above code change.\n""",
                    },
                ],
                max_tokens=50,
                temperature=0.8,
                n=30,
                top_p=0.95,
            )
            num_answers = 30
            msgGPTs = []
            for i in range(num_answers):
                msgGPT = completion.choices[i].message.content
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

            # Add diff and msg, score to the list
            data = {"sha": sha, "msg": f"{msg}"}
            for i in range(30):
                data[f"msgGPT{i}"] = f"{msgGPTs[i]}"

            results.append(data)
        except:
            traceback.print_exc()
            print(f"{item} has been retried 3 times and still failed.")
            break

    with open(f"{args.output}/gpt.jsonl", "a") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    main()
