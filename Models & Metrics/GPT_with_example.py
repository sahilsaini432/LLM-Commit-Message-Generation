from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import json
import re
import numpy as np
import nltk
import traceback
from openai import OpenAI
import time

lan = "java.jsonl"
output_filenames = {
    1: "javaadd16k_gpt_1_noselect.jsonl",
    3: "javaadd16k_gpt_3_noselect.jsonl",
    5: "javaadd16k_gpt_5_noselect.jsonl",
    10: "javaadd16k_gpt_10_noselect.jsonl",
}
best_file = "javabest_no_select.jsonl"
key_list = []


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def to_Underline(x):
    """Convert to underscore naming"""
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
    if len(tokens) > 600:
        return " ".join(tokens[:600])
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


# Open JSONL file and read data
with open(lan, "r", encoding="utf8") as f:
    json_data = f.readlines()
data = {
    "diff_id": 0,
    "msg": f"0",
    "msgGPT": f"0",
    "METEOR Score": f"0",
    "BLEU Score": f"0",
    "ROUGE-L Score": f"0",
}

# Iterate through JSON data, extract and store diff and msg
num = 0
temp = 0
for item in json_data:
    attempts = 0
    while attempts < 5:
        key_list = key_list
        key = key_list[temp]
        client = OpenAI(
            api_key=key,
        )
        temp += 1
        if temp == 20:
            temp = 0
        # Parse JSON data
        data = json.loads(item)

        # Extract diff and msg
        diff_id = data["diff_id"]
        diff = data["diff"]
        result = remove_between_identifiers(diff, "mmm a", "<nl>")
        diff = get_tokens(remove_between_identifiers(result, "ppp b", "<nl>"))
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
        # Example usage:
        # Extract corresponding best_diff and best_msg
        best_diffs_msgs = []
        with open(best_file, "r", encoding="utf8") as file:
            for line in file:
                best_data = json.loads(line)
                if best_data["diff_id"] == diff_id:
                    for i in range(1, 11):
                        diff_key = f"best_diff{i}"
                        msg_key = f"best_msg{i}"
                        if diff_key in best_data and msg_key in best_data:
                            # Apply same preprocessing steps
                            result_b = remove_between_identifiers(best_data[diff_key], "mmm a", "<nl>")
                            best_diff = get_tokens(remove_between_identifiers(result_b, "ppp b", "<nl>"))
                            best_msg = best_data[msg_key]
                            best_diffs_msgs.append((best_diff, best_msg))
                    break

        if num < 4:
            num += 1
        elif num >= 4:
            num = 0
        try:
            for num_examples in [1, 3, 5, 10]:
                if len(best_diffs_msgs) >= num_examples:
                    # 构建prompt
                    prompt = ""
                    for best_diff, best_msg in best_diffs_msgs[:num_examples]:
                        prompt += f"{best_diff}\nPlease write a commit message for the above code change.\n{best_msg}\n\n"
                    prompt += f"{diff}\nPlease write a commit message for the above code change.\n"

                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo-16k",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a programmer who makes the above code changes.",
                            },
                            {"role": "user", "content": f"""{prompt}"""},
                        ],
                        max_tokens=50,
                        temperature=0.8,
                        n=5,
                        top_p=0.95,
                    )
                    num_answers = 5
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
                    print(msgGPTs)

                    # Add diff and msg, score to list
                    data = {"diff_id": diff_id, "msg": f"{msg}"}
                    for i in range(5):
                        data[f"msgGPT{i}"] = f"{msgGPTs[i]}"
                        output_data = {"diff_id": diff_id, "msg": msg}

                        for i in range(5):
                            output_data[f"msgGPT{i}"] = msgGPTs[i]
                    with open(output_filenames[num_examples], "a", encoding="utf8") as f:
                        json.dump(output_data, f)
                        f.write("\n")
            break
        except:
            traceback.print_exc()
            time.sleep(1)
            attempts += 1
            if attempts == 5:
                print(f"{item} has been retried 3 times and still failed.")
                # Here you can choose to log failed items or perform other error handling
                # ...
                break  # After 3 retries, break out of inner loop and process next item
