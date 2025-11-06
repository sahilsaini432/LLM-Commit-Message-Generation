import json
import json
import os
import openai
import re
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
)
import nltk

def calculate_meteor(sentence1, sentence2):
    """
    Calculate METEOR score between two sentences
    """
    # Convert both sentences to word frequency vectors
    vectorizer = CountVectorizer().fit([sentence1, sentence2])
    sentence1_vector = vectorizer.transform([sentence1])
    sentence2_vector = vectorizer.transform([sentence2])

    # Calculate cosine similarity between the two vectors
    similarity = cosine_similarity(sentence1_vector, sentence2_vector)[0][0]

    # Calculate score based on METEOR formula
    score = 2 * similarity * len(sentence1) * len(sentence2) / (len(sentence1) + len(sentence2))
    return score


def calculate_bleu(reference, translation):
    """
    Calculate BLEU score
    """
    bleu_score = sentence_bleu([reference], translation)
    return bleu_score


def calculate_rouge_l(reference, translation):
    """
    Calculate ROUGE-L score
    """
    rouge = Rouge()
    rouge_l_score = rouge.get_scores(translation, reference, avg=True)["rouge-l"]
    return rouge_l_score


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def to_Underline(x):
    """Convert to space-separated naming"""
    return re.sub("(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])", " \g<0>", x).lower()


def get_tokens(text):
    tokens = nltk.word_tokenize(text)
    if len(tokens) > 1024:
        return " ".join(tokens[:1024])
    else:
        return " ".join(tokens)


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


def main():
    # Read JSONL file
    with open(file, "r") as f:
        lines = f.readlines()
    with open(nlp_file_path, "w") as f:
        f.write("")
    # Process each line of JSON data
    new_lines = []
    for line in lines:
        data = json.loads(line)
        # Check if both msg and msgGPT are the string '0'
        if (
            isinstance(data["msg"], str)
            and data["msg"] == "0"
            and isinstance(data["msgGPT"], str)
            and data["msgGPT"] == "0"
        ):
            # If so, delete this line of data
            continue
        new_lines.append(line)

    # Write processed JSON data back to file
    with open(file, "w") as f:
        f.writelines(new_lines)

    # Open JSONL file and read data
    with open(file, "r") as f:
        json_data = f.readlines()

    for item in json_data:
        # Parse JSON data
        data = json.loads(item)
        diff_id = data["diff_id"]
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
        msgGPT = data["msgGPT0"]
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

        bleu_score = calculate_bleu(msg, msgGPT)
        rouge_l_score = calculate_rouge_l(msg, msgGPT)
        meteor_score = calculate_meteor(msg, msgGPT)

        # Add diff and msg, score to list
        data = {
            "diff_id": diff_id,
            "msg": f"{msg}",
            "msgGPT": f"{msgGPT}",
            "METEOR Score": f"{meteor_score}",
            "BLEU Score": f"{bleu_score}",
            "ROUGE-L Score": f"{rouge_l_score['f']}",
        }
        with open(nlp_file_path, "a") as f:
            json.dump(data, f)
            f.write("\n")

    # Initialize variables to save total scores
    total_meteor_score = 0
    total_bleu_score = 0
    total_rouge_l_score = 0

    # File handle
    def count_jsonl_lines(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
        return len(lines)

    # Put your JSONL file path here
    x = count_jsonl_lines(nlp_file_path)

    with open(nlp_file_path, "r") as f:
        # Read file line by line
        for line in f:
            # Decode each line to get a json object
            json_obj = json.loads(line)

            # Get scores from json object
            meteor_score = float(json_obj.get("METEOR Score", 0))
            bleu_score = float(json_obj.get("BLEU Score", 0))
            rouge_l_score = float(json_obj.get("ROUGE-L Score", 0))

            # Add to total scores
            total_meteor_score += meteor_score
            total_bleu_score += bleu_score
            total_rouge_l_score += rouge_l_score

        # Calculate average scores
    average_meteor_score = total_meteor_score / x
    average_bleu_score = total_bleu_score / x
    average_rouge_l_score = total_rouge_l_score / x

    # Output average scores
    print(f"Average METEOR Score: {average_meteor_score}")
    print(f"Average BLEU Score: {average_bleu_score}")
    print(f"Average ROUGE-L Score: {average_rouge_l_score}")

if __name__ == "__main__":
    main()