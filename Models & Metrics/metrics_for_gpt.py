import json
import json
import ssl
import statistics
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import tqdm
import nltk
from nltk.translate.meteor_score import meteor_score

#  Bypass SSL verification for NLTK downloads on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet")


def calculate_meteor(sentence1, sentence2):
    """
    Calculate a custom METEOR-like score (cosine similarity with F1 weighting), normalized to [0, 1].
    """
    # Tokenize for better handling
    words1 = sentence1.split()
    words2 = sentence2.split()

    if not words1 or not words2:
        return 0.0

    # Use CountVectorizer on tokenized input for consistency
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"\b\w+\b")
    vec1 = vectorizer.fit_transform([" ".join(words1)])
    vec2 = vectorizer.transform([" ".join(words2)])

    similarity = cosine_similarity(vec1, vec2)[0][0]

    # Word lengths
    len1 = len(words1)
    len2 = len(words2)

    # F1-like score
    if len1 + len2 == 0:
        score = 0.0
    else:
        score = 2 * similarity * len1 * len2 / (len1 + len2)

    # Normalize to [0, 1] by dividing by the maximum possible score (when sim=1 and len1=len2)
    max_score = max(len1, len2)  # Approximation; exact max is len1 when len1==len2
    return score / max_score if max_score > 0 else 0.0


def calculate_bleu(reference, translation):
    """
    Calculate BLEU score
    """
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference], translation, smoothing_function=smoothing)
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
    return re.sub(r"(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])", r" \g<0>", x).lower()


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--gptFile", required=True, help="Input JSONL file path")
    args = parser.parse_args()

    # Open JSONL file and read data
    with open(args.gptFile, "r", encoding="utf8") as f:
        json_data = f.readlines()

    # Metrics Calculation
    all_meteor_scores = []
    all_bleu_scores = []
    all_rouge_l_scores = []

    for item in tqdm.tqdm(json_data):
        # Parse JSON data
        data = json.loads(item)

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

        m_score = calculate_meteor(msg, msgGPT)
        all_meteor_scores.append(m_score)

        b_score = calculate_bleu(msg.split(), msgGPT.split())
        all_bleu_scores.append(b_score)

        r_score = calculate_rouge_l(msg, msgGPT)
        all_rouge_l_scores.append(r_score["f"])

    # Output average scores
    print(f"\nAverage METEOR Score: {statistics.mean(all_meteor_scores)}")
    print(f"Average BLEU Score: {statistics.mean(all_bleu_scores)}")
    print(f"Average ROUGE-L Score: {statistics.mean(all_rouge_l_scores)}")

    # Output average scores
    print(f"\nMedian METEOR Score: {statistics.median(all_meteor_scores)}")
    print(f"Median BLEU Score: {statistics.median(all_bleu_scores)}")
    print(f"Median ROUGE-L Score: {statistics.median(all_rouge_l_scores)}")


if __name__ == "__main__":
    main()
