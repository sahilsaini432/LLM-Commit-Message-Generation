import json
import statistics
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import tqdm

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


def main():
    # Get file path from as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--truth", required=True, help="Ground Truth file path")
    parser.add_argument("-g", "--predict", required=True, help="Predicted file path")
    args = parser.parse_args()

    # Read .gold file (assuming one sentence per line)
    with open(args.truth, "r", encoding="utf8") as f:
        truth_lines = [line.strip() for line in f.readlines()]

    with open(args.predict, "r", encoding="utf8") as f:
        predicted_lines = [line.strip() for line in f.readlines()]

    # Pair up the lines
    data_pairs = list(zip(truth_lines, predicted_lines))

    # Metrics Calculation
    all_meteor_scores = []
    all_bleu_scores = []
    all_rouge_l_scores = []

    for truth, predict in tqdm.tqdm(data_pairs):
        print("Truth: ", truth)
        print("Predict: ", predict)
    #     # Process reference message
    #     words = msg.split()
    #     msg_list = []
    #     for word in words:
    #         if len(word) > 1:
    #             if is_camel_case(word):
    #                 msg_list.append(to_Underline(word))
    #             else:
    #                 msg_list.append(word)
    #         else:
    #             msg_list.append(word)
    #     msg = " ".join(msg_list)

    #     # Process generated message
    #     wordsGPT = msgGPT.split()
    #     msgGPT_list = []
    #     for wordGPT in wordsGPT:
    #         if len(wordGPT) > 1:
    #             if is_camel_case(wordGPT):
    #                 msgGPT_list.append(to_Underline(wordGPT))
    #             else:
    #                 msgGPT_list.append(wordGPT)
    #         else:
    #             msgGPT_list.append(wordGPT)
    #     msgGPT = " ".join(msgGPT_list)

    #     m_score = calculate_meteor(msg, msgGPT)
    #     all_meteor_scores.append(m_score)

    #     b_score = calculate_bleu(msg, msgGPT)
    #     all_bleu_scores.append(b_score)

    #     r_score = calculate_rouge_l(msg, msgGPT)
    #     all_rouge_l_scores.append(r_score["f"])

    # # Output average scores
    # print(f"Median METEOR Score: {statistics.median(all_meteor_scores)}")
    # print(f"Median BLEU Score: {statistics.median(all_bleu_scores)}")
    # print(f"Median ROUGE-L Score: {statistics.median(all_rouge_l_scores)}")


if __name__ == "__main__":
    main()
