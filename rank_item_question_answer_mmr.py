import os
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from util import preprocess_text


from sklearn.metrics.pairwise import cosine_similarity

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input directory")
    parser.add_argument("-ld", "--lambda_score", type=float, default=0.5, help="Lambda score for MRR scoring function")
    return parser.parse_args()

args = parse_arguments()

def mmr_rank(question, answers, lambda_score=0.5):
    ranked_answers, scores = [], []
    # build similarity matrix
    corpus = []
    corpus.append(preprocess_text(question))
    for answer in answers:
        corpus.append(preprocess_text(answer))
    tfidf = TfidfVectorizer().fit_transform(corpus)
    similarity = cosine_similarity(tfidf)
    unranked_answer_ids = [i+1 for i in range(len(answers))]
    selected_answer_ids = []
    while len(unranked_answer_ids) > 0:
        score = -1e9
        selected_answer_idx = None
        for aid in unranked_answer_ids:
            mmr_score = lambda_score * similarity[0, aid] 
            if len(selected_answer_ids) > 0:
                mmr_score -= (1 - lambda_score) * similarity[aid, selected_answer_ids].sum()
            if mmr_score > score:
                score = mmr_score
                selected_answer_idx = aid
        if selected_answer_idx is None:
            selected_answer_idx = unranked_answer_ids[0]
        unranked_answer_ids.remove(selected_answer_idx)
        selected_answer_ids.append(selected_answer_idx)
        scores.append(mmr_score)
        ranked_answers.append(answers[selected_answer_idx-1])
    return ranked_answers, scores, selected_answer_ids

qas = []
with open(os.path.join(args.input, 'qa.txt'), 'r') as f:
    for l in tqdm(f, desc="Processing"):
        toks = l.split('\t\t')
        new_line = [toks[0]]
        for question_answers in toks[1:]:
            question_answers_toks = question_answers.split('\t')
            question_answer_texts, voting_scores = question_answers_toks[::2], question_answers_toks[1::2]
            question = question_answer_texts[0]
            answers = question_answer_texts[1:]
            if len(answers) <= 1:
                new_line.append(question_answers)
            else:
                ranked_answers, scores, ranked_answer_ids = mmr_rank(question, answers, args.lambda_score)
                answer_voting_scores = [voting_scores[idx] for idx in ranked_answer_ids]
0                new_line.append(
                    '\t'.join(
                        [question, voting_scores[0]] 
                        + sum([[t, s] for t, s in zip(ranked_answers, answer_voting_scores)], [])
                    )
                )
        qas.append('\t\t'.join(new_line))

with open(os.path.join(args.input, 'qa-ranked_answers_mmr_ld_{}.txt'.format(args.lambda_score)), 'w') as f:
    for l in tqdm(qas, desc="exporting"):
        f.write('{}\n'.format(l))


