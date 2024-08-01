import os
import cornac
from cornac.data import Reader
from cornac.eval_methods import StratifiedSplit
from cornac.data import ReviewAndItemQAModality
from cornac.data.text import BaseTokenizer
import pandas as pd
import numpy as np

import tensorflow as tf


physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input directory')
    parser.add_argument('-mu', '--min_user_freq', type=int, default=1)
    parser.add_argument('-mi', '--min_item_freq', type=int, default=1)
    parser.add_argument('-ct', '--cluster_threshold', type=float, default=0.8)
    parser.add_argument('-km', '--kmean', type=int, default=50)
    parser.add_argument('-k', '--n_factors', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=str, default='1,2')
    parser.add_argument('-s', '--model_selection', type=str, choices=['best', 'last'], default='best')
    parser.add_argument('-lr', '--learning_rates', type=str, default='0.001')
    return parser.parse_args()

args = parse_arguments()
feedback = Reader(min_user_freq=args.min_user_freq, min_item_freq=args.min_item_freq).read(os.path.join(args.input, "rating.txt"), fmt="UIRT", sep="\t")
reviews = Reader().read(os.path.join(args.input, "review.txt"), fmt="UIReview", sep="\t")

data_dir = "{}_crawled_kmeans_{}".format(args.input, args.kmean)

MAX_VOCAB = 4000
EMB_SIZE = 100
ID_EMB_SIZE = args.n_factors
N_FACTORS = args.n_factors
ATTENTION_SIZE = 8
BATCH_SIZE = 64
MAX_TEXT_LENGTH = 128
DROPOUT_RATE = 0.5
TEST_SIZE = 0.1
VAL_SIZE = 0.1
KERNEL_SIZES = [3]
N_FILTERS = 64
centroid_questions_file = open(os.path.join(data_dir, 'centroid_questions.txt'), 'r')
centroid_questions = centroid_questions_file.readlines()
cluster_label_in_order = []
cluster_count = []
with open(os.path.join(data_dir, 'cluster.count'), 'r') as f:
    for line in f:
        tokens = line.split(',')
        cluster_label_in_order.append(int(tokens[0]))
        cluster_count.append(int(tokens[1]))

pct = np.array(cluster_count) / sum(cluster_count)
max_keep_idx = 0
for i in range(len(pct)):
    if pct[:i+1].sum() >= args.cluster_threshold:
        max_keep_idx = i+1
        break

print('Max keep idx (coverage:{}): {}'.format(args.cluster_threshold, max_keep_idx))

item_question_clusters = {}
with open(os.path.join(data_dir, 'item_question_clusters.txt'), 'r') as f:
    for line in f:
        tokens = line.split(',')
        item_question_clusters[tokens[0]] = [int(cluster) for cluster in tokens[1:]]
qas = []
with open(os.path.join(data_dir, 'qa.txt'), 'r') as f:
    for line in f:
        tokens = line.split('\t\t')
        asin = tokens[0]
        qas.append((
            asin,
            [
                tuple([qtoken for q_inc, qtoken in enumerate(question.split('\t')) if q_inc % 2 == 0])
                for question, cluster_label in zip(tokens[1:], item_question_clusters.get(asin, [])) if cluster_label in cluster_label_in_order[:max_keep_idx]
            ]
        ))

mean_question = ' '.join(centroid_questions[max_keep_idx:]).replace('\n', ' ')

item_with_qas = [x[0] for x in qas]
item_without_qas = list(set([x[1] for x in reviews if x[1] not in item_with_qas]))
[x[1].append((mean_question,)) for x in qas]
qas = qas + [(x, [(mean_question,)])for x in item_without_qas]

review_and_item_qa_modality = ReviewAndItemQAModality(
    data=reviews,
    qa_data=qas,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=MAX_VOCAB,
)

eval_method = StratifiedSplit(
    data=feedback,
    group_by="item",
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    exclude_unknowns=True,
    review_and_item_qa_text=review_and_item_qa_modality,
    verbose=True,
    seed=123,
)

pretrained_word_embeddings = {}
with open(f"/data/shared/download/glove/glove.6B.{EMB_SIZE}d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        pretrained_word_embeddings[word] = coefs


models = [
        cornac.models.HRDRQ(
            name=f"HRDRQ_KMEAN_{args.kmean}_THRESHOLD_{args.cluster_threshold}_MINUSER_{args.min_user_freq}_MINITEM_{args.min_item_freq}_MAXNREVIEW_{max_num_review}_MAXNQA_{max_num_qa}_EMB_{EMB_SIZE}_IDEMB_{ID_EMB_SIZE}_F_{args.n_factors}_ATT_{ATTENTION_SIZE}_MAXTXTLEN_{MAX_TEXT_LENGTH}_BS_{BATCH_SIZE}_E_{max_iter}_OPTIMIZER_{optimizer}_LR_{learning_rate}_MODEL_SELECTION_{args.model_selection}_SEED_{seed}",
            embedding_size=EMB_SIZE,
            n_factors=args.n_factors,
            id_embedding_size=ID_EMB_SIZE,
            attention_size=ATTENTION_SIZE,
            kernel_sizes=KERNEL_SIZES,
            n_filters=N_FILTERS,
            max_text_length=MAX_TEXT_LENGTH,
            max_num_review=max_num_review,
            max_num_qa=max_num_qa,
            batch_size=BATCH_SIZE,
            dropout_rate=DROPOUT_RATE,
            optimizer=optimizer,
            learning_rate=learning_rate,
            max_iter=max_iter,
            model_selection=args.model_selection,
            init_params={"pretrained_word_embeddings": pretrained_word_embeddings},
            verbose=True,
            seed=seed,
        )
        for max_num_qa in [32]
        for max_num_review in [32]
        for optimizer in ['adam']
        for learning_rate in [float(lr) for lr in args.learning_rates.split(',')]
        for max_iter in [int(e) for e in args.epochs.split(',')]
        for seed in [123]
]

exp = cornac.Experiment(
    eval_method=eval_method,
    models=models,
    metrics=[
        cornac.metrics.RMSE(),
        cornac.metrics.MAE(),
        cornac.metrics.MSE(),
    ],
)

exp.run()
print(data_dir)
export_dir = args.input
selected_model = models[0]
epoch = selected_model.best_epoch if args.model_selection == 'best' else args.epochs.split(',')[0]
model_name = 'hrdrq_k_{}_e_{}'.format(args.n_factors, epoch)
export_dir = os.path.join(args.input, model_name)
os.makedirs(export_dir, exist_ok=True)
import util
from importlib import reload
if args.model_selection == 'best':
    util.export_ranked_questions_from_narreq(selected_model, os.path.join(export_dir, 'ranked_questions.txt'))
    util.export_important_question_ranking_from_narreq(selected_model, os.path.join(export_dir, 'important_question_ranking.txt'))
    util.export_hftq_item_explanations(selected_model, export_dir)
