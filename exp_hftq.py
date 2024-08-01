import os
import cornac
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import QAModality, TextModality
from cornac.data.text import BaseTokenizer
import numpy as np

MAX_VOCAB = 4000

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input directory')
    parser.add_argument('-mu', '--min_user_freq', type=int, default=1)
    parser.add_argument('-mi', '--min_item_freq', type=int, default=1)
    parser.add_argument('-ct', '--cluster_threshold', type=float, default=0.8)
    parser.add_argument('-km', '--kmean', type=int, default=50)
    parser.add_argument('-k', '--n_topics', type=int, default=50)
    parser.add_argument('-e', '--epochs', type=str, default='1,2')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    return parser.parse_args()

args = parse_arguments()


args = parse_arguments()
feedback = Reader(min_user_freq=args.min_user_freq, min_item_freq=args.min_item_freq).read(os.path.join(args.input, "rating.txt"), fmt="UIRT", sep="\t")
reviews = Reader().read(os.path.join(args.input, "review.txt"), fmt="UIReview", sep="\t")

data_dir = "{}_crawled_kmeans_{}".format(args.input, args.kmean)

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


qa_modality = QAModality(
    data=qas,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=MAX_VOCAB,
)

item_ids = [
    qa[0]
    for qa in qas
]
item_docs = [
    ' '.join([' '.join(q) for q in qa[1]])
    for qa in qas
]
item_text_modality = TextModality(
    corpus=item_docs,
    ids=item_ids,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=MAX_VOCAB,
)

eval_method = StratifiedSplit(
    data=feedback,
    group_by="item",
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    item_text=item_text_modality,
    qa_text=qa_modality,
    verbose=True,
    seed=123,
)


models = [
        cornac.models.HFT(
            name=f'HFTQ_MINUSER_{args.min_user_freq}_MINITEM_{args.min_item_freq}_K_{args.n_topics}_E_{max_iter}_REG_{reg}_TEXTREG_{text_reg}',
            k=args.n_topics,
            max_iter=max_iter,
            grad_iter=5,
            l2_reg=reg,
            lambda_text=text_reg,
            vocab_size=item_text_modality.vocab.size,
            verbose=True,
            seed=seed,
        )
        for max_iter in [int(tok) for tok in args.epochs.split(',')]
        for reg in [0.001]
        for text_reg in [0.01]
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
model_name = 'hftq_k_{}'.format(args.n_topics)
export_dir = os.path.join(args.input, model_name)
os.makedirs(export_dir, exist_ok=True)
import util
from importlib import reload
util.export_ranked_questions_from_hftq(selected_model, os.path.join(export_dir, 'ranked_questions.txt'))
util.export_important_question_ranking_from_hftq(selected_model, os.path.join(export_dir, 'important_question_ranking.txt'))
util.export_hftq_item_explanations(selected_model, export_dir)
import pdb; pdb.set_trace()
