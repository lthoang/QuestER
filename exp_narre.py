import os
import cornac
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import ReviewModality
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
    parser.add_argument('-k', '--n_factors', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=str, default='1,2')
    parser.add_argument('-s', '--model_selection', type=str, choices=['best', 'last'], default='best')
    parser.add_argument('-lr', '--learning_rates', type=str, default='0.001')
    return parser.parse_args()

# k = 8
emb_size = 100
ATTENTION_SIZE = 8
BATCH_SIZE = 64
MAX_TEXT_LENGTH = 128
MAX_NUM_REVIEW = 32
args = parse_arguments()

data_dir = args.input
print(data_dir)
feedback = Reader(min_user_freq=args.min_user_freq, min_item_freq=args.min_item_freq).read(
    os.path.join(data_dir, "rating.txt"), fmt="UIRT", sep="\t"
)
reviews = Reader().read(os.path.join(data_dir, "review.txt"), fmt="UIReview", sep="\t")

pretrained_word_embeddings = {}
with open(
    "/data/shared/download/glove/glove.6B.{}d.txt".format(emb_size), encoding="utf-8"
) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        pretrained_word_embeddings[word] = coefs

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

# eval_method = StratifiedSplit(
#     data=feedback,
#     group_by="item",
#     test_size=0.1,
#     val_size=0.1,
#     exclude_unknowns=True,
#     review_text=review_modality,
#     verbose=True,
#     seed=123,
# )

eval_method = StratifiedSplit(
    data=feedback,
    group_by="user",
    test_size=0.2,
    val_size=0.16,
    exclude_unknowns=True,
    review_text=review_modality,
    verbose=True,
    # seed=123,
    seed=42,
)


models = [
        cornac.models.NARRE(
            name=f"NARRE_MINUSER_{args.min_user_freq}_MINITEM_{args.min_item_freq}_MAXNREVIEW_{MAX_NUM_REVIEW}_EMB_{100}_IDEMB_{args.n_factors}_K_{args.n_factors}_ATT_{ATTENTION_SIZE}_MAXTXTLEN_{MAX_TEXT_LENGTH}_BS_{BATCH_SIZE}_E_{max_iter}_OPTIMIZER_{optimizer}_LR_{learning_rate}_MODEL_SELECTION_{args.model_selection}_SEED_{seed}",
            embedding_size=100,
            n_factors=args.n_factors,
            id_embedding_size=args.n_factors,
            attention_size=ATTENTION_SIZE,
            kernel_sizes=[3],
            n_filters=64,
            batch_size=BATCH_SIZE,
            max_text_length=MAX_TEXT_LENGTH,
            max_num_review=MAX_NUM_REVIEW,
            dropout_rate=0.5,
            optimizer=optimizer,
            learning_rate=learning_rate,
            model_selection=args.model_selection,
            max_iter=max_iter,
            init_params={"pretrained_word_embeddings": pretrained_word_embeddings},
            seed=seed,
        )
        for optimizer in ['adam']
        for learning_rate in [float(lr) for lr in args.learning_rates.split(',')]
        for max_iter in [int(e) for e in args.epochs.split(',')]
        # for seed in [118, 203, 220, 243, 245, 261, 270, 273, 285, 304]
        # for seed in [316, 325, 345, 356, 371, 400, 458, 487, 490, 501]
        # for seed in [578, 602, 641, 710, 780, 796, 803, 845, 867, 888]
        # for seed in [123, 118]
        for seed in [123]
]

exp = cornac.Experiment(
    eval_method=eval_method,
    models=models,
    metrics=[
        cornac.metrics.MSE(),
        cornac.metrics.RMSE(),
        # cornac.metrics.MAE(),
        cornac.metrics.AUC(),
        # cornac.metrics.NDCG(k=50),
        cornac.metrics.NDCG(),
    ],
    # save_dir='./dist/testsavenarre'
)

exp.run()
print(data_dir)
export_dir = args.input
selected_model = models[0]
epoch = selected_model.best_epoch if args.model_selection == 'best' else args.epochs.split(',')[0]
model_name = 'narre_k_8_e_{}_lr_{}'.format(epoch, args.learning_rates.split(',')[0])
export_dir = os.path.join(args.input, model_name)
os.makedirs(export_dir, exist_ok=True)
import util
from importlib import reload
if args.model_selection == 'best':
    util.export_useful_review_ranking_from_narre(selected_model, os.path.join(export_dir, 'useful_review_ranking.txt'))
    util.export_most_useful_review_from_narre(selected_model, os.path.join(export_dir, 'most_useful_review.txt'))
    util.export_narre_item_explanations(selected_model, export_dir)
