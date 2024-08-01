import os
import cornac
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.data import ReviewModality
from cornac.data.text import BaseTokenizer
import numpy as np

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input directory')
    parser.add_argument('-mu', '--min_user_freq', type=int, default=1)
    parser.add_argument('-mi', '--min_item_freq', type=int, default=1)
    parser.add_argument('-k', '--n_topics', type=int, default=50)
    parser.add_argument('-e', '--epochs', type=str, default='1,2')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    return parser.parse_args()

args = parse_arguments()


data_dir = args.input
print(data_dir)
feedback = Reader(min_user_freq=args.min_user_freq, min_item_freq=args.min_item_freq).read(
    os.path.join(data_dir, "rating.txt"), fmt="UIRT", sep="\t"
)
reviews = Reader().read(os.path.join(data_dir, "review.txt"), fmt="UIReview", sep="\t")

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
)

item_text_modality = ReviewModality(
    data=reviews,
    group_by='item',
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
)

eval_method = StratifiedSplit(
    data=feedback,
    group_by="item",
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    item_text=item_text_modality,
    review_text=review_modality,
    verbose=True,
    seed=123,
)


models = [
        cornac.models.HFT(
            name=f'HFT_MINUSER_{args.min_user_freq}_MINITEM_{args.min_item_freq}_K_{args.n_topics}_E_{max_iter}_REG_{reg}_TEXTREG_{text_reg}',
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
model_name = 'hft_k_{}_e_{}'.format(args.n_topics, args.epochs.split(',')[0])
export_dir = os.path.join(args.input, model_name)
os.makedirs(export_dir, exist_ok=True)
import util
from importlib import reload
util.export_useful_review_ranking_from_hft(selected_model, os.path.join(export_dir, 'useful_review_ranking.txt'))
util.export_most_useful_review_from_hft(selected_model, os.path.join(export_dir, 'most_useful_review.txt'))
util.export_hft_item_explanations(selected_model, export_dir)
import pdb; pdb.set_trace()
