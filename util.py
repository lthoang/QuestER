import os
import numpy as np
import re
import string


def attention_review_count(model, max_num_review=32):
    train_set = model.train_set
    inc = 1
    for iix in train_set.item_iter(1):
        iid = iix[0]
        if np.count_nonzero(model.Gamma[iid]) != min(
            max_num_review,
            len(train_set.review_and_item_qa_text.item_review[iid].items()),
        ):
            inc += 1
            print(
                iid,
                np.count_nonzero(model.Gamma[iid]),
                min(
                    max_num_review,
                    len(train_set.review_and_item_qa_text.item_review[iid].items()),
                ),
            )
            print(model.Gamma[iid])
    print(inc)


def export_train_ui(rs, path):
    uid2rawuid = list(rs.train_set.user_ids)
    iid2rawiid = list(rs.train_set.item_ids)
    with open(path, "w") as f:
        for u, i, r in rs.train_set.uir_iter(1):
            f.write("{},{}\n".format(uid2rawuid[u[0]], iid2rawiid[i[0]]))


def export_ranked_questions(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            line = str(raw_iid)
            for ranked_id in (
                -model.Gamma[iid][: min(len(item_qas_ids), max_num_question)]
            ).argsort():
                item_qa_ids = item_qas_ids[ranked_id]
                qid = item_qa_ids[0]
                question_answer_text = model.train_set.review_and_item_qa_text.qas[qid]
                if len(item_qa_ids) > 1:
                    first_answer_idx = item_qa_ids[1]
                    question_answer_text = "{}\t{}".format(
                        question_answer_text,
                        model.train_set.review_and_item_qa_text.qas[first_answer_idx],
                    )
                line = "{}\t\t{}".format(line, question_answer_text)
            f.write("{}\n".format(line))


def export_ranked_questions_from_narreq(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            line = str(raw_iid)
            for ranked_id in (
                -model.A[iid, : min(len(item_qas_ids), max_num_question)]
            ).argsort():
                item_qa_ids = item_qas_ids[ranked_id]
                qid = item_qa_ids[0]
                question_answer_text = model.train_set.review_and_item_qa_text.qas[qid]
                if len(item_qa_ids) > 1:
                    first_answer_idx = item_qa_ids[1]
                    question_answer_text = "{}\t{}".format(
                        question_answer_text,
                        model.train_set.review_and_item_qa_text.qas[first_answer_idx],
                    )
                line = "{}\t\t{}".format(line, question_answer_text)
            f.write("{}\n".format(line))


def export_ranked_questions_from_narqre(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            line = str(raw_iid)
            for ranked_id in (
                -model.Aq[iid, : min(len(item_qas_ids), max_num_question)]
            ).argsort():
                item_qa_ids = item_qas_ids[ranked_id]
                qid = item_qa_ids[0]
                question_answer_text = model.train_set.review_and_item_qa_text.qas[qid]
                if len(item_qa_ids) > 1:
                    first_answer_idx = item_qa_ids[1]
                    question_answer_text = "{}\t{}".format(
                        question_answer_text,
                        model.train_set.review_and_item_qa_text.qas[first_answer_idx],
                    )
                line = "{}\t\t{}".format(line, question_answer_text)
            f.write("{}\n".format(line))


def export_ranked_questions_from_hftq(
    model, path, max_num_question=32, max_num_answer=1
):
    from collections import Counter

    theta = np.exp(model.model.kappa * model.model.gamma_i) / np.expand_dims(
        np.exp(model.model.kappa * model.model.gamma_i).sum(1), -1
    )
    topic_assignment = model.model.topic_assignment
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.qa_text.item_qas[iid][:max_num_question]
            line = str(raw_iid)
            item_qas = [
                model.train_set.item_text.vocab.to_idx(tokens)
                for tokens in model.train_set.item_text.tokenizer.batch_tokenize(
                    [
                        " ".join(
                            [
                                model.train_set.qa_text.corpus[qa_idx]
                                for qa_idx in qa_ids[: 1 + max_num_answer]
                            ]
                        )
                        for qa_ids in item_qas_ids
                    ]
                )
            ]
            item_document = model.documents[iid]
            scores = []
            for qa in item_qas:
                qa_word_topic = [
                    topic_assignment[iid][np.where(item_document == word_idx)[0][0]]
                    for word_idx in qa
                    if word_idx in item_document
                ]
                topic_freq = Counter(qa_word_topic)
                freq_dist = np.zeros(model.k)
                freq_dist[list(topic_freq.keys())] = np.array(list(topic_freq.values()))
                freq_dist = (
                    freq_dist / freq_dist.sum() if freq_dist.sum() > 0 else freq_dist
                )
                distance = ((theta[iid] - freq_dist) ** 2).sum()
                scores.append(distance)
            scores = np.array(scores)
            ranked = (-scores).argsort()
            for ranked_id in ranked:
                item_qa_ids = item_qas_ids[ranked_id]
                qid = item_qa_ids[0]
                question_answer_text = model.train_set.qa_text.corpus[qid]
                if len(item_qa_ids) > 1:
                    first_answer_idx = item_qa_ids[1]
                    question_answer_text = "{}\t{}".format(
                        question_answer_text,
                        model.train_set.qa_text.corpus[first_answer_idx],
                    )
                line = "{}\t\t{}".format(line, question_answer_text)
            f.write("{}\n".format(line))


def export_useful_review_ranking(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            if iid not in model.train_set.review_and_item_qa_text.item_review:
                continue
            top_question_id = model.Gamma[iid].argmax()
            item_reviewer_ids = list(
                model.train_set.review_and_item_qa_text.item_review[iid].keys()
            )
            ranked = (
                -model.Beta[iid, top_question_id, : len(item_reviewer_ids)]
            ).argsort()[: len(item_reviewer_ids)]
            f.write(
                "{},{}\n".format(
                    raw_iid,
                    ",".join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked]),
                )
            )


def export_important_question_ranking(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            n_questions = min(max_num_question, len(item_qas_ids) - 1)
            if n_questions > 0:
                ranked = (-model.Gamma[iid][:n_questions]).argsort()
                f.write("{},{}\n".format(raw_iid, ",".join([str(x) for x in ranked])))


def export_important_question_ranking_from_narreq(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            n_questions = min(max_num_question, len(item_qas_ids) - 1)
            if n_questions > 0:
                ranked = (-model.A[iid, :n_questions]).argsort()[:n_questions]
                f.write("{},{}\n".format(raw_iid, ",".join([str(x) for x in ranked])))


def export_important_question_ranking_from_narqre(model, path, max_num_question=32):
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            n_questions = min(max_num_question, len(item_qas_ids) - 1)
            if n_questions > 0:
                ranked = (-model.Aq[iid, :n_questions]).argsort()[:n_questions]
                f.write("{},{}\n".format(raw_iid, ",".join([str(x) for x in ranked])))


def export_important_question_ranking_from_hftq(
    model, path, max_num_question=32, max_num_answer=1
):
    from collections import Counter

    theta = np.exp(model.model.kappa * model.model.gamma_i) / np.expand_dims(
        np.exp(model.model.kappa * model.model.gamma_i).sum(1), -1
    )
    topic_assignment = model.model.topic_assignment
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.qa_text.item_qas[iid]
            n_questions = min(max_num_question, len(item_qas_ids) - 1)
            if n_questions > 0:
                item_qas = [
                    model.train_set.item_text.vocab.to_idx(tokens)
                    for tokens in model.train_set.item_text.tokenizer.batch_tokenize(
                        [
                            " ".join(
                                [
                                    model.train_set.qa_text.corpus[qa_idx]
                                    for qa_idx in qa_ids[: 1 + max_num_answer]
                                ]
                            )
                            for qa_ids in item_qas_ids[:n_questions]
                        ]
                    )
                ]
                item_document = model.documents[iid]
                scores = []
                for qa in item_qas:
                    qa_word_topic = [
                        topic_assignment[iid][np.where(item_document == word_idx)[0][0]]
                        for word_idx in qa
                        if word_idx in item_document
                    ]
                    topic_freq = Counter(qa_word_topic)
                    freq_dist = np.zeros(model.k)
                    freq_dist[list(topic_freq.keys())] = np.array(
                        list(topic_freq.values())
                    )
                    freq_dist = (
                        freq_dist / freq_dist.sum()
                        if freq_dist.sum() > 0
                        else freq_dist
                    )
                    distance = ((theta[iid] - freq_dist) ** 2).sum()
                    scores.append(distance)

                scores = np.array(scores)
                ranked = (-scores).argsort()
                f.write("{},{}\n".format(raw_iid, ",".join([str(x) for x in ranked])))


def export_most_useful_review(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        f.write("asin\treviewerID\treviewText\n")
        for raw_iid, iid in model.train_set.iid_map.items():
            top_question_id = model.Gamma[iid].argmax()
            item_review_ids = list(
                model.train_set.review_and_item_qa_text.item_review[iid].items()
            )
            reviewer_id, review_id = item_review_ids[
                model.Beta[iid, top_question_id, : len(item_review_ids)].argmax()
            ]
            f.write(
                "{}\t{}\t{}\n".format(
                    raw_iid,
                    str(uid2rawuid[reviewer_id]),
                    model.train_set.review_and_item_qa_text.corpus[review_id],
                )
            )


def export_useful_review_ranking_from_narre(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(
                model.train_set.review_text.item_review[iid].keys()
            )
            ranked = (-model.A[iid, : len(item_reviewer_ids)]).argsort()[
                : len(item_reviewer_ids)
            ]
            f.write(
                "{},{}\n".format(
                    raw_iid,
                    ",".join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked]),
                )
            )


def export_useful_review_ranking_from_hft(model, path, max_num_review=32):
    from collections import Counter

    uid2rawuid = list(model.train_set.user_ids)
    theta = np.exp(model.model.kappa * model.model.gamma_i) / np.expand_dims(
        np.exp(model.model.kappa * model.model.gamma_i).sum(1), -1
    )
    topic_assignment = model.model.topic_assignment
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(
                model.train_set.review_text.item_review[iid].keys()
            )[:max_num_review]
            item_review_ids = list(
                model.train_set.review_text.item_review[iid].values()
            )[:max_num_review]
            item_reviews = [
                model.train_set.item_text.vocab.to_idx(tokens)
                for tokens in model.train_set.item_text.tokenizer.batch_tokenize(
                    [
                        model.train_set.review_text.corpus[review_id]
                        for review_id in item_review_ids
                    ]
                )
            ]
            item_document = model.documents[iid]
            scores = []
            for review in item_reviews:
                review_word_topic = [
                    topic_assignment[iid][np.where(item_document == word_idx)[0][0]]
                    for word_idx in review
                    if word_idx in item_document
                ]
                topic_freq = Counter(review_word_topic)
                freq_dist = np.zeros(model.k)
                freq_dist[list(topic_freq.keys())] = np.array(list(topic_freq.values()))
                freq_dist = (
                    freq_dist / freq_dist.sum() if freq_dist.sum() > 0 else freq_dist
                )
                distance = ((theta[iid] - freq_dist) ** 2).sum()
                scores.append(distance)
            scores = np.array(scores)
            ranked = (-scores).argsort()
            f.write(
                "{},{}\n".format(
                    raw_iid,
                    ",".join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked]),
                )
            )


def export_useful_review_ranking_from_hrdr(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(
                model.train_set.review_text.item_review[iid].keys()
            )
            ranked = (-model.A[iid, : len(item_reviewer_ids)]).argsort()[
                : len(item_reviewer_ids)
            ]
            f.write(
                "{},{}\n".format(
                    raw_iid,
                    ",".join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked]),
                )
            )


def export_useful_review_ranking_from_narqre(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(
                model.train_set.review_and_item_qa_text.item_review[iid].keys()
            )
            ranked = (-model.Ar[iid, : len(item_reviewer_ids)]).argsort()[
                : len(item_reviewer_ids)
            ]
            f.write(
                "{},{}\n".format(
                    raw_iid,
                    ",".join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked]),
                )
            )


def export_useful_review_ranking_from_narqre1(model, path):
    pass
    # uid2rawuid = list(model.train_set.user_ids)
    # with open(path, 'w') as f:
    #     for raw_iid, iid in model.train_set.iid_map.items():
    #         item_reviewer_ids = list(model.train_set.review_and_item_qa_text.item_review[iid].keys())
    #         ranked = (-model.Ar[iid, :len(item_reviewer_ids)]).argsort()[:len(item_reviewer_ids)]
    #         f.write('{},{}\n'.format(raw_iid, ','.join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked])))


def export_most_useful_review_from_hft(model, path, max_num_review=32):
    from collections import Counter

    uid2rawuid = list(model.train_set.user_ids)
    theta = np.exp(model.model.kappa * model.model.gamma_i) / np.expand_dims(
        np.exp(model.model.kappa * model.model.gamma_i).sum(1), -1
    )
    topic_assignment = model.model.topic_assignment
    with open(path, "w") as f:
        f.write("asin\treviewerID\treviewText\n")
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(
                model.train_set.review_text.item_review[iid].keys()
            )[:max_num_review]
            item_review_ids = list(
                model.train_set.review_text.item_review[iid].values()
            )[:max_num_review]
            item_reviews = [
                model.train_set.item_text.vocab.to_idx(tokens)
                for tokens in model.train_set.item_text.tokenizer.batch_tokenize(
                    [
                        model.train_set.review_text.corpus[review_id]
                        for review_id in item_review_ids
                    ]
                )
            ]
            item_document = model.documents[iid]
            scores = []
            for review in item_reviews:
                review_word_topic = [
                    topic_assignment[iid][np.where(item_document == word_idx)[0][0]]
                    for word_idx in review
                    if word_idx in item_document
                ]
                topic_freq = Counter(review_word_topic)
                freq_dist = np.zeros(model.k)
                freq_dist[list(topic_freq.keys())] = np.array(list(topic_freq.values()))
                freq_dist = (
                    freq_dist / freq_dist.sum() if freq_dist.sum() > 0 else freq_dist
                )
                distance = ((theta[iid] - freq_dist) ** 2).sum()
                scores.append(distance)
            scores = np.array(scores)
            top_id = (-scores).argsort()[0]
            f.write(
                "{}\t{}\t{}\n".format(
                    raw_iid,
                    str(uid2rawuid[item_reviewer_ids[top_id]]),
                    model.train_set.review_text.corpus[item_review_ids[top_id]],
                )
            )


def export_most_useful_review_from_narre(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        f.write("asin\treviewerID\treviewText\n")
        for raw_iid, iid in model.train_set.iid_map.items():
            item_review_ids = list(model.train_set.review_text.item_review[iid].items())
            top_review_id = model.A[iid, : len(item_review_ids)].argmax()
            reviewer_id, review_id = item_review_ids[top_review_id]
            f.write(
                "{}\t{}\t{}\n".format(
                    raw_iid,
                    str(uid2rawuid[reviewer_id]),
                    model.train_set.review_text.corpus[review_id],
                )
            )


def export_most_useful_review_from_hrdr(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        f.write("asin\treviewerID\treviewText\n")
        for raw_iid, iid in model.train_set.iid_map.items():
            item_review_ids = list(model.train_set.review_text.item_review[iid].items())
            top_review_id = model.A[iid, : len(item_review_ids)].argmax()
            reviewer_id, review_id = item_review_ids[top_review_id]
            f.write(
                "{}\t{}\t{}\n".format(
                    raw_iid,
                    str(uid2rawuid[reviewer_id]),
                    model.train_set.review_text.corpus[review_id],
                )
            )


def export_most_useful_review_from_narqre(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, "w") as f:
        f.write("asin\treviewerID\treviewText\n")
        for raw_iid, iid in model.train_set.iid_map.items():
            item_review_ids = list(
                model.train_set.review_and_item_qa_text.item_review[iid].items()
            )
            top_review_id = model.Ar[iid, : len(item_review_ids)].argmax()
            reviewer_id, review_id = item_review_ids[top_review_id]
            f.write(
                "{}\t{}\t{}\n".format(
                    raw_iid,
                    str(uid2rawuid[reviewer_id]),
                    model.train_set.review_and_item_qa_text.corpus[review_id],
                )
            )


def export_most_useful_review_from_narqre1(model, path):
    pass
    # uid2rawuid = list(model.train_set.user_ids)
    # with open(path, 'w') as f:
    #     f.write('asin\treviewerID\treviewText\n')
    #     for raw_iid, iid in model.train_set.iid_map.items():
    #         item_review_ids = list(model.train_set.review_and_item_qa_text.item_review[iid].items())
    #         top_review_id = model.Ar[iid, :len(item_review_ids)].argmax()
    #         reviewer_id, review_id = item_review_ids[top_review_id]
    #         f.write('{}\t{}\t{}\n'.format(raw_iid, str(uid2rawuid[reviewer_id]), model.train_set.review_and_item_qa_text.corpus[review_id]))


def export_quester_explanations(selected_model, output_dir, max_num_question=32, max_num_review=32, max_num_answer=32):
    import json
    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        questions = []
        answers = []
        for inc, item_question_answers_ids in enumerate(selected_model.train_set.review_and_item_qa_text.item_qas[mapped_iid]):
            if max_num_question is not None and inc == max_num_question:
                break
            qidx = item_question_answers_ids[0]
            aids = item_question_answers_ids[1:]
            question = {
                'questionText': selected_model.train_set.review_and_item_qa_text.qas[qidx],
                'answerText': '',
                'answers': [selected_model.train_set.review_and_item_qa_text.qas[aidx] for aidx in aids]
            }
            if len(item_question_answers_ids) > 1:
                # answer = selected_model.train_set.review_and_item_qa_text.qas[item_question_answers_ids[1]] # first answer
                top_answer_idx = item_question_answers_ids[1:][selected_model.Xi[mapped_iid, inc].argmax()]
                answer = selected_model.train_set.review_and_item_qa_text.qas[top_answer_idx]
                question['answerText'] = answer
            questions.append(question)
        reviews = [selected_model.train_set.review_and_item_qa_text.corpus[rid] for _, rid in list(selected_model.train_set.review_and_item_qa_text.item_review[mapped_iid].items())]
        n_question = min(len(questions), max_num_question)
        n_answer = min(len(answers), max_num_answer)
        n_review = min(len(reviews), max_num_review)
        questions = questions[:n_question]
        reviews = reviews[:n_review]
        saved_dir = os.path.join(output_dir, '/'.join(str(raw_iid)))
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, '{}.json'.format(raw_iid)), 'w') as f:
            data = {
                "questions": questions,
                "reviews": reviews,
                "Gamma": selected_model.Gamma[mapped_iid][:n_review].tolist(),
                "Beta": selected_model.Beta[mapped_iid, :n_review, :n_question].tolist(),
                "Xi": selected_model.Xi[mapped_iid, :n_question, :n_answer].tolist()
            }
            json.dump(data, f)



def export_erqa_explanations(selected_model, explanation_dir):
    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        item_question_ids = sum(
            selected_model.train_set.qa_text.item_qas[mapped_iid], []
        )
        item_questions = [
            selected_model.train_set.qa_text.corpus[idx] for idx in item_question_ids
        ]
        item_reviews = [
            selected_model.train_set.review_text.corpus[review_idx]
            for review_idx in selected_model.train_set.review_text.item_review[
                mapped_iid
            ].values()
        ]
        top_question_idx = selected_model.Gamma[mapped_iid].argmax()
        top_review_idx = selected_model.Beta[mapped_iid, top_question_idx].argmax()
        n_questions = len(item_questions)
        n_reviews = len(item_reviews)
        question_attention_weights = selected_model.Gamma[mapped_iid, :n_questions]
        review_attention_weights = selected_model.Beta[
            mapped_iid, :n_questions, :n_reviews
        ]
        with open(os.path.join(explanation_dir, "{}.txt".format(raw_iid)), "w") as f:
            f.write("Top question:\n{}\n".format(item_questions[top_question_idx]))
            f.write("Top review:\n{}\n".format(item_reviews[top_review_idx]))
            f.write("\n")
            f.write(
                "Question attention weights:\n{}\n".format(
                    ",".join([str(weight) for weight in question_attention_weights])
                )
            )
            f.write(
                "Review attention weights:\n{}\n".format(
                    "\n".join(
                        [
                            ",".join(
                                [
                                    str(review_attention_weights[qidx, ridx])
                                    for ridx in range(n_reviews)
                                ]
                            )
                            for qidx in range(n_questions)
                        ]
                    )
                )
            )
            f.write("Questions:\n")
            f.write("{}\n".format("\n".join(item_questions)))
            f.write("Reviews:\n")
            f.write("{}\n".format("\n".join(item_reviews)))


def export_erqa_sentence_level_explanations(
    selected_model, explanation_dir, top_k_question=1, top_k_sentence=1
):
    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        item_question_ids = sum(
            selected_model.train_set.review_sentence_and_item_qa_text.item_qas[
                mapped_iid
            ],
            [],
        )
        item_questions = [
            selected_model.train_set.review_sentence_and_item_qa_text.corpus[idx]
            for idx in item_question_ids
        ]

        sentence_ids = []
        for inc, (jdx, j_sentence_ids) in enumerate(
            selected_model.train_set.review_sentence_and_item_qa_text.item_review_sentence[
                mapped_iid
            ].items()
        ):
            if (
                selected_model.max_num_review is not None
                and inc == selected_model.max_num_review
            ):
                break
            for s_inc, sentence_idx in enumerate(j_sentence_ids):
                if (
                    selected_model.max_num_sentence is not None
                    and s_inc == selected_model.max_num_sentence
                ):
                    break
                sentence_ids.append(sentence_idx)

        item_sentences = [
            selected_model.train_set.review_sentence_and_item_qa_text.corpus[
                sentence_idx
            ]
            for sentence_idx in sentence_ids
        ]
        # item_reviews = [selected_model.train_set.review_text.corpus[review_idx] for review_idx in selected_model.train_set.review_text.item_review[mapped_iid].values()]

        n_questions = len(item_questions)
        n_sentences = len(item_sentences)
        top_question_ids = selected_model.Gamma[mapped_iid].argsort()[::-1][
            : min(top_k_question, n_questions)
        ]
        top_sentence_ids = selected_model.Beta[
            mapped_iid, top_question_ids[0]
        ].argsort()[::-1][: min(top_k_sentence, n_sentences)]
        question_attention_weights = selected_model.Gamma[mapped_iid, :n_questions]
        sentence_attention_weights = selected_model.Beta[
            mapped_iid, :n_questions, :n_sentences
        ]
        with open(os.path.join(explanation_dir, "{}.txt".format(raw_iid)), "w") as f:
            f.write(
                "Top question(s):\n{}\n".format(
                    "\n".join(
                        [
                            item_questions[top_question_idx]
                            for top_question_idx in top_question_ids
                        ]
                    )
                )
            )
            f.write(
                "Top sentence(s):\n{}\n".format(
                    "\n".join(
                        [
                            item_sentences[top_sentence_idx]
                            for top_sentence_idx in top_sentence_ids
                        ]
                    )
                )
            )
            f.write("\n")
            f.write(
                "Question attention weights:\n{}\n".format(
                    ",".join([str(weight) for weight in question_attention_weights])
                )
            )
            f.write(
                "Sentence attention weights:\n{}\n".format(
                    "\n".join(
                        [
                            ",".join(
                                [
                                    str(sentence_attention_weights[qidx, sidx])
                                    for sidx in range(n_sentences)
                                ]
                            )
                            for qidx in range(n_questions)
                        ]
                    )
                )
            )
            f.write("Questions:\n")
            f.write("{}\n".format("\n".join(item_questions)))
            f.write("Reviews:\n")
            f.write("{}\n".format("\n".join(item_sentences)))


def export_erqa_item_explanations(
    selected_model, output_dir, max_num_question=32, max_num_review=32
):
    import json

    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        # questions = [selected_model.train_set.review_and_item_qa_text.qas[qid] for qid in np.array(sum(selected_model.train_set.review_and_item_qa_text.item_qas[mapped_iid], []))]
        questions = []
        for inc, item_question_answers_ids in enumerate(
            selected_model.train_set.review_and_item_qa_text.item_qas[mapped_iid]
        ):
            if max_num_question is not None and inc == max_num_question:
                break
            question = {
                "questionText": selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[0]
                ],
                "answerText": "",
            }
            if len(item_question_answers_ids) > 1:
                firstAnswer = selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[1]
                ]
                question["answerText"] = firstAnswer
            questions.append(question)
        reviews = [
            selected_model.train_set.review_and_item_qa_text.corpus[rid]
            for _, rid in list(
                selected_model.train_set.review_and_item_qa_text.item_review[
                    mapped_iid
                ].items()
            )
        ]
        n_question = min(len(questions), max_num_question)
        n_review = min(len(reviews), max_num_review)
        questions = questions[:n_question]
        reviews = reviews[:n_review]
        saved_dir = os.path.join(output_dir, "/".join(str(raw_iid)))
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, "{}.json".format(raw_iid)), "w") as f:
            data = {
                "questions": questions,
                "reviews": reviews,
                "Gamma": selected_model.Gamma[mapped_iid][:n_question].tolist(),
                "Beta": selected_model.Beta[
                    mapped_iid, :n_question, :n_review
                ].tolist(),
            }
            json.dump(data, f)


def export_item_explanations_from_narqre(
    selected_model, output_dir, max_num_question=32, max_num_review=32
):
    import json

    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        questions = []
        for inc, item_question_answers_ids in enumerate(
            selected_model.train_set.review_and_item_qa_text.item_qas[mapped_iid]
        ):
            if max_num_question is not None and inc == max_num_question:
                break
            question = {
                "questionText": selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[0]
                ],
                "answerText": "",
            }
            if len(item_question_answers_ids) > 1:
                firstAnswer = selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[1]
                ]
                question["answerText"] = firstAnswer
            questions.append(question)
        reviews = [
            selected_model.train_set.review_and_item_qa_text.corpus[rid]
            for _, rid in list(
                selected_model.train_set.review_and_item_qa_text.item_review[
                    mapped_iid
                ].items()
            )
        ]
        n_question = min(len(questions), max_num_question)
        n_review = min(len(reviews), max_num_review)
        questions = questions[:n_question]
        reviews = reviews[:n_review]
        saved_dir = os.path.join(output_dir, "/".join(str(raw_iid)))
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, "{}.json".format(raw_iid)), "w") as f:
            data = {
                "questions": questions,
                "reviews": reviews,
                "Ar": selected_model.Ar[mapped_iid][:n_review].tolist(),
                "Aq": selected_model.Aq[mapped_iid][:n_question].tolist(),
            }
            json.dump(data, f)


def export_narre_item_explanations(selected_model, output_dir, max_num_review=32):
    import json

    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        reviews = [
            selected_model.train_set.review_text.corpus[rid]
            for _, rid in list(
                selected_model.train_set.review_text.item_review[mapped_iid].items()
            )
        ]
        n_review = min(len(reviews), max_num_review)
        reviews = reviews[:n_review]
        saved_dir = os.path.join(output_dir, "/".join(str(raw_iid)))
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, "{}.json".format(raw_iid)), "w") as f:
            data = {
                "reviews": reviews,
                "A": selected_model.A[mapped_iid][:n_review].tolist(),
            }
            json.dump(data, f)


def export_narreq_item_explanations(selected_model, output_dir, max_num_question=32):
    import json

    for raw_iid, mapped_iid in selected_model.train_set.iid_map.items():
        questions = []
        for inc, item_question_answers_ids in enumerate(
            selected_model.train_set.review_and_item_qa_text.item_qas[mapped_iid]
        ):
            if max_num_question is not None and inc == max_num_question:
                break
            question = {
                "questionText": selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[0]
                ],
                "answerText": "",
            }
            if len(item_question_answers_ids) > 1:
                firstAnswer = selected_model.train_set.review_and_item_qa_text.qas[
                    item_question_answers_ids[1]
                ]
                question["answerText"] = firstAnswer
            questions.append(question)
        n_question = min(len(questions), max_num_question)
        questions = questions[:n_question]
        saved_dir = os.path.join(output_dir, "/".join(str(raw_iid)))
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, "{}.json".format(raw_iid)), "w") as f:
            data = {
                "questions": questions,
                "A": selected_model.A[mapped_iid][:n_question].tolist(),
            }
            json.dump(data, f)


def export_hrdr_item_explanations(selected_model, output_dir, max_num_review=32):
    export_narre_item_explanations(selected_model, output_dir, max_num_review)


def export_hrdrq_item_explanations(selected_model, output_dir, max_num_question=32):
    export_narreq_item_explanations(selected_model, output_dir, max_num_question)


def export_item_ids(selected_model, path, label):
    with open(path, "a") as f:
        for raw_iid in selected_model.train_set.iid_map.keys():
            f.write("{},{}\n".format(raw_iid, label))


def export_item_question_count(selected_model, path, label):
    with open(path, "w") as f:
        for raw_iid, iid in selected_model.train_set.iid_map.items():
            item_qa_ids = np.array(
                sum(selected_model.train_set.review_and_item_qa_text.item_qas[iid], [])
            )
            f.write("{},{},{}\n".format(raw_iid, len(item_qa_ids), label))


def rm_tags(t: str) -> str:
    """
    Remove html tags.
    e,g, rm_tags("<i>Hello</i> <b>World</b>!") -> "Hello World".
    """
    return re.sub("<([^>]+)>", "", t)


def rm_numeric(t: str) -> str:
    """
    Remove digits from `t`.
    """
    return re.sub("[0-9]+", " ", t)


def rm_punctuation(t: str) -> str:
    """
    Remove "!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" from t.
    """
    return t.translate(str.maketrans("", "", string.punctuation))


def rm_dup_spaces(t: str) -> str:
    """
    Remove duplicate spaces in `t`.
    """
    return re.sub(" {2,}", " ", t)


def rp_tab(t: str) -> str:
    """
    Replace tab with space
    """
    return re.sub("\t", " ", t)


DEFAULT_PRE_RULES = [
    lambda t: t.lower(),
    rm_tags,
    rm_numeric,
    rm_punctuation,
    rp_tab,
    rm_dup_spaces,
]


def preprocess_text(t: str) -> str:
    for rule in DEFAULT_PRE_RULES:
        t = rule(t)
    return t
