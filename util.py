import os
import numpy as np
import re
import string

def attention_review_count(model, max_num_review=32):
    train_set = model.train_set
    inc = 1
    for iix in train_set.item_iter(1):
        iid = iix[0]
        if np.count_nonzero(model.Gamma[iid]) != min(max_num_review, len(train_set.review_and_item_qa_text.item_review[iid].items())):
            inc += 1
            print(iid, np.count_nonzero(model.Gamma[iid]), min(max_num_review, len(train_set.review_and_item_qa_text.item_review[iid].items())))
            print(model.Gamma[iid])
    print(inc)

def export_train_ui(rs, path):
    uid2rawuid = list(rs.train_set.user_ids)
    iid2rawiid = list(rs.train_set.item_ids)
    with open(path, 'w') as f:
        for u, i, r in rs.train_set.uir_iter(1):
            f.write('{},{}\n'.format(uid2rawuid[u[0]], iid2rawiid[i[0]]))

def export_ranked_questions(model, path, max_num_question=32):
    with open(path, 'w') as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            line = str(raw_iid)
            for ranked_id in (-model.Gamma[iid][:min(len(item_qas_ids), max_num_question)]).argsort():
                item_qa_ids = item_qas_ids[ranked_id]
                qid = item_qa_ids[0]
                question_answer_text = model.train_set.review_and_item_qa_text.qas[qid]
                if len(item_qa_ids) > 1:
                    # answer_idx = item_qa_ids[1] # first answer idx
                    answer_idx = item_qa_ids[model.Xi[iid, ranked_id].argmax()] # top answer idx
                    question_answer_text = '{}\t{}'.format(question_answer_text, model.train_set.review_and_item_qa_text.qas[answer_idx])
                line = '{}\t\t{}'.format(line, question_answer_text)
            f.write('{}\n'.format(line))

def export_useful_review_ranking(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, 'w') as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_reviewer_ids = list(model.train_set.review_and_item_qa_text.item_review[iid].keys())
            ranked = (-model.Gamma[iid][:len(item_reviewer_ids)]).argsort()
            f.write('{},{}\n'.format(raw_iid, ','.join([str(uid2rawuid[item_reviewer_ids[i]]) for i in ranked])))

def export_important_question_ranking(model, path, max_num_question=32):
    with open(path, 'w') as f:
        for raw_iid, iid in model.train_set.iid_map.items():
            item_qas_ids = model.train_set.review_and_item_qa_text.item_qas[iid]
            n_questions = min(max_num_question, len(item_qas_ids)-1)
            top_review_id = model.Gamma[iid].argmax()
            if n_questions > 0:
                ranked = (-model.Beta[iid, top_review_id, :n_questions]).argsort()[:n_questions]
                f.write('{},{}\n'.format(raw_iid, ','.join([str(x) for x in ranked])))

def export_most_useful_review(model, path):
    uid2rawuid = list(model.train_set.user_ids)
    with open(path, 'w') as f:
        f.write('asin\treviewerID\treviewText\n')
        for raw_iid, iid in model.train_set.iid_map.items():
            top_review_id = model.Gamma[iid].argmax()
            item_review_ids = list(model.train_set.review_and_item_qa_text.item_review[iid].items())
            reviewer_id, review_id = item_review_ids[top_review_id]
            f.write('{}\t{}\t{}\n'.format(raw_iid, str(uid2rawuid[reviewer_id]), model.train_set.review_and_item_qa_text.corpus[review_id]))


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


def export_item_ids(selected_model, path, label):
    with open(path, 'a') as f: 
        for raw_iid in selected_model.train_set.iid_map.keys():
            f.write('{},{}\n'.format(raw_iid, label))

def export_item_question_count(selected_model, path, label):
    with open(path, 'w') as f:
        for raw_iid, iid in selected_model.train_set.iid_map.items():
            item_qa_ids = np.array(sum(selected_model.train_set.review_and_item_qa_text.item_qas[iid], []))
            f.write('{},{},{}\n'.format(raw_iid, len(item_qa_ids), label))



def rm_tags(t: str) -> str:
    """
    Remove html tags.
    e,g, rm_tags("<i>Hello</i> <b>World</b>!") -> "Hello World".
    """
    return re.sub('<([^>]+)>', '', t)


def rm_numeric(t: str) -> str:
    """
    Remove digits from `t`.
    """
    return re.sub('[0-9]+', ' ', t)


def rm_punctuation(t: str) -> str:
    """
    Remove "!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" from t.
    """
    return t.translate(str.maketrans('', '', string.punctuation))


def rm_dup_spaces(t: str) -> str:
    """
    Remove duplicate spaces in `t`.
    """
    return re.sub(' {2,}', ' ', t)

def rp_tab(t: str) -> str:
    """
    Replace tab with space
    """
    return re.sub('\t', ' ', t)

DEFAULT_PRE_RULES = [lambda t: t.lower(), rm_tags, rm_numeric, rm_punctuation, rp_tab, rm_dup_spaces]

def preprocess_text(t: str) -> str:
    for rule in DEFAULT_PRE_RULES:
        t = rule(t)
    return t