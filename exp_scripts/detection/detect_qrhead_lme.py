import argparse
from tqdm import tqdm
from itertools import product
import json
import numpy as np
from qrretriever.attn_retriever import FullHeadRetriever


def truncate_text_by_space(text, limit):
    if limit <= 0:
        return text
    words = text.split(' ')
    if len(words) > limit:
        print('number of words being truncated: ', len(words) - limit, flush=True)
    return ' '.join(words[:limit])


def prepare_docs_for_detection(
    data,
    truncate_by_space=0,
    evidence_preserving_truncation=False,
    gold_truncate_by_space=0,
):
    docs = data["paragraphs"]
    gt_doc_ids = set(data.get("gt_docs", []))

    for p in docs:
        paragraph_text = p['paragraph_text'].strip()
        is_gold_doc = p.get("idx") in gt_doc_ids or p.get("is_supporting") is True

        if evidence_preserving_truncation and is_gold_doc:
            p['paragraph_text'] = truncate_text_by_space(paragraph_text, gold_truncate_by_space)
        else:
            p['paragraph_text'] = truncate_text_by_space(paragraph_text, truncate_by_space)

    return docs


def lme_eval(retrieval_results, data_instances):
    """
    retrieval_results: a dict of qid -> {doc_id -> score}, retrieval results from a specific head
    data_instances: a list of dicts, each dict represents an instance
    """
    all_score_over_gold = []

    for data in data_instances:
        qid = data['idx']
        gt_docs = data["gt_docs"] # a list of doc ids

        doc_id2score = retrieval_results[qid] # doc_id -> score

        if len(gt_docs) == 0:
            score_over_gold = 0
        else:
            score_over_gold = np.sum([doc_id2score[doc_id] for doc_id in gt_docs])
            sorted_docs_ids = sorted(doc_id2score.items(), key=lambda x: x[1], reverse=True)
            sorted_docs_ids = [doc_id for doc_id, _ in sorted_docs_ids]

        all_score_over_gold.append(score_over_gold)

    mean_score_over_gold = np.mean(all_score_over_gold)
    return mean_score_over_gold # QRScore for a specific head


def get_doc_scores_per_head(
    full_head_retriever,
    data_instances,
    truncate_by_space=0,
    evidence_preserving_truncation=False,
    gold_truncate_by_space=0,
):
    """
    data_instances: a list of dicts, each dict represents an instance
    """
    doc_scores_per_head = {} # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    for i, data in enumerate(tqdm(data_instances)):

        query = data["question"]
        docs = prepare_docs_for_detection(
            data,
            truncate_by_space=truncate_by_space,
            evidence_preserving_truncation=evidence_preserving_truncation,
            gold_truncate_by_space=gold_truncate_by_space,
        )

        retrieval_scores = full_head_retriever.score_docs_per_head_for_detection(query, docs) # doc_id -> score tensor with shape (n_layers, n_heads)
        doc_scores_per_head[data['idx']] = retrieval_scores

    return doc_scores_per_head



def score_heads(doc_scores_per_head, data_instances):
    """
    doc_scores_per_head: a dict of dicts, outer dict key is question idx, inner dict key is doc idx, value is a (n_layers, n_heads) tensor
    """

    # pick first qid
    first_qid = next(iter(doc_scores_per_head))
    first_doc_id = next(iter(doc_scores_per_head[first_qid]))
    example_tensor = doc_scores_per_head[first_qid][first_doc_id]
    num_layers, num_heads = example_tensor.shape

    # score by head
    layer_head = product(range(num_layers), range(num_heads))
    head_scores = {}

    for layer, head in tqdm(layer_head, total=num_layers * num_heads):
        retrieval_results = {} # get new retrieval results for this head, qid -> {doc_id -> score}

        for qid, per_doc_score_tensors in doc_scores_per_head.items():
            # per_doc_score_tensors: a dict of doc_id -> (n_layers, n_heads) tensor
            doc_id2score = {}
            for doc_id, score_tensor in per_doc_score_tensors.items():
                score = score_tensor[layer][head]
                doc_id2score[doc_id] = score.item()

            retrieval_results[qid] = doc_id2score
            
        head_score = lme_eval(retrieval_results, data_instances) # QRScore for this head
        head_scores[(layer, head)] = head_score

    # replace key with layer-head
    head_scores_list = [(f"{layer}-{head}", score) for (layer, head), score in head_scores.items()]
    # sort heads by scores
    head_scores_list.sort(key=lambda x: x[1], reverse=True)

    return head_scores_list # a list of tuples (head, score)





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file to find QRHead.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save scores for each head.")

    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")
    parser.add_argument("--evidence_preserving_truncation", action="store_true", help="For LME, apply --truncate_by_space to non-gold docs only and preserve gt_docs/is_supporting docs.")
    parser.add_argument("--gold_truncate_by_space", type=int, default=0, help="When using --evidence_preserving_truncation, optionally truncate gold docs to this many words. Default is 0 (preserve full gold docs).")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")

    args = parser.parse_args()

    full_head_retriever = FullHeadRetriever(
        config_or_config_path=args.config_or_config_path,
        model_name_or_path=args.model_name_or_path,
    )

    # read input file
    print(f"Reading input file: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)

    doc_scores_per_head = get_doc_scores_per_head(
        full_head_retriever,
        data_instances,
        truncate_by_space=args.truncate_by_space,
        evidence_preserving_truncation=args.evidence_preserving_truncation,
        gold_truncate_by_space=args.gold_truncate_by_space,
    ) # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    head_scores_list = score_heads(doc_scores_per_head, data_instances)

    with open(args.output_file, "w") as f:
        json.dump(head_scores_list, f, indent=4)
