#!/usr/bin/python

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import json
import argparse
import string
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from rgnet.utils.compute_tool import compute_gen_ans,split_equation_tt
from rgnet.utils.util import split_equation_evuation
import thulac

thu1 = thulac.thulac(seg_only=True)

# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())

PUN = string.punctuation.replace('/','').replace('+','').replace('-','').replace('*','').replace('<','').replace(
    '>','').replace('<','').replace('(','').replace(')','').replace('[','').replace(']','').replace('^','').replace('{','').replace('}','').replace(
    '%','')
EXCLUDE = set(PUN)

def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str) -> List[str]:
    return re.split(" ", text)

def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text: str) -> bool:
    try:
        if re.search(r'\d',text) and ('/' in text or '%' in text):
            return True
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(text)
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags



def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]],
                history_answer: Dict,
                answer_type: List,
                node_bag: Dict):
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    type_dict = {'type': 0}
    type_EM = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}
    type_f1 = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}
    type_count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
    supporting_exact = 0.0
    supporting_count = 0.0
    compute_right = 0.0
    compute_number = 0.0


    cost, edge_em, edge_f1 = evaluate_edge(node_bag['node_dict'], node_bag['evidence_edges'], node_bag['predict_edges'],
                                           node_bag['question_id'])
    total_em = 1
    if set(re.findall(r'<a\d>', gold[0][0])):
        supporting_exact = 0.0
        supporting_count = 1
        supporting_predicted = str(predicted.replace(' ',''))
        for key, value in history_answer.items():
            if key in supporting_predicted:
                if set(re.findall(r'<a\d>', str(gold[0][0]).lower())) == set(
                        re.findall(r'<a\d>', supporting_predicted.lower())) and set(re.findall(r'<a\d>', str(gold[0][0]))):
                    print("supporting_facts", str(predicted).lower(), str(gold[0][0]).lower())
                    supporting_exact = 1.0
                    supporting_count = 1
                    break
    else:
        supporting_exact = 0.0
        supporting_count = 0
    if len(gold) == 2:
        predicted = predicted.replace(' ','')
        tt = _answer_to_bags(str(' '.join(split_equation_evuation(predicted))))
        for key, value in history_answer.items():
            predicted = predicted.replace(key.lower(), value, 2)
        # print("predicted",predicted)
        predicted_bags_formula = _answer_to_bags(predicted)
        predicted = str(compute_gen_ans(predicted_bags_formula[0][0]))
        predicted_bags = _answer_to_bags(predicted)
        gold_bags = _answer_to_bags(str(gold[1]))
        gold_bags_formula = _answer_to_bags(str(' '.join(split_equation_evuation(str(gold[0][0])))))
        try:
            if 'error' in predicted_bags[0][0]:
                # print("generation_false", gold_bags[0][0], predicted_bags[0][0])
                exact_match = 0.0
                compute_right = 0.0
                compute_number = 1.0
            elif abs(float(re.sub(r'[a-z]', '', gold_bags[0][0])) - float(predicted_bags[0][0])) < 1e-5:
                # print("generation_true",gold_bags[0][0],predicted_bags[0][0])
                exact_match = 1.0
                compute_right = 1.0
                compute_number = 1.0
            else:
                # print("generation_false", gold_bags[0][0], predicted_bags[0][0])
                exact_match = 0.0
                compute_right = 0.0
                compute_number = 1.0
        except ValueError:
            # print("generation_false", gold_bags[0][0], predicted_bags[0][0])
            exact_match = 0.0
            compute_right = 0.0
            compute_number = 1.0

        # print("predicted_bags_formula[1]",[set(split_equation_tt(split_equation_tt(tt[1][0])[0]))])
        # print("gold_bags_formula[1]",[set(split_equation_tt(split_equation_tt(tt[1][0])[0]))])
        # f1_per_bag = _align_bags(predicted_bags_formula[1], gold_bags_formula[1])
        # print("gold_bags_formula",gold_bags_formula[1])
        f1_per_bag = _align_bags(tt[1], gold_bags_formula[1])
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        if answer_type == [4, 0]:
            type_EM['7'] = exact_match
            type_count['7'] = 1
            type_f1['7'] = f1
        else:
            type_EM[str(answer_type[0])] = exact_match
            type_count[str(answer_type[0])] = 1
            type_f1[str(answer_type[0])] = f1
    else:
        predicted = predicted.replace(' ','')
        gold_f1 = (gold[0].replace(' ',''),)

        f1_predicted_bags = _answer_to_bags(thu1.cut(predicted, text=True))
        f1_gold_bags = _answer_to_bags(thu1.cut(gold[0], text=True))

        predicted_bags = _answer_to_bags(predicted)
        gold_bags = _answer_to_bags(gold_f1)

        # print("set(predicted_bags[0])",set(predicted_bags[0]))
        # print("set(gold_bags[0])",set(gold_bags[0]))
        if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
            # print(predicted_bags[0],gold_bags[0])
            # print('span')
            exact_match = 1.0
        else:
            exact_match = 0.0

        f1_per_bag = _align_bags(f1_predicted_bags[1], f1_gold_bags[1])
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        type_EM[str(answer_type[0])] = exact_match
        type_count[str(answer_type[0])] = 1
        type_f1[str(answer_type[0])] = f1
    
    step_em = {
        'question_id': node_bag['question_id'],
        'passage_id': node_bag['passage_id'],
        'em': exact_match
    }
    with open("step_robert_cn.json", 'a', encoding='utf-8') as f:
        #json.dump(edge,f, ensure_ascii=False, indent=4)
        j = json.dumps(step_em, ensure_ascii=False)
        f.write(j + '\n')
        f.close()
    return exact_match, f1, supporting_exact, supporting_count, type_EM, type_f1, type_count, edge_em, total_em

def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """

    if "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"

    elif "generation" in answer and answer['generation']:
        return tuple([answer['generation'],answer['arithmetic_ans']]), "generation" if len(answer["generation"]) == 1 else "generation"
    else:
        raise ValueError(f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}")


def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations (note that these are somewhat deep in the JSON for the
    gold annotations, but must be top-level keys in the predicted answers).

    The ``annotations`` are assumed to have the format of the dev set in the DROP data release.
    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a string
    (or list of strings) that is the answer.
    """
    instance_exact_match = []
    instance_f1 = []
    # for each type as well
    type_to_em: Dict[str, List[float]] = defaultdict(list)
    type_to_f1: Dict[str, List[float]] = defaultdict(list)
    for _, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0.0
            max_f1_score = 0.0
            max_type = None
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]
                candidate_answers = [qa_pair["answer"]]
                if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    candidate_answers += qa_pair["validated_answers"]
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    em_score, f1_score = get_metrics(predicted, gold_answer)
                    if gold_answer[0].strip() != "":
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1_score)
                        if max_em_score == em_score and max_f1_score == f1_score:
                            max_type = gold_type
            else:
                print("Missing prediction for question: {}".format(query_id))
                if qa_pair and qa_pair["answer"]:
                    max_type = answer_json_to_strings(qa_pair["answer"])[1]
                else:
                    max_type = "number"
                max_em_score = 0.0
                max_f1_score = 0.0
            instance_exact_match.append(max_em_score)
            instance_f1.append(max_f1_score)
            type_to_em[max_type].append(max_em_score)
            type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(typ, len(type_to_em[typ]), 100. * len(type_to_em[typ])/total))
        print("  Exact-match accuracy {0:.3f}".format(100. * np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100. * np.mean(type_to_f1[typ])))
    return global_em, global_f1


def evaluate_prediction_file(prediction_path: str, gold_path: str,
                             output_path: Optional[str] = None) -> Tuple[float, float]:
    """
    Takes a prediction file and a gold file and evaluates the predictions for each question in the
    gold file.  Both files must be json formatted and must have query_id keys, which are used to
    match predictions to gold annotations.  The gold file is assumed to have the format of the dev
    set in the DROP data release.  The prediction file must be a JSON dictionary keyed by query id,
    where the value is either a JSON dictionary with an "answer" key, or just a string (or list of
    strings) that is the answer. Writes a json with global_em and global_f1 metrics to file at
    the specified output path, unless None is passed as output path.
    """
    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    global_em, global_f1 = evaluate_json(annotations, predicted_answers)

    # Output predictions to file if an output path is given
    if output_path is not None:
        output_dict = {"global_em": global_em,
                       "global_f1": global_f1}

        with open(output_path, "w", encoding="utf8") as outfile:
            json.dump(output_dict, outfile)

    return (global_em, global_f1)


def evaluate_edge(node_dict, evidence_edges, predict_edges, question_id):

    evidence_edges = [['<s>', '<q>']] + evidence_edges

    predict_edges = [['<s>', '<q>']] + predict_edges

    remove_predict_edges = []
    for p_ed in predict_edges:
        if p_ed[0] in node_dict.keys() and p_ed[1] in node_dict.keys():
            remove_predict_edges.append(p_ed)
    predict_edges = remove_predict_edges

    gold_node_string = 'gold_edge: ' + ' '.join(['{}->{}'.format(edge[0],edge[1]) for edge in evidence_edges[1:]]) + ' \n'
    predict_node_string = 'predict_edge: ' + ' '.join(['{}->{}'.format(edge[0],edge[1]) for edge in predict_edges[1:]]) + ' \n'
    write_list = ['question_id:' + ' ' + question_id + ' \n', gold_node_string, predict_node_string, "*" * 87 + ' \n']

    edge = {
        'question_id':question_id,
        'gold_edges': evidence_edges[1:],
        'predict_edges': predict_edges[1:]
    }
    with open("edge_roberta.json", 'a', encoding='utf-8') as f:
        #json.dump(edge,f, ensure_ascii=False, indent=4)
        j = json.dumps(edge, ensure_ascii=False)
        f.write(j + '\n')
        f.close()

    evidence_nodes = []

    

    #print("G1",G1.nodes,G1.edges)
    #print("G2", G2.nodes, G2.edges)
    #cost = tuple(nx.optimize_edit_paths(G1, G2))[0][2]
    cost = 1.0

    predict_set = set(['{}-{}'.format(edge[0],edge[1]) for edge in predict_edges])
    evidence_set = set(['{}-{}'.format(edge[0],edge[1]) for edge in evidence_edges])

    if predict_set == evidence_set and len(predict_set) == len(evidence_set):
        EM = 1
    else:
        EM = 0

    intersection = len(evidence_set.intersection(predict_set))
    if not predict_set:
        precision = 1.0
    else:
        precision = intersection / float(len(predict_set))
    if not evidence_set:
        recall = 1.0
    else:
        recall = intersection / float(len(evidence_set))

    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0

    return cost, EM, f1


'''
if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='evaluate on drop dataset')
    parser.add_argument("--gold_path",
                        type=str,
                        required=False,
                        default="drop_dataset_test.gold.json",
                        help='location of the gold file')
    parser.add_argument("--prediction_path",
                        type=str,
                        required=False,
                        default="sample_predictions.json",
                        help='location of the prediction file')
    parser.add_argument("--output_path",
                        type=str,
                        required=False,
                        default=None,
                        help='location of the output metrics file')

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path, args.output_path)
'''
