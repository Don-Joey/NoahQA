import argparse
import json
import logging
import numpy as np
import random
import collections

import pulp
import editdistance
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules
from tqdm import tqdm
from allennlp.models.model import Model
from allennlp.common.params import Params
from naqanet_generation_model.ComwpReader import ComwpReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.training.util import *
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader


def read_json(dir_):
    with open(dir_) as f:
        return json.load(f)

def editdist(p_pred, p_gold):
    return 1 - (editdistance.eval(p_pred.lower(), p_gold.lower()) / max(len(p_pred), len(p_gold)))


def best_alignment(di, dj):
    problem = pulp.LpProblem("Problem-1", pulp.LpMaximize)

    # Variable
    alignment = [[pulp.LpVariable("align_{}_{}".format(i, j), 0, 1, pulp.LpBinary) for j in range(len(dj))] for i in
                 range(len(di))]

    #
    # Constraints

    # Each node has one out going edge
    for i in range(len(di)):
        y = 0

        if len(dj) == 0:
            continue

        for j in range(len(dj)):
            y += alignment[i][j]

        problem.addConstraint(y <= 1)

    # Each node has one out going edge
    for i in range(len(dj)):
        y = 0

        if len(di) == 0:
            continue

        for j in range(len(di)):
            y += alignment[j][i]

        problem.addConstraint(y <= 1)

    # Set objective function.
    obj_vars = []
    obj_coefs = collections.defaultdict(dict)

    for i in range(len(di)):
        for j in range(len(dj)):
            coefs = []

            coefs += [editdist(di[i], dj[j])]

            coef = np.mean(coefs)

            obj_coefs[i][j] = coef

            if coef > 0.0:
                obj_vars += [coef * alignment[i][j]]

    #     print (obj_coefs)

    if len(obj_vars) == 0:
        return 0.0, {}, {}

    problem.setObjective(sum(obj_vars))
    problem.solve()

    alignment_pred, alignment_true = {}, {}

    for i in range(len(di)):
        alignment_pred[i] = None, 0.0

        for j in range(len(dj)):
            if pulp.value(alignment[i][j]) == 1.0:
                alignment_pred[i] = j, obj_coefs[i][j]

    for i in range(len(dj)):
        alignment_true[i] = None, 0.0

        for j in range(len(di)):
            if pulp.value(alignment[j][i]) == 1.0:
                alignment_true[i] = j, obj_coefs[j][i]

    num_cor = pulp.value(problem.objective)
    return num_cor, alignment_pred, alignment_true


def convert_to_adjacency_graph(graph_l):
    graph = {'<q>': [], '<p1>': [], '<p2>': [], '<p3>': [], '<p4>': [], '<p5>': [], '<p6>': [], '<p7>': [], '<p8>': [],
             '<q1>': [], '<q2>': [], '<q3>': [], '<q4>': [], '<q5>': [], '<q6>': [], '<p9>': [], '<p10>': [],
             '<q7>': [], '<q8>': [], '<q9>': [], '<q10>': []}
    for l_r in graph_l:
        l = l_r[0]
        r = l_r[1]
        #         if l not in graph:
        #             graph[l] = [r]
        #         else:
        graph[l].append(r)
    return graph


def decom(graph, node):
    visited = set()  # Set to keep track of visited nodes.
    path = []
    paths = []

    def dfs(visited, graph, node, path):
        path.append(node)
        if graph[node] == []:
            #             print (path)
            paths.append(path[:])
        if node not in visited:
            #             print (node)
            visited.add(node)
            for neighbour in graph[node]:
                dfs(visited, graph, neighbour, path)
        path.pop()

    dfs(visited, graph, node, path)
    return paths

def write_graph(node_dict, evidence_edges, predict_edges, question_id, predictions_graph_path):

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
    with open(predictions_graph_path+".json", 'a', encoding='utf-8') as f:
        #json.dump(edge,f, ensure_ascii=False, indent=4)
        j = json.dumps(edge, ensure_ascii=False)
        f.write(j + '\n')
        f.close()


def evaluate(
    model: Model,
    data_loader: DataLoader,
    cuda_device: int = -1,
    batch_weight_key: str = None,
    output_file: str = None,
    predictions_output_file: str = None,
    predictions_graph_path: str = None, 
) -> Dict[str, Any]:
    
    check_for_gpu(cuda_device)
    predictions_file = (
        None if predictions_output_file is None else open(predictions_output_file, "w")
    )
    if os.path.isfile(predictions_graph_path+".json"):
        os.remove(predictions_graph_path+".json")

    with torch.no_grad():
        model.eval()

        iterator = iter(data_loader)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator)

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")
            graph = output_dict.get("node_bag")
            write_graph(graph['node_dict'], graph['evidence_edges'], graph['predict_edges'],
                                        graph['question_id'], predictions_graph_path)

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if not HasBeenWarned.tqdm_ignores_underscores and any(
                metric_name.startswith("_") for metric_name in metrics
            ):
                logger.warning(
                    'Metrics with names beginning with "_" will '
                    "not be logged to the tqdm progress bar."
                )
                HasBeenWarned.tqdm_ignores_underscores = True
            description = (
                ", ".join(
                    [
                        "%s: %.2f" % (name, value)
                        for name, value in metrics.items()
                        if not name.startswith("_")
                    ]
                )
                + " ||"
            )
            generator_tqdm.set_description(description, refresh=False)

            if predictions_file is not None:
                predictions = json.dumps(sanitize(model.make_output_human_readable(output_dict)))
                predictions_file.write(predictions + "\n")

        if predictions_file is not None:
            predictions_file.close()

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError(
                    "The model you are trying to evaluate only sometimes produced a loss!"
                )
            final_metrics["loss"] = total_loss / total_weight

        if output_file is not None:
            dump_metrics(output_file, final_metrics, log=True)

        return final_metrics

def evalute_reasoning(evaluate_data, test_data_dict):
    precs, recalls, fs = [], [], []
    scores = []
    total = 0

    with tqdm(total=len(evaluate_data)) as pbar:
        for data_i, one_eva_data in enumerate(evaluate_data):
            qid = one_eva_data['question_id']
            gold_edges = one_eva_data['gold_edges']
            predict_edges = one_eva_data['predict_edges']
            if qid not in test_data_dict:  # the whole annotated data don't need to skip
                continue

            q_text = test_data_dict[qid]['q']
            content = test_data_dict[qid]['content']
            tmp_dict = {}
            tmp_dict['<q>'] = q_text
            tmp_dict.update(content)
            #         print (tmp_dict)

            #             print ('gold_edges: ',gold_edges)
            #             print ('predict_edges: ',predict_edges)

            if len(gold_edges) >= len(predict_edges):
                big_graph = 1
            else:
                big_graph = 2

            gold_graph = convert_to_adjacency_graph(gold_edges)
            predict_edges = convert_to_adjacency_graph(predict_edges)

            gold_paths = decom(gold_graph, '<q>')
            predict_paths = decom(predict_edges, '<q>')

            #         print ('gold: ',gold_paths)
            #         print ('pred: ',predict_paths)

            weights_big = {}
            if big_graph == 1:
                big_graph_paths = gold_paths[:]
            else:
                big_graph_paths = predict_paths[:]
            for big_path in big_graph_paths:
                for big_node in big_path:
                    if big_node not in weights_big:
                        weights_big[big_node] = 1
                    else:
                        weights_big[big_node] += 1

            weights_path = []
            for big_path in big_graph_paths:
                weights = 0
                for node in big_path:
                    weights += 1.0 / weights_big[node]
                weights = weights / len(weights_big)
                weights_path.append(weights)
            #         print (sum(weights_path), weights_path)

            bipartite_matrix = np.zeros((len(gold_paths), len(predict_paths)))
            for pp_i, predict_path in enumerate(predict_paths):
                for gp_i, gold_path in enumerate(gold_paths):
                    try:
                        predict_path_ = []
                        for l_r in predict_path:
                            #                     print ('d',l_r, predict_path)
                            predict_path_.append(tmp_dict[l_r])
                        gold_path_ = []
                        for l_r in gold_path:
                            gold_path_.append(tmp_dict[l_r])
                        num_cor, alignment_pred, alignment_true = best_alignment(predict_path_, gold_path_)
                        #                     print ('--',pp_i, gp_i,'|',num_cor, alignment_pred, alignment_true)
                        bipartite_matrix[gp_i][pp_i] = num_cor
                    except:
                        continue
            row_ind, col_ind = linear_sum_assignment(-bipartite_matrix)
            #             print (row_ind, col_ind)

            num_cor_list = []
            for n_i in range(len(row_ind)):
                num_cor_list.append(bipartite_matrix[row_ind[n_i]][col_ind[n_i]])
            #             print ("num_cor_list:", num_cor_list)
            #             for rid_ in row_ind:
            #                 print ('gold path:',gold_paths[rid_])
            for n_i, rid_ in enumerate(col_ind):
                predict_path = predict_paths[rid_]
                #                 print (predict_path)
                negative_v = 0.
                for node_i in range(len(predict_path) - 1):
                    left_node = predict_path[node_i]
                    right_node = predict_path[node_i + 1]
                    #                     print (left_node, right_node)
                    if len(left_node) > 3 and len(right_node) > 3 and left_node[1] == 'q' and right_node[1] == 'q':
                        if left_node[2] <= right_node[2]:
                            negative_v -= 1.0
                alpha = 1.0
                negative_v *= alpha
                if negative_v + num_cor_list[n_i] < 0:
                    negative_v = -num_cor_list[n_i]
                num_cor_list[n_i] = num_cor_list[n_i] + negative_v
            #                 print (negative_v)

            prec_ = []
            recall_ = []
            f_ = []
            score_ = []
            for ni, num_cor_ in enumerate(num_cor_list):
                y_pred = predict_paths[col_ind[ni]]
                y_true = gold_paths[row_ind[ni]]
                if big_graph == 1:
                    y_final = y_true
                    final_ni = row_ind[ni]
                else:
                    y_final = y_pred
                    final_ni = col_ind[ni]
                score = weights_path[final_ni] * num_cor_ / len(y_final) if len(y_final) > 0 else 0
                score_.append(score)
            #             prec = num_cor_ / len(y_pred) if len(y_pred) > 0 else 0
            #             recall = num_cor_ / len(y_true) if len(y_true) > 0 else 0
            #                 prec = weights_path[col_ind[ni]]*num_cor_ / len(y_pred) if len(y_pred) > 0 else 0
            #                 recall = weights_path[col_ind[ni]]*num_cor_ / len(y_true) if len(y_true) > 0 else 0
            #                 f = (2 * prec * recall) / (prec + recall) if prec + recall > 0 else 0
            #                 prec_.append(prec)
            #                 recall_.append(recall)
            #                 f_.append(f)
            #         print (prec_, recall_, f_)
            #         print ()
            score = np.mean(score_)

            scores += [score]
            pbar.update(1)

            # total += 1
            # if total > 3:
            #     break

    return {"score": np.mean(scores)}

if __name__ == "__main__":
    # Parse arguments
    cuda_device = 13
    parser = argparse.ArgumentParser()

    parser.add_argument("--archive_file", type=str, required=True,
                        help="dir")
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')
    parser.add_argument('--language',
                        type=str,
                        default="cn",
                        help='test dataset language version')

    args = parser.parse_args()

    for package_name in getattr(args, 'include_package', ()):
        import_submodules(package_name)

    address = args.archive_file

    archive = load_archive(args.archive_file)

    config = archive.config.duplicate()

    model_type = config.get("model").get("type")

    dataset_reader_params = config["validation_dataset_reader"]

    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    if args.language == "cn":
        test_dataset = dataset_reader.read("test_cn.json")
    else:
        test_dataset = dataset_reader.read("test_en.json")
    #iterator = DataIterator.from_params(config['iterator'])
    model = archive.model
    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()
    vocab_address = address + '/vocabulary'
    vocab = Vocabulary.from_files(vocab_address)
    iterator = BasicIterator(batch_size=8)
    iterator.index_with(vocab)
    predictions_graph_path = os.path.join("evaluate_cache", args.archive_file+"_"+args.include_package+"_"+args.language)
    metrics = evaluate(model, test_dataset, iterator, cuda_device, batch_weight_key="", predictions_graph_path = predictions_graph_path)

    if args.language == "cn":
        test_data = read_json("test_cn.json")
    else:
        test_data = read_json("test_en.json")
    
    test_data_dict = {}
    for qid, q_elem in test_data.items():
        passage = q_elem['passage']
        qa_pairs = q_elem['qa_pairs']
        tmp_dict = {}
        for psg_i, sent in enumerate(passage):
            tmp_dict['<p'+str(psg_i+1)+'>'] = sent[1]
        for qa_i, qa_pair in enumerate(qa_pairs):
            tmp_dict['<q'+str(qa_i+1)+'>'] = qa_pair['question']+ (qa_pair['ans'] if 'ans' in qa_pair else qa_pair['answer'])
    #     print (tmp_dict)
        for qa_i, qa_pair in enumerate(qa_pairs):
            test_data_dict[qa_pair['query_id']] = {'q':qa_pair['question']+ (qa_pair['ans'] if 'ans' in qa_pair else qa_pair['answer']),\
                                                'content': tmp_dict.copy()}
    
    to_be_evaluated_data = []
    with open(predictions_graph_path+".json", 'r') as f:
        for line in f:
            to_be_evaluated_data.append(json.loads(line))
    
    print (evalute_reasoning(to_be_evaluated_data, test_data_dict))
