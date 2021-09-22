import numpy as np
from util import *
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any
import json


from overrides import overrides
from word2number.w2n import word_to_num
import re

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import (
    Field,
    MetadataField,
    LabelField,
    ListField,
    SequenceLabelField,
    SpanField,
    IndexField,
    ArrayField,
)
from common.fields.text_field import TextField
import itertools

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer#, WordpieceIndexer
from common import WordpieceIndexer
from allennlp.data.tokenizers import Token, Tokenizer#, WordTokenizer
from common import WordTokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from common.tokenizer.spacy_tokenizer import SpacyTokenizer
from pytorch_pretrained_bert import WordpieceTokenizer
from transformers import RobertaTokenizer, XLMRobertaTokenizer

from rgnet_xlm.utils.util import (
    IGNORED_TOKENS,
    STRIPPED_CHARACTERS,
    make_reading_comprehension_instance,
    split_tokens_by_hyphen,
    clipped_passage_num,
)

logger = logging.getLogger(__name__)

AUGMENT = '未知 | 是 | 否 | 不知道 | ( ) { } [ ] + - * / ^'

WORD_NUMBER_MAP = {
    "零": 0,
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16,
    "十七": 17,
    "十八": 18,
    "十九": 19,
}
@Tokenizer.register("xlm-zh-edge")
class RoBertaDropTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]

@TokenIndexer.register("xlm-zh-edge")
class RoBertaDropTokenIndexer(WordpieceIndexer):
    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 800) -> None:
        roberta_tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
        wordpiece_tokenizer = WordpieceTokenizer(vocab=roberta_tokenizer.get_vocab(),unk_token='<unk>')
        super().__init__(vocab=roberta_tokenizer.get_vocab(),
                         wordpiece_tokenizer=wordpiece_tokenizer.tokenize,
                         max_pieces=max_pieces,
                         namespace="roberta",
                         separator_token="</s>")

@DatasetReader.register("zh_xlm-edge")
class ComwpReader(DatasetReader):

    def __init__(
            self,
            tokenizer: Tokenizer = None,
            roberta_tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = False,
            passage_length_limit: int = None,
            question_length_limit: int = None,
            skip_when_all_empty: List[str] = None,
            instance_format: str = "drop",
            relaxed_span_match_for_finding_labels: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._bert_tokenizer = roberta_tokenizer
        self._answer_tokenizer = WordTokenizer()

        self.max_pieces = 800
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in [
                "passage_span",
                "question_span",
                "generation",
            ], f"Unsupported skip type: {item}"
        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path, 'rb') as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        kept_count, skip_count = 0, 0
        for passage_id, passage_info in dataset.items():
            passage_info_list = ['<p>']
            for token in passage_info['passage']:
                if token[1]:
                    passage_info_list.append('<t>')
                    passage_info_list.append(str(token[1]))
                    passage_info_list.append('</t>')
            passage_length = len(passage_info['passage'])
            passage_info_list.append(AUGMENT)
            passage_text = " ".join(passage_info_list)

            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)


            num_list = [n_value for n_key, n_value in passage_info['num_list'].items()]
            history = []
            history_evidence_buffer = {}
            for qa_number, question_answer in enumerate(passage_info["qa_pairs"]):
                passage_evidence = [0] * passage_length
                evidence_facts = question_answer["evidence"]
                question_id = question_answer["query_id"]
                passage_evidence_nodes = []
                for evi in evidence_facts:
                    if evi < 1 and (evi * 100) % 10 == 0:
                        try:
                            passage_evidence[int(evi * 10) - 1] = 1
                            passage_evidence_nodes.append('<p{}>'.format(int(evi * 10)))
                        except IndexError:
                            print("question_id", question_id)
                    elif evi < 1 and (evi * 100) % 10 != 0:
                        try:
                            passage_evidence[int(evi * 100) - 2] = 1
                            passage_evidence_nodes.append('<p{}>'.format(int(evi * 100) - 1))
                        except IndexError:
                            print("question_id", question_id)
                passage_evidence.append(-1)



                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                answer_txt = ""
                if "answer" in question_answer:
                    answer = dict()

                    if question_answer["answer_type"] == [6, 1]:
                        question_answer["answer"] = '未知'
                        answer_txt = '未知'
                    answer[str(question_answer["answer_type"][0])] = str(question_answer["answer"]).lower()
                    if question_answer["answer_type"][0] == 6 or question_answer["answer_type"] == [2, 0] or \
                            question_answer["answer_type"][0] == 1 or question_answer["answer_type"] == [5, 0] or \
                            question_answer['answer_type'] == [4,1]:
                        answer['spans'] = str(question_answer["answer"]).rstrip('了').rstrip('。').lower()
                        answer_txt = str(question_answer["answer"]).rstrip('了').rstrip('。').lower()

                    if question_answer["answer_type"][0] == 3 or question_answer["answer_type"] == [4,0]:
                        try:
                            answer[str(question_answer["answer_type"][0])] = str(question_answer["answer_temp"]).lower()
                            answer['generation'] = str(question_answer['answer_temp']).lower()
                            answer_txt = str(question_answer['answer_temp']).lower()
                            answer_txt = ' '.join(split_equation(str(question_answer['answer_temp']).lower(), num_list))
                        except KeyError:
                            try:
                                answer[str(question_answer["answer_type"][0])] = str(question_answer["answer_formula"]).lower()
                                answer['generation'] = str(question_answer['answer_formula']).lower()
                                answer_txt = str(question_answer['answer_formula']).lower()
                                answer_txt = ' '.join(
                                    split_equation(str(question_answer['answer_formula']).lower(), num_list))
                            except KeyError:
                                answer[str(question_answer["answer_type"][0])] = str(question_answer["answer"]).lower()
                                answer['generation'] = str(question_answer['answer']).lower()
                                answer_txt = str(question_answer['answer']).lower()
                                answer_txt = ' '.join(
                                    split_equation(str(question_answer['answer']).lower(), num_list))
                        try:
                            answer['arithmetic_ans'] = question_answer['ans']
                        except:
                            answer['arithmetic_ans'] = answer['generation']

                    answer_annotations.append(answer)

                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                question_type = question_answer["answer_type"]

                history.append((question_text, answer_txt))
                history_evidence_buffer[qa_number + 1] = evidence_facts

                try:
                    edge_list = self.construct_graph(qa_number + 1, history_evidence_buffer)
                except RecursionError:
                    print("question_id", question_id)
                    new_evidence_facts = []
                    for evid in evidence_facts:
                        if evid != 1.0:
                            new_evidence_facts.append(evid)
                    history_evidence_buffer[qa_number + 1] = new_evidence_facts
                    edge_list = self.construct_graph(qa_number + 1, history_evidence_buffer)

                instance = None
                if question_type[0] == 6 or question_type == [2, 0] or question_type[0] == 1 or question_type == [5, 0] or question_type[0] == 3 or question_type[0] == 4:
                    instance, length_of_input = self.text_to_instance(
                        question_text,
                        passage_text,
                        passage_length,
                        question_id,
                        passage_id,
                        answer_annotations,
                        passage_tokens,
                        question_type,
                        history,
                        num_list,
                        passage_evidence,
                        passage_evidence_nodes,
                        evidence_facts,
                        edge_list,
                    )
                
                if instance is not None and length_of_input <= 512:
                    kept_count += 1
                    yield instance
                else:
                    skip_count += 1
        print(kept_count, skip_count)
        logger.info(f"Skipped {skip_count} questions, kept {kept_count} questions.")

    @overrides
    def text_to_instance(
            self,  # type: ignore
            question_text: str,
            passage_text: str,
            passage_length: int,
            question_id: str = None,
            passage_id: str = None,
            answer_annotations: List[Dict] = None,
            passage_tokens: List[Token] = None,
            question_type: List[int] = None,
            history: List[tuple] = None,
            num_list: List = None,
            passage_evidence: List = None,
            passage_evidence_nodes: List = None,
            evidence_facts: List = None,
            edge_list: List[List] = None,
    ) -> Union[Instance, None]:
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)

        para_sentences_start_map = []
        para_sentences_end_map = []
        para_sentences_start_len = []
        para_sentences_end_len = []
        numbers_in_passage = []
        number_indices = []
        number_words = []
        number_len = []
        passage_tokens_replace = []
        curr_index = 0
        for word_token_index,word_token in enumerate(passage_tokens):
            number = self.convert_word_to_number(word_token.text,True)
            wordpieces = self._bert_tokenizer.tokenize(word_token.text)
            for word_piece_index, word_piece in enumerate(wordpieces):
                if passage_tokens_replace == [] and word_piece_index==0:
                    wordpieces[0] = Token(word_piece.text,0,text_id=0)#word_piece
                elif passage_tokens_replace != [] and word_piece_index==0:
                    wordpieces[0] = Token(word_piece.text,idx=passage_tokens_replace[-1].idx + len(passage_tokens_replace[-1].text) + 1,
                                          text_id=passage_tokens_replace[-1].text_id + 1)#word_piece
                elif word_piece_index > 0:
                    wordpieces[word_piece_index] = Token(word_piece.text,
                                                         idx=(wordpieces[word_piece_index-1].idx + len(wordpieces[word_piece_index-1].text)),
                                                         text_id=(wordpieces[word_piece_index-1].text_id + 1))#word_piece
            num_wordpieces = len(wordpieces)
            if word_token.text == '<t>':
                para_sentences_start_map.append(curr_index)
                para_sentences_start_len.append(num_wordpieces)
            if word_token.text == '</t>':
                para_sentences_end_map.append(curr_index)
                para_sentences_end_len.append(num_wordpieces)
            if number is not None:
                numbers_in_passage.append(number)
                number_indices.append(curr_index)
                number_words.append(word_token.text)
                number_len.append(num_wordpieces)
            if word_token_index >= (len(passage_tokens) - 11) and word_token.text in AUGMENT[-21:]:
                numbers_in_passage.append(number)
                number_indices.append(curr_index)
                number_words.append(word_token.text)
                number_len.append(num_wordpieces)
            passage_tokens_replace += wordpieces
            curr_index += num_wordpieces
        passage_tokens = passage_tokens_replace

        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)


        question_tokens.insert(0, self._tokenizer.tokenize('<q>')[0])
        question_text = '<q> ' + question_text
        history_answer = {}
        question_evidence_nodes = []
        if len(history) > 1:
            question_evidence = [0] * (len(history) - 1)
            for evi in evidence_facts:
                if evi >= 1.0:
                    try:
                        question_evidence[int(evi - 1)] = 1
                        question_evidence_nodes.append('<q{}>'.format(int(evi)))
                    except:
                        print("question_id", question_id)
            question_evidence.append(-1)
            temp = []
            temp_txt = []
            for i in range(1, len(history)):
                d = i#len(history) - i
                temp.append(self._tokenizer.tokenize('<q{}>'.format(d))[0])
                temp.extend(split_tokens_by_hyphen(self._tokenizer.tokenize(str(history[i - 1][0]).lower())))
                temp.append(self._tokenizer.tokenize('<a{}>'.format(d))[0])
                temp.extend(split_tokens_by_hyphen(self._tokenizer.tokenize(str(history[i - 1][1]).lower())))
                temp.append(self._tokenizer.tokenize('</q{}>'.format(d))[0])

                temp_txt.append('<q{}>'.format(d))
                temp_txt.append(str(history[i - 1][0]).lower())
                temp_txt.append('<a{}>'.format(d))
                history_answer['<a{}>'.format(d)]=str(history[i - 1][1]).lower()
                temp_txt.append(str(history[i - 1][1]).lower())
                temp_txt.append('</q{}>'.format(d))
            question_text = " ".join(temp_txt) + ' ' + question_text
            temp.extend(question_tokens)
            question_tokens = temp
        else:
            question_evidence = [-1]

        evidence_nodes = passage_evidence_nodes + question_evidence_nodes

        ques_sentences_start_map = []
        ques_sentences_end_map = []
        question_start_map = []
        ques_sentences_start_len = []
        ques_sentences_end_len = []
        question_start_len = []
        numbers_in_question = []
        question_number_indices = []
        question_number_words = []
        question_number_len = []
        question_tokens_replace = []
        question_curr_index = 0
        for question_word_token_index, question_word_token in enumerate(question_tokens):

            question_number = self.convert_word_to_number(question_word_token.text,True)
            question_wordpieces = self._bert_tokenizer.tokenize(question_word_token.text)
            for question_word_piece_index, question_word_piece in enumerate(question_wordpieces):
                if question_tokens_replace == [] and question_word_piece_index==0:
                    question_wordpieces[0] = Token(question_word_piece.text,0,text_id=0)
                elif question_tokens_replace != [] and question_word_piece_index==0:
                    question_wordpieces[0] = Token(question_word_piece.text,idx=question_tokens_replace[-1].idx + len(question_tokens_replace[-1].text) + 1,
                                                   text_id=question_tokens_replace[-1].text_id + 1)
                elif question_word_piece_index > 0:
                    question_wordpieces[question_word_piece_index] = Token(question_word_piece.text,
                                                                           question_wordpieces[question_word_piece_index-1].idx + len(question_wordpieces[question_word_piece_index-1].text),
                                                                           text_id=question_wordpieces[question_word_piece_index-1].text_id + 1)
            num_question_wordpieces = len(question_wordpieces)
            if '<q' in question_word_token.text and question_word_token.text != '<q>':
                ques_sentences_start_map.append(question_curr_index)
                ques_sentences_start_len.append(num_question_wordpieces)
            elif '</q' in question_word_token.text:
                ques_sentences_end_map.append(question_curr_index)
                ques_sentences_end_len.append(num_question_wordpieces)
            elif question_word_token.text == '<q>':
                question_start_map.append(question_curr_index)
                question_start_len.append(num_question_wordpieces)

            if question_number is not None:
                numbers_in_question.append(question_number)
                question_number_indices.append(question_curr_index)
                question_number_words.append(question_word_token.text)
                question_number_len.append(num_question_wordpieces)
            question_tokens_replace += question_wordpieces
            question_curr_index += num_question_wordpieces
        question_tokens = question_tokens_replace

        if self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        pp_graph = []
        for graph_id in range(passage_length + 1):
            if passage_length > 1:
                node_relation = [0] * (passage_length + 1)
                node_relation[graph_id + 1] = 1
                node_relation[graph_id] = 2
                pp_graph.append(node_relation)
                if graph_id == (passage_length - 2):
                    pp_graph.append([0] * (passage_length - 1) + [2] + [0])
                    pp_graph.append([0] * (passage_length + 1))
                    break
            else:
                pp_graph = [[2, 0], [0, 0]]

        qq_graph = []
        if (len(history) - 1) > 0:
            for graph_id in range(len(history) - 1):
                if (len(history) - 1) > 1:
                    node_relation = [1] * (len(history))
                    node_relation[graph_id] = 2
                    node_relation[-1] = 0
                    qq_graph.append(node_relation)
                    if graph_id == (len(history) - 1 - 2):
                        qq_graph.append([1] * (len(history) - 2) + [2] + [0])
                        qq_graph.append([0] * len(history))
                        break
                else:
                    qq_graph = [[2, 0], [0, 0]]

        else:
            qq_graph = [[0]]

        pq_graph_list = []
        p_len = len(pp_graph)
        q_len = len(qq_graph)

        if q_len == 1:
            pq_graph = np.zeros([p_len, q_len], int)
        else:
            for _ in range(0, p_len - 1):
                pq_graph_list.append([1] * (q_len - 1) + [0])
            pq_graph_list.append([0] * q_len)
            pq_graph = np.array(pq_graph_list)

        question_p_graph = np.ones([1, p_len], int)
        question_p_graph[0, p_len - 1] = 0
        question_q_graph = np.ones([1, q_len], int)
        question_q_graph[0, q_len - 1] = 0

        p_question_graph = question_p_graph.T
        q_question_graph = question_q_graph.T

        qp_graph = pq_graph.T
        pp_graph = np.array(pp_graph)
        qq_graph = np.array(qq_graph)

        if p_len > 2:
            for p_l in range(1, p_len - 1):
                pp_graph[p_l, p_l - 1] = 1
        question_node = np.array([[2]])

        graph = {
            "pp_graph": pp_graph,
            "qq_graph": qq_graph,
            "pq_graph": pq_graph,
            "qp_graph": qp_graph,
            "question_p_graph": question_p_graph,
            "question_q_graph": question_q_graph,
            "p_question_graph": p_question_graph,
            "q_question_graph": q_question_graph,
            "question_node": question_node
        }

        pp_graph_evidence = np.zeros(list(pp_graph.shape), int)
        pq_graph_evidence = np.zeros(list(pq_graph.shape), int)
        qq_graph_evidence = np.zeros(list(qq_graph.shape), int)
        p_question_evidence = np.zeros(list(p_question_graph.shape), int)
        q_question_evidence = np.zeros(list(q_question_graph.shape), int)
        if edge_list != []:
            for edge in edge_list:
                node_1 = edge[0]
                node_2 = edge[1]
                if node_2 < 1 and node_1 != q_len:
                    if node_2 > 0.1 and node_2 < 0.2:
                        pq_graph_evidence[int(node_2 * 100 - 2), int(node_1 - 1)] = 1
                    else:
                        pq_graph_evidence[int(node_2 * 10 - 1), int(node_1 - 1)] = 1
                elif node_2 >= 1 and node_1 != q_len:
                    qq_graph_evidence[int(node_1 - 1), int(node_2 - 1)] = 1
                    qq_graph_evidence[int(node_2 - 1), int(node_1 - 1)] = 1
                elif node_2 < 1 and node_1 == q_len:
                    if node_2 > 0.1 and node_2 < 0.2:
                        p_question_evidence[int(node_2 * 100 - 2), 0] = 1
                    else:
                        p_question_evidence[int(node_2 * 10 - 1), 0] = 1
                elif node_2 >= 1 and node_1 == q_len:
                    q_question_evidence[int(node_2 - 1), 0] = 1

        qp_graph_evidence = pq_graph_evidence.T
        question_p_evidence = p_question_evidence.T
        question_q_evidence = q_question_evidence.T
        question_node_evidence = np.zeros([1, 1], int)

        graph_evidence = {
            "pp_graph": pp_graph_evidence,
            "qq_graph": qq_graph_evidence,
            "pq_graph": pq_graph_evidence,
            "qp_graph": qp_graph_evidence,
            "question_p_graph": question_p_evidence,
            "question_q_graph": question_q_evidence,
            "p_question_graph": p_question_evidence,
            "q_question_graph": q_question_evidence,
            "question_node": question_node_evidence
        }

        plen = len(passage_tokens)
        qlen = len(question_tokens)

        question_passage_tokens = [Token('<s>',idx=0,text_id=0)] + [Token(token.text,idx=(token.idx+4),text_id=(token.text_id+1)) for token in question_tokens]
        end_question_token = question_passage_tokens[-1]
        question_passage_tokens += [Token('</s>',idx=(end_question_token.idx+len(end_question_token.text)+1),text_id=(end_question_token.text_id+1))]
        end_question_token = question_passage_tokens[-1]
        SEP_indices = [end_question_token.text_id]


        question_passage_tokens += [Token(token.text,idx=(token.idx+end_question_token.idx+5),
                                          text_id=(token.text_id+end_question_token.text_id+1)) for token in passage_tokens]
        if len(question_passage_tokens) > self.max_pieces - 1:
            question_passage_tokens = question_passage_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - qlen - 3]
            plen = len(passage_tokens)
            number_indices, number_len, numbers_in_passage = \
                clipped_passage_num(number_indices, number_len, numbers_in_passage, plen)
        end_question_token = question_passage_tokens[-1]
        question_passage_tokens += [Token('</s>',idx=(end_question_token.idx+len(end_question_token.text)+1),text_id=(end_question_token.text_id+1))]
        SEP_indices.append(question_passage_tokens[-1].text_id)
        SEP_indices.append(-1)
        number_indices = [index + qlen + 2 for index in number_indices] + [-1]
        number_len = number_len + [1]
        numbers_in_passage = numbers_in_passage + [0]
        number_tokens = [Token(str(number)) for number in numbers_in_passage]
        question_number_indices = [index + 1 for index in question_number_indices] + [-1]
        question_number_len = question_number_len + [1]
        numbers_in_question = numbers_in_question + [0]

        para_sentences_start_map = [index + qlen + 2 for index in para_sentences_start_map] + [-1]
        para_sentences_start_len = para_sentences_start_len + [1]
        para_sentences_end_map = [index + qlen + 2 for index in para_sentences_end_map] + [-1]
        para_sentences_end_len = para_sentences_end_len + [1]

        ques_sentences_start_map = [index + 1 for index in ques_sentences_start_map] + [-1]
        ques_sentences_start_len = ques_sentences_start_len + [1]
        ques_sentences_end_map = [index+2 for index in ques_sentences_end_map] + [-1]
        ques_sentences_end_len = ques_sentences_end_len + [1]

        question_start_map = [index + 1 for index in question_start_map]

        answer_type: str = None
        answer_texts: List[str] = []

        if answer_annotations:
            answer_type, answer_texts = self.extract_answer_info_from_annotation(
                answer_annotations[0], question_type
            )

        if answer_type == 'spans':
            answer_annotations[0] = {'spans': answer_texts}  # mark注意
        elif answer_type == 'generation':
            answer_annotations[0] = {'generation': answer_texts[:-1],'arithmetic_ans':answer_texts[-1]}


        tokenized_answer_texts = []

        if answer_type == 'generation':
            answer_texts.pop(-1)
            answer_texts[0] = ' '.join(split_equation(answer_texts[0], num_list))
            for answer_text in answer_texts:
                answer_tokens = self._tokenizer.tokenize(str(answer_text).lower())
                answer_tokens = split_tokens_by_hyphen(answer_tokens)
                answer_tokens_replace = []
                answer_curr_index = 0
                for answer_word_token_index, answer_word_token in enumerate(answer_tokens):
                    answer_wordpieces = self._bert_tokenizer.tokenize(answer_word_token.text)
                    for answer_word_piece_index, answer_word_piece in enumerate(answer_wordpieces):
                        if answer_tokens_replace == [] and answer_word_piece_index == 0:
                            answer_wordpieces[0] = Token(answer_word_piece.text, 0, text_id=0)
                        elif answer_tokens_replace != [] and answer_word_piece_index == 0:
                            answer_wordpieces[0] = Token(answer_word_piece.text,
                                                         idx=answer_tokens_replace[-1].idx + len(
                                                             answer_tokens_replace[-1].text) + 1,
                                                         text_id=answer_tokens_replace[-1].text_id + 1)
                        elif answer_word_piece_index > 0:
                            answer_wordpieces[answer_word_piece_index] = Token(answer_word_piece.text,
                                                                               idx=answer_wordpieces
                                                                                   [answer_word_piece_index - 1].idx +
                                                                                   len(answer_wordpieces
                                                                                       [
                                                                                           answer_word_piece_index - 1].text),
                                                                               text_id=answer_wordpieces
                                                                                       [
                                                                                           answer_word_piece_index - 1].text_id
                                                                                       + 1)
                    num_answer_wordpieces = len(answer_wordpieces)
                    answer_tokens_replace += answer_wordpieces
                    answer_curr_index += num_answer_wordpieces
                answer_tokens = answer_tokens_replace
                tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))
        else:
            for answer_text in answer_texts:
                answer_tokens = self._tokenizer.tokenize(str(answer_text).lower())
                answer_tokens = split_tokens_by_hyphen(answer_tokens)
                answer_tokens_replace = []
                answer_curr_index = 0
                for answer_word_token_index, answer_word_token in enumerate(answer_tokens):
                    answer_wordpieces = self._bert_tokenizer.tokenize(answer_word_token.text)
                    for answer_word_piece_index, answer_word_piece in enumerate(answer_wordpieces):
                        if answer_tokens_replace == [] and answer_word_piece_index==0:
                            answer_wordpieces[0] = Token(answer_word_piece.text,0,text_id=0)
                        elif answer_tokens_replace != [] and answer_word_piece_index==0:
                            answer_wordpieces[0] = Token(answer_word_piece.text,
                                                         idx=answer_tokens_replace[-1].idx + len(answer_tokens_replace[-1].text) + 1,
                                                         text_id = answer_tokens_replace[-1].text_id + 1)
                        elif answer_word_piece_index > 0:
                            answer_wordpieces[answer_word_piece_index] = Token(answer_word_piece.text,
                                                                               idx=answer_wordpieces
                                                                                   [answer_word_piece_index-1].idx +
                                                                                   len(answer_wordpieces
                                                                                       [answer_word_piece_index-1].text),
                                                                               text_id=answer_wordpieces
                                                                                       [answer_word_piece_index-1].text_id
                                                                                       + 1)
                    num_answer_wordpieces = len(answer_wordpieces)
                    answer_tokens_replace += answer_wordpieces
                    answer_curr_index += num_answer_wordpieces
                answer_tokens = answer_tokens_replace
                answer_annotations[0]['spans'] = [str(" ".join(token.text for token in answer_tokens))]
                tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))

        
        if self.instance_format == "drop":
            

            mask_indices = [0, qlen + 1, len(question_passage_tokens) - 1]
            
            valid_passage_spans = (
                self.find_valid_spans(passage_tokens, tokenized_answer_texts)
                if tokenized_answer_texts
                else []
            )
            for span_id, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_id] = (span[0]+qlen+2,span[1]+qlen+2)

            valid_question_spans = (
                self.find_valid_spans(question_tokens, tokenized_answer_texts)
                if tokenized_answer_texts
                else []
            )
            for span_id, span in enumerate(valid_question_spans):
                valid_question_spans[span_id] = (span[0]+1,span[1]+1)

            generation_mask_index = 0
            valid_generation = []
            if valid_question_spans == [] and valid_passage_spans == []:
                valid_generation = answer_tokens
                generation_mask_index = 1

            type_to_answer_map = {
                "passage_span": valid_passage_spans,
                "question_span": valid_question_spans,
                "generation": valid_generation,
            }

            if self.skip_when_all_empty and not any(
                    type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty
            ):
                return None
            answer_info = {
                "answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                "answer_passage_spans": valid_passage_spans,
                "answer_question_spans": valid_question_spans,
                "answer_as_generation": valid_generation,
            }
            if 'spans' in answer_annotations[0].keys():
                answer_annotations[0]['spans'][0] = str(answer_annotations[0]['spans'][0]).replace(' ', '').replace('▁',
                                                                                                                    ' ').strip()

            return self.make_marginal_drop_instance(
                passage_tokens,
                question_tokens,
                question_passage_tokens,
                answer_tokens,
                number_indices,
                number_len,
                question_number_indices,
                question_number_len,
                mask_indices,
                generation_mask_index,
                SEP_indices,
                para_sentences_start_map,
                para_sentences_end_map,
                ques_sentences_start_map,
                ques_sentences_end_map,
                question_start_map,
                para_sentences_start_len,
                para_sentences_end_len,
                ques_sentences_start_len,
                ques_sentences_end_len,
                question_start_len,
                graph,
                graph_evidence,
                self._token_indexers,
                question_text,
                passage_text,
                answer_info,
                additional_metadata={
                    "original_passage": passage_text,
                    "original_question": question_text,
                    "original_numbers": numbers_in_passage,
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "answer_info": answer_info,
                    "answer_annotations": answer_annotations,
                    "history_answer": history_answer,
                    "type": question_type,
                    "evidence_edges": edge_list
                },
            )
        else:
            raise ValueError(
                f'Expect the instance format to be "drop", "squad" or "bert", '
                f"but got {self.instance_format}"
            )

    @staticmethod
    def make_marginal_drop_instance(
            passage_tokens: List[Token],
            question_tokens: List[Token],
            question_passage_tokens: List[Token],
            answer_tokens: List[Token],
            number_indices: List[int],
            number_len: List[int],
            question_number_indices: List[int],
            question_number_len: List[int],
            mask_indices: List[int],
            generation_mask_index: int,
            SEP_indices: List[int],
            para_sentences_start_map: List[int],
            para_sentences_end_map: List[int],
            ques_sentences_start_map: List[int],
            ques_sentences_end_map: List[int],
            question_start_map: List[int],
            para_sentences_start_len: List[int],
            para_sentences_end_len: List[int],
            ques_sentences_start_len: List[int],
            ques_sentences_end_len: List[int],
            question_start_len: List[int],
            graph: Dict,
            graph_evidence: Dict,
            token_indexers: Dict[str, TokenIndexer],
            question_text: str,
            passage_text: str,
            answer_info: Dict[str, Any] = None,
            additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        question_passage_offsets = [(token.idx, token.idx + len(token.text)) for token in question_passage_tokens]

        # This is separate so we can reference it later with a known type.
        question_passage_field = TextField(question_passage_tokens, token_indexers)
        fields["question_passage"] = question_passage_field

        number_index_fields = \
            [ArrayField(np.arange(start_ind, start_ind + number_len[i]), padding_value=-1)
             for i, start_ind in enumerate(number_indices)]
        fields["number_indices"] = ListField(number_index_fields)

        question_number_index_fields = \
            [ArrayField(np.arange(start_ind, start_ind + question_number_len[i]), padding_value=-1)
             for i, start_ind in enumerate(question_number_indices)]
        fields["question_number_indices"] = ListField(question_number_index_fields)

        answer_tokens = [Token('<s>',idx=0,text_id=0)] + [Token(token.text,idx=token.idx+4,text_id=token.text_id+1) for token in answer_tokens]
        end_token = answer_tokens[-1]
        answer_tokens+=[Token('</s>',idx=end_token.idx+len(end_token.text)+1,text_id=end_token.text_id+1)]
        generation_fields = TextField(answer_tokens, token_indexers)
        generation_mask_index_fields: List[Field] = [IndexField(generation_mask_index, generation_fields)]
        fields["generation_mask_index"] = ListField(generation_mask_index_fields)
        mask_index_fields: List[Field] = [IndexField(index, question_passage_field) for index in mask_indices]
        fields["mask_indices"] = ListField(mask_index_fields)
        SEP_indices_fields: List[Field] = [IndexField(index, question_passage_field) for index in SEP_indices]
        fields['SEP_indices'] = ListField(SEP_indices_fields)

        para_start_index_fields: List[Field] = [
            IndexField(index, question_passage_field) for index in para_sentences_start_map
        ]
        fields['para_start_index'] = ListField(para_start_index_fields)

        para_end_index_fields: List[Field] = [
            IndexField(index, question_passage_field) for index in para_sentences_end_map
        ]
        fields['para_end_index'] = ListField(para_end_index_fields)

        ques_start_index_fields: List[Field] = [
            IndexField(index, question_passage_field) for index in ques_sentences_start_map
        ]
        fields['ques_start_index'] = ListField(ques_start_index_fields)

        ques_end_index_fields: List[Field] = [
            IndexField(index, question_passage_field) for index in ques_sentences_end_map
        ]
        fields['ques_end_index'] = ListField(ques_end_index_fields)

        question_start_index_fields: List[Field] = [
            IndexField(index, question_passage_field) for index in question_start_map
        ]
        fields['question_start_index'] = ListField(question_start_index_fields)

        pp_graph_array = np.array(graph['pp_graph'])
        pp_graph_fields = ArrayField(pp_graph_array, 0)
        fields["pp_graph"] = pp_graph_fields

        qq_graph_array = np.array(graph['qq_graph'])
        qq_graph_fields = ArrayField(qq_graph_array, 0)
        fields["qq_graph"] = qq_graph_fields

        pq_graph_array = np.array(graph['pq_graph'])
        pq_graph_fields = ArrayField(pq_graph_array, 0)
        fields["pq_graph"] = pq_graph_fields

        question_p_graph_array = np.array(graph['question_p_graph'])
        question_p_graph_fields = ArrayField(question_p_graph_array, 0)
        fields["question_p_graph"] = question_p_graph_fields

        question_q_graph_array = np.array(graph['question_q_graph'])
        question_q_graph_fields = ArrayField(question_q_graph_array, 0)
        fields["question_q_graph"] = question_q_graph_fields

        question_node_array = np.array(graph['question_node'])
        question_node_fields = ArrayField(question_node_array, 0)
        fields["question_node_graph"] = question_node_fields

        pp_graph_evidence_array = np.array(graph_evidence['pp_graph'])
        pp_graph_evidence_fields = ArrayField(pp_graph_evidence_array, 0)
        fields["pp_graph_evidence"] = pp_graph_evidence_fields

        qq_graph_evidence_array = np.array(graph_evidence['qq_graph'])
        qq_graph_evidence_fields = ArrayField(qq_graph_evidence_array, 0)
        fields["qq_graph_evidence"] = qq_graph_evidence_fields

        pq_graph_evidence_array = np.array(graph_evidence['pq_graph'])
        pq_graph_evidence_fields = ArrayField(pq_graph_evidence_array, 0)
        fields["pq_graph_evidence"] = pq_graph_evidence_fields

        question_p_graph_evidence_array = np.array(graph_evidence['question_p_graph'])
        question_p_graph_evidence_fields = ArrayField(question_p_graph_evidence_array, 0)
        fields["question_p_graph_evidence"] = question_p_graph_evidence_fields

        question_q_graph_evidence_array = np.array(graph_evidence['question_q_graph'])
        question_q_graph_evidence_fields = ArrayField(question_q_graph_evidence_array, 0)
        fields["question_q_graph_evidence"] = question_q_graph_evidence_fields

        question_node_evidence_array = np.array(graph_evidence['question_node'])
        question_node_evidence_fields = ArrayField(question_node_evidence_array, 0)
        fields["question_node_graph_evidence"] = question_node_evidence_fields
        metadata = {
            "original_passage": passage_text,
            "question_passage_text": "<s> "+question_text+" </s> "+passage_text + " </s>",
            "question_passage_token_offsets": question_passage_offsets,
            "question_tokens": [token.text for token in question_tokens],
            "passage_tokens": [token.text for token in passage_tokens],
            "number_indices": number_indices,
            "generation_tokens": [token.text for token in answer_tokens],
        }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]

            passage_span_fields: List[Field] = [
                SpanField(span[0], span[1], question_passage_field)
                for span in answer_info["answer_passage_spans"]
            ]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = [
                SpanField(span[0], span[1], question_passage_field)
                for span in answer_info["answer_question_spans"]
            ]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            fields["answer_as_generation"] = generation_fields

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields),len(question_passage_tokens)




    @staticmethod
    def extract_answer_info_from_annotation(
            answer_annotation: Dict[str, Any],
            question_type: List[int]
    ) -> Tuple[str, List[str]]:
        answer_type = None
        if question_type[0] == 6:
            # if question_type[1] == 0:  #mark
            answer_type = "spans"
            answer_annotation[answer_type] = answer_annotation['6']
        elif question_type[0] == 5:
            answer_type = "spans"
            answer_annotation[answer_type] = answer_annotation['5']
        elif question_type == [4,0]:
            answer_type = "generation"
            answer_annotation[answer_type] = answer_annotation['4']
        elif question_type[0] == 3:
            answer_type = "generation"
            answer_annotation[answer_type] = answer_annotation['3']
        elif question_type == [4, 1]:
            answer_type = "spans"
            answer_annotation[answer_type] = answer_annotation['4']
        elif question_type[0] == 2:
            answer_type = "spans"
            answer_annotation[answer_type] = answer_annotation['2']
        elif question_type[0] == 1:
            answer_type = "spans"
            answer_annotation[answer_type] = answer_annotation['1']

        answer_content = str(answer_annotation[answer_type]).lower() if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = [answer_content]

        elif answer_type == "generation":
            answer_texts = [answer_content]
            answer_texts.append(answer_annotation['arithmetic_ans'])
        return answer_type, answer_texts

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace("-", "")
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None

            if '/' in str(word):
                number = word
                return number
            if re.match('a\d',word):
                word = '<'+word+'>'
                return word
            '''
            try:
                number = word_to_num(word)
            except ValueError:
            '''
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    @staticmethod
    def find_valid_spans(
            passage_tokens: List[Token], answer_texts: List[str]
    ) -> List[Tuple[int, int]]:
        normalized_tokens = [
            token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens
        ]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []

        for answer_text in answer_texts:
            answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans


    @staticmethod
    def find_valid_add_sub_expressions(
        numbers: List[int], targets: List[int], max_number_of_numbers_to_consider: int = 2
    ) -> List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(
                enumerate(numbers), number_of_numbers_to_consider
            ):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    if eval_value in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = (
                                1 if sign == 1 else 2
                            )  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices
    # @staticmethod
    # def find_valid_generation(answer_tokens: List[Token]):

    @staticmethod
    def find_valid_generation(answer_tokens:List[Token]) -> List[Token]:
        Operation_list = ['+','-','*','/','^']
        valid_generation_bool = False
        for word_token in answer_tokens:
            if word_token.text in Operation_list:
                valid_generation_bool = True
        if valid_generation_bool:
            return answer_tokens
        else:
            return []

    def construct_graph(self, p_start_node: int, evidence_buffer: Dict) -> List[List]:
        edge_list = []
        evidence = evidence_buffer[p_start_node]
        for p_end in evidence:
            if p_end < 1.0:
                edge_list.append([p_start_node, p_end])
            else:
                edge_list.append([p_start_node, int(p_end)])
                edge_list.extend(self.construct_graph(int(p_end), evidence_buffer))
        return edge_list





