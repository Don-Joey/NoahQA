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
    TextField,
    MetadataField,
    LabelField,
    ListField,
    SequenceLabelField,
    SpanField,
    IndexField,
)
import itertools
from rgnet_base_en.fields_test import SourceTextField, SourceQuestionTextField, TargetTextField
from rgnet_base_en.custom_instance import SyncedFieldsInstance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rgnet_base_en.spacy_tokenizer import SpacyTokenizer

from rgnet_base_en.util import (
    IGNORED_TOKENS,
    STRIPPED_CHARACTERS,
    make_reading_comprehension_instance,
    split_tokens_by_hyphen,
)

logger = logging.getLogger(__name__)

AUGMENT = 'unknown | yes | no | do not know | ( ) { } [ ] + - * / ^'

WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}


@DatasetReader.register("comwp")
class ComwpReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = False,
            passage_length_limit: int = None,
            question_length_limit: int = None,
            skip_when_all_empty: List[str] = None,
            instance_format: str = "drop",
            relaxed_span_match_for_finding_labels: bool = True,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or SpacyTokenizer()

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.row_number = 1

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
            passage_info_list = []
            for token in passage_info['passage']:
                if token[1]:
                    passage_info_list.append(str(token[1]))
            passage_info_list.append(AUGMENT)
            passage_text = " ".join(passage_info_list)

            # passage_text = for passage_info["passage"]  \mark
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)

            num_list = [n_value for n_key, n_value in passage_info['num_list'].items()]
            history = []

            answer_history = []
            for qa_number, question_answer in enumerate(passage_info["qa_pairs"]):
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                answer_txt = {}
                if "answer" in question_answer:
                    answer = dict()
                    if question_answer["answer_type"] == [6, 1]:
                        question_answer["answer"] = 'unknown'
                        answer_txt = 'unknown'
                    answer[str(question_answer["answer_type"][0])] = str(question_answer["answer"]).lower()
                    if question_answer["answer_type"][0] == 6 or question_answer["answer_type"] == [2, 0] or \
                            question_answer["answer_type"][0] == 1 or question_answer["answer_type"] == [5, 0] or \
                            question_answer["answer_type"] == [4, 1]:

                        answer['spans'] = str(question_answer["answer"]).lower()
                        answer_txt = str(question_answer["answer"]).lower()
                    # if "answer_formula" in question_answer:

                    if question_answer["answer_type"][0] == 3 or question_answer["answer_type"] == [4, 0]:

                        try:
                            answer[str(question_answer["answer_type"][0])] = str(question_answer["answer_temp"]).lower()
                            answer['generation'] = str(question_answer['answer_temp']).lower()
                            answer_txt = str(question_answer['answer_temp']).lower()
                            answer_txt = ' '.join(split_equation(str(question_answer['answer_temp']).lower(), num_list))
                        except KeyError:
                            try:
                                answer[str(question_answer["answer_type"][0])] = str(
                                    question_answer["answer_formula"]).lower()
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

                instance = None
                if question_type[0] == 6 or question_type == [2, 0] or question_type[0] == 1 or question_type == [5,
                                                                                                                  0] or \
                        question_type[0] == 3 or question_type[0] == 4:
                    # question_type[0] == 3 or question_type[0] == 4:
                    # question_type[0] == 6 or question_type == [2, 0] or question_type[0] == 1 or question_type == [5, 0]:
                    # if question_type[0] != 1 and question_type[0] != 5 and question_type != [6,1] and question_type[0] != 2 and question_type[0] != 4\
                    #       and question_type[0] != 3:
                    instance = self.text_to_instance(
                        question_text,
                        passage_text,
                        question_id,
                        passage_id,
                        answer_annotations,
                        passage_tokens,
                        question_type,
                        history,
                        num_list,
                    )
                if instance is not None:
                    kept_count += 1
                    yield instance
                else:
                    skip_count += 1
            '''
            if 'train' in file_path and kept_count > 1000:
                print(1111111)
                break
            elif ('valid' in file_path or 'test' in file_path) and kept_count > 100:
                break
            '''
        print(kept_count, skip_count)
        # self.wb1.save('supplment_1.13.xlsx')
        logger.info(f"Skipped {skip_count} questions, kept {kept_count} questions.")

    @overrides
    def text_to_instance(
            self,  # type: ignore
            question_text: str,
            passage_text: str,
            question_id: str = None,
            passage_id: str = None,
            answer_annotations: List[Dict] = None,
            passage_tokens: List[Token] = None,
            question_type: List[int] = None,
            history: List[str] = None,
            num_list: List = None,
    ) -> Union[Instance, None]:

        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)

        question_tokens = self._tokenizer.tokenize(str(question_text).lower())
        question_tokens = split_tokens_by_hyphen(question_tokens)

        question_tokens.insert(0, self._tokenizer.tokenize('<q>')[0])
        question_text = '<q> ' + question_text
        history_answer = {}
        # print(question_tokens)
        # print('ob')
        if len(history) > 1:
            temp = []
            temp_txt = []
            for i in range(1, len(history)):
                d = i  # len(history) - i
                temp.append(self._tokenizer.tokenize('<q{}>'.format(d))[0])
                temp.extend(split_tokens_by_hyphen(self._tokenizer.tokenize(str(history[i - 1][0]).lower())))
                temp.append(self._tokenizer.tokenize('<a{}>'.format(d))[0])
                temp.extend(split_tokens_by_hyphen(self._tokenizer.tokenize(str(history[i - 1][1]).lower())))
                # temp.extend(self.tokenize_function(answer_history[i-1]))

                temp_txt.append('<q{}>'.format(d))
                temp_txt.append(str(history[i - 1][0]).lower())
                temp_txt.append('<a{}>'.format(d))
                history_answer['<a{}>'.format(d)] = str(history[i - 1][1]).lower()
                temp_txt.append(str(history[i - 1][1]).lower())
            question_text = " ".join(temp_txt) + ' ' + question_text
            temp += question_tokens
            question_tokens = temp

        # print('ob_1')
        if self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        answer_type: str = None
        answer_texts: List[str] = []

        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(
                answer_annotations[0], question_type
            )

        if answer_type == 'spans':
            answer_annotations[0] = {'spans': answer_texts}  # mark注意
        elif answer_type == 'generation':
            answer_annotations[0] = {'generation': answer_texts[:-1], 'arithmetic_ans': answer_texts[-1]}

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []

        if answer_type == 'generation':
            answer_texts.pop(-1)
            answer_texts[0] = ' '.join(split_equation(answer_texts[0], num_list))
            for answer_text in answer_texts:
                answer_tokens = self._tokenizer.tokenize(str(answer_text).lower())
                answer_tokens = split_tokens_by_hyphen(answer_tokens)
                tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))
        else:
            for answer_text in answer_texts:
                answer_tokens = self._tokenizer.tokenize(str(answer_text).lower())
                answer_tokens = split_tokens_by_hyphen(answer_tokens)
                tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))

        if self.instance_format == "drop":
            numbers_in_passage = []
            number_indices = []
            passage_text_coverage = ""
            for token_index, token in enumerate(passage_tokens):
                passage_text_coverage += token.text
                number = self.convert_word_to_number(token.text, True)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(token_index)
                if token_index >= (len(passage_tokens) - 11) and token.text in AUGMENT[-21:]:
                    numbers_in_passage.append(number)
                    number_indices.append(token_index)

            numbers_in_question = []
            question_number_indices = []
            for token_index, token in enumerate(question_tokens):
                number = self.convert_word_to_number(token.text, True)
                if number is not None:
                    numbers_in_question.append(number)
                    question_number_indices.append(token_index)

            # all_number = numbers_in_passage + numbers_in_question

            numbers_in_passage.append(0)
            number_indices.append(-1)
            # numbers_as_tokens = [Token(str(number)) for number in numbers_in_passage]

            numbers_in_question.append(0)
            question_number_indices.append(-1)
            # numbers_as_tokens = [Token(str(number)) for number in numbers_in_question]

            numbers_as_tokens = [Token(str(number)) for number in numbers_in_passage]

            # print(passage_tokens,tokenized_answer_texts)
            valid_passage_spans = (
                self.find_valid_spans(passage_tokens, tokenized_answer_texts)
                if tokenized_answer_texts
                else []
            )
            # print(valid_passage_spans)

            valid_question_spans = (
                self.find_valid_spans(question_tokens, tokenized_answer_texts)
                if tokenized_answer_texts
                else []
            )
            # print(valid_question_spans)

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

            return self.make_marginal_drop_instance(
                question_tokens,
                passage_tokens,
                answer_tokens,
                numbers_as_tokens,
                number_indices,
                question_number_indices,
                generation_mask_index,
                self._token_indexers,
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
                    "type": question_type
                },
            )
        else:
            raise ValueError(
                f'Expect the instance format to be "drop", "squad" or "bert", '
                f"but got {self.instance_format}"
            )

    @staticmethod
    def make_marginal_drop_instance(
            question_tokens: List[Token],
            passage_tokens: List[Token],
            answer_tokens: List[Token],
            number_tokens: List[Token],
            number_indices: List[int],
            question_number_indices: List[int],
            generation_mask_index: int,
            token_indexers: Dict[str, TokenIndexer],
            passage_text: str,
            answer_info: Dict[str, Any] = None,
            additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        passage_field = SourceTextField(passage_tokens, token_indexers)
        question_field = SourceQuestionTextField(question_tokens, token_indexers)

        fields["passage"] = passage_field
        fields["question"] = question_field

        number_index_fields: List[Field] = [
            IndexField(index, passage_field) for index in number_indices
        ]
        fields["number_indices"] = ListField(number_index_fields)

        question_number_index_fields: List[Field] = [IndexField(index, question_field) for index in
                                                     question_number_indices]
        fields["question_number_indices"] = ListField(question_number_index_fields)

        answer_tokens.insert(0, Token(START_SYMBOL))
        answer_tokens.append(Token(END_SYMBOL))

        generation_fields = TargetTextField(answer_tokens, token_indexers)
        generation_mask_index_fields: List[Field] = [IndexField(generation_mask_index, generation_fields)]
        fields["generation_mask_index"] = ListField(generation_mask_index_fields)

        # This field is actually not required in the model,
        # it is used to create the `answer_as_plus_minus_combinations` field, which is a `SequenceLabelField`.
        # We cannot use `number_indices` field for creating that, because the `ListField` will not be empty
        # when we want to create a new empty field. That will lead to error.
        numbers_in_passage_field = TextField(number_tokens, token_indexers)
        metadata = {
            "original_passage": passage_text,
            "passage_token_offsets": passage_offsets,
            "question_token_offsets": question_offsets,
            "question_tokens": [token.text for token in question_tokens],
            "passage_tokens": [token.text for token in passage_tokens],
            "number_tokens": [token.text for token in number_tokens],
            "number_indices": number_indices,
            "generation_tokens": [token.text for token in answer_tokens],
        }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]

            passage_span_fields: List[Field] = [
                SpanField(span[0], span[1], passage_field)
                for span in answer_info["answer_passage_spans"]
            ]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = [
                SpanField(span[0], span[1], question_field)
                for span in answer_info["answer_question_spans"]
            ]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            fields["answer_as_generation"] = generation_fields

        metadata.update(additional_metadata)
        # print(metadata)
        fields["metadata"] = MetadataField(metadata)

        return SyncedFieldsInstance(fields)

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
        elif question_type == [4, 0]:
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
            if re.match('a\d', word):
                word = '<' + word + '>'
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
    def find_valid_generation(answer_tokens: List[Token]) -> List[Token]:
        Operation_list = ['+', '-', '*', '/', '^']
        valid_generation_bool = False
        for word_token in answer_tokens:
            if word_token.text in Operation_list:
                valid_generation_bool = True
        if valid_generation_bool:
            return answer_tokens
        else:
            return []

