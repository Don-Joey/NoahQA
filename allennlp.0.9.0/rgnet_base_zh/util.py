import json
import random

from collections import Counter, defaultdict
import logging
import string
from typing import Any, Dict, List, Tuple, Optional

from allennlp.data.fields import (
    Field,
    TextField,
    IndexField,
    MetadataField,
    LabelField,
    ListField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

def read_json(file):
    with open(file, 'rb') as f:
        data = json.load(f)
    return data

def write_json(data,file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def split_dataset(dataset,random_seed=2020):
    random.seed(random_seed)
    total_num = len(dataset.keys())
    data_list = []
    for i,j in dataset.items():
        data_list.append((i,j))
    random.shuffle(data_list)
    train_list = data_list[:int(total_num / 10 * 8)]
    valid_list = data_list[int(total_num / 10 * 8):int(total_num / 10 * 9)]
    test_list = data_list[int(total_num / 10 * 9):]
    train_dataset=dict(train_list)
    valid_dataset=dict(valid_list)
    test_dataset=dict(test_list)
    del data_list
    return train_dataset,valid_dataset,test_dataset

logger = logging.getLogger(__name__)

# These are tokens and characters that are stripped by the standard SQuAD and TriviaQA evaluation
# scripts.
IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


def normalize_text(text: str) -> str:
    """
    Performs a normalization that is very similar to that done by the normalization functions in
    SQuAD and TriviaQA.

    This involves splitting and rejoining the text, and could be a somewhat expensive operation.
    """
    return " ".join(
        [
            token
            for token in text.lower().strip(STRIPPED_CHARACTERS).split()
            if token not in IGNORED_TOKENS
        ]
    )


def char_span_to_token_span(
    token_offsets: List[Optional[Tuple[int, int]]], character_span: Tuple[int, int]
) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).

    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.

    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.

    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether the token spans match the input character spans exactly.  If this is ``False``, it
        means there was an error in either the tokenization or the annotated character span.
    """
    # We have token offsets into the passage from the tokenizer; we _should_ be able to just find
    # the tokens that have the same offsets as our span.
    error = False
    start_index = 0
    while start_index < len(token_offsets) and (
        token_offsets[start_index] is None or token_offsets[start_index][0] < character_span[0]
    ):
        start_index += 1
    if start_index >= len(token_offsets):
        raise ValueError(f"Character span %r outside the range of the given tokens.")
    # start_index should now be pointing at the span start index.
    if token_offsets[start_index][0] > character_span[0]:
        if start_index <= 0:
            raise ValueError(f"Character span %r outside the range of the given tokens.")
        # In this case, a tokenization or labeling issue made us go too far - the character span we're looking for
        # actually starts in the previous token. We'll back up one. Note that this might have us starting at a None
        # token.
        logger.debug("Bad labelling or tokenization - start offset doesn't match")
        start_index -= 1
    if token_offsets[start_index] is None or token_offsets[start_index][0] != character_span[0]:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and (
        token_offsets[end_index] is None or token_offsets[end_index][1] < character_span[1]
    ):
        end_index += 1
    if end_index >= len(token_offsets):
        raise ValueError(f"Character span %r outside the range of the given tokens.")
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def find_valid_answer_spans(
    passage_tokens: List[Token], answer_texts: List[str]
) -> List[Tuple[int, int]]:
    """
    Finds a list of token spans in ``passage_tokens`` that match the given ``answer_texts``.  This
    tries to find all spans that would evaluate to correct given the SQuAD and TriviaQA official
    evaluation scripts, which do some normalization of the input text.

    Note that this could return duplicate spans!  The caller is expected to be able to handle
    possible duplicates (as already happens in the SQuAD dev set, for instance).
    """
    normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
    # Because there could be many `answer_texts`, we'll do the most expensive pre-processing
    # step once.  This gives us a map from tokens to the position in the passage they appear.
    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        # For each answer, we'll first find all valid start positions in the passage.  Then
        # we'll grow each span to the same length as the number of answer tokens, and see if we
        # have a match.  We're a little tricky as we grow the span, skipping words that are
        # already pruned from the normalized answer text, and stopping early if we don't match.
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        num_answer_tokens = len(answer_tokens)
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index] == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans


def make_reading_comprehension_instance(
    question_tokens: List[Token],
    passage_tokens: List[Token],
    token_indexers: Dict[str, TokenIndexer],
    passage_text: str,
    token_spans: List[Tuple[int, int]] = None,
    answer_texts: List[str] = None,
    additional_metadata: Dict[str, Any] = None,
) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.

    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.

    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans : ``List[Tuple[int, int]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list because there might be several possible correct answer spans in the passage.
        Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
        annotations on the dev set; this will select the span that the most annotators gave as
        correct).
    answer_texts : ``List[str]``, optional
        All valid answer strings for the given question.  In SQuAD, e.g., the training set has
        exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
        possible answers, which are the aliases for the known correct entity.  This is put into the
        metadata for use with official evaluation scripts, but not used anywhere else.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields["passage"] = passage_field
    fields["question"] = TextField(question_tokens, token_indexers)
    metadata = {
        "original_passage": passage_text,
        "token_offsets": passage_offsets,
        "question_tokens": [token.text for token in question_tokens],
        "passage_tokens": [token.text for token in passage_tokens],
    }
    if answer_texts:
        metadata["answer_texts"] = answer_texts

    if token_spans:
        # There may be multiple answer annotations, so we pick the one that occurs the most.  This
        # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
        # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
        # all of the annotations.  This is why we have a separate official SQuAD metric calculation
        # (the "em" and "f1" metrics use the official script).
        candidate_answers: Counter = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        fields["span_start"] = IndexField(span_start, passage_field)
        fields["span_end"] = IndexField(span_end, passage_field)

    metadata.update(additional_metadata)
    fields["metadata"] = MetadataField(metadata)
    return Instance(fields)

def handle_cannot(reference_answers: List[str]):
    """
    Process a list of reference answers.
    If equal or more than half of the reference answers are "CANNOTANSWER", take it as gold.
    Otherwise, return answers that are not "CANNOTANSWER".
    """
    num_cannot = 0
    num_spans = 0
    for ref in reference_answers:
        if ref == "CANNOTANSWER":
            num_cannot += 1
        else:
            num_spans += 1
    if num_cannot >= num_spans:
        reference_answers = ["CANNOTANSWER"]
    else:
        reference_answers = [x for x in reference_answers if x != "CANNOTANSWER"]
    return reference_answers

def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]

def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~", '.']
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def split_equation(equation,num_list):
    equation.replace('pi', '3.14', 3).replace('PI', '3.14', 3)
    num_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%', '.', 'P', 'I','a','<','>']
    start = ''
    equ_list = []
    for char in equation:
        if char not in num_set:
            if start != '':
                equ_list.append(start)
            equ_list.append(char)
            start = ''
        else:
            start += char
        

    if start != '':
        equ_list.append(start)
    while ('km' in equ_list):
        equ_list.remove('km')
    while ' ' in equ_list:
        equ_list.remove(' ')
    for i,ip in enumerate(equ_list):
        if ip == '/':
            for num in num_list:
                if ''.join(equ_list[i-1:i+2]) in str(num):
                    #print(''.join(equ_list[i-1:i+2]))
                    equ_list.insert(i-1,''.join(equ_list[i-1:i+2]))
                    del equ_list[i]
                    del equ_list[i]
                    try:
                        del equ_list[i]
                    except IndexError:
                        equ_list[i-1]=equ_list[i-1][:(len(equ_list[i-1])-1)]
                        equ_list.insert(i,')')
                        print(equ_list)

    return equ_list

def split_equation_evuation(equation):
    equation.replace('pi', '3.14', 3).replace('PI', '3.14', 3)
    num_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%', '.', 'P', 'I', 'a', '<', '>']
    char_set = ['a','b','c','d','e','f','g','h','i','k','l','m','n','j','o','q','r','s','t','u','v','w','x','y','z']
    start = ''
    word_start = ''
    equ_list = []
    for char in equation:
        if char == ' ':
            continue
        if char not in num_set and char not in char_set:
            if start != '':
                equ_list.append(start)
            if word_start != '':
                equ_list.append(word_start)
            equ_list.append(char)
            start = ''
            word_start = ''
        elif char == 'a':
            if start != '' and start[-1] == '<':
                start += char
            else:
                word_start += char
        elif char in num_set:
            start+=char
        else:
            word_start+=char
    if start != '':
        equ_list.append(start)
    if word_start != '':
        equ_list.append(word_start)
    for equ_id,equ in enumerate(equ_list):
        if len(equ)>20:
            equ_list[equ_id] = equ[:21]
    return equ_list


def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
    for elem in equ_list:
        elem = str(elem)
        if elem == '(' or elem=='[' or elem=='{':
            stack.append('(')
        elif elem == ')' or elem==']' or elem=='}':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if stack == []:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            if elem=='Π':
                post_equ.append('3.14')
            elif elem=='π':
                post_equ.append('3.14')
            else:
                post_equ.append(elem)
            #if elem == 'PI':
            #    post_equ.append('3.14')
            #else:
            #    post_equ.append(elem)

    while stack != []:
        post_equ.append(stack.pop())
    return post_equ

