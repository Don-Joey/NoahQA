from typing import List

import jieba
import thulac
from overrides import overrides
from allennlp.data.tokenizers.token_class import Token
from common import WordSplitter

thu1 = thulac.thulac(seg_only=True)


@WordSplitter.register('thunlp')
class THUNLPSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses THUNLP's tokenizer. To Split Chinese sentences.
    simplify:Convert traditional characters to simplified characters
    filt:Remove meaningless words
    user_dict:a txt file, one word in a line.
    """

    def __init__(self):
        pass

    @overrides
    def split_words(self, sentence: str) -> List[Token]:

        offset = 0
        tokens = []

        words_sentence = thu1.cut(sentence, text=True).split(' ')
        # print(words_sentence)
        i = 0
        while i < len(words_sentence):
            word = words_sentence[i].strip()
            if not word:
                i+=1
                continue
            if word == '<' and (
                    words_sentence[i + 1].strip() == 'a' or words_sentence[i + 1].strip() == 'q9' or words_sentence[i + 1].strip() == 'p' or words_sentence[
                i + 1].strip() == 'q' or words_sentence[i + 1].strip() == 's' or words_sentence[i + 1].strip() == 't'):
                start = sentence.find(word, offset)
                if words_sentence[i + 2].strip() == '>':
                    word = ''.join(words_sentence[i:i + 3]).strip()
                    i += 2
                elif words_sentence[i + 3].strip() == '>':
                    word = ''.join(words_sentence[i:i + 4]).strip()
                    i += 3
                # word = word +  words_sentence[i+1].strip() + words_sentence[i+2].strip() + words_sentence[i+3].strip()
                tokens.append(Token(word, start))
                offset = start + len(word)
            elif word == "<" and words_sentence[i + 1].strip() == '/':
                start = sentence.find(word, offset)
                if words_sentence[i + 3].strip() == '>':
                    word = ''.join(words_sentence[i:i + 4]).strip()
                    i += 3
                elif words_sentence[i + 4].strip() == '>':
                    word = ''.join(words_sentence[i:i + 5]).strip()
                    i += 4
                tokens.append(Token(word, start))
                offset = start + len(word)
            else:
                start = sentence.find(word, offset)
                tokens.append(Token(word, start))

                offset = start + len(word)

            i += 1
        return tokens