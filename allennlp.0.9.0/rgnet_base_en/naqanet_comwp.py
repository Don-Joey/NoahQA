from typing import Any, Dict, List, Optional
import logging
from overrides import overrides

import torch
import torch.nn as nn
import numpy

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from rgnet_base_en.drop_em_and_f1 import DropEmAndF1
from allennlp.data.dataset import Batch
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rgnet_base_en.CopyDecoder import Attention,CopyMechanism
from rgnet_base_en.compute_tool import compute_gen_ans

import torch.nn.functional as F

import random


logger = logging.getLogger(__name__)

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

@Model.register("comwp")
class NumericallyAugmentedQaNet(Model):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.

    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None) -> None:
        super().__init__(vocab, regularizer)


        #if answering_abilities is None:
        self.answering_abilities = ["passage_span_extraction", "question_span_extraction", "generation"] #edit
        self.PAD_ID = vocab.get_token_index(vocab._padding_token)
        self.vocab_size = vocab.get_vocab_size()
        #else:
        #self.answering_abilities = answering_abilities


        text_embed_dim = text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()
        modeling_out_dim = modeling_layer.get_output_dim()

        self.register_buffer("true_rep", torch.tensor(1.0))
        self.register_buffer("false_rep", torch.tensor(0.0))


        self._text_field_embedder = text_field_embedder

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        self._passage_weights_predictor = torch.nn.Linear(modeling_out_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(encoding_out_dim, 1)

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FeedForward(modeling_out_dim + encoding_out_dim,
                                                         activations=[Activation.by_name('relu')(),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[modeling_out_dim,
                                                                      len(self.answering_abilities)],
                                                         num_layers=2,
                                                         dropout=dropout_prob)
        

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = FeedForward(modeling_out_dim * 2,
                                                             activations=[Activation.by_name('relu')(),
                                                                          Activation.by_name('linear')()],
                                                             hidden_dims=[modeling_out_dim, 1],
                                                             num_layers=2)
            self._passage_span_end_predictor = FeedForward(modeling_out_dim * 2,
                                                           activations=[Activation.by_name('relu')(),
                                                                        Activation.by_name('linear')()],
                                                           hidden_dims=[modeling_out_dim, 1],
                                                           num_layers=2)

        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = FeedForward(modeling_out_dim * 2,
                                                              activations=[Activation.by_name('relu')(),
                                                                           Activation.by_name('linear')()],
                                                              hidden_dims=[modeling_out_dim, 1],
                                                              num_layers=2)
            self._question_span_end_predictor = FeedForward(modeling_out_dim * 2,
                                                            activations=[Activation.by_name('relu')(),
                                                                         Activation.by_name('linear')()],
                                                            hidden_dims=[modeling_out_dim, 1],
                                                            num_layers=2)

        

        if "generation" in self.answering_abilities:
            self._generation_index = self.answering_abilities.index("generation")
            self.encoding_out_dim =encoding_out_dim
            self.PAD_ID = vocab.get_token_index(vocab._padding_token)
            self.OOV_ID = vocab.get_token_index(vocab._oov_token)
            self.START_ID = vocab.get_token_index(START_SYMBOL)
            self.END_ID = vocab.get_token_index(END_SYMBOL)
            self.fuse_layer = nn.Sequential(
                nn.Linear(4*modeling_out_dim+modeling_out_dim+encoding_out_dim, 2*modeling_out_dim),
                nn.ReLU()
            )
            self.fuse_h_layer = nn.Sequential(
                nn.Linear(2 * encoding_out_dim, modeling_out_dim),
                nn.ReLU()
            ) #add
            self.attention_layer = Attention(3 * modeling_out_dim, 2 * modeling_out_dim, 2 * modeling_out_dim)
            #self.attention_layer=Attention(3*modeling_out_dim, 2*modeling_out_dim, 2*modeling_out_dim)
            self.copymech = CopyMechanism(modeling_out_dim, modeling_out_dim, encoding_in_dim)
            self.decoder_rnn = torch.nn.GRU(input_size=encoding_in_dim, hidden_size=2*modeling_out_dim,
                                             num_layers=1, batch_first=False, bidirectional=False)
            self.statenctx_to_prefinal = nn.Linear(5 * modeling_out_dim, modeling_out_dim, bias=True)
            #self.statenctx_to_prefinal = nn.Linear(5 * modeling_out_dim, modeling_out_dim, bias=True)
            self.project_to_decoder_input = nn.Linear(encoding_in_dim + 3 * modeling_out_dim, encoding_in_dim,
                                                      bias=True)
            #self.project_to_decoder_input = nn.Linear(encoding_in_dim + 3*modeling_out_dim, encoding_in_dim, bias=True)

            self.output_projector = torch.nn.Conv1d(modeling_out_dim, self.vocab_size, kernel_size=1, bias=True)
            self.softmax = nn.Softmax(dim=-1)

            self.input_encoder = nn.Sequential(
                torch.nn.GRU(input_size=encoding_in_dim, hidden_size=encoding_out_dim,
                              num_layers=1, batch_first=True, bidirectional=True),
            ) #add



        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                generation_mask_index: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_generation: Dict[str, torch.LongTensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        inp_passage_with_unks = passage["ids_with_unks"]
        inp_passage_with_oovs = passage["ids_with_oovs"]

        inp_question_with_unks = question["ids_with_unks"]
        inp_question_with_oovs = question["ids_with_oovs"]

        max_oovs = int(torch.max(question["num_oovs"]))


        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        #question_mask = torch.where(inp_question_with_unks!=0, self.true_rep, self.false_rep)
        #passage_mask = torch.where(inp_passage_with_unks!=0, self.true_rep, self.false_rep)
        answer_mask = util.get_text_field_mask(answer_as_generation).float()

        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))

        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        batch_size = embedded_question.size(0)


        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = masked_softmax(passage_question_similarity,
                                                    question_mask,
                                                    memory_efficient=True)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = masked_softmax(passage_question_similarity.transpose(1, 2),
                                                    passage_mask,
                                                    memory_efficient=True)

        question_passage_vectors = util.weighted_sum(encoded_passage, question_passage_attention)

        # Shape: (batch_size, passage_length, passage_length)
        passsage_attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = util.weighted_sum(encoded_passage, passsage_attention_over_attention)

        question_attention_over_attention = torch.bmm(question_passage_attention, passage_question_attention)
        question_question_vectors = util.weighted_sum(encoded_question, question_attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
                torch.cat([encoded_passage, passage_question_vectors,
                           encoded_passage * passage_question_vectors,
                           encoded_passage * passage_passage_vectors],
                          dim=-1))

        merged_question_attention_vectors = self._dropout(
            torch.cat([encoded_question, question_passage_vectors,
                       encoded_question * question_passage_vectors,
                       encoded_question * question_question_vectors],
                      dim=-1))

        modeled_question_list = [self._modeling_proj_layer(merged_question_attention_vectors)]
        for _ in range(4):
            modeled_question = self._dropout(self._modeling_layer(modeled_question_list[-1],question_mask))
            modeled_question_list.append(modeled_question)
        modeled_question_list.pop(0)

        # The recurrent modeling layers. Since these layers share the same parameters,
        # we don't construct them conditioned on answering abilities.
        modeled_passage_list = [self._modeling_proj_layer(merged_passage_attention_vectors)]
        for _ in range(4):
            modeled_passage = self._dropout(self._modeling_layer(modeled_passage_list[-1], passage_mask))
            modeled_passage_list.append(modeled_passage)
        # Pop the first one, which is input
        modeled_passage_list.pop(0)

        # The first modeling layer is used to calculate the vector representation of passage
        passage_weights = self._passage_weights_predictor(modeled_passage_list[0]).squeeze(-1)
        passage_weights = masked_softmax(passage_weights, passage_mask)
        passage_vector = util.weighted_sum(modeled_passage_list[0], passage_weights)
        # The vector representation of question is calculated based on the unmatched encoding,
        # because we may want to infer the answer ability only based on the question words.
        question_weights = self._question_weights_predictor(encoded_question).squeeze(-1)
        question_weights = masked_softmax(question_weights, question_mask)
        question_vector = util.weighted_sum(encoded_question, question_weights)


        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        if "passage_span_extraction" in self.answering_abilities:
            # Shape: (batch_size, passage_length, modeling_dim * 2))
            passage_for_span_start = torch.cat([modeled_passage_list[0], modeled_passage_list[1]], dim=-1)
            # Shape: (batch_size, passage_length)
            passage_span_start_logits = self._passage_span_start_predictor(passage_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            passage_for_span_end = torch.cat([modeled_passage_list[0], modeled_passage_list[2]], dim=-1)
            # Shape: (batch_size, passage_length)
            passage_span_end_logits = self._passage_span_end_predictor(passage_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)
            passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
            passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)
            #print("passage_span_start_log_probs",passage_span_start_log_probs.size(),passage_span_start_log_probs)
            #print("passage_span_end_log_probs", passage_span_end_log_probs.size(), passage_span_end_log_probs)

            # Info about the best passage span prediction
            passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
            passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)
            # Shape: (batch_size, 2)
            best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
            # Shape: (batch_size, 2)
            best_passage_start_log_probs = \
                torch.gather(passage_span_start_log_probs, 1, best_passage_span[:, 0].unsqueeze(-1)).squeeze(-1)
            best_passage_end_log_probs = \
                torch.gather(passage_span_end_log_probs, 1, best_passage_span[:, 1].unsqueeze(-1)).squeeze(-1)
            # Shape: (batch_size,)
            best_passage_span_log_prob = best_passage_start_log_probs + best_passage_end_log_probs
            if len(self.answering_abilities) > 1:
                best_passage_span_log_prob += answer_ability_log_probs[:, self._passage_span_extraction_index]

        if "question_span_extraction" in self.answering_abilities:
            # Shape: (batch_size, question_length)
            encoded_question_for_span_prediction = \
                torch.cat([encoded_question,
                           passage_vector.unsqueeze(1).repeat(1, encoded_question.size(1), 1)], -1)
            question_span_start_logits = \
                self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
            # Shape: (batch_size, question_length)
            question_span_end_logits = \
                self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
            #print("question_span_start_logits",question_span_start_logits)
            question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
            #print("question_span_start_log_probs",question_span_start_log_probs)
            question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)
            #print("question_span_start_log_probs", question_span_start_log_probs.size(), question_span_start_log_probs)
            #print("question_span_end_log_probs", question_span_end_log_probs.size(), question_span_end_log_probs)

            # Info about the best question span prediction
            question_span_start_logits = \
                util.replace_masked_values(question_span_start_logits, question_mask, -1e7)
            question_span_end_logits = \
                util.replace_masked_values(question_span_end_logits, question_mask, -1e7)
            # Shape: (batch_size, 2)
            best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)
            # Shape: (batch_size, 2)
            best_question_start_log_probs = \
                torch.gather(question_span_start_log_probs, 1, best_question_span[:, 0].unsqueeze(-1)).squeeze(-1)
            best_question_end_log_probs = \
                torch.gather(question_span_end_log_probs, 1, best_question_span[:, 1].unsqueeze(-1)).squeeze(-1)
            # Shape: (batch_size,)
            best_question_span_log_prob = best_question_start_log_probs + best_question_end_log_probs
            if len(self.answering_abilities) > 1:
                best_question_span_log_prob += answer_ability_log_probs[:, self._question_span_extraction_index]

        if "generation" in self.answering_abilities:
            #target_tensor = answer_as_generation["tokens"][:, 1:]
            number_indices = number_indices.squeeze(-1)
            number_mask = (number_indices != -1).long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            #encoded_passage_for_numbers = modeled_passage_list[0]
            encoded_passage_for_numbers = torch.cat([modeled_passage_list[0], modeled_passage_list[3]], dim=-1)
            #encoded_passage_for_numbers = inp_passage_enc_seq
            # Shape: (batch_size, # of numbers in the passage, encoding_dim)
            encoded_numbers = torch.gather(
                encoded_passage_for_numbers,
                1,
                clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))
            #print("encoded_numbers",encoded_numbers)
            passage_numbers = torch.gather(
                inp_passage_with_oovs,
                1,
                clamped_number_indices)
            clamped_passage_numbers = util.replace_masked_values(passage_numbers, number_mask, 0)
            passage_encoded_numbers = torch.cat(
                [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

            question_number_indices = question_number_indices.squeeze(-1)
            question_number_mask = (question_number_indices != -1).long()
            clamped_question_number_indices = util.replace_masked_values(question_number_indices, question_number_mask,0)
            #encoded_question_for_numbers = modeled_question_list[0]
            encoded_question_for_numbers = torch.cat([modeled_question_list[0],modeled_question_list[3]],-1)
            question_encoded_numbers = torch.gather(
                encoded_question_for_numbers,
                1,
                clamped_question_number_indices.unsqueeze(-1).expand(-1, -1, encoded_question_for_numbers.size(-1)))
            question_numbers = torch.gather(
                inp_question_with_oovs,
                1,
                clamped_question_number_indices)
            clamped_question_numbers = util.replace_masked_values(question_numbers, question_number_mask, 0)
            question_encoded_numbers = torch.cat(
                [question_encoded_numbers, question_vector.unsqueeze(1).repeat(1, question_encoded_numbers.size(1), 1)],
                -1)

            feed_tensor = {"ids_with_unks":answer_as_generation["ids_with_unks"][:, :-1],"token_characters":answer_as_generation["token_characters"][:, :-1,:]}

            passage_question_encoded_numbers = torch.cat((passage_encoded_numbers,question_encoded_numbers),1)

            x = torch.cat([clamped_passage_numbers, clamped_question_numbers], -1)
            #x = torch.cat([passage_numbers, question_numbers], -1) #mark.
            #x = torch.cat((inp_passage_with_oovs,inp_question_with_oovs),1)

            input_pad_mask = torch.where(x != 0, self.true_rep, self.false_rep)

            output_embedded = self._dropout(self._text_field_embedder(feed_tensor))
            output_embedded = self._highway_layer(self._embedding_proj_layer(output_embedded))

            # feed_tensor = answer_as_generation["tokens"][:, :-1]
            # seq_len*batch*embedded_size
            seqlen_first = output_embedded.permute(1, 0, 2)
            output_seq_len = seqlen_first.size(0)

            #new_add_part
            passage_encoded_hidden = torch.cat((encoded_passage_for_numbers[:, -1, :], passage_vector),dim=-1)
            question_encoded_hidden = torch.cat((encoded_question_for_numbers[:, -1, :], question_vector),dim=-1)
            #new_add_part
            encoder_hidden = torch.cat((passage_encoded_hidden, question_encoded_hidden),dim=-1).unsqueeze(0).contiguous()
            #encoder_hidden = torch.cat((encoded_passage_for_numbers[:, -1, :],encoded_question_for_numbers[:,-1,:]),dim=-1).unsqueeze(0).contiguous()
            encoder_hidden_fused = self.fuse_layer(encoder_hidden)

            #encoder_hidden_fused = torch.cat((last_passage_h_value,last_question_h_value),-1)
            #print(encoder_hidden.size())
            #encoder_hidden = passage_question_encoded_numbers[:, -1, :].unsqueeze(0).contiguous()

            decoder_hidden_state = encoder_hidden_fused
            decoder_hstates_batchfirst = decoder_hidden_state.permute(1, 0, 2)

            '''
            print("passage_question_encoded_numbers",passage_question_encoded_numbers.size())
            print("decoder_hstates_batchfirst",decoder_hstates_batchfirst.size())
            print("input_pad_mask",input_pad_mask.size())
            print("modeling_out_dim",)
            '''
            context_vector, _ = self.attention_layer(passage_question_encoded_numbers, decoder_hstates_batchfirst,
                                                     input_pad_mask)
            #print("context_vector:", context_vector)
            #print("context_vector_size", context_vector.size())
            step_generation_for_prob: List[torch.Tensor] = []
            coverages = [torch.zeros_like(x).type(torch.float).cuda()]
            step_generation_for_prediction: List[torch.Tensor] = []
            all_attn_weights = []
            for _i in range(output_seq_len):
                seqlen_first_onetimestep = seqlen_first[_i:_i + 1]  # shape is 1xbatchsizexembsize

                context_vector_seqlenfirst = context_vector.permute(1, 0, 2)  # seqlen is 1 always
                pre_input_to_decoder = torch.cat([seqlen_first_onetimestep, context_vector_seqlenfirst], dim=-1)
                #print("pre_input_to_decoder",pre_input_to_decoder.size())
                input_to_decoder = self.project_to_decoder_input(pre_input_to_decoder)  # shape is 1xbatchsizexembsize

                decoder_h_values, decoder_hidden_state = self.decoder_rnn(input_to_decoder, decoder_hidden_state)
                # decoder_h_values is shape 1XbatchsizeXhiddensize
                #print("decoder_h_value", decoder_h_values)
                #print("decoder_h_value", decoder_h_values.size())
                #print("decoder_hidden_state", decoder_hidden_state)
                decoder_h_values_batchfirst = decoder_h_values.permute(1, 0, 2)

                decoder_hstates_batchfirst = decoder_hidden_state.permute(1, 0, 2)

                # concatenated_decoder_states = torch.cat([decoder_cstates_batchfirst, decoder_hstates_batchfirst], dim=-1)
                #prev_coverage = coverages[-1]
                context_vector, attn_weights = self.attention_layer(passage_question_encoded_numbers, decoder_hstates_batchfirst,
                                                                    input_pad_mask)
                all_attn_weights.append(attn_weights.squeeze(1))

                #coverages.append(prev_coverage + attn_weights.squeeze(1))

                #print("context_vector_{}".format(_i), context_vector, context_vector.size())
                #print("attn_weights_{}".format(_i), attn_weights, attn_weights.size())
                #attn_weights = attn_weights.scatter_(2,prediction_after_copying.unsqueeze(1),0)
                decstate_and_context = torch.cat([decoder_h_values_batchfirst, context_vector],
                                                 dim=-1)  # batchsizeXdec_seqlenX3*hidden_size

                prefinal_tensor = self.statenctx_to_prefinal(decstate_and_context)
                seqlen_last = prefinal_tensor.permute(0, 2, 1)  # batchsizeXpre_output_dimXdec_seqlen
                #print("seqlen_last", seqlen_last, seqlen_first.size())
                logits = self.output_projector(seqlen_last)
                #print("logits", logits, logits.size())
                logits = logits.permute(0, 2, 1)  # batchXdec_seqlenXvocab

                # now executing copymechanism
                probs_after_copying = self.copymech(logits, attn_weights, decoder_hstates_batchfirst,
                                                    input_to_decoder.permute(1, 0, 2), context_vector, x, max_oovs)
                #print("probs_after_copying:",probs_after_copying,probs_after_copying.size())

                #probs_after_copying = probs_after_copying.scatter_(2,max_mask,-10)
                step_generation_for_prob.append(probs_after_copying)

            #print("step_generation_for_prob",step_generation_for_prob)
            generation_for_prob = torch.cat(step_generation_for_prob,1)
            best_for_generation = torch.argmax(generation_for_prob, -1)
            best_generation_log_probs = torch.gather(generation_for_prob, 2, best_for_generation.unsqueeze(-1)).squeeze(-1)
            best_generation_log_probs = util.replace_masked_values(best_generation_log_probs, answer_mask[:, 1:], -1e7)
            best_combination_generation_log_prob = best_generation_log_probs.sum(-1)
            if len(self.answering_abilities) > 1:
                best_combination_generation_log_prob += answer_ability_log_probs[:, self._generation_index]



            #[*4]

        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_generation is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    #print("gold_passage_span_starts",gold_passage_span_starts)
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    #print("gold_passage_span_ends",gold_passage_span_ends)
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    #print("gold_passage_span_starts",gold_passage_span_starts)
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()
                    #print("gold_passage_span_mask",gold_passage_span_mask)
                    #print("gold_passage_span_mask",gold_passage_span_mask)
                    clamped_gold_passage_span_starts = \
                        util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = \
                        util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = \
                        torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
                    #print("log_likelihood_for_passage_span_starts",log_likelihood_for_passage_span_starts.size(),log_likelihood_for_passage_span_starts)
                    log_likelihood_for_passage_span_ends = \
                        torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
                    #print("log_likelihood_for_passage_span_ends",log_likelihood_for_passage_span_ends.size(),log_likelihood_for_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = \
                        log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    #print("log_likelihood_for_passage_spans",log_likelihood_for_passage_spans.size(),log_likelihood_for_passage_spans)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = \
                        util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)
                    #print("masked_log_likelihood_for_passage_spans",log_likelihood_for_passage_spans.size(),log_likelihood_for_passage_spans)

                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
                    #print("log_marginal_likelihood_for_passage_span",log_marginal_likelihood_for_passage_span)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)
                    #print("log_marginal_likelihood_list",log_marginal_likelihood_list)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    #print("gold_question_span_starts",gold_question_span_starts)
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    #print("gold_question_span_ends",gold_question_span_ends)
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    #print("gold_question_span_mask",gold_question_span_mask)
                    clamped_gold_question_span_starts = \
                        util.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = \
                        util.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = \
                        torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
                    #print("log_likelihood_for_question_span_starts",log_likelihood_for_question_span_starts.size(),log_likelihood_for_question_span_starts)
                    log_likelihood_for_question_span_ends = \
                        torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
                    #print("log_likelihood_for_question_span_ends",log_likelihood_for_question_span_ends.size(),log_likelihood_for_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = \
                        log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    #print("log_likelihood_for_question_spans",log_likelihood_for_question_spans)
                    # For those padded spans, we set their log probabilities to be very small negative value

                    log_likelihood_for_question_spans = \
                        util.replace_masked_values(log_likelihood_for_question_spans,
                                                   gold_question_span_mask,
                                                   -1e7)
                    #print("masked_log_likelihood_for_question_spans",log_likelihood_for_question_spans)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = \
                        util.logsumexp(log_likelihood_for_question_spans)
                    #print("log_marginal_likelihood_for_question_span",log_marginal_likelihood_for_question_span)

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)
                    #print("log_marginal_likelihood_list",log_marginal_likelihood_list)
                elif answering_ability == 'generation':
                    target_tensor = answer_as_generation['ids_with_oovs'][:, 1:]
                    targets_tensor_seqfirst = target_tensor.permute(1, 0)
                    pad_mask = torch.where(targets_tensor_seqfirst != 0, self.true_rep, self.false_rep)
                    step_log_list = []
                    step_masked_log_list = []
                    # print("step_generation_for_prob:",step_generation_for_prob)
                    for _i in range(len(step_generation_for_prob)):
                        predicted_probs = step_generation_for_prob[_i].squeeze(1)
                        # print("predicted_probs",predicted_probs)
                        true_labels = targets_tensor_seqfirst[_i]
                        mask_labels = pad_mask[_i]
                        selected_probs = torch.gather(predicted_probs, 1, true_labels.unsqueeze(1))
                        selected_probs = selected_probs.squeeze(1)
                        # print("selected_probs:",selected_probs)
                        selected_neg_logprobs = torch.log(selected_probs)
                        # print("selected_neg_logprobs:",selected_neg_logprobs)
                        step_log_list.append(selected_neg_logprobs * mask_labels)

                        # step_masked_log_list.append(util.replace_masked_values(selected_neg_logprobs, mask_labels, -1e7))

                    # print("step_non_log_list:",step_non_log_list)
                    log_likelihood_for_generation = torch.stack(step_log_list, 1)
                    #print("log_likelihood_for_generation",log_likelihood_for_generation)
                    # log_marginal_likelihood_for_generation = torch.stack(step_masked_log_list, 1)
                    # log_marginal_likelihood_for_generation_1 = util.logsumexp(log_marginal_likelihood_for_generation, 1)

                    log_marginal_likelihood_for_generation_2 = log_likelihood_for_generation.sum(1,keepdim=True)
                    #print("log_marginal_likelihood_for_generation",log_marginal_likelihood_for_generation_2)
                    #print("generation_mask_index",generation_mask_index)
                    log_marginal_likelihood_for_generation_2 = util.replace_masked_values(log_marginal_likelihood_for_generation_2,generation_mask_index.squeeze(1),-1e7)
                    #print("masked_log_marginal_likelihood_for_generation", log_marginal_likelihood_for_generation_2)
                    # log_marginal_likelihood_for_generation_2 = torch.sum(log_likelihood_for_generation, -1)
                    log_marginal_likelihood_for_generation_2 = util.logsumexp(log_marginal_likelihood_for_generation_2)
                    # log_marginal_likelihood_for_generation = util.logsumexp(log_likelihood_for_generation)
                    #print("log_marginal_likelihood_for_generation",log_marginal_likelihood_for_generation_2)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_generation_2)
                    #print("log_marginal_likelihood_list",log_marginal_likelihood_list)
                    # true_labels = util.replace_masked_values(target_tensor,answer_mask[:,1:],0)
                    # log_likelihood_for_generation = torch.gather(generation_for_prob,2,true_labels.unsqueeze(-1))
                    # log_likelihood_for_generation = util.replace_masked_values(log_likelihood_for_generation,answer_mask[:,1:],-1e7)
                    # log_marginal_likelihood_for_generation = util.logsumexp(log_likelihood_for_generation)
                    # log_marginal_likelihood_list.append(log_marginal_likelihood_for_generation)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                #print("all_log_marginal_likelihoods",all_log_marginal_likelihoods.size(),all_log_marginal_likelihoods)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                #print("all_log_marginal_likelihoods", all_log_marginal_likelihoods.size(), all_log_marginal_likelihoods)
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
                #print("marginal_log_likelihood",marginal_log_likelihood.size(),marginal_log_likelihood)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]

            output_dict["loss"] = - marginal_log_likelihood.mean()
            #print("output_dict[loss]",output_dict["loss"])

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            #best_answer_ability_list =
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]

                answer_json: Dict[str, Any] = {}

                # We did not consider multi-mention answers here
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    passage_str = metadata[i]['original_passage']
                    offsets = metadata[i]['passage_token_offsets']
                    predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    predicted_answer = passage_str[start_offset:end_offset]
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = [(start_offset, end_offset)]
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    question_str = metadata[i]['original_question']
                    offsets = metadata[i]['question_token_offsets']
                    predicted_span = tuple(best_question_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    predicted_answer = question_str[start_offset:end_offset]
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = [(start_offset, end_offset)]
                elif predicted_ability_str == 'generation':
                    answer_json["answer_type"] = "generation"
                    predicted_indices = best_for_generation[i].detach().cpu().numpy()#output_dict["predictions"]
                    #pq=torch.cat([question['tokens'],passage['tokens']],-1)
                    indices = list(predicted_indices)#.gather(0,predicted_indices[i]))
                    if self.END_ID in indices:
                        indices = indices[:indices.index(self.END_ID)]
                    try:
                        predicted_tokens = [self.vocab.get_token_from_index(x, namespace="tokens") for x in indices]
                    except KeyError:
                        predicted_tokens = []
                        for x in indices:
                            try:
                                predicted_tokens.append(self.vocab.get_token_from_index(x))
                            except KeyError:
                                pass
                    answer_json["generation"] = ""
                    for token_id,token in enumerate(predicted_tokens):
                        if token in WORD_NUMBER_MAP:
                            token = str(WORD_NUMBER_MAP[token])
                            predicted_tokens[token_id] = token
                    answer_json["generation"] = " ".join(predicted_tokens)
                    predicted_answer = answer_json['generation'].strip().replace(' ','')
                    answer_json["generation"] = "".join([str(x) for x in indices])
                    #predicted_answer = str(answer_json["generation"])
                    #print("predicted_answer",predicted_answer)
                    answer_json["value"] = indices
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                answer_history = metadata[i].get('history_answer',{})
                answer_type = metadata[i].get("type", [])
                #answer_history = list(answer_history.values())
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations,answer_history,answer_type)
            # This is used for the demo.
            #output_dict["passage_question_attention"] = passage_question_attention
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens
        return output_dict
    def encode(self, inp):
        '''Get the encoding of input'''
        batch_size = inp.size(0)
        inp_seq_len = inp.size(1)
        inp_encoded = self.input_encoder(inp)
        output_seq = inp_encoded[0]
        h_value = inp_encoded[1]
        h_value_layerwise = h_value.reshape(1, 2, batch_size,self.encoding_out_dim)
        last_layer_h = h_value_layerwise[-1:, :, :, :]
        last_layer_h = last_layer_h.permute(0, 2, 1, 3).contiguous().view(1, batch_size, 2 * self.encoding_out_dim)
        last_layer_h_fused = self.fuse_h_layer(last_layer_h)
        '''
        h_value, c_value = inp_encoded[1]

        h_value_layerwise = h_value.reshape(self.num_encoder_layers, 2, batch_size,
                                            self.hidden_size)  # numlayersXbidirecXbatchXhid
        c_value_layerwise = c_value.reshape(self.num_encoder_layers, 2, batch_size,
                                            self.hidden_size)  # numlayersXbidirecXbatchXhid

        last_layer_h = h_value_layerwise[-1:, :, :, :]
        last_layer_c = c_value_layerwise[-1:, :, :, :]

        last_layer_h = last_layer_h.permute(0, 2, 1, 3).contiguous().view(1, batch_size, 2 * self.hidden_size)
        last_layer_c = last_layer_c.permute(0, 2, 1, 3).contiguous().view(1, batch_size, 2 * self.hidden_size)

        last_layer_h_fused = self.fuse_h_layer(last_layer_h)
        last_layer_c_fused = self.fuse_c_layer(last_layer_c)
        '''
        return output_seq, last_layer_h_fused

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score,supporting_ratio,supporting_total,judge_em,comp_em,open_em,unsure_em,extract_em,arithmetic_em,arithmetic_open_em, judge_f1,comp_f1,open_f1,unsure_f1,extract_f1,arithmetic_f1,arithmetic_open_f1 = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 
                'f1': f1_score, 
                'supporting_ritio':supporting_ratio,
                'supporting_total':supporting_total,
                'judge_em': judge_em,
                "judge_f1": judge_f1,
                "comp_em": comp_em,
                "comp_f1":comp_f1,
                "open_em": open_em,
                "open_f1": open_f1,
                "unsure_em": unsure_em,
                "unsure_f1": unsure_f1,
                "extract_em": extract_em,
                'extract_f1': extract_f1,
                'arithmetic_em': arithmetic_em,
                'arithmetic_f1': arithmetic_f1,
                'arithmetic_open_em': arithmetic_open_em,
                'arithmetic_open_f1': arithmetic_open_f1}

    def _get_loss(self,logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:

        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask, average= None)
    '''
    @overrides
    def forward_on_instance(self, instance: SyncedFieldsInstance) -> Dict[str, numpy.ndarray]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        return self.forward_on_instances([instance])[0]
    @overrides
    def forward_on_instances(self,
                             instances: List[SyncedFieldsInstance]) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.

        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.

        Returns
        -------
        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element

            if instance_separated_output[0]['answer']['answer_type'] == 'generation':
                output_words = []
                for _id in instance_separated_output[0]['answer']['value']:
                    if _id < self.vocab_size:
                        output_words.append(self.vocab.get_token_from_index(_id))
                    else:
                        output_words.append(instances[0].oov_list[_id - self.vocab_size])
                print("indices",instance_separated_output[0]['answer']['value'])
                print("output_words",output_words)
                if output_words != []:
                    instance_separated_output[0]['answer']['value']=" ".join(output_words)
            return instance_separated_output
    '''