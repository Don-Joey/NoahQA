from typing import Any, Dict, List, Optional
import logging
from overrides import overrides

import torch
import torch.nn as nn
import numpy

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from common import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from rgnet.evaluate.drop_em_and_f1 import DropEmAndF1
from allennlp.data import Batch
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rgnet.model.CopyDecoder import Attention,CopyMechanism
from rgnet.utils.compute_tool import compute_gen_ans
from rgnet.model.gnn import CharmGNN

from transformers import RobertaTokenizer,RobertaModel,RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from transformers import BertTokenizer,BertModel,BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

import torch.nn.functional as F

import random


logger = logging.getLogger(__name__)


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

@Model.register("comwp-roberta-edge")
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
                 roberta_pretrained_model: str,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None) -> None:
        super().__init__(vocab, regularizer)


        #if answering_abilities is None:
        self.answering_abilities = ["passage_span_extraction", "question_span_extraction", "generation"] #edit

        self._bert_tokenizer = BertTokenizer(vocab_file='/home/leiwang/naqanet_generation_cn/roberta_zh/vocab.txt')#RobertaTokenizer.from_pretrained(roberta_pretrained_model)
        self.vocab_size = self._bert_tokenizer.vocab_size

        Roberta_Config = BertConfig.from_json_file('/home/leiwang/naqanet_generation_cn/roberta_zh/bert_config_large.json')#.from_pretrained(roberta_pretrained_model)
        self.BERT = BertModel.from_pretrained('/home/leiwang/naqanet_generation_cn/roberta_zh/roberta_zh_large_model.ckpt.index', from_tf=True, config=Roberta_Config)

        self.dropout = dropout_prob

        self.register_buffer("true_rep", torch.tensor(1.0))
        self.register_buffer("false_rep", torch.tensor(0.0))

        bert_dim = 1024

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)

        self._gnn = CharmGNN(2 * bert_dim, 3)
        self._proj_fc = torch.nn.Linear(bert_dim * 2, bert_dim, bias=True)

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = torch.nn.Sequential(torch.nn.Linear(2*bert_dim, bert_dim),
                                                                   torch.nn.ReLU(),
                                                                   torch.nn.Dropout(self.dropout),
                                                                   torch.nn.Linear(bert_dim, len(self.answering_abilities)))
        

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = torch.nn.Linear(bert_dim,1)
            self._passage_span_end_predictor = torch.nn.Linear(bert_dim,1)

        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = torch.nn.Sequential(torch.nn.Linear(2*bert_dim, bert_dim),
                                                                       torch.nn.ReLU(),
                                                                       torch.nn.Dropout(self.dropout),
                                                                       torch.nn.Linear(bert_dim, 1))
            self._question_span_end_predictor = torch.nn.Sequential(torch.nn.Linear(2*bert_dim, bert_dim),
                                                                       torch.nn.ReLU(),
                                                                       torch.nn.Dropout(self.dropout),
                                                                       torch.nn.Linear(bert_dim, 1))




        if "generation" in self.answering_abilities:
            self.embedded = BertEmbeddings(Roberta_Config)
            self._generation_index = self.answering_abilities.index("generation")
            self.encoding_out_dim =bert_dim
            self.fuse_layer = nn.Sequential(
                nn.Linear(2*bert_dim, bert_dim),
                nn.ReLU()
            )
            self.fuse_h_layer = nn.Sequential(
                nn.Linear(2 * bert_dim, bert_dim),
                nn.ReLU()
            ) #add
            self.attention_layer = Attention(2*bert_dim, bert_dim, bert_dim)

            self.copymech = CopyMechanism(bert_dim, bert_dim, bert_dim)
            self.decoder_rnn = torch.nn.GRU(input_size=bert_dim, hidden_size=bert_dim,
                                             num_layers=1, batch_first=False, bidirectional=False)
            self.statenctx_to_prefinal = nn.Linear(3 * bert_dim, bert_dim, bias=True)
            self.project_to_decoder_input = nn.Linear(bert_dim + bert_dim + bert_dim, bert_dim,
                                                      bias=True)

            self.output_projector = torch.nn.Conv1d(bert_dim, self.vocab_size, kernel_size=1, bias=True)
            self.softmax = nn.Softmax(dim=-1)

            self.input_encoder = nn.Sequential(
                torch.nn.GRU(input_size=bert_dim, hidden_size=bert_dim,
                              num_layers=1, batch_first=True, bidirectional=True),
            ) #add



        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        initializer(self)

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                generation_mask_index: torch.LongTensor,
                mask_indices: torch.LongTensor,
                SEP_indices: torch.LongTensor,
                para_start_index: torch.LongTensor,
                para_end_index: torch.LongTensor,
                ques_start_index: torch.LongTensor,
                ques_end_index: torch.LongTensor,
                question_start_index: torch.LongTensor,
                pp_graph: torch.LongTensor,
                qq_graph: torch.LongTensor,
                pq_graph: torch.LongTensor,
                question_p_graph: torch.LongTensor,
                question_q_graph: torch.LongTensor,
                question_node_graph: torch.LongTensor,
                pp_graph_evidence: torch.LongTensor,
                qq_graph_evidence: torch.LongTensor,
                pq_graph_evidence: torch.LongTensor,
                question_p_graph_evidence: torch.LongTensor,
                question_q_graph_evidence: torch.LongTensor,
                question_node_graph_evidence: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_generation: Dict[str, torch.LongTensor] = None,
                mask_test_indices: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        question_passage_tokens = question_passage["tokens"]
        pad_mask = question_passage["mask"]
        seqlen_ids = question_passage["tokens-type-ids"]

        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]

        mask = mask_indices.squeeze(-1)

        cls_sep_mask = \
            torch.ones(pad_mask.shape, device=pad_mask.device).long().scatter(1, mask, torch.zeros(mask.shape,
                                                                                                   device=mask.device).long())
        passage_mask = seqlen_ids * pad_mask * cls_sep_mask
        # Shape: (batch_size, seqlen)
        question_mask = (1 - seqlen_ids) * pad_mask * cls_sep_mask
        bert_out = self.BERT(input_ids=question_passage_tokens,attention_mask =pad_mask, token_type_ids = seqlen_ids).last_hidden_state#, token_type_ids=seqlen_ids)

        question_end = max(mask[:,1])
        question_out = bert_out[:,:question_end]
        question_mask = question_mask[:, :question_end]
        
        
        q_node_start_indices = ques_start_index.squeeze(-1)#[:,:,0].squeeze(-1).long()
        q_node_start_mask = (q_node_start_indices > -1).bool()#long()
        clamped_q_node_start_indices = util.replace_masked_values(q_node_start_indices, q_node_start_mask, 0)
        q_node_end_indices = ques_end_index.squeeze(-1)#[:,:,0].squeeze(-1).long()
        q_node_end_mask = (q_node_end_indices > -1).bool()#long()
        clamped_q_node_end_indices = util.replace_masked_values(q_node_end_indices, q_node_end_mask, 0)

        encoded_question_for_nodes = question_out
        question_encoded_start_nodes = torch.gather(
            encoded_question_for_nodes,
            1,
            clamped_q_node_start_indices.unsqueeze(-1).expand(-1, -1, encoded_question_for_nodes.size(-1)))
        question_encoded_end_nodes = torch.gather(
            encoded_question_for_nodes,
            1,
            clamped_q_node_end_indices.unsqueeze(-1).expand(-1, -1, encoded_question_for_nodes.size(-1)))

        question_node_indice = question_start_index.squeeze(-1)#[:,:,0].squeeze(-1).long()
        question_node_mask = (question_node_indice > -1).bool()#long()
        clamped_question_indice = util.replace_masked_values(question_node_indice, question_node_mask, 0)#.unsqueeze(-1)
        question_for_node = torch.gather(
            encoded_question_for_nodes,
            1,
            clamped_question_indice.unsqueeze(-1).expand(-1, -1, encoded_question_for_nodes.size(-1))
        )
        

        SEP_indices = SEP_indices.squeeze(-1)
        SEP_mask = (SEP_indices != -1).bool()#long()
        clamped_SEP_indices = util.replace_masked_values(SEP_indices, SEP_mask, 0)
        encoded_SEP = torch.gather(
            bert_out,
            1,
            clamped_SEP_indices.unsqueeze(-1).expand(-1, -1, bert_out.size(-1))
        )
        SEP_1 = encoded_SEP[:,0,:]
        SEP_2 = encoded_SEP[:,1,:]
        passage_out = bert_out
        del bert_out
        
        p_node_start_indices = para_start_index.squeeze(-1)#[:,:,0].squeeze(-1).long()
        p_node_start_mask = (p_node_start_indices > -1).bool()#long()
        clamped_p_node_start_indices = util.replace_masked_values(p_node_start_indices, p_node_start_mask, 0)
        p_node_end_indices = para_end_index.squeeze(-1)#[:,:,0].squeeze(-1).long()
        p_node_end_mask = (p_node_end_indices > -1).bool()#long()
        clamped_p_node_end_indices = util.replace_masked_values(p_node_end_indices, p_node_end_mask, 0)

        encoded_passage_for_nodes = passage_out
        encoded_start_nodes = torch.gather(
            encoded_passage_for_nodes,
            1,
            clamped_p_node_start_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_nodes.size(-1)))
        encoded_end_nodes = torch.gather(
            encoded_passage_for_nodes,
            1,
            clamped_p_node_end_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_nodes.size(-1)))

        p_node = torch.cat([encoded_start_nodes, encoded_end_nodes], -1)
        q_node = torch.cat([question_encoded_start_nodes, question_encoded_end_nodes], -1)
        question_node = torch.cat([question_for_node, question_for_node], -1)

        p_node, q_node, question_node, edge_loss, y_pred_edges = self._gnn(
            p_node=p_node,
            q_node=q_node,
            question_node=question_node,
            p_node_mask=p_node_start_mask,
            q_node_mask=q_node_start_mask,
            question_node_mask=question_node_mask,
            pp_graph=pp_graph,
            qq_graph=qq_graph,
            pq_graph=pq_graph,
            question_p_graph=question_p_graph,
            question_q_graph=question_q_graph,
            question_node_graph=question_node_graph,
            pp_graph_evidence=pp_graph_evidence,
            qq_graph_evidence=qq_graph_evidence,
            pq_graph_evidence=pq_graph_evidence,
            question_p_graph_evidence=question_p_graph_evidence,
            question_q_graph_evidence=question_q_graph_evidence,
            question_node_graph_evidence=question_node_graph_evidence,
        )

        y_edges = F.softmax(y_pred_edges, dim=3).argmax(dim=3)

        passage_start_nodes = p_node  # [:,:,:self.encoding_out_dim]
        passage_end_nodes = p_node  # [:,:,self.encoding_out_dim:]
        question_start_nodes = q_node
        question_end_nodes = q_node
        question_start_node = question_node  # [:,:,:self.encoding_out_dim]

        gnn_info_passage_vec = torch.zeros(
            (batch_size, passage_out.size(1) + 1, passage_out.size(-1)), dtype=torch.float,
            device=passage_start_nodes.device)
        gnn_info_question_vec = torch.zeros(
            (batch_size, question_out.size(1) + 1, question_out.size(-1)), dtype=torch.float,
            device=question_start_node.device)
        clamped_p_node_start_indices = util.replace_masked_values(p_node_start_indices, p_node_start_mask,
                                                                  gnn_info_passage_vec.size(1) - 1)
        clamped_p_node_end_indices = util.replace_masked_values(p_node_end_indices, p_node_end_mask,
                                                                gnn_info_passage_vec.size(1) - 1)

        clamped_q_node_start_indices = util.replace_masked_values(q_node_start_indices, q_node_start_mask,
                                                                  gnn_info_question_vec.size(1) - 1)
        clamped_q_node_end_indices = util.replace_masked_values(q_node_end_indices, q_node_end_mask,
                                                                gnn_info_question_vec.size(1) - 1)
        clamped_question_indice = util.replace_masked_values(question_node_indice, question_node_mask,
                                                             gnn_info_question_vec.size(1) - 1)#.unsqueeze(-1)

        gnn_info_passage_vec.scatter_(1, clamped_p_node_start_indices.unsqueeze(-1).expand(-1, -1,
                                                                                           passage_start_nodes.size(
                                                                                               -1)),
                                      passage_start_nodes)
        gnn_info_passage_vec.scatter_(1, clamped_p_node_end_indices.unsqueeze(-1).expand(-1, -1,
                                                                                         passage_end_nodes.size(-1)),
                                      passage_end_nodes)
        gnn_info_question_vec.scatter_(1, clamped_q_node_start_indices.unsqueeze(-1).expand(-1, -1,
                                                                                            question_start_nodes.size(
                                                                                                -1)),
                                       question_start_nodes)
        gnn_info_question_vec.scatter_(1, clamped_q_node_end_indices.unsqueeze(-1).expand(-1, -1,
                                                                                          question_end_nodes.size(-1)),
                                       question_end_nodes)
        gnn_info_question_vec.scatter_(1, clamped_question_indice.unsqueeze(-1).expand(-1, -1,
                                                                                       question_start_node.size(-1)),
                                       question_start_node)

        gnn_info_passage_vec = gnn_info_passage_vec[:, :-1, :]
        gnn_info_question_vec = gnn_info_question_vec[:, :-1, :]
        
        question_out = self._proj_fc(torch.cat((question_out, gnn_info_question_vec), dim=-1))
        passage_out = self._proj_fc(torch.cat((passage_out, gnn_info_passage_vec), dim=-1))

        
        
        question_weights = self._question_weights_predictor(question_out).squeeze(-1)
        question_weights = masked_softmax(question_weights, question_mask)
        question_vector = util.weighted_sum(question_out, question_weights)

        passage_weights = self._passage_weights_predictor(passage_out).squeeze(-1)
        passage_weights = masked_softmax(passage_weights, passage_mask)
        passage_vector = util.weighted_sum(passage_out, passage_weights)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)


        if "passage_span_extraction" in self.answering_abilities:

            passage_span_start_logits = self._passage_span_start_predictor(passage_out).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)

            passage_span_end_logits = self._passage_span_end_predictor(passage_out).squeeze(-1)
            # Shape: (batch_size, passage_length)
            passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
            passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

            # Info about the best passage span prediction
            passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask.bool(), -1e7)
            passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask.bool(), -1e7)
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
                torch.cat([question_out,
                           passage_vector.unsqueeze(1).repeat(1, question_out.size(1), 1)], -1)
            question_span_start_logits = \
                self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
            # Shape: (batch_size, question_length)
            question_span_end_logits = \
                self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
            question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
            question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)

            # Info about the best question span prediction
            question_span_start_logits = \
                util.replace_masked_values(question_span_start_logits, question_mask.bool(), -1e7)
            question_span_end_logits = \
                util.replace_masked_values(question_span_end_logits, question_mask.bool(), -1e7)
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

            number_indices = number_indices[:,:,0].long()
            number_mask = (number_indices != -1).bool()#long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            encoded_passage_for_numbers = passage_out
            # Shape: (batch_size, # of numbers in the passage, encoding_dim)
            encoded_numbers = torch.gather(
                encoded_passage_for_numbers,
                1,
                clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))
            passage_numbers = torch.gather(
                question_passage_tokens,
                1,
                clamped_number_indices)
            #print("passage_number",passage_number)
            clamped_passage_numbers = util.replace_masked_values(passage_numbers, number_mask, 0)

            passage_encoded_numbers = torch.cat(
                [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

            question_number_indices = question_number_indices[:,:,0].long()
            question_number_mask = (question_number_indices != -1).bool()#long()
            clamped_question_number_indices = util.replace_masked_values(question_number_indices, question_number_mask,0)
            encoded_question_for_numbers = question_out
            question_encoded_numbers = torch.gather(
                encoded_question_for_numbers,
                1,
                clamped_question_number_indices.unsqueeze(-1).expand(-1, -1, encoded_question_for_numbers.size(-1)))
            question_numbers = torch.gather(
                question_passage_tokens,
                1,
                clamped_question_number_indices)
            clamped_question_numbers = util.replace_masked_values(question_numbers, question_number_mask, 0)

            question_encoded_numbers = torch.cat(
                [question_encoded_numbers, question_vector.unsqueeze(1).repeat(1, question_encoded_numbers.size(1), 1)],
                -1)

            feed_tensor = {"tokens":answer_as_generation['tokens'][:,:-1],
                           "mask":answer_as_generation['mask'][:,:-1],
                           "tokens-type-ids" : answer_as_generation["tokens-type-ids"][:,:-1]}


            passage_question_encoded_numbers = torch.cat((passage_encoded_numbers,question_encoded_numbers),1)

            x = torch.cat([clamped_passage_numbers, clamped_question_numbers], -1) #..................................

            input_pad_mask = torch.where(x != 0, self.true_rep, self.false_rep)

            #decoder_input
            output_embedded = self.embedded(input_ids=answer_as_generation['tokens'][:, :-1],
                                            token_type_ids=answer_as_generation['tokens-type-ids'][:, :-1])#self.BERT(input_ids=answer_as_generation['tokens'],

            seqlen_first = output_embedded.permute(1, 0, 2)
            output_seq_len = seqlen_first.size(0)

            encoder_hidden = torch.cat((SEP_1,SEP_2),dim=-1).unsqueeze(0).contiguous()
            encoder_hidden_fused = self.fuse_layer(encoder_hidden)

            decoder_hidden_state = encoder_hidden_fused
            decoder_hstates_batchfirst = decoder_hidden_state.permute(1, 0, 2)

            context_vector, _ = self.attention_layer(passage_question_encoded_numbers, decoder_hstates_batchfirst,
                                                     input_pad_mask)
            step_generation_for_prob: List[torch.Tensor] = []
            coverages = [torch.zeros_like(x).type(torch.float).cuda()]
            step_generation_for_prediction: List[torch.Tensor] = []
            all_attn_weights = []
            for _i in range(output_seq_len):
                seqlen_first_onetimestep = seqlen_first[_i:_i + 1]  # shape is 1xbatchsizexembsize

                context_vector_seqlenfirst = context_vector.permute(1, 0, 2)  # seqlen is 1 always
                pre_input_to_decoder = torch.cat([seqlen_first_onetimestep, context_vector_seqlenfirst], dim=-1)

                input_to_decoder = self.project_to_decoder_input(pre_input_to_decoder)  # shape is 1xbatchsizexembsize

                decoder_h_values, decoder_hidden_state = self.decoder_rnn(input_to_decoder, decoder_hidden_state)
                # decoder_h_values is shape 1XbatchsizeXhiddensize
                decoder_h_values_batchfirst = decoder_h_values.permute(1, 0, 2)

                decoder_hstates_batchfirst = decoder_hidden_state.permute(1, 0, 2)

                context_vector, attn_weights = self.attention_layer(passage_question_encoded_numbers, decoder_hstates_batchfirst,
                                                                    input_pad_mask)
                all_attn_weights.append(attn_weights.squeeze(1))
                decstate_and_context = torch.cat([decoder_h_values_batchfirst, context_vector],
                                                 dim=-1)  # batchsizeXdec_seqlenX3*hidden_size

                prefinal_tensor = self.statenctx_to_prefinal(decstate_and_context)
                seqlen_last = prefinal_tensor.permute(0, 2, 1)  # batchsizeXpre_output_dimXdec_seqlen
                logits = self.output_projector(seqlen_last)
                logits = logits.permute(0, 2, 1)  # batchXdec_seqlenXvocab

                # now executing copymechanism
                probs_after_copying = self.copymech(logits, attn_weights, decoder_hstates_batchfirst,
                                                    input_to_decoder.permute(1, 0, 2), context_vector, x)
                step_generation_for_prob.append(probs_after_copying)

            generation_for_prob = torch.cat(step_generation_for_prob,1)
            best_for_generation = torch.argmax(generation_for_prob, -1)
            best_generation_log_probs = torch.gather(generation_for_prob, 2, best_for_generation.unsqueeze(-1)).squeeze(-1)
            best_generation_log_probs = util.replace_masked_values(best_generation_log_probs, answer_as_generation['mask'][:,1:].bool(), -1e7)
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
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).bool()#.long()
                    clamped_gold_passage_span_starts = \
                        util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = \
                        util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = \
                        torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = \
                        torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = \
                        log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = \
                        util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)

                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).bool()#long()
                    clamped_gold_question_span_starts = \
                        util.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = \
                        util.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = \
                        torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = \
                        torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = \
                        log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value

                    log_likelihood_for_question_spans = \
                        util.replace_masked_values(log_likelihood_for_question_spans,
                                                   gold_question_span_mask,
                                                   -1e7)
                    log_marginal_likelihood_for_question_span = \
                        util.logsumexp(log_likelihood_for_question_spans)

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)


                elif answering_ability == 'generation':
                    target_tensor = answer_as_generation['tokens'][:, 1:]
                    targets_tensor_seqfirst = target_tensor.permute(1, 0)
                    pad_mask = torch.where(targets_tensor_seqfirst != 0, self.true_rep, self.false_rep)
                    step_log_list = []
                    step_masked_log_list = []
                    for _i in range(len(step_generation_for_prob)):
                        predicted_probs = step_generation_for_prob[_i].squeeze(1)
                        true_labels = targets_tensor_seqfirst[_i]
                        mask_labels = pad_mask[_i]
                        selected_probs = torch.gather(predicted_probs, 1, true_labels.unsqueeze(1))
                        selected_probs = selected_probs.squeeze(1)
                        selected_neg_logprobs = torch.log(selected_probs)
                        step_log_list.append(selected_neg_logprobs * mask_labels)

                    log_likelihood_for_generation = torch.stack(step_log_list, 1)

                    log_marginal_likelihood_for_generation_2 = log_likelihood_for_generation.sum(1,keepdim=True)
                    log_marginal_likelihood_for_generation_2 = util.replace_masked_values(log_marginal_likelihood_for_generation_2,generation_mask_index.squeeze(1).bool(),-1e7)
                    log_marginal_likelihood_for_generation_2 = util.logsumexp(log_marginal_likelihood_for_generation_2)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_generation_2)


                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")


            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]

            output_dict["loss"] = - marginal_log_likelihood.mean() + edge_loss

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            
            question_passage_y_edges = y_edges[:,pp_graph.size(1):-1,:pp_graph.size(1)]
            question_question_y_edges = y_edges[:,pp_graph.size(1):-1,pp_graph.size(1):-1]
            question_passage_y_edge = y_edges[:,-1:,:pp_graph.size(1)]
            question_question_y_edge = y_edges[:, -1:, pp_graph.size(1):-1]
            
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                
                predict_edges = []
                question_passage_predict_edges = question_passage_y_edges[i].detach().cpu().numpy()
                question_question_predict_edges = question_question_y_edges[i].detach().cpu().numpy()
                question_passage_predict_edge = question_passage_y_edge[i].detach().cpu().numpy()
                question_question_predict_edge = question_question_y_edge[i].detach().cpu().numpy()
                for qp_y, qpedges in enumerate(question_passage_predict_edges):
                    for qp_x, pnode in enumerate(qpedges):
                        if pnode == 1:
                            predict_edges.append(['<q{}>'.format(int(qp_y+1)),'<p{}>'.format(int(qp_x+1))])
                for qq_y, qqedges in enumerate(question_question_predict_edges):
                    for qq_x, qnode in enumerate(qqedges):
                        if qnode == 1:
                            predict_edges.append(['<q{}>'.format(int(qq_y+1)),'<q{}>'.format(int(qq_x+1))])
                for p, pedges in enumerate(question_passage_predict_edge[0]):
                    if pedges == 1:
                        predict_edges.append(['<q>', '<p{}>'.format(int(p + 1))])
                for q, qedges in enumerate(question_question_predict_edge[0]):
                    if qedges == 1:
                        predict_edges.append(['<q>', '<q{}>'.format(int(q + 1))])
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
                    passage_str = metadata[i]['question_passage_text']
                    offsets = metadata[i]['question_passage_token_offsets']
                    (predicted_start, predicted_end) = tuple(best_passage_span[i].detach().cpu().numpy())
                    answer_passage_token_ids = question_passage_tokens[i,predicted_start:predicted_end+1].detach().cpu().numpy()
                    predicted_answer = ' '.join(self._bert_tokenizer.convert_ids_to_tokens(answer_passage_token_ids,skip_special_tokens=True))

                    answer_json["value"] = predicted_answer
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    question_str = metadata[i]['question_passage_text']
                    offsets = metadata[i]['question_passage_token_offsets']
                    (predicted_start, predicted_end) = tuple(best_question_span[i].detach().cpu().numpy())
                    answer_question_token_ids = question_passage_tokens[i,predicted_start:predicted_end + 1].detach().cpu().numpy()
                    predicted_answer = ' '.join(self._bert_tokenizer.convert_ids_to_tokens(answer_question_token_ids, skip_special_tokens=True))

                    answer_json["value"] = predicted_answer


                elif predicted_ability_str == 'generation':
                    answer_json["answer_type"] = "generation"
                    predicted_indices = best_for_generation[i].detach().cpu().numpy()#output_dict["predictions"]
                    indices = list(predicted_indices)#.gather(0,predicted_indices[i]))
                    predicted_tokens = self._bert_tokenizer.convert_ids_to_tokens(indices,skip_special_tokens=True)
                    answer_json["generation"] = ""
                    for token_id,token in enumerate(predicted_tokens):
                        if token in WORD_NUMBER_MAP:
                            token = str(WORD_NUMBER_MAP[token])
                            predicted_tokens[token_id] = token
                    answer_json["generation"] = ' '.join(predicted_tokens)

                    predicted_answer = answer_json['generation'].strip()#.replace(' ','') mark!
                    answer_json["generation"] = indices
                    answer_json["value"] = predicted_answer
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                answer_json["predict_edges"] = predict_edges
                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                answer_history = metadata[i].get('history_answer',{})
                answer_type = metadata[i].get("type", [])
                answer_edges = metadata[i].get("evidence_edges", [])
                node_dict = metadata[i].get('node_dict', {})
                node_bag = {
                    "node_dict": node_dict,
                    "evidence_edges": answer_edges,
                    "predict_edges": predict_edges,
                    "question_id": metadata[i].get("question_id", ''),
                    'passage_id': metadata[i].get("passage_id",'')
                }
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations,answer_history,answer_type,node_bag)
            # This is used for the demo.
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

        return output_seq, last_layer_h_fused

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score, supporting_ratio, supporting_total, judge_em, comp_em, open_em, unsure_em, extract_em, arithmetic_em, arithmetic_open_em, judge_f1, comp_f1, open_f1, unsure_f1, extract_f1, arithmetic_f1, arithmetic_open_f1, edge_em = self._drop_metrics.get_metric(
            reset)
        return {'em': exact_match,
                'f1': f1_score,
                'supporting_ritio': supporting_ratio,
                'supporting_total': supporting_total,
                'judge_em': judge_em,
                "judge_f1": judge_f1,
                "comp_em": comp_em,
                "comp_f1": comp_f1,
                "open_em": open_em,
                "open_f1": open_f1,
                "unsure_em": unsure_em,
                "unsure_f1": unsure_f1,
                "extract_em": extract_em,
                'extract_f1': extract_f1,
                'arithmetic_em': arithmetic_em,
                'arithmetic_f1': arithmetic_f1,
                'arithmetic_open_em': arithmetic_open_em,
                'arithmetic_open_f1': arithmetic_open_f1,
                'edge_em': edge_em}

    def _get_loss(self,logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:

        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask, average= None)
