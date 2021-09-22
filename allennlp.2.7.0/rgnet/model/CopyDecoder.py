import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, total_encoder_hidden_size, total_decoder_hidden_size, attn_vec_size):
        super(Attention, self).__init__()
        self.total_encoder_hidden_size = total_encoder_hidden_size
        self.total_decoder_hidden_size = total_decoder_hidden_size
        self.attn_vec_size = attn_vec_size

        self.Wh_layer = nn.Linear(total_encoder_hidden_size, attn_vec_size, bias=False)
        self.Ws_layer = nn.Linear(total_decoder_hidden_size, attn_vec_size, bias=True)
        self.selector_vector_layer = nn.Linear(attn_vec_size, 1, bias=False)  # called 'v' in see et al

        self.Wc_layer = nn.Linear(1, attn_vec_size, bias=False)
        torch.nn.init.zeros_(self.Wc_layer.weight)
    def forward(self, encoded_seq, decoder_state, input_pad_mask, coverage=None):
        '''
        encoded seq is batchsize x enc_seqlen x total_encoder_hidden_size
        decoder_state is batchsize x dec_seqlen x total_decoder_hidden_size
        '''

        projected_decstates = self.Ws_layer(decoder_state)
        projected_encstates = self.Wh_layer(encoded_seq)

        added_projections = projected_decstates.unsqueeze(2) + projected_encstates.unsqueeze(1)  # batchsize X declen X enclen X attnvecsize
        
        added_projections = torch.tanh(added_projections)

        attn_logits = self.selector_vector_layer(added_projections)
        attn_logits = attn_logits.squeeze(3)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # shape=batchXdec_lenXenc_len
        attn_weights2 = attn_weights * input_pad_mask.unsqueeze(1)
        attn_weights_renormalized = attn_weights2 / torch.sum(attn_weights2, dim=-1,
                                                              keepdim=True)  # shape=batchx1x1     # TODO - why is there a division without EPS ?

        context_vector = torch.sum(encoded_seq.unsqueeze(1) * attn_weights_renormalized.unsqueeze(-1), dim=-2)
        # shape batchXdec_seqlenXhiddensize

        return context_vector, attn_weights_renormalized


class CopyMechanism(nn.Module):
    def __init__(
            self, encoder_hidden_size, decoder_hidden_size, decoder_input_size):
        super(CopyMechanism, self).__init__()
        self.pgen = nn.Sequential(
            nn.Linear(encoder_hidden_size + 3 * decoder_hidden_size + decoder_input_size, 1),
            nn.Sigmoid()
        )
        self.output_probs = nn.Softmax(dim=-1)
        self.m = nn.Softmax(dim=-1)

    def forward(
            self, output_logits, attn_weights, decoder_hidden_state, decoder_input,
            context_vector, encoder_input):
        '''output_logits = batchXseqlenXoutvocab
            attn_weights = batchXseqlenXenc_len
            decoder_hidden_state = batchXseqlenXdecoder_hidden_size
            context_vector = batchXseqlenXencoder_hidden_dim
            encoder_input = batchxenc_len'''
        output_probabilities = self.output_probs(output_logits)
        #print("copydecoder_output_probabilities:",output_probabilities)



        return output_probabilities  # batchXseqlenXoutvocab , batchsizeXseqlenX1
