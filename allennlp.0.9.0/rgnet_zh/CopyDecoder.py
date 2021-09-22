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
        '''
        if coverage is not None:
            projected_coverage = self.Wc_layer(
                coverage.unsqueeze(-1))  # shape = batchsize X dec_seqlen x enc_seqlen X attn_vec_size
            added_projections += projected_coverage
        '''
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
        '''
        self.pgen = nn.Sequential(
            nn.Linear(encoder_hidden_size + 4*decoder_hidden_size + decoder_input_size, 1),
            nn.Sigmoid()
        )
        '''
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
'''
class Attention(nn.Module):
    """ Simple Attention
    This Attention is learned from weight
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

        # Declare the Attention Weight
        self.W = nn.Linear(dim, 1)

        # Declare the coverage feature
        self.coverage_feature = nn.Linear(1, dim)

    def forward(self, output, context, coverage):
        # declare the size
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # Expand the output to the num of timestep
        output_expand = output.expand(batch_size, input_size, hidden_size)

        # reshape to 2-dim
        output_expand = output_expand.reshape([-1, hidden_size])
        context = context.reshape([-1, hidden_size])

        # transfer the coverage to features
        coverage_feature = self.coverage_feature(coverage.reshape(-1, 1))

        # Learning the attention
        attn = self.W(output_expand + context + coverage_feature)
        attn = attn.reshape(-1, input_size)
        attn = F.softmax(attn, dim=1)

        # update the coverage
        coverage = coverage + attn

        context = context.reshape(batch_size, input_size, hidden_size)
        attn = attn.reshape(batch_size, -1, input_size)

        # get the value of a
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn, coverage
class AttentionDecoder(torch.nn.Module):

    def __init__(self, num_words,embedding_dim,hidden_size):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_words,
                                            embedding_dim=self.embedding_dim)

        self.gru = torch.nn.LSTM(input_size=self.embedding_dim + self.hidden_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=1,
                                 bidirectional=False,
                                 batch_first=True)

        self.att = Attention(self.hidden_size)

    def forward(self, input, hidden, encoder_output, z, content, coverage):
        # Embedding
        embedding = self.embedding(input)
        # print(embedding.squeeze().size())

        combine = torch.cat([embedding, z], 2)
        # print(combine.squeeze().size())
        # Call the GRU
        out, hidden = self.gru(combine, hidden)

        # call the attention
        output, attn, coverage = self.att(output=out, context=encoder_output, coverage=coverage)

        index = content
        attn = attn.view(attn.size(0), -1)
        attn_value = torch.zeros([attn.size(0), self.configure["num_words"]]).to(self.device)
        attn_value = attn_value.scatter_(1, index, attn)

        # print(torch.cat([embedding.squeeze(), combine.squeeze()], 1).size(), )
        # print(p)
        out = attn_value
        # print(attn_value.size(), output.size())

        return out, hidden, output, attn, coverage
'''
