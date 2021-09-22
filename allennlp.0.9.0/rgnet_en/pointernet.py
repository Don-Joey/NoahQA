import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, units, bias=False)
        self.W2 = nn.Linear(hidden_size, units, bias=False)
        self.V = nn.Linear(units, 1, bias=False)

    def forward(self,
                encoder_out: torch.Tensor,
                decoder_hidden: torch.Tensor):
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # decoder_hidden: (BATCH, HIDDEN_SIZE)

        # Add time axis to decoder hidden state
        # in order to make operations compatible with encoder_out
        # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
        decoder_hidden_time = decoder_hidden.unsqueeze(1)

        # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
        # Note: we can add the both linear outputs thanks to broadcasting

        uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
        uj = torch.tanh(uj)

        # uj: (BATCH, ARRAY_LEN, 1)
        uj = self.V(uj)

        # Attention mask over inputs
        # aj: (BATCH, ARRAY_LEN, 1)
        aj = F.softmax(uj, dim=1)

        # di_prime: (BATCH, HIDDEN_SIZE)
        di_prime = aj * encoder_out
        di_prime = di_prime.sum(1)

        return di_prime, uj.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 attention_units=10):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, attention_units)

    def forward(self,
                x: torch.Tensor,
                hidden,
                encoder_out: torch.Tensor):
        # x: (BATCH, 1, 1)
        # hidden: (1, BATCH, HIDDEN_SIZE)
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # For a better understanding about hidden shapes read: https://pytorch.org/docs/stable/nn.html#lstm

        # Get hidden states (not cell states)
        # from the first and unique LSTM layer
        ht = hidden[0][0]  # ht: (BATCH, HIDDEN_SIZE)
        # di: Attention aware hidden state -> (BATCH, HIDDEN_SIZE)
        # att_w: Not 'softmaxed', torch will take care of it -> (BATCH, ARRAY_LEN)
        di, att_w = self.attention(encoder_out, ht)

        # Append attention aware hidden state to our input
        # x: (BATCH, 1, 1 + HIDDEN_SIZE)
        x = torch.cat((di.unsqueeze(1), x), dim=2)

        # Generate the hidden state for next timestep
        _, hidden = self.lstm(x, hidden)
        return hidden, att_w