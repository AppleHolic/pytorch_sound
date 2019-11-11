import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


#
# Multi Head Self Attention Modules
#
class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention module. https://arxiv.org/abs/1706.03762
    This version has no normalization module and suppose self-attention
    """
    def __init__(self, hidden_dim: int, heads: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        # linear projection layers
        self.linear_kvq = nn.Conv1d(self.hidden_dim, self.hidden_dim * 3, 1, bias=False)
        self.linear = nn.Conv1d(self.hidden_dim,  self.hidden_dim, 1, bias=False)

        # dropout layer
        if 0 < dropout_rate < 1:
            self.drop_out = nn.Dropout(dropout_rate)
        else:
            self.drop_out = None

    def forward(self, input: torch.tensor, mask: torch.tensor = None) -> Tuple[torch.tensor, torch.tensor]:
        # linear and split k, v, q
        k, v, q = self.linear_kvq(input).chunk(3, 1)

        # split heads and concatenate on batch-wise dimension
        # to calculate 'scale dot product' at once
        k, v, q = [torch.cat(x.chunk(self.heads, 1), dim=0) for x in [k, v, q]]

        # repeat mask tensor for supporting above line
        if mask is not None:
            mask = mask.repeat(self.heads, 1)

        # dot product
        x, att = self.scale_dot_att(k, v, q, att_mask=mask)

        # re-arrange tensor
        x = torch.cat(x.chunk(self.heads, 0), dim=1)

        # linear operation
        x = self.linear(x)

        # dropout
        if self.drop_out is not None:
            x = self.drop_out(x)

        return input + x, att

    @staticmethod
    def scale_dot_att(k: torch.tensor, v: torch.tensor, q: torch.tensor, att_mask: torch.tensor) -> torch.tensor:

        # matmul and scale
        att = torch.bmm(k.transpose(1, 2), q) / (k.size(1)**0.5)

        # apply mask
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att.data.masked_fill_(att_mask.transpose(1, 2).data, -float('inf'))

        # apply softmax
        att = F.softmax(att, 1)
        if att_mask is not None:
            att.data.masked_fill_(att_mask.data, 0)

        # apply attention
        return torch.bmm(v, att), att


class PointwiseFeedForward(nn.Module):
    """
    Point-wise FeedForward module. https://arxiv.org/abs/1706.03762
    This version has no normalization module
    """

    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ff = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 4, 1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim * 4, self.hidden_dim, 1),
        )

        self.act = nn.ReLU()

        # dropout layer
        if 0 < dropout_rate < 1:
            self.drop_out = nn.Dropout(dropout_rate)
        else:
            self.drop_out = None

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = self.ff(input)
        if self.drop_out is not None:
            x = self.drop_out(x)

        return self.act(x + input)


class PositionalEncoder(nn.Module):
    """
    - reference : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    """
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.d_model = dim

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, dim)
        for pos in range(max_seq_len):
            for i in range(0, dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / dim)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
