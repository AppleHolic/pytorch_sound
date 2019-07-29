import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

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

    def forward(self, input: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        # linear and split k, v, q
        k, v, q = self.linear_kvq(input).chunk(3, 1)
        k, v, q = [torch.cat(x.chunk(self.heads, 1), dim=0) for x in [k, v, q]]

        # do attention at once
        x, att = self.scale_dot_att(k, v, q, mask.repeat(self.heads, 1))
        x = torch.cat(x.chunk(self.heads, 0), dim=1)

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

    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ff = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 4, 1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim * 4, self.hidden_dim, 1),
        )

        self.act = nn.ReLU()
        # self.norm = nn.BatchNorm1d(hidden_dim)

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


class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.attention = MultiHeadAttention(hidden_dim, heads)
        self.ff = PointwiseFeedForward(hidden_dim)

    def forward(self, input: torch.tensor) -> torch.tensor:
        feature, att = self.attention(input)
        return self.ff(feature), att
