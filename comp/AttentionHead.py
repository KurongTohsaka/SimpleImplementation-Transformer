import torch
from torch import nn
from torch import Tensor

from math import sqrt


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: [Tensor, None] = None) -> Tensor:
    dim_k = q.size(-1)
    attention_scores = torch.bmm(q, k.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(attention_scores, dim=-1)
    return weights.bmm(v)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_states: Tensor, attention_mask: [Tensor, None] = None) -> Tensor:
        attention_outputs = scaled_dot_product_attention(self.q(hidden_states), self.k(hidden_states),
                                                         self.v(hidden_states), mask=attention_mask)
        return attention_outputs
