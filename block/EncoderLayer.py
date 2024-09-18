import torch.nn as nn
from torch import Tensor
from transformers import BertConfig

from comp.MultiHeadAttention import MultiHeadAttention
from comp.FeedForward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, attention_mask: [Tensor, None] = None) -> Tensor:
        hidden_states = self.layer_norm_1(x)
        x += self.attention(hidden_states, attention_mask=attention_mask)
        x += self.feed_forward(self.layer_norm_2(x))
        return x
