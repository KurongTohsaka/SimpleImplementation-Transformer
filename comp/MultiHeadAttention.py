import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertConfig

from AttentionHead import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states: Tensor, attention_mask: [Tensor, None] = None) -> Tensor:
        multi_head_outputs = torch.cat([h(hidden_states, attention_mask=attention_mask) for h in self.heads], dim=-1)
        outputs = self.output_layer(multi_head_outputs)
        return outputs
