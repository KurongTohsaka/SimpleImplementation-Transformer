import torch.nn as nn
from torch import Tensor
from transformers import BertConfig


class FeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gelu(self.linear_1(x))
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
