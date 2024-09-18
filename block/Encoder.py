import torch.nn as nn
from torch import Tensor
from transformers import BertConfig

from EncoderLayer import EncoderLayer
from comp.Embeddings import Embeddings


class Encoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: Tensor, attention_mask: [Tensor, None] = None) -> Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x
