import torch.nn as nn
from torch import Tensor
from transformers import BertConfig

from block.Encoder import Encoder


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids: Tensor, attention_mask: [Tensor, None] = None) -> Tensor:
        x = self.encoder(input_ids, attention_mask=attention_mask)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
