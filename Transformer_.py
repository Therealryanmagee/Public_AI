import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TramsformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_head: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'transformer'
        self.pos_encoder = TransformerEncoder(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_head, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


