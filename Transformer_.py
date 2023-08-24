# Main ML framework to be built here.
import math
import torch
import setuptools

class ml_transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.TransformerEncoderLayer()