import torch.nn as nn
from experiment.Transformer.ffn.baseSwiGLU import BaseSwiGLU
from experiment.Transformer.linear.hyperLoRALinear import HyperLoRALinear

class HybridLoRASwiGLU(BaseSwiGLU):
    def __init__(self, input_dim, output_dim, up_proj_dim, compressed_feature_dim, rank, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            rank=rank,
            **kwargs
        )

    def _init_sublayers(self, compressed_feature_dim, rank, **kwargs):
        self.gate_proj = nn.Linear(self.input_dim, self.up_proj_dim)
        self.up_proj = HyperLoRALinear(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            compressed_feature_dim=compressed_feature_dim,
            rank=rank
        )
        self.down_proj = nn.Linear(self.up_proj_dim, self.output_dim)