import torch.nn as nn
from experiment.Transformer.ffn.baseSwiGLU import BaseSwiGLU
from experiment.Transformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridMoMixSwiGLU(BaseSwiGLU):
    def __init__(self, input_dim, output_dim, up_proj_dim, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            **kwargs
        )

    def _init_sublayers(self, compressed_feature_dim, num_monarchs, use_checkpointing, **kwargs):
        self.gate_proj = nn.Linear(self.input_dim, self.up_proj_dim)

        self.up_proj = HyperMoMixLinear(
            in_features=self.input_dim,
            out_features=self.up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )

        self.down_proj = nn.Linear(self.up_proj_dim, self.output_dim)