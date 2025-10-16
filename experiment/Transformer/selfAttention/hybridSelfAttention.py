import torch.nn as nn
from experiment.Transformer.selfAttention.baseSelfAttention import BaseSelfAttention
from experiment.Transformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(d_model, num_heads,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_projections(self, compressed_feature_dim, num_monarchs, use_checkpointing, **kwargs):
        self.q_proj = HyperMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)