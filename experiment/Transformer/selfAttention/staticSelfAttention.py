import torch.nn as nn
from experiment.Transformer.selfAttention.baseSelfAttention import BaseSelfAttention

class StaticSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads):
        super().__init__(d_model, num_heads)

    def _init_projections(self, **kwargs):
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)