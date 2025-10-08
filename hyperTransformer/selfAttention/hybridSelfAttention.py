import torch.nn as nn
from hyperTransformer.selfAttention.baseSelfAttention import BaseSelfAttention
from hyperTransformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridSelfAttention(BaseSelfAttention):
    """
    混合自注意力实现，Q 投影层使用 HyperMoMixLinear。
    """
    def __init__(self, d_model, nheads, compressed_feature_dim, num_experts):
        # 将特定于此子类的参数传递给 _init_projections
        super().__init__(d_model, nheads,
                         compressed_feature_dim=compressed_feature_dim,
                         num_experts=num_experts)

    def _init_projections(self, compressed_feature_dim, num_experts):
        # 实现父类的抽象方法，定义自己的投影层
        self.q_proj = HyperMoMixLinear(self.d_model, self.d_model, compressed_feature_dim, num_experts)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)