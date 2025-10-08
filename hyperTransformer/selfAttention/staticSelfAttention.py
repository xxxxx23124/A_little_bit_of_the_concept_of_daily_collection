import torch.nn as nn
from hyperTransformer.selfAttention.baseSelfAttention import BaseSelfAttention

class StandardSelfAttention(BaseSelfAttention):
    """
    标准自注意力实现，所有投影层都使用 nn.Linear。
    """
    def __init__(self, d_model, nheads):
        # 这里没有额外的参数，可以直接调用
        super().__init__(d_model, nheads)

    def _init_projections(self, **kwargs):
        # 实现父类的抽象方法，定义自己的投影层
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)