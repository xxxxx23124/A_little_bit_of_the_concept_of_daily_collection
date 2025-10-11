import torch.nn as nn
from hyperTransformer.ffn.baseSwiGLU import BaseSwiGLU
from hyperTransformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridMoMixSwiGLU(BaseSwiGLU):
    """
    一个混合 SwiGLU 实现，其特点是：
    - 门控投影 (gate_proj): 静态 (nn.Linear)
    - 内容上采样 (up_proj):  动态 (HyperMoMixLinear)
    - 降维投影 (down_proj):   静态 (nn.Linear)
    """
    def __init__(self, input_dim, output_dim, up_proj_dim, compressed_feature_dim, num_experts, **kwargs):
        """
        Args:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度 (通常等于 input_dim)。
            up_proj_dim (int): 隐藏层的扩展维度。
            compressed_feature_dim (int): 动态层使用的压缩特征维度。
            num_experts (int): 动态层使用的专家数量。
        """
        # 将所有参数传递给父类的构造函数，父类会再将它们传递给 _init_sublayers
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_experts=num_experts,
            **kwargs
        )

    def _init_sublayers(self, input_dim, output_dim, up_proj_dim, **kwargs):
        """
        初始化具体的静态和动态层。
        """
        # 从kwargs中获取use_checkpointing标志，默认为False
        use_checkpointing = kwargs.get('use_checkpointing', False)

        # 从 kwargs 中安全地提取参数
        compressed_feature_dim = kwargs.get('compressed_feature_dim')
        num_experts = kwargs.get('num_experts')

        # 检查必需的参数是否存在
        if any(p is None for p in [compressed_feature_dim, num_experts]):
            raise ValueError(
                "HybridMoMixSwiGLU requires 'compressed_feature_dim' and 'num_experts' to be provided in kwargs."
            )

        # 1. 静态门控 (Static Gate)
        self.gate_proj = nn.Linear(input_dim, up_proj_dim)

        # 2. 动态内容 (Dynamic Content)
        self.up_proj = HyperMoMixLinear(
            in_features=input_dim,
            out_features=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_experts,
            use_checkpointing=use_checkpointing
        )

        # 3. 静态降维 (Static Down-projection)
        self.down_proj = nn.Linear(up_proj_dim, output_dim)