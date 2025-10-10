from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention
from hyperTransformer.ffn.hybridMoMixSwiGLU import HybridMoMixSwiGLU
import math


class HalfHybridEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, num_experts, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         num_experts=num_experts)

    def _init_sublayers(self, d_model, **kwargs):

        # 0. 从 kwargs 中安全地提取参数
        # .get() 方法比直接用 ['key'] 更安全，如果键不存在不会报错
        num_heads = kwargs.get('num_heads')
        d_ff = kwargs.get('d_ff')
        num_experts = kwargs.get('num_experts')

        # 检查必需的参数是否存在
        if any(p is None for p in [num_heads, d_ff, num_experts]):
            raise ValueError(
                "HalfHybridEncoderLayer requires 'num_heads', 'd_ff', and 'num_experts' to be provided in kwargs."
            )
        # 1. 初始化混合自注意力模块
        self.attention = StaticSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        # 计算压缩特征维度
        compressed_feature_dim = math.isqrt(d_model)
        # 2. 初始化混合SwiGLU前馈网络模块
        self.ffn = HybridMoMixSwiGLU(
            input_dim=d_model,
            output_dim=d_model,  # SwiGLU的输入和输出维度通常与d_model相同
            up_proj_dim=d_ff,
            compressed_feature_dim=compressed_feature_dim,
            num_experts=num_experts
        )