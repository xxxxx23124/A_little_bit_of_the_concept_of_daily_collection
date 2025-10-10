from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention
from hyperTransformer.ffn.hybridSwiGLU import HybridSwiGLU
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

        self.attention = StaticSelfAttention(d_model, num_heads)
        # 计算压缩特征维度
        compressed_feature_dim = math.isqrt(d_model)

        self.ffn = HybridSwiGLU(d_model, d_model, d_ff, compressed_feature_dim, num_experts)