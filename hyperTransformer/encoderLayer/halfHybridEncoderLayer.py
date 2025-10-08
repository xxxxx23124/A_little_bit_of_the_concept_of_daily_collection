from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention
from hyperTransformer.ffn.hybridSwiGLU import HybridSwiGLU
import math


class HalfHybridEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff)

    def _init_sublayers(self, d_model, num_heads, d_ff):
        self.attention = StaticSelfAttention(d_model, num_heads)
        # 计算压缩特征维度
        compressed_feature_dim = math.isqrt(d_model)

        self.ffn = HybridSwiGLU(d_model, d_model, d_ff, compressed_feature_dim)