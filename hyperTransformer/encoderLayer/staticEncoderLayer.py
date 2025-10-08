from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention
from hyperTransformer.ffn.hybridSwiGLU import HybridSwiGLU

class StaticEncoderLayer(BaseEncoderLayer):
    """使用静态注意力和标准SwiGLU的编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff)

    def _init_sublayers(self, d_model, num_heads, d_ff):
        self.attention = StaticSelfAttention(d_model, num_heads)
        compressed_feature_dim = d_ff // 16
        assert compressed_feature_dim > 1, f"d_ff//16 ({d_ff // 16}) must be > 1"
        self.ffn = HybridSwiGLU(d_model, d_model, d_ff, compressed_feature_dim) # 假设SwiGLU的构造函数是这样的