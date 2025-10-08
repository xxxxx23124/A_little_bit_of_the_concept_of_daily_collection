from hyperTransformer.decoderLayer.baseDecoderLayer import BaseDecoderLayer
from hyperTransformer.ffn.hybridSwiGLU import HybridSwiGLU
from hyperTransformer.selfAttention.hybridSelfAttention import HybridSelfAttention
import math


class HybridDecoderLayer(BaseDecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, num_experts=2):
        # 将特定于此子类的参数通过 kwargs 传递给父类的 _init_sublayers
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         num_experts=num_experts)

    def _init_sublayers(self, d_model, num_heads, d_ff, num_experts):
        """
        实现父类的抽象方法，定义具体的 attention 和 ffn 模块。
        """
        # 计算压缩特征维度
        compressed_feature_dim = math.isqrt(d_model)

        # 1. 初始化混合自注意力模块
        self.self_attention = HybridSelfAttention(
            d_model,
            num_heads,
            compressed_feature_dim,
            num_experts=num_experts
        )

        # 2. 初始化混合交叉注意力模块
        self.cross_attention = HybridSelfAttention(
            d_model,
            num_heads,
            compressed_feature_dim,
            num_experts=num_experts
        )

        # 3. 初始化混合SwiGLU前馈网络模块
        self.ffn = HybridSwiGLU(
            d_model,
            d_model, # SwiGLU的输入和输出维度通常与d_model相同
            d_ff,
            compressed_feature_dim
        )