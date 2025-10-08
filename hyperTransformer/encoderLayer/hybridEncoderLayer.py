from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from hyperTransformer.ffn.hybridSwiGLU import HybridSwiGLU
from hyperTransformer.selfAttention.hybridSelfAttention import HybridSelfAttention

class PreNormEncoderLayer(BaseEncoderLayer):
    """
    一个具体的 Pre-Norm Transformer 编码器层实现。
    它使用 HybridSelfAttention 和 HybridSwiGLU 作为其子模块。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, num_experts=2):
        """
        Args:
            d_model (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            d_ff (int): FFN的中间层维度。
            dropout_rate (float): 子层输出后的 dropout 比率。
            num_experts (int): HybridSwiGLU模型中专家的数量。
        """
        # 将特定于此子类的参数通过 kwargs 传递给父类的 _init_sublayers
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         num_experts=num_experts)

    def _init_sublayers(self, d_model, num_heads, d_ff, num_experts):
        """
        实现父类的抽象方法，定义具体的 attention 和 ffn 模块。
        """
        # 计算并验证压缩特征维度
        compressed_feature_dim = d_ff // 16
        assert compressed_feature_dim > 1, f"d_ff//16 ({d_ff//16}) must be > 1"

        # 1. 初始化混合自注意力模块
        self.attention = HybridSelfAttention(
            d_model,
            num_heads,
            compressed_feature_dim,
            num_experts=num_experts
        )

        # 2. 初始化混合SwiGLU前馈网络模块
        self.ffn = HybridSwiGLU(
            d_model,
            d_model, # SwiGLU的输入和输出维度通常与d_model相同
            d_ff,
            compressed_feature_dim
        )