from experiment.hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from experiment.hyperTransformer.ffn.dualMoMixSwiGLU import DualMoMixSwiGLU
from experiment.hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention

class HalfDualEncoderLayer(BaseEncoderLayer):
    """
    一个具体的 Pre-Norm Transformer 编码器层实现。
    它使用 StaticSelfAttention 和 DualMoMixSwiGLU 作为其子模块。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, compressed_feature_dim, num_monarchs, **kwargs):
        """
        Args:
            d_model (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            d_ff (int): FFN的中间层维度。
            dropout_rate (float): 子层输出后的 dropout 比率。
            num_monarchs (int): DualMoMixSwiGLU模型中专家的数量。
        """
        # 将特定于此子类的参数通过 kwargs 传递给父类的 _init_sublayers
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_sublayers(self, d_model, **kwargs):
        """
        实现父类的抽象方法，定义具体的 attention 和 ffn 模块。
        """

        # 0. 从 kwargs 中安全地提取参数
        # .get() 方法比直接用 ['key'] 更安全，如果键不存在不会报错
        num_heads = kwargs.get('num_heads')
        d_ff = kwargs.get('d_ff')
        num_monarchs = kwargs.get('num_monarchs')
        compressed_feature_dim = kwargs.get('compressed_feature_dim')

        # 检查必需的参数是否存在
        if any(p is None for p in [num_heads, d_ff, num_monarchs, compressed_feature_dim]):
            raise ValueError(
                "HybridEncoderLayer requires 'num_heads', 'd_ff', 'compressed_feature_dim', and 'num_monarchs' to be provided in kwargs."
            )

        # 从kwargs中获取use_checkpointing标志，默认为False
        use_checkpointing = kwargs.get('use_checkpointing', False)

        # 1. 初始化自注意力模块
        self.attention = StaticSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        # 2. 初始化混合SwiGLU前馈网络模块
        self.ffn = DualMoMixSwiGLU(
            input_dim=d_model,
            output_dim=d_model, # SwiGLU的输入和输出维度通常与d_model相同
            up_proj_dim=d_ff,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )