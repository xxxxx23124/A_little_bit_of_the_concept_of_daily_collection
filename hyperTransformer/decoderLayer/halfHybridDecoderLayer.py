from hyperTransformer.decoderLayer.baseDecoderLayer import BaseDecoderLayer
from hyperTransformer.ffn.hybridMoMixSwiGLU import HybridMoMixSwiGLU
from hyperTransformer.selfAttention.staticSelfAttention import StaticSelfAttention
from hyperTransformer.crossAttention.hybridCrossAttention import HybridCrossAttention
import math


class HalfHybridDecoderLayer(BaseDecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, compressed_feature_dim, num_experts, **kwargs):
        # 将特定于此子类的参数通过 kwargs 传递给父类的 _init_sublayers
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         compressed_feature_dim=compressed_feature_dim,
                         num_experts=num_experts,
                         **kwargs
                         )

    def _init_sublayers(self, d_model, **kwargs):
        # 0. 从 kwargs 中安全地提取参数
        # .get() 方法比直接用 ['key'] 更安全，如果键不存在不会报错
        num_heads = kwargs.get('num_heads')
        d_ff = kwargs.get('d_ff')
        num_experts = kwargs.get('num_experts')
        compressed_feature_dim=kwargs.get('compressed_feature_dim')

        # 检查必需的参数是否存在
        if any(p is None for p in [num_heads, d_ff, num_experts, compressed_feature_dim]):
            raise ValueError(
                "HalfHybridDecoderLayer requires 'num_heads', 'd_ff', 'compressed_feature_dim', and 'num_experts' to be provided in kwargs."
            )

        # 从kwargs中获取use_checkpointing标志，默认为False
        use_checkpointing = kwargs.get('use_checkpointing', False)

        # 1. 初始化混合自注意力模块
        self.self_attention = StaticSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        # 2. 初始化混合交叉注意力模块
        self.cross_attention = HybridCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            compressed_feature_dim=compressed_feature_dim,
            num_experts=num_experts,
            use_checkpointing=use_checkpointing
        )

        # 3. 初始化混合SwiGLU前馈网络模块
        self.ffn = HybridMoMixSwiGLU(
            input_dim=d_model,
            output_dim=d_model, # SwiGLU的输入和输出维度通常与d_model相同
            up_proj_dim=d_ff,
            compressed_feature_dim=compressed_feature_dim,
            num_experts=num_experts,
            use_checkpointing=use_checkpointing
        )