from experiment.hyperTransformer.selfAttention.baseSelfAttention import BaseSelfAttention
from experiment.hyperTransformer.linear.dualMoMixLinear import DualMoMixLinear

class DualSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads, compressed_feature_dim, num_monarchs, **kwargs):
        # 将特定于此子类的参数传递给 _init_projections
        super().__init__(d_model, num_heads,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_projections(self, compressed_feature_dim, num_monarchs, **kwargs):
        # 从kwargs中获取use_checkpointing标志，默认为False
        use_checkpointing = kwargs.get('use_checkpointing', False)
        # 实现父类的抽象方法，定义自己的投影层
        self.q_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )
        self.k_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )
        self.v_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )