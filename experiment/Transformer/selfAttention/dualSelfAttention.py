from experiment.Transformer.selfAttention.baseSelfAttention import BaseSelfAttention
from experiment.Transformer.linear.dualMoMixLinear import DualMoMixLinear

class DualSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(d_model, num_heads,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_projections(self, compressed_feature_dim, num_monarchs, use_checkpointing, dropout_rate, **kwargs):
        self.q_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )
        self.k_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )
        self.v_proj = DualMoMixLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )