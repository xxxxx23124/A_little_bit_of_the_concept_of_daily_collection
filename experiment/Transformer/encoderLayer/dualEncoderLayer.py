from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from experiment.Transformer.ffn.dualMoMixSwiGLU import DualMoMixSwiGLU
from experiment.Transformer.selfAttention.dualSelfAttention import DualSelfAttention

class DualEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_sublayers(self, num_heads, d_ff, num_monarchs, compressed_feature_dim, dropout_rate, use_checkpointing, **kwargs):
        self.attention = DualSelfAttention(
            d_model=self.d_model,
            num_heads=num_heads,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )

        self.ffn = DualMoMixSwiGLU(
            input_dim=self.d_model,
            output_dim=self.d_model,
            up_proj_dim=d_ff,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )