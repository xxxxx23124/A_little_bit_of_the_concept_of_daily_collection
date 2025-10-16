from experiment.Transformer.decoderLayer.baseDecoderLayer import BaseDecoderLayer
from experiment.Transformer.ffn.hybridMoMixSwiGLU import HybridMoMixSwiGLU
from experiment.Transformer.selfAttention.hybridSelfAttention import HybridSelfAttention
from experiment.Transformer.crossAttention.hybridCrossAttention import HybridCrossAttention

class HybridDecoderLayer(BaseDecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         compressed_feature_dim=compressed_feature_dim,
                         num_monarchs=num_monarchs,
                         **kwargs
                         )

    def _init_sublayers(self, num_heads, d_ff, num_monarchs, compressed_feature_dim, use_checkpointing, **kwargs):
        self.self_attention = HybridSelfAttention(
            d_model=self.d_model,
            num_heads=num_heads,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )

        self.cross_attention = HybridCrossAttention(
            d_model=self.d_model,
            num_heads=num_heads,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )

        self.ffn = HybridMoMixSwiGLU(
            input_dim=self.d_model,
            output_dim=self.d_model,
            up_proj_dim=d_ff,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )