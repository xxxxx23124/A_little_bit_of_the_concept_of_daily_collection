from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from experiment.Transformer.ffn.staticCompositeSwiGLU import StaticCompositeSwiGLU
from experiment.Transformer.selfAttention.staticCompositeSelfAttention import StaticCompositeSelfAttention

class HybridEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, num_linear, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         num_linear=num_linear,
                         **kwargs
                         )

    def _init_sublayers(self, num_heads, d_ff, num_linear, use_checkpointing, **kwargs):
        self.attention = StaticCompositeSelfAttention(
            d_model=self.d_model,
            num_heads=num_heads,
            num_linear=num_linear,
            use_checkpointing=use_checkpointing
        )

        self.ffn = StaticCompositeSwiGLU(
            input_dim=self.d_model,
            output_dim=self.d_model,
            up_proj_dim=d_ff,
            num_linear=num_linear,
            use_checkpointing=use_checkpointing
        )