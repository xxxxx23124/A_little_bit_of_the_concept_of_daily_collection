from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from experiment.Transformer.ffn.staticCompositeSwiGLU import StaticCompositeSwiGLU
from experiment.Transformer.selfAttention.staticSelfAttention import StaticSelfAttention

class HalfStaticCompositeEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, num_linears, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         num_linears=num_linears,
                         **kwargs
                         )

    def _init_sublayers(self, num_heads, d_ff, num_linears, **kwargs):
        self.attention = StaticSelfAttention(
            d_model=self.d_model,
            num_heads=num_heads
        )

        self.ffn = StaticCompositeSwiGLU(
            input_dim=self.d_model,
            output_dim=self.d_model,
            up_proj_dim=d_ff,
            num_linears=num_linears,
        )