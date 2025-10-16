from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer
from experiment.Transformer.ffn.staticSwiGLU import StaticSwiGLU
from experiment.Transformer.selfAttention.staticSelfAttention import StaticSelfAttention

class StaticEncoderLayer(BaseEncoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, **kwargs):
        super().__init__(d_model, dropout_rate,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         **kwargs
                         )

    def _init_sublayers(self, num_heads, d_ff, **kwargs):
        self.attention = StaticSelfAttention(
            d_model=self.d_model,
            num_heads=num_heads
        )

        self.ffn = StaticSwiGLU(
            input_dim=self.d_model,
            output_dim=self.d_model,
            up_proj_dim=d_ff,
        )