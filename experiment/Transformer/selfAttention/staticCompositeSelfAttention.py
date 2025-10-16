from experiment.Transformer.selfAttention.baseSelfAttention import BaseSelfAttention
from experiment.Transformer.linear.staticCompositeLinear import StaticCompositeLinear

class StaticCompositeSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads, num_linear, **kwargs):
        super().__init__(d_model=d_model, num_heads=num_heads,
                         num_linear=num_linear,
                         **kwargs
                         )

    def _init_projections(self, num_linear, use_checkpointing, **kwargs):
        self.q_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linear=num_linear,
            use_checkpointing=use_checkpointing
        )
        self.k_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linear=num_linear,
            use_checkpointing=use_checkpointing
        )
        self.v_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linear=num_linear,
            use_checkpointing=use_checkpointing
        )