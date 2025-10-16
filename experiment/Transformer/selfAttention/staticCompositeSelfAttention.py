from experiment.Transformer.selfAttention.baseSelfAttention import BaseSelfAttention
from experiment.Transformer.linear.staticCompositeLinear import StaticCompositeLinear

class StaticCompositeSelfAttention(BaseSelfAttention):
    def __init__(self, d_model, num_heads, num_linears, **kwargs):
        super().__init__(d_model=d_model, num_heads=num_heads,
                         num_linears=num_linears,
                         **kwargs
                         )

    def _init_projections(self, num_linears, **kwargs):
        self.q_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linears=num_linears,
        )
        self.k_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linears=num_linears,
        )
        self.v_proj = StaticCompositeLinear(
            in_features=self.d_model,
            out_features=self.d_model,
            num_linears=num_linears,
        )