from experiment.Transformer.ffn.baseSwiGLU import BaseSwiGLU
from experiment.Transformer.linear.staticCompositeLinear import StaticCompositeLinear

class StaticCompositeSwiGLU(BaseSwiGLU):
    def __init__(self, input_dim, output_dim, up_proj_dim, num_linears, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            num_linears=num_linears,
            **kwargs
        )

    def _init_sublayers(self, num_linears, **kwargs):
        self.gate_proj = StaticCompositeLinear(
            in_features=self.input_dim,
            out_features=self.up_proj_dim,
            num_linears=num_linears,
        )
        self.up_proj = StaticCompositeLinear(
            in_features=self.input_dim,
            out_features=self.up_proj_dim,
            num_linears=num_linears,
        )
        self.down_proj = StaticCompositeLinear(
            in_features=self.up_proj_dim,
            out_features=self.output_dim,
            num_linears=num_linears,
        )