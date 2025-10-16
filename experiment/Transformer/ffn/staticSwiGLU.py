from torch import nn
from experiment.Transformer.ffn.baseSwiGLU import BaseSwiGLU

class StaticSwiGLU(BaseSwiGLU):
    def __init__(self, input_dim, output_dim, up_proj_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            **kwargs
        )

    def _init_sublayers(self, **kwargs):
        self.gate_proj = nn.Linear(self.input_dim, self.up_proj_dim)
        self.up_proj = nn.Linear(self.input_dim, self.up_proj_dim)
        self.down_proj = nn.Linear(self.up_proj_dim, self.output_dim)