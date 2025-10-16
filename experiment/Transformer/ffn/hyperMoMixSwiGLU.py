from experiment.Transformer.ffn.baseSwiGLU import BaseSwiGLU
from experiment.Transformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HyperMoMixSwiGLU(BaseSwiGLU):
    def __init__(self, input_dim, output_dim, up_proj_dim, compressed_feature_dim, num_monarchs, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            **kwargs
        )

    def _init_sublayers(self, compressed_feature_dim, num_monarchs, use_checkpointing, **kwargs):
        self.gate_proj = HyperMoMixLinear(
            in_features=self.input_dim,
            out_features=self.up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )
        self.up_proj = HyperMoMixLinear(
            in_features=self.input_dim,
            out_features=self.up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )

        down_proj_compressed_feature_dim = self.up_proj_dim//(self.input_dim//compressed_feature_dim)
        assert down_proj_compressed_feature_dim > 0, "DualMoMixSwiGLU 的 down_proj_compressed_feature_dim 自动计算出错"

        self.down_proj = HyperMoMixLinear(
            in_features=self.up_proj_dim,
            out_features=self.output_dim,
            compressed_feature_dim=down_proj_compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )