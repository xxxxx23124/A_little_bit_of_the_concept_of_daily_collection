from experiment.hyperTransformer.ffn.baseSwiGLU import BaseSwiGLU
from experiment.hyperTransformer.linear.dualMoMixLinear import DualMoMixLinear

class DualMoMixSwiGLU(BaseSwiGLU):
    """
    一个混合 SwiGLU 实现，其特点是：
    - 门控投影 (gate_proj): 动态静态并行
    - 内容上采样 (up_proj):  动态静态并行
    - 降维投影 (down_proj):   动态静态并行
    """
    def __init__(self, input_dim, output_dim, up_proj_dim,
                 compressed_feature_dim, num_monarchs,
                 dropout_rate,
                 **kwargs):
        """
        Args:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度 (通常等于 input_dim)。
            up_proj_dim (int): 隐藏层的扩展维度。
            compressed_feature_dim (int): 动态层使用的压缩特征维度。
            num_monarchs (int): 动态层使用的专家数量。
        """
        # 将所有参数传递给父类的构造函数，父类会再将它们传递给 _init_sublayers
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            up_proj_dim=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def _init_sublayers(self, input_dim, output_dim, up_proj_dim, **kwargs):
        """
        初始化具体的静态和动态层。
        """
        # 从kwargs中获取use_checkpointing标志，默认为False
        use_checkpointing = kwargs.get('use_checkpointing', False)

        # 从 kwargs 中安全地提取参数
        compressed_feature_dim = kwargs.get('compressed_feature_dim')
        num_monarchs = kwargs.get('num_monarchs')
        dropout_rate = kwargs.get('dropout_rate')

        # 检查必需的参数是否存在
        if any(p is None for p in [compressed_feature_dim, num_monarchs, dropout_rate]):
            raise ValueError(
                "HybridMoMixSwiGLU requires 'compressed_feature_dim' 'num_monarchs' and 'dropout_rate' to be provided in kwargs."
            )

        # 1. 混合门控
        self.gate_proj = DualMoMixLinear(
            in_features=input_dim,
            out_features=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )

        # 2. 混合内容
        self.up_proj = DualMoMixLinear(
            in_features=input_dim,
            out_features=up_proj_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )

        down_proj_compressed_feature_dim = up_proj_dim//(input_dim//compressed_feature_dim)
        assert down_proj_compressed_feature_dim > 0, "DualMoMixSwiGLU 的 down_proj_compressed_feature_dim 自动计算出错"

        # 3. 混合降维
        self.down_proj = DualMoMixLinear(
            in_features=up_proj_dim,
            out_features=output_dim,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            dropout_rate=dropout_rate,
            use_checkpointing=use_checkpointing
        )