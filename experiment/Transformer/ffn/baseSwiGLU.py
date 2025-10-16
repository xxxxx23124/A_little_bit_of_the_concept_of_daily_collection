import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSwiGLU(nn.Module, ABC):
    """
    SwiGLU (Swish-Gated Linear Unit) 前馈网络的抽象基类。

    该结构遵循标准实现: FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    子类需要实现 _init_sublayers 方法来定义具体的 gate_proj, up_proj, 和 down_proj 层。
    这允许我们轻松地创建不同变体，例如：
    - 全静态层 (使用 nn.Linear)
    - 全动态层 (使用 Hyper-networks)
    - 混合层 (静态与动态结合)
    """
    def __init__(self, input_dim, output_dim, up_proj_dim, **kwargs):
        """
        构造函数。

        Args:
            input_dim: 输入的维度
            output_dim: 输出的维度
            up_proj_dim: 升维的维度
            **kwargs: 传递给 _init_sublayers 的配置参数，例如 num_experts 等。
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_proj_dim = up_proj_dim


        self.silu = nn.SiLU()

        # 初始化投影层为 None，由子类负责定义
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

        # 调用抽象方法，强制子类实现层的初始化
        self._init_sublayers(input_dim=self.input_dim, output_dim=self.output_dim, up_proj_dim=self.up_proj_dim, **kwargs)

        # 确保子类已经正确初始化了所有层
        assert self.gate_proj is not None, "self.gate_proj must be initialized in _init_sublayers"
        assert self.up_proj is not None, "self.up_proj must be initialized in _init_sublayers"
        assert self.down_proj is not None, "self.down_proj must be initialized in _init_sublayers"

    @abstractmethod
    def _init_sublayers(self, **kwargs):
        """
        抽象方法，子类必须实现此方法来初始化 self.gate_proj,
        self.up_proj, 和 self.down_proj 模块。

        例如:
            self.gate_proj = nn.Linear(input_dim, up_proj_dim)
            self.up_proj = HyperMoMixLinear(...)
            self.down_proj = nn.Linear(up_proj_dim, output_dim)
        """
        raise NotImplementedError("Subclasses must implement the _init_sublayers method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通用的前向传播逻辑。
        """
        # 门控和内容投影
        gate = self.gate_proj(x)
        up_content = self.up_proj(x)

        # 门控激活
        gated_hidden = self.silu(gate) * up_content

        # 降维投影
        output = self.down_proj(gated_hidden)

        return output