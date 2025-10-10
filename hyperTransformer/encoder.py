import torch
import torch.nn as nn
from torch import Tensor
from hyperTransformer.rmsNorm import RMSNorm
from hyperTransformer.rotaryEmbedding import RotaryEmbedding
from hyperTransformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer


class Encoder(nn.Module):
    """
    一个通用的、配置驱动的 Transformer 编码器堆栈。

    该类被设计为高度灵活，它不硬编码任何特定的编码器层类型。
    相反，它接收一个“配方”（recipe）——一个包含层类型的列表，
    并根据这个配方动态地构建编码器堆栈。
    这使得实验不同的层组合（例如，混合不同类型的注意力层）变得非常容易，
    而无需修改此类本身的代码。
    """

    def __init__(self,
                 layer_recipe: list[type[BaseEncoderLayer]],
                 d_model: int,
                 **layer_kwargs
                 ):
        """
        Args:
            layer_recipe (List[Type[BaseEncoderLayer]]):
                一个类型列表，定义了编码器堆栈中每一层的类。
                列表的顺序决定了层在堆栈中的顺序。
                例如: [LayerTypeA, LayerTypeB, LayerTypeB]
                将创建一个三层编码器，第一层是 LayerTypeA，后两层是 LayerTypeB。

            d_model (int):
                模型的维度，这是所有层共享的基本参数。

            **layer_kwargs:
                一个包含关键字参数的字典，它将被“透传”给 `layer_recipe` 中
                每一个层的构造函数 (__init__)。这允许所有层共享通用配置，
                如 n_heads, d_ff, dropout_rate 等。
        """
        super().__init__()

        # --- 1. 动态层实例化 ---
        # 使用列表推导式遍历 `layer_recipe`。
        # 对于配方中的每一个 `layer_cls` (层类型/类)，我们都调用它的构造函数
        # 来创建一个实例。`d_model` 和 `layer_kwargs` 被传递给每个构造函数。
        # 这种模式将层的构建逻辑从 Encoder 类本身转移到了调用它的外部配置中。
        layers = [
            layer_cls(d_model=d_model, **layer_kwargs)
            for layer_cls in layer_recipe
        ]

        # 将创建的层实例列表包装在 `nn.ModuleList` 中。
        # 这是至关重要的，因为它能确保所有层都被正确注册为 Encoder 模块的
        # 子模块，从而使 .parameters(), .to(device), .state_dict() 等方法能够
        # 正常工作。如果使用普通的 Python 列表，这些功能会失效。
        self.layers = nn.ModuleList(layers)

        # --- 2. 最终归一化层 ---
        # 在所有编码器层之后应用一个最终的归一化层是一种标准实践。
        # 它可以稳定下一模块（如解码器）的输入，有助于改善训练动态。
        self.norm = RMSNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                rotary_emb: RotaryEmbedding | None,
                padding_mask: Tensor | None = None) -> torch.Tensor:
        """
        编码器堆栈的前向传播。

        Args:
            x (torch.Tensor):
                输入张量，形状为 (batch_size, seq_len, d_model)。
                这通常是词嵌入和位置编码相加后的结果。

            rotary_emb (RotaryEmbedding | None):
                旋转位置编码模块。这个对象将被原封不动地传递给堆栈中的
                每一个层，由层内部的注意力模块决定如何使用它。

            padding_mask (Tensor | None): padding 掩码

        Returns:
            torch.Tensor:
                编码器的最终输出，形状与输入 x 相同。
        """
        # --- 1. 顺序通过层堆栈 ---
        # 简单地遍历 `self.layers` 中的每一个层模块，
        # 并将上一层的输出作为下一层的输入。
        for layer in self.layers:
            x = layer(x,
                      rotary_emb=rotary_emb,
                      padding_mask=padding_mask
                      )

        # --- 2. 应用最终归一化 ---
        # 在通过所有层之后，将最终的输出通过归一化层。
        x = self.norm(x)

        return x
