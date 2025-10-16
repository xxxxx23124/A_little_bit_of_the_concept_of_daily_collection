import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from experiment.Transformer.rmsNorm import RMSNorm
from experiment.Transformer.rotaryEmbedding import RotaryEmbedding
from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer


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

        self.use_checkpointing = layer_kwargs.get('use_checkpointing', False)

        layers = [
            layer_cls(d_model=d_model, **layer_kwargs)
            for layer_cls in layer_recipe
        ]

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                rotary_emb: RotaryEmbedding | None,
                attention_mask: Tensor | None = None) -> torch.Tensor:
        """
        编码器堆栈的前向传播。

        Args:
            x (torch.Tensor):
                输入张量，形状为 (batch_size, seq_len, d_model)。
                这通常是词嵌入和位置编码相加后的结果。

            rotary_emb (RotaryEmbedding | None):
                旋转位置编码模块。这个对象将被原封不动地传递给堆栈中的
                每一个层，由层内部的注意力模块决定如何使用它。

            attention_mask (Tensor | None): padding 掩码

        Returns:
            torch.Tensor:
                编码器的最终输出，形状与输入 x 相同。
        """

        for layer in self.layers:
            if self.training and self.use_checkpointing:
                # 使用 checkpoint
                x = checkpoint(
                    layer,
                    x=x,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask
                )

        x = self.norm(x)

        return x
