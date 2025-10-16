import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from experiment.Transformer.rmsNorm import RMSNorm
from experiment.Transformer.rotaryEmbedding import RotaryEmbedding
from experiment.Transformer.kvCache import KVCache
from experiment.Transformer.decoderLayer.baseDecoderLayer import BaseDecoderLayer

class Decoder(nn.Module):
    """
    一个通用的、配置驱动的 Transformer 解码器堆栈。

    类似于通用 Encoder，此类通过接收一个定义了层类型的“配方”来动态构建
    解码器堆栈。这使得混合不同类型的解码器层（例如，标准层与专门用于
    特定任务的层）变得非常简单。
    """

    def __init__(self,
                 layer_recipe: list[type[BaseDecoderLayer]],
                 d_model: int,
                 **layer_kwargs
                 ):
        """
        Args:
            layer_recipe (List[Type[BaseDecoderLayer]]):
                一个类型列表，定义了解码器堆栈中每一层的类。
                列表的顺序决定了层在堆栈中的顺序。

            d_model (int):
                模型的维度，这是所有层共享的基本参数。

            **layer_kwargs:
                一个包含关键字参数的字典，它将被“透传”给 `layer_recipe` 中
                每一个层的构造函数 (__init__)。
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
                context: torch.Tensor,
                rotary_emb: RotaryEmbedding | None,
                all_kv_caches: list[tuple[KVCache, KVCache]] | None = None,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        解码器堆栈的前向传播。

        Args:
            x (torch.Tensor):
                目标序列的嵌入，形状为 (batch, target_seq_len, d_model)。
            context (torch.Tensor):
                编码器的输出，形状为 (batch, source_seq_len, d_model)。
            rotary_emb (RotaryEmbedding | None):
                旋转位置编码模块，将传递给每一层。
            all_kv_caches (List[Tuple[KVCache, KVCache]] | None, optional):
                一个列表，包含了所有层的KV缓存。
                `all_kv_caches[i]` 是第 `i` 层的缓存元组 `(self_attn_cache, cross_attn_cache)`。
                在训练时通常为 None，在自回归推理时使用。Defaults to None.

        Returns:
            torch.Tensor:
                解码器的最终输出，形状与输入 x 相同。
        """

        for i, layer in enumerate(self.layers):
            layer_kv_caches = all_kv_caches[i] if all_kv_caches is not None else None

            if self.training and self.use_checkpointing:
                x = checkpoint(
                    layer,
                    x=x,
                    context=context,
                    rotary_emb=rotary_emb,
                    self_attn_kv_cache=layer_kv_caches[0],
                    cross_attn_kv_cache=layer_kv_caches[1],
                    attention_mask=attention_mask,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    context=context,
                    rotary_emb=rotary_emb,
                    self_attn_kv_cache=layer_kv_caches[0],
                    cross_attn_kv_cache=layer_kv_caches[1],
                    attention_mask=attention_mask
                )

        x = self.norm(x)

        return x