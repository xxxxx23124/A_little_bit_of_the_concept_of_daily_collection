import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from experiment.Transformer.rmsNorm import RMSNorm
from experiment.Transformer.rotaryEmbedding import RotaryEmbedding
from experiment.Transformer.encoderLayer.baseEncoderLayer import BaseEncoderLayer


class Encoder(nn.Module):
    def __init__(self,
                 layer_recipe: list[type[BaseEncoderLayer]],
                 d_model: int,
                 **layer_kwargs
                 ):
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
                self_attention_mask: Tensor | None = None) -> torch.Tensor:

        if self_attention_mask is not None:
            # 假设输入 self_attention_mask 是 (batch_size, seq_len)，值 0 为 padding
            # 转换为 PyTorch 注意力掩码：(batch_size, 1, 1, seq_len)，True 为屏蔽位置
            self_attention_mask = (self_attention_mask == 0).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            if self.training and self.use_checkpointing:
                # 使用 checkpoint
                x = checkpoint(
                    layer,
                    x=x,
                    rotary_emb=rotary_emb,
                    self_attention_mask=self_attention_mask,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    rotary_emb=rotary_emb,
                    self_attention_mask=self_attention_mask
                )

        x = self.norm(x)

        return x
