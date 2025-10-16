import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from experiment.Transformer.rmsNorm import RMSNorm
from experiment.Transformer.rotaryEmbedding import RotaryEmbedding
from experiment.Transformer.kvCache import KVCache
from experiment.Transformer.decoderLayer.baseDecoderLayer import BaseDecoderLayer

class Decoder(nn.Module):
    def __init__(self,
                 layer_recipe: list[type[BaseDecoderLayer]],
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
                context: torch.Tensor,
                rotary_emb: RotaryEmbedding | None,
                all_kv_caches: list[tuple[KVCache, KVCache]] | None = None,
                self_attention_mask: torch.Tensor | None = None,
                cross_attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:

        batch_size, tgt_seq_len, _ = x.shape

        # 生成因果掩码：上三角为 True (屏蔽未来 token)
        causal_mask = torch.triu(
            torch.ones((tgt_seq_len, tgt_seq_len), device=x.device, dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0).unsqueeze(1)  # 形状: (1, 1, tgt_seq_len, tgt_seq_len)

        if self_attention_mask is not None:
            # 转换 padding mask 为键填充掩码：(batch_size, 1, 1, tgt_seq_len)，True 为 padding 位置
            key_padding_mask = (self_attention_mask == 0).unsqueeze(1).unsqueeze(1)

            # 扩展键填充掩码到 (batch_size, 1, tgt_seq_len, tgt_seq_len)
            key_padding_mask = key_padding_mask.expand(batch_size, 1, tgt_seq_len, tgt_seq_len)

            # 结合因果掩码和填充掩码：逻辑或 (任何需要屏蔽的位置都为 True)
            self_attention_mask = causal_mask | key_padding_mask
        else:
            # 无 padding mask，只用因果掩码（广播到 batch_size）
            self_attention_mask = causal_mask.expand(batch_size, 1, tgt_seq_len, tgt_seq_len)

        if cross_attention_mask is not None:
            # 假设输入 cross_attention_mask 是 (batch_size, seq_len)，值 0 为 padding
            # 转换为 PyTorch 注意力掩码：(batch_size, 1, 1, seq_len)，True 为屏蔽位置
            cross_attention_mask = (cross_attention_mask == 0).unsqueeze(1).unsqueeze(2)

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
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=cross_attention_mask,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    context=context,
                    rotary_emb=rotary_emb,
                    self_attn_kv_cache=layer_kv_caches[0],
                    cross_attn_kv_cache=layer_kv_caches[1],
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=cross_attention_mask
                )

        x = self.norm(x)

        return x