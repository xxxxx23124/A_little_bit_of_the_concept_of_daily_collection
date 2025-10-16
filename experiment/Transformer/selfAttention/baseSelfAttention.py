import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod


from experiment.Transformer.kvCache import KVCache
from experiment.Transformer.rotaryEmbedding import RotaryEmbedding


class BaseSelfAttention(nn.Module, ABC):
    """
    自注意力机制的基类，封装了通用的计算流程。

    子类需要通过调用 super().__init__() 并实现 _init_projections 方法
    来定义具体的 Q, K, V 投影层。
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0, "d_model must be divisible by nheads"

        # 初始化 Q, K, V 投影层为 None，由子类负责定义
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        # 调用抽象方法，强制子类实现投影层的初始化
        self._init_projections(d_model=d_model, num_heads=num_heads, **kwargs)

        # 确保子类已经正确初始化了投影层
        assert all(p is not None for p in [self.q_proj, self.k_proj, self.v_proj]), \
            "Q, K, V projection layers must be initialized in _init_projections"

        # 输出投影层是共有的
        self.out_proj = nn.Linear(d_model, d_model)

    @abstractmethod
    def _init_projections(self, **kwargs):
        """
        抽象方法，子类必须实现此方法来初始化 Q, K, V 投影层。
        例如:
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        """
        raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                rotary_emb: RotaryEmbedding | None = None,
                kv_cache: KVCache | None = None):
        """
        通用自注意力前向传播。通过参数组合自动适应不同模式。

        Args:
            x (torch.Tensor):
                输入张量。形状 (B, S, D)。
                - 训练/编码时, S > 1。
                - 推理时, S = 1。

            attention_mask (torch.Tensor, optional):
                用于屏蔽特定位置的注意力。
                - 在 Encoder 中，这是 Padding Mask。
                - 在 Decoder 训练中，这是 Causal + Padding Mask。
                - 在 Decoder 推理中，为 None。
                形状通常为 (B, 1, S_q, S_k) 或可广播的形状。
                PyTorch's F.scaled_dot_product_attention 期望的 mask 是布尔类型，
                True 表示该位置被忽略（masked）。

            rotary_emb (RotaryEmbedding, optional):
                旋转位置编码模块。如果提供，则应用RoPE。

            kv_cache (KVCache, optional):
                Key-Value 缓存。如果提供，则模块进入“自回归推理”模式。
        """
        B, S, D = x.shape
        is_inference = kv_cache is not None

        # 1. 计算 Q, K, V
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # 2. 重塑以适应多头
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.num_heads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.num_heads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.num_heads)

        # 3. 应用旋转位置编码 (RoPE)
        past_len = len(kv_cache) if is_inference else 0
        if rotary_emb is not None:
            # 在推理时，只对当前新 token (S=1) 应用RoPE，但要传入正确的 past_len
            # 在训练时，对整个序列 (S>1) 应用RoPE，past_len=0
            query = rotary_emb(query, seq_len=S, past_len=past_len)
            key = rotary_emb(key, seq_len=S, past_len=past_len)

        # 4. 更新并使用 KV 缓存 (仅在推理模式下)
        if is_inference:
            # 在推理模式下，attention_mask 应为 None，因为 Q 长度为1，天然因果。
            # 即使有 padding，也应在生成结束时通过特殊 token 处理，而不是通过 mask。
            assert attention_mask is None, "attention_mask should not be used during autoregressive inference"

            kv_cache.update(key, value)
            key, value = kv_cache.get()

        # 5. 执行缩放点积注意力
        # is_causal 标志是一个方便的快捷方式，但它不能与 attention_mask 同时使用。
        # 当我们需要处理 padding 时，必须自己构建掩码并传入 attention_mask。
        # 在这里，我们假设外部调用者会正确构建并传入所需的 `attention_mask`。
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False  # 让 attention_mask 完全控制掩码行为
        )

        # 6. 整理输出并应用输出投影
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)

"""

如何调用它：

Encoder Layer:
# 在外部创建 padding_mask
# x 是 (B, S, D), padding_tokens 是 (B, S) 的布尔张量 (True for padding)
padding_mask = create_padding_mask(padding_tokens) # (B, 1, 1, S)
output = self_attention(x, attention_mask=padding_mask, rotary_emb=None, kv_cache=None)

Decoder Layer (训练时):
# 在外部创建组合掩码
# target 是 (B, S, D), target_padding 是 (B, S)
combined_mask = create_causal_and_padding_mask(target_padding)
output = self_attention(x, attention_mask=combined_mask, rotary_emb=rope, kv_cache=None)

Decoder Layer (推理时):
# x 是当前 token (B, 1, D)
# kv_cache 是从上层传入的
output = self_attention(x, attention_mask=None, rotary_emb=rope, kv_cache=kv_cache)

"""
