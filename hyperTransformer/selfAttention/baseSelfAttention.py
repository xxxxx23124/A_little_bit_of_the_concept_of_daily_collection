import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod


from hyperTransformer.kvCache import KVCache
from hyperTransformer.rotaryEmbedding import RotaryEmbedding


class BaseSelfAttention(nn.Module, ABC):
    """
    自注意力机制的基类，封装了通用的计算流程。

    子类需要通过调用 super().__init__() 并实现 _init_projections 方法
    来定义具体的 Q, K, V 投影层。
    """

    def __init__(self, d_model, nheads, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.d_model = d_model
        assert d_model % self.nheads == 0, "d_model must be divisible by nheads"

        # 初始化 Q, K, V 投影层为 None，由子类负责定义
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        # 调用抽象方法，强制子类实现投影层的初始化
        self._init_projections(**kwargs)

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

    def forward(self, x, rotary_emb:RotaryEmbedding|None, kv_cache:KVCache|None, use_causal_mask):
        """
        通用的前向传播逻辑。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, S, D)
            rotary_emb: 旋转位置编码模块
            kv_cache (KVCache, optional): Key-Value 缓存。
            use_causal_mask (bool): 是否使用因果掩码
        """
        B, S, D = x.shape

        # 1. 使用子类定义的投影层计算 Q, K, V
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # 2. 重塑张量以适应多头注意力
        # (B, S, D) -> (B, nheads, S, head_dim)
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.nheads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.nheads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.nheads)

        # 3. 应用旋转位置编码 (RoPE)
        past_len = len(kv_cache) if kv_cache is not None else 0
        if rotary_emb is not None:
            query = rotary_emb(query, seq_len=S, past_len=past_len)
            key = rotary_emb(key, seq_len=S, past_len=past_len)

        # 4. 更新并使用 KV 缓存 (用于推理加速)
        if kv_cache is not None:
            kv_cache.update(key, value)
            key, value = kv_cache.get()

        # 5. 执行缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=use_causal_mask and S > 1
        )

        # 6. 整理输出并应用输出投影
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)
