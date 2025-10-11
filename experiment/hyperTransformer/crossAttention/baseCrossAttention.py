import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod

from experiment.hyperTransformer.kvCache import KVCache


class BaseCrossAttention(nn.Module, ABC):
    """
    交叉注意力机制的基类，封装了通用的计算流程。

    子类需要通过调用 super().__init__() 并实现 _init_projections 方法
    来定义具体的 Q, K, V 投影层。
    """
    def __init__(self, d_model, num_heads, **kwargs):
        """
        初始化交叉注意力模块。

        Args:
            d_model (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            **kwargs: 传递给 _init_projections 的额外参数。
        """
        super().__init__()
        self.nheads = num_heads
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
        交叉注意力的 K 和 V 投影可能需要处理不同维度的输入（如果 d_kv 与 d_model 不同）。

        例如:
        d_kv = kwargs.get('d_kv', self.d_model) # 假设 context 的维度是 d_kv
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(d_kv, self.d_model)
        self.v_proj = nn.Linear(d_kv, self.d_model)
        """
        raise NotImplementedError

    def forward(self, x, context, kv_cache: KVCache | None):
        """
        通用的交叉注意力前向传播逻辑。

        Args:
            x (torch.Tensor): 查询序列 (Query)，形状为 (B, S_q, D)。
                              通常是解码器的输入。
            context (torch.Tensor): 键/值序列 (Key/Value)，形状为 (B, S_kv, D_kv)。
                                     通常是编码器的输出。
            kv_cache (KVCache | None): Key-Value 缓存。用于在推理时缓存 `context`
                                       的 K, V 投影，避免重复计算。
        """
        B_q, S_q, D_q = x.shape

        # 1. 计算 Query
        query = self.q_proj(x)
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.nheads)

        # 2. 计算 Key 和 Value
        # 如果提供了缓存且缓存非空，则直接从缓存获取 K, V
        if kv_cache is not None and len(kv_cache) > 0:
            key, value = kv_cache.get()
        # 否则，从 context 计算 K, V，并更新缓存
        else:
            key = self.k_proj(context)
            value = self.v_proj(context)

            key = rearrange(key, 'b s (h d) -> b h s d', h=self.nheads)
            value = rearrange(value, 'b s (h d) -> b h s d', h=self.nheads)

            if kv_cache is not None:
                kv_cache.update(key, value)

        # 3. 执行缩放点积注意力
        # is_causal=False，因为查询可以关注上下文中的任何位置。
        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=False
        )

        # 4. 整理输出并应用输出投影
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)
