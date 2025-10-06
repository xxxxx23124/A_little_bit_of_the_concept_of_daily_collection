import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hyperTransformer.hyperLinear import HyperLinear

class HyperSelfAttention(nn.Module):
    def __init__(self, d_model, nheads, rank, dynamic_dim, ratio_dim):
        super().__init__()
        self.nheads = nheads
        assert d_model % self.nheads == 0, "d_model must be divisible by nheads"

        self.static_Q = nn.Linear(d_model, d_model)
        self.hyper_K = HyperLinear(d_model, d_model, dynamic_dim, rank, ratio_dim)
        self.hyper_V = HyperLinear(d_model, d_model, dynamic_dim, rank, ratio_dim)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, rotary_emb, kv_cache):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, S, D)
            rotary_emb: 旋转位置编码模块
            kv_cache (KVCache, optional): Key-Value 缓存。默认为 None。
        """
        B, S, D = x.shape

        query = self.static_Q(x)
        key = self.hyper_K(x)
        value = self.hyper_V(x)

        # (B, S, D) -> (B, nheads, S, head_dim)
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.nheads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.nheads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.nheads)

        # 如果使用缓存，需要告知RoPE当前的偏移量（已缓存的长度）
        past_len = len(kv_cache) if kv_cache is not None else 0
        if rotary_emb is not None:
            query = rotary_emb(query, seq_len=S, past_len=past_len)
            key = rotary_emb(key, seq_len=S, past_len=past_len)

        # 更新并使用 KV 缓存
        if kv_cache is not None:
            # 将新计算的 key 和 value 更新到缓存中
            kv_cache.update(key, value)
            # 从缓存中获取完整的 key 和 value
            key, value = kv_cache.get()

        # is_causal 只在没有缓存（即第一次预填充）时为 True
        # 在生成阶段（S=1），causal mask 不是必需的，因为Query只会关注包括自己在内的所有Key
        use_causal_mask = (kv_cache is None)
            
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=use_causal_mask)
        # 整理输出
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)