import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, base, head_dim, max_seq_len):
        super().__init__()
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # emb 形状为 (max_seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册 buffer，形状为 (1, 1, max_seq_len, head_dim) 以便广播
        self.register_buffer("freqs_cos", emb.cos()[None, None, :, :])
        self.register_buffer("freqs_sin", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        # x shape: (..., dim)
        # x1, x2 shape: (..., dim / 2)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # shape: (..., dim)

    def forward(self, x, seq_len, seq_len_offset=0):
        # x shape: (batch, n_heads, seq_len, d_k)
        cos = self.freqs_cos[:, :, seq_len_offset: seq_len_offset + seq_len, :]  # shape: (1, 1, seq_len, d_k)
        sin = self.freqs_sin[:, :, seq_len_offset: seq_len_offset + seq_len, :]  # shape: (1, 1, seq_len, d_k)
        # RoPE 旋转的数学等价实现
        # (x * cos) + (rotate_half(x) * sin)
        # 这等价于复数乘法 (x_r + i*x_i) * (cos + i*sin) 的实部和虚部
        rotated_x = (x * cos) + (self.rotate_half(x) * sin)
        return rotated_x
    