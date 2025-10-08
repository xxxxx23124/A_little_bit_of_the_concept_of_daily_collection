import torch
import torch.nn as nn
from einops import rearrange
import math


class HyperMonarchLinear(nn.Module):
    """
    实现一个两阶段的Monarch矩阵 W = M_2 * M_1。
    这对应于论文中提到的 (d_in, d_out) = (n*n, n*n) 的情况。
    """

    def __init__(self, in_features, out_features, compressed_feature_dim):
        super().__init__()

        if not math.isqrt(in_features) ** 2 == in_features:
            raise ValueError(f"in_features ({in_features}) must be a perfect square.")
        if not math.isqrt(out_features) ** 2 == out_features:
            raise ValueError(f"out_features ({out_features}) must be a perfect square.")

        self.in_features = in_features
        self.out_features = out_features
        self.compressed_feature_dim = compressed_feature_dim

        self.n_in = math.isqrt(in_features)
        self.n_out = math.isqrt(out_features)

        # 权重生成器
        self.M1_gen = nn.Linear(compressed_feature_dim, self.n_in * self.n_in * self.n_out)
        self.M2_gen = nn.Linear(compressed_feature_dim, self.n_out * self.n_in * self.n_out)

    def forward(self, x, compressed_features):
        """
        标准的 forward 方法，先生成权重再计算。
        """
        M1_flat = self.M1_gen(compressed_features)
        M2_flat = self.M2_gen(compressed_features)
        return self.forward_with_weights(x, M1_flat, M2_flat)

    def forward_with_weights(self, x, M1_flat, M2_flat):
        """
        使用外部传入的扁平化权重进行计算。
        这允许我们在混合专家模型中先融合权重，再进行计算，以提高效率。
        """
        # M1: (b, ..., n_in, n_in, n_out)
        M1 = rearrange(M1_flat, '... (g i o) -> ... g i o', g=self.n_in, i=self.n_in, o=self.n_out)
        # M2: (b, ..., n_out, n_in, n_out)
        M2 = rearrange(M2_flat, '... (g i o) -> ... g i o', g=self.n_out, i=self.n_in, o=self.n_out)

        # --- Stage 1: 应用 M1 ---
        # x1_in: (b, ..., n_in, n_in)
        x1_in = rearrange(x, '... (h w) -> ... h w', h=self.n_in, w=self.n_in)
        # x1_out: (b, ..., n_in, n_out)
        x1_out = torch.einsum('... gi, ... gio -> ... go', x1_in, M1)

        # --- Stage 2: 应用 M2 (带置换) ---
        # x2_in: (b, ..., n_out, n_in)
        x2_in = rearrange(x1_out, '... g o -> ... o g')
        # x2_out: (b, ..., n_out, n_out)
        x2_out = torch.einsum('... gi, ... gio -> ... go', x2_in, M2)

        # --- 逆置换并展平 ---
        # y: (b, ..., out_features)
        y = rearrange(x2_out, '... h w -> ... (h w)')
        return y


if __name__ == '__main__':
    # 假设输入是 (batch, seq_len, features)
    # d_in = 64 (8*8), d_out = 256 (16*16)
    d_model = 64
    d_ffn = 256
    comp_dim = 32
    seq_len = 10
    batch_size = 4

    # 创建模型
    model = HyperMonarchLinear(in_features=d_model, out_features=d_ffn, compressed_feature_dim=comp_dim)

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_ffn)

    # 测试二维输入
    x_2d = torch.randn(batch_size, d_model)
    output_2d = model(x_2d)
    print(f"\nInput shape (2D): {x_2d.shape}")
    print(f"Output shape (2D): {output_2d.shape}")
    assert output_2d.shape == (batch_size, d_ffn)
    print("\n优化后的代码运行成功，且能处理多维输入！")