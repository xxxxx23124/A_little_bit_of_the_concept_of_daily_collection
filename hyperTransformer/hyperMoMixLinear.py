import torch
import torch.nn as nn
from einops import rearrange
import math


class HyperMoMixLinear(nn.Module):
    """
    先对专家的扁平化权重进行加权求和，然后执行一次Monarch矩阵乘法。
    """

    def __init__(self, in_features, out_features, compressed_feature_dim, num_monarchs=2):
        super().__init__()
        n_in = math.isqrt(in_features)
        n_out = math.isqrt(out_features)
        if not n_in ** 2 == in_features:
            raise ValueError(f"in_features ({in_features}) must be a perfect square.")
        if not n_out ** 2 == out_features:
            raise ValueError(f"out_features ({out_features}) must be a perfect square.")

        self.n_in = n_in
        self.n_out = n_out

        self.num_monarchs = num_monarchs

        self.in_features = in_features
        self.out_features = out_features
        self.compressed_feature_dim = compressed_feature_dim

        # 压缩输入特征的模块
        self.compressor = nn.Sequential(nn.Linear(in_features, compressed_feature_dim), nn.Tanh())

        # 多个权重生成器
        self.M1_gens = nn.ModuleList(
            [nn.Linear(compressed_feature_dim, self.n_in * self.n_in * self.n_out) for _ in range(num_monarchs)]
        )
        self.M2_gens = nn.ModuleList(
            [nn.Linear(compressed_feature_dim, self.n_out * self.n_in * self.n_out) for _ in range(num_monarchs)]
        )

        # 混合器、缩放器和偏置生成器
        self.mixer = nn.Linear(compressed_feature_dim, num_monarchs)
        self.biasor = nn.Linear(compressed_feature_dim, out_features)
        self.ratio_gen = nn.Linear(compressed_feature_dim, 1)

    def forward(self, x):
        # 1. 压缩特征
        # x: (b, ..., in_features)
        compressed_features = self.compressor(x)  # (b, ..., compressed_features_dim)

        # 2. 生成所有专家的扁平化权重
        # 每个 M1_flat_k 的形状: (b, ..., weight_size)
        all_M1_flat = [gen(compressed_features) for gen in self.M1_gens]
        all_M2_flat = [gen(compressed_features) for gen in self.M2_gens]

        # 堆叠权重: (k, b, ..., weight_size)
        stacked_M1_flat = torch.stack(all_M1_flat, dim=0)
        stacked_M2_flat = torch.stack(all_M2_flat, dim=0)

        # 3. 获取混合系数
        coeffs = self.mixer(compressed_features)  # (b, ..., k)
        coeffs = torch.softmax(coeffs, dim=-1)

        # 4. 加权融合权重
        # 'k' 是专家维度, '...' 代表批次和序列等任意维度, 'd' 是权重大小
        # fused_M1_flat: (b, ..., weight_size)
        fused_M1_flat = torch.einsum('k...d, ...k -> ...d', stacked_M1_flat, coeffs)
        fused_M2_flat = torch.einsum('k...d, ...k -> ...d', stacked_M2_flat, coeffs)

        # 5. 使用融合后的权重进行一次 Monarch 计算
        y = self.forward_with_weights(x, fused_M1_flat, fused_M2_flat)

        # 6. 缩放
        ratio = self.ratio_gen(compressed_features)  # (b, ..., 1)
        y = y * ratio

        # 7. 添加动态偏置
        bias = self.biasor(compressed_features)  # (b, ..., out_features)
        y = y + bias

        return y

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
    model = HyperMoMixLinear(in_features=d_model, out_features=d_ffn, compressed_feature_dim=comp_dim, num_monarchs=4)

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