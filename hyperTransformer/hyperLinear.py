import torch
import torch.nn as nn
from einops import rearrange


class LoRAHyperParams(nn.Module):
    """为MainNet的单层生成权重的超网络"""

    def __init__(self, input_dim, output_dim, dynamic_dim, rank, ratio_dim):
        super(LoRAHyperParams, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # 阶段一：两个子网络用于压缩输入特征
        self.compressor = nn.Sequential(nn.Linear(input_dim, dynamic_dim), nn.Tanh())

        # 阶段二：使用两个独立的、职责单一的生成器
        self.a_generator = nn.Linear(dynamic_dim, input_dim * rank)
        self.b_generator = nn.Linear(dynamic_dim, rank * output_dim)

        # 第二个子网络从compressor的输出生成偏置b
        self.bias_generator = nn.Linear(dynamic_dim, output_dim)

        # --- 动态 ratio 生成器 ---
        # 标准 LoRA 对权重矩阵 W 的更新如下：
        # W_adapted = W_0 + ΔW
        #
        # 其中 ΔW = (alpha / rank) * (B @ A)
        #
        # W_0：预训练的、冻结的权重。
        # A 和 B：可训练的低秩矩阵。
        # rank：低秩维度。
        # alpha：一个固定的、标量超参数。

        # 一个非常小的网络，只输出一个标量
        self.ratio_generator = nn.Sequential(
            nn.Linear(dynamic_dim, ratio_dim), # 极小的中间层
            nn.Tanh(),
            nn.Linear(ratio_dim, 1)
        )

    def forward(self, x):
        # 注意：这里的输入x是MainNet每一层的输入

        # 阶段一：压缩
        compressed_feat = self.compressor(x)

        # 阶段二：独立生成 A, B 和偏置 b
        a_flat = self.a_generator(compressed_feat)
        b_flat = self.b_generator(compressed_feat)

        # 重塑 A 和 B
        # a_flat: [b, i*r] -> A: [b, i, r]
        A = rearrange(a_flat, 'b (i r) -> b i r', i=self.input_dim, r=self.rank)
        # b_flat: [b, r*o] -> B: [b, r, o]
        B = rearrange(b_flat, 'b (r o) -> b r o', r=self.rank, o=self.output_dim)

        # 生成动态 ratio
        # ratio 的形状: [batch_size, 1] -> [b, 1, 1] 用于广播
        dynamic_ratio = self.ratio_generator(compressed_feat).unsqueeze(-1)

        # [batch_size, output_dim] -> [batch_size, 1, output_dim]
        b = self.bias_generator(compressed_feat).unsqueeze(1)

        # 返回分解后的矩阵 A, B 和其他组件，而不是 W
        return {'A': A, 'B': B, 'b': b, 'ratio': dynamic_ratio}

class HyperLinear(nn.Module):
    def __init__(self, input_dim, output_dim, dynamic_dim, rank, ratio_dim):
        super().__init__()
        # 一个小型的 HyperLayer 动态生成精炼内容的参数
        self.hyper_layer = LoRAHyperParams(input_dim, output_dim, dynamic_dim, rank, ratio_dim)

    def forward(self, x):
        # 为当前层动态生成分解后的权重矩阵 A, B
        # A: [b, in, r], B: [b, r, out], b: [b, 1, out], ratio: [b, 1, 1]
        params = self.hyper_layer(x)
        A = params['A']
        B = params['B']
        bias = params['b']
        ratio = params['ratio']

        # --- 执行优化的矩阵乘法 ---
        # 原始输入 x: [b, in] -> [b, 1, in]
        x_unsqueezed = x.unsqueeze(1)

        # 步骤 1: (x @ A)
        # (b, 1, in) @ (b, in, r) -> (b, 1, r)
        tmp = torch.bmm(x_unsqueezed, A)

        # 步骤 2: (tmp @ B)
        # (b, 1, r) @ (b, r, out) -> (b, 1, out)
        delta_W_x = torch.bmm(tmp, B)

        # 应用动态 ratio
        # (b, 1, out) * (b, 1, 1) -> (b, 1, out)
        delta_W_x_scaled = delta_W_x * ratio

        # 加上偏置
        # (b, 1, out) + (b, 1, out) -> (b, 1, out)
        y = delta_W_x_scaled + bias

        # 移除多余的维度
        y = y.squeeze(1)

        return y