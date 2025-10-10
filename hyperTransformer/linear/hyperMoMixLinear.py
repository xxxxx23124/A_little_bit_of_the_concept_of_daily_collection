import torch
import torch.nn as nn
from einops import rearrange
import math
from torch.utils.checkpoint import checkpoint

class HyperMoMixLinear(nn.Module):
    """
    内置了一个辅助损失（entropy loss）来鼓励专家使用多样性。
    先对专家的扁平化权重进行加权求和，然后执行一次Monarch矩阵乘法。
    """

    def __init__(self,
                 in_features,
                 out_features,
                 compressed_feature_dim,
                 num_monarchs,
                 reg_strength=1e-2,
                 use_checkpointing:bool=True
                 ):
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
        self.reg_strength = reg_strength
        self.auxiliary_losses = []

        self.in_features = in_features
        self.out_features = out_features
        self.compressed_feature_dim = compressed_feature_dim

        # 压缩输入特征的模块
        self.compressor = nn.Sequential(
            nn.Linear(in_features, compressed_feature_dim),
            nn.SiLU()
        )

        # 多个权重生成器
        self.M1_gens = nn.ModuleList(
            [nn.Linear(compressed_feature_dim, self.n_in * self.n_in * self.n_out) for _ in range(num_monarchs)]
        )
        self.M2_gens = nn.ModuleList(
            [nn.Linear(compressed_feature_dim, self.n_out * self.n_in * self.n_out) for _ in range(num_monarchs)]
        )

        # 混合器、缩放器和偏置生成器
        self.mixer = nn.Sequential(
            nn.Linear(compressed_feature_dim, compressed_feature_dim),
            nn.SiLU(),
            nn.Linear(compressed_feature_dim, num_monarchs)
        )
        self.biasor = nn.Linear(compressed_feature_dim, out_features)
        self.ratio_gen = nn.Linear(compressed_feature_dim, 1)

        self.use_checkpointing = use_checkpointing
        self.module_name = f"{self.__class__.__name__}_{hex(id(self))}"

    def forward(self, x):
        # 压缩特征
        # x: (b, ..., in_features)
        compressed_features = self.compressor(x)  # (b, ..., compressed_features_dim)
        # 这确保了不会累积上一个batch的损失
        self.auxiliary_losses = []
        # 在这里应用鼓励多专家的正则化
        self._apply_routing_regularization(compressed_features)

        if self.training and self.use_checkpointing:
            # 对权重生成和融合部分进行 checkpoint
            fused_M1_flat, fused_M2_flat = checkpoint(
                self._generate_and_fuse_weights, compressed_features
            )
        else:
            fused_M1_flat, fused_M2_flat = self._generate_and_fuse_weights(compressed_features)

        # 使用融合后的权重进行一次 Monarch 计算
        y = self.forward_with_weights(x, fused_M1_flat, fused_M2_flat)

        # 缩放
        ratio = self.ratio_gen(compressed_features)  # (b, ..., 1)
        y = y * ratio

        # 添加动态偏置
        bias = self.biasor(compressed_features)  # (b, ..., out_features)
        y = y + bias

        return y

    def _generate_and_fuse_weights(self, compressed_features):
        """
        这个辅助函数包含了最消耗激活值内存的部分。
        我们将对它进行 checkpoint。
        """
        # 2. 生成所有专家的扁平化权重
        all_M1_flat = [gen(compressed_features) for gen in self.M1_gens]
        all_M2_flat = [gen(compressed_features) for gen in self.M2_gens]

        stacked_M1_flat = torch.stack(all_M1_flat, dim=0)
        stacked_M2_flat = torch.stack(all_M2_flat, dim=0)

        # 3. 获取混合系数
        coeffs = self.mixer(compressed_features)
        coeffs = torch.softmax(coeffs, dim=-1)

        # 4. 加权融合权重
        fused_M1_flat = torch.einsum('k...d, ...k -> ...d', stacked_M1_flat, coeffs)
        fused_M2_flat = torch.einsum('k...d, ...k -> ...d', stacked_M2_flat, coeffs)

        return fused_M1_flat, fused_M2_flat

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

    def _apply_routing_regularization(self, compressed_features):
        """
        计算并注册一个只作用于 mixer 的“负载均衡损失”。
        """
        if not (self.training and self.num_monarchs > 1 and self.reg_strength > 0):
            return

        # 使用 .detach() 来切断与 compressor 的梯度连接
        detached_features = compressed_features.detach()

        # 在分离的计算图上重新计算 mixer 的输出
        logits = self.mixer(detached_features)
        coeffs = torch.softmax(logits, dim=-1)

        # 将所有非专家维度展平，得到 (N, k) 的形状，N是token总数
        flat_coeffs = rearrange(coeffs, '... k -> (...) k')

        # 计算每个专家在批次中处理的token的平均比例 (f_i)
        avg_expert_prob = torch.mean(flat_coeffs, dim=0)

        # 计算每个专家在其被路由到的token上的平均门控值 (P_i)
        avg_expert_gate = torch.mean(flat_coeffs, dim=0)

        # 负载均衡损失是 f_i 和 P_i 的点积，乘以专家数量进行缩放
        # 这是为了鼓励所有专家被均匀使用
        load_balancing_loss = self.num_monarchs * torch.sum(avg_expert_prob * avg_expert_gate)

        self.auxiliary_losses.append(load_balancing_loss * self.reg_strength)

    def _check_expert_health(self, coeffs, threshold=0.1):
        """
        一个简单的诊断函数，用于检查是否有专家的权重过低。
        这是一个“暴力”的检查，会直接打印到控制台。
        """
        # 只在训练模式下检查，避免在推理时产生不必要的IO开销
        if not self.training:
            return

        # 检查批次中是否有任何一个token的任何一个专家的权重低于阈值
        if torch.any(coeffs < threshold):
            # 找出权重低于阈值的专家索引
            # (coeffs < threshold)会返回一个布尔张量，.nonzero()找到所有True的坐标
            problematic_experts = (coeffs < threshold).nonzero(as_tuple=False)

            # 为了避免信息刷屏，我们只打印第一个检测到的问题
            first_problem = problematic_experts[0]

            # 获取有问题的专家索引
            expert_idx = first_problem[-1].item()

            # 获取该专家在那个具体位置的权重值
            problematic_value = coeffs[tuple(first_problem[:-1])][expert_idx].item()

            # 打印警报
            print(f"\n/!\\ --- EXPERT HEALTH WARNING --- /!\\")
            print(f"  - Module: '{self.module_name}'")
            print(f"  - Problem: Expert {expert_idx} has a dangerously low activation weight.")
            print(f"  - Value: {problematic_value:.4f} (Threshold: {threshold})")
            print(
                f"  - All expert weights at this position: {[f'{c:.4f}' for c in coeffs[tuple(first_problem[:-1])].tolist()]}")
            print(f"/!\\ ----------------------------- /!\\\n")


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