import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class StaticCompositeLinear(nn.Module):
    """
    一个复合线性层，根据输入动态地加权融合一组静态的线性层（专家）的参数。

    核心机制:
    1. 内部维护一组固定的、可学习的专家线性层（Static Experts）。
    2. 一个门控网络（Mixer）根据输入生成每个专家的混合权重。
    3. 在权重空间（Weight-Space）中，将所有专家的权重和偏置根据混合权重进行加权求和，
       生成一个动态的、为当前输入定制的权重矩阵和偏置。
    4. 使用这个动态合成的权重和偏置执行一次标准的线性计算。
    5. （可选）内置一个辅助损失（负载均衡损失），以鼓励专家使用的多样性。
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_softExperts: int,
                 compressed_feature_dim: int,
                 reg_strength: float = 1e-2,
                 use_checkpointing: bool = False
                 ):
        super().__init__()

        # --- 基本属性 ---
        self.in_features = in_features
        self.out_features = out_features
        self.num_softExperts = num_softExperts
        self.compressed_feature_dim = compressed_feature_dim
        self.reg_strength = reg_strength
        self.use_checkpointing = use_checkpointing
        self.auxiliary_losses = []
        self.module_name = f"{self.__class__.__name__}_{hex(id(self))}"

        # --- 核心组件 ---
        # 静态专家组 (Static Experts)
        # 我们将权重和偏置分开存储，以便于融合
        self.expert_weights = nn.Parameter(torch.empty(num_softExperts, out_features, in_features))
        self.expert_biases = nn.Parameter(torch.empty(num_softExperts, out_features))

        # 动态混合器 (Mixer)
        self.mixer = nn.Linear(compressed_feature_dim, num_softExperts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, ..., in_features)
        # --- 应用负载均衡损失 (仅在训练时) ---
        if self.training:
            self._apply_routing_regularization(x)
        # --- 获取混合系数 ---
        # coeffs: (b, ..., num_experts)
        coeffs = self.mixer(x)
        coeffs = torch.softmax(coeffs, dim=-1)

        # --- 动态融合专家参数 ---
        if self.training and self.use_checkpointing:
            # 对权重融合部分进行 checkpoint 以节省显存
            fused_weight, fused_bias = checkpoint(
                self._fuse_experts, coeffs, use_reentrant=False
            )
        else:
            fused_weight, fused_bias = self._fuse_experts(coeffs)

        # --- 执行一次线性计算 ---
        # 核心操作：y = xW^T + b
        # 为了处理任意批次维度，我们使用 einsum
        # x: (..., in), fused_weight: (..., out, in) -> y: (..., out)
        y = torch.einsum('...i, ...oi -> ...o', x, fused_weight) + fused_bias

        return y

    def _fuse_experts(self, coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        这个辅助函数包含了参数融合的核心逻辑，是 checkpointing 的目标。
        """
        # coeffs: (b, ..., num_experts)
        # expert_weights: (num_experts, out_features, in_features)

        # 使用 einsum 进行高效的加权求和
        # '...k, koi -> ...oi'
        # k: num_experts, o: out_features, i: in_features
        # ...: 批次和序列维度
        fused_weight = torch.einsum('...k, koi -> ...oi', coeffs, self.expert_weights)
        fused_bias = torch.einsum('...k, ko -> ...o', coeffs, self.expert_biases)

        return fused_weight, fused_bias

    def clear_auxiliary_losses(self):
        """在每个训练步骤开始时调用，以清除上一批次的辅助损失。"""
        self.auxiliary_losses = []

    def _apply_routing_regularization(self, x: torch.Tensor):
        """
        计算并注册一个“负载均衡损失”，以鼓励所有专家被均匀使用。
        这与稀疏MoE中的负载均衡损失思想一致。
        """
        if not (self.training and self.num_softExperts > 1 and self.reg_strength > 0):
            return

        # 使用 .detach() 来确保这个损失只影响 mixer 的参数
        detached_features = x.detach()

        # 在分离的计算图上重新计算 mixer 的输出
        logits = self.mixer(detached_features)
        coeffs = torch.softmax(logits, dim=-1)

        # 将所有非专家维度展平，得到 (N, k) 的形状，N是token总数
        flat_coeffs = rearrange(coeffs, '... k -> (...) k')

        # 计算每个专家在批次中被分配的平均概率
        avg_expert_prob = torch.mean(flat_coeffs, dim=0)

        # 计算每个专家在其被路由到的token上的平均门控值
        # 这里为了简化，直接用 avg_expert_prob
        avg_expert_gate = avg_expert_prob

        # 负载均衡损失是 f_i 和 P_i 的点积，乘以专家数量进行缩放
        load_balancing_loss = self.num_softExperts * torch.sum(avg_expert_prob * avg_expert_gate)

        self.auxiliary_losses.append(load_balancing_loss * self.reg_strength)


if __name__ == '__main__':
    import math

    # --- 测试参数 ---
    d_model = 64
    d_ffn = 256
    num_experts = 8
    comp_dim = 32
    seq_len = 10
    batch_size = 4

    print("--- 实例化 StaticCompositeLinear ---")
    model = StaticCompositeLinear(
        in_features=d_model,
        out_features=d_ffn,
        num_softExperts=num_experts,
        compressed_feature_dim=comp_dim,
        reg_strength=0.01,
        use_checkpointing=True  # 测试 checkpointing
    )
    model.train()  # 设置为训练模式以激活辅助损失和checkpointing

    # --- 测试 3D 输入 (batch, seq, features) ---
    print("\n--- 测试 3D 输入 ---")
    x_3d = torch.randn(batch_size, seq_len, d_model)

    # 清除可能存在的旧损失
    model.clear_auxiliary_losses()

    output_3d = model(x_3d)

    print(f"Input shape: {x_3d.shape}")
    print(f"Output shape: {output_3d.shape}")
    assert output_3d.shape == (batch_size, seq_len, d_ffn)

    # 检查辅助损失是否生成
    print(f"Generated auxiliary losses: {model.auxiliary_losses}")
    assert len(model.auxiliary_losses) == 1

    # --- 测试 2D 输入 (batch, features) ---
    print("\n--- 测试 2D 输入 ---")
    x_2d = torch.randn(batch_size, d_model)

    model.clear_auxiliary_losses()
    output_2d = model(x_2d)

    print(f"Input shape (2D): {x_2d.shape}")
    print(f"Output shape (2D): {output_2d.shape}")
    assert output_2d.shape == (batch_size, d_ffn)
    print(f"Generated auxiliary losses: {model.auxiliary_losses}")
    assert len(model.auxiliary_losses) == 1

    print("\n`StaticCompositeLinear` 类实现成功，并通过了基本测试！")

    # --- 打印模型结构 ---
    # print("\n--- 模型结构 ---")
    # print(model)
