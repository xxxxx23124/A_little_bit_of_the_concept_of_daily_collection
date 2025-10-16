import torch
import torch.nn as nn
from einops import rearrange


class StaticCompositeLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_linears: int,
                 reg_strength: float = 1e-2,
                 ):
        super().__init__()

        # --- 基本属性 ---
        self.in_features = in_features
        self.out_features = out_features
        self.num_linears = num_linears
        self.reg_strength = reg_strength
        self.auxiliary_losses = []
        self.module_name = f"{self.__class__.__name__}_{hex(id(self))}"

        # --- 核心组件 ---
        # 静态专家组 (Static Experts)
        self.expert_weights = nn.Parameter(torch.empty(num_linears, out_features, in_features))
        self.expert_biases = nn.Parameter(torch.empty(num_linears, out_features))

        # 动态混合器 (Mixer)
        self.mixer = nn.Linear(in_features, num_linears)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, ..., in_features)
        if self.training:
            self._apply_routing_regularization(x)

        # coeffs: (b, ..., num_experts)
        coeffs = self.mixer(x)
        coeffs = torch.softmax(coeffs, dim=-1)

        expert_outputs = torch.einsum('...i, koi -> ...ko', x, self.expert_weights)

        # 加上每个专家的偏置
        # expert_biases: (k, o)
        # expert_outputs: (..., k, o)
        # PyTorch 的广播机制会自动处理
        expert_outputs += self.expert_biases  # (..., k, o) + (k, o) -> (..., k, o)

        # 根据混合系数对专家输出进行加权求和
        # coeffs: (..., k) -> 扩展为 (..., k, 1)
        # expert_outputs: (..., k, o)
        # 逐元素相乘然后求和: (..., k, 1) * (..., k, o) -> sum(dim=-2)
        # 使用 einsum 再次简化:
        # '...k, ...ko -> ...o'
        y = torch.einsum('...k, ...ko -> ...o', coeffs, expert_outputs)

        return y

    def clear_auxiliary_losses(self):
        """在每个训练步骤开始时调用，以清除上一批次的辅助损失。"""
        self.auxiliary_losses = []

    def _apply_routing_regularization(self, x: torch.Tensor):
        """
        计算并注册一个“负载均衡损失”，以鼓励所有专家被均匀使用。
        这与稀疏MoE中的负载均衡损失思想一致。
        """
        if not (self.training and self.num_linears > 1 and self.reg_strength > 0):
            return

        # 使用 .detach() 来确保这个损失只影响 mixer 的参数
        detached_x = x.detach()

        # 在分离的计算图上重新计算 mixer 的输出
        logits = self.mixer(detached_x)
        coeffs = torch.softmax(logits, dim=-1)

        # 将所有非专家维度展平，得到 (N, k) 的形状，N是token总数
        flat_coeffs = rearrange(coeffs, '... k -> (...) k')

        # 计算每个专家在批次中处理的token的平均比例 (f_i)
        avg_expert_prob = torch.mean(flat_coeffs, dim=0)

        # 计算每个专家在其被路由到的token上的平均门控值 (P_i)
        avg_expert_gate = torch.mean(flat_coeffs, dim=0)

        # 负载均衡损失是 f_i 和 P_i 的点积，乘以专家数量进行缩放
        # 这是为了鼓励所有专家被均匀使用
        load_balancing_loss = self.num_linears * torch.sum(avg_expert_prob * avg_expert_gate)

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
        num_linears=num_experts,
        reg_strength=0.01,
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
