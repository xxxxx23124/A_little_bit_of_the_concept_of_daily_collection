import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
import numpy as np


class LowRankExperts(nn.Module):
    """一个模块，一次性为所有专家生成低秩权重因子 A 和 B"""

    def __init__(self, num_experts, input_dim, output_dim, rank, expert_hidden_dim=16):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # 每个专家一个独立的生成器网络
        self.factor_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.Tanh(),
                nn.Linear(expert_hidden_dim, (input_dim * rank) + (rank * output_dim))
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [batch_size, input_dim]

        all_factors_flat = []
        for generator in self.factor_generators:
            # generator(x) -> [b, (in*r + r*out)]
            all_factors_flat.append(generator(x))

        # factors_stack: [e, b, (in*r + r*out)]
        factors_stack = torch.stack(all_factors_flat, dim=0)

        # 切分 A 和 B
        a_flat = factors_stack[..., :self.input_dim * self.rank]
        b_flat = factors_stack[..., self.input_dim * self.rank:]

        # 重塑 A 和 B
        # a_flat: [e, b, i*r] -> A: [e, b, i, r]
        A = rearrange(a_flat, 'e b (i r) -> e b i r', i=self.input_dim, r=self.rank)
        # b_flat: [e, b, r*o] -> B: [e, b, r, o]
        B = rearrange(b_flat, 'e b (r o) -> e b r o', r=self.rank, o=self.output_dim)

        return A, B


class MoELayerHyperNet(nn.Module):
    """使用 MoE 思想为单层生成低秩权重的超网络"""

    def __init__(self, input_dim, output_dim, rank, num_experts, top_k, hyper_hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank

        # 专家列表 直接使用一个 LowRankExperts 实例
        self.experts = LowRankExperts(
            num_experts=num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            rank=rank,
            expert_hidden_dim=hyper_hidden_dim
        )

        # 门控网络，用于为每个专家打分
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hyper_hidden_dim),
            nn.Tanh(),
            nn.Linear(hyper_hidden_dim, num_experts)
        )

        # 偏置生成器可以保持不变，或者也用MoE
        self.bias_generator = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [b, i] where b=batch_size, i=input_dim

        # 1. 门控：获取专家的权重
        # logits: [b, num_experts]
        logits = self.gating_network(x)
        weights = F.softmax(logits, dim=1)

        # 2. Top-k 选择
        # top_k_weights: [b, k], top_k_indices: [b, k]
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)

        # --- 以下是向量化优化的核心 ---

        # 3. 批量处理所有专家
        # 创建一个 [num_experts, b, i] 的张量，让每个专家处理完整的 batch
        x_expanded = x.unsqueeze(0).expand(self.num_experts, -1, -1)  # [e, b, i]

        # 3. 批量处理所有专家
        # 直接调用 self.experts 一次，传入 [b, i] 的 x
        # stack_A: [e, b, i, r], stack_B: [e, b, r, o]
        stack_A, stack_B = self.experts(x)

        # 4. 使用 gather 高效收集 top-k 专家的输出
        # top_k_indices 需要被扩展以匹配 stack_A 和 stack_B 的维度
        # a_indices: [k, b, i, r], b_indices: [k, b, r, o]
        # 我们将 top_k_indices 从 [b, k] -> [k, b] -> [k, b, 1, 1] -> expand
        idx_A = rearrange(top_k_indices, 'b k -> k b 1 1').expand(-1, -1, self.input_dim, self.rank)
        idx_B = rearrange(top_k_indices, 'b k -> k b 1 1').expand(-1, -1, self.rank, self.output_dim)

        # 沿专家维度(dim=0)收集
        # top_k_A: [k, b, i, r], top_k_B: [k, b, r, o]
        top_k_A = stack_A.gather(0, idx_A)
        top_k_B = stack_B.gather(0, idx_B)

        # 5. 加权求和
        # 调整 top_k_weights 的形状以进行广播乘法
        # w: [b, k] -> [k, b] -> [k, b, 1, 1]
        w = rearrange(top_k_weights, 'b k -> k b 1 1')

        # 加权并求和
        # (w * top_k_A) -> [k, b, i, r]
        # .sum(dim=0) -> [b, i, r]
        final_A = (w * top_k_A).sum(dim=0)
        final_B = (w * top_k_B).sum(dim=0)

        # 6. 重构最终权重 W 和偏置 b
        # W: [b, i, o]
        W = torch.bmm(final_A, final_B)
        b = self.bias_generator(x).unsqueeze(1)  # [b, 1, o]

        return {'W': W, 'b': b, 'logits': logits}

class HyperNetV3(nn.Module):
    def __init__(self, layer_sizes, rank, num_experts, top_k, hyper_hidden_dim):
        super(HyperNetV3, self).__init__()
        self.layer_sizes = layer_sizes
        self.relu = nn.ReLU()

        # 创建一个分层的超网络列表
        self.hyper_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            self.hyper_layers.append(
                MoELayerHyperNet(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=rank,  # 显式传递 rank
                    num_experts=num_experts,
                    top_k=top_k,
                    hyper_hidden_dim=hyper_hidden_dim # 显式传递 hyper_hidden_dim
                )
            )

    def forward(self, x):
        # x的形状: [batch_size, input_dim]
        current_input = x
        gating_logits_list = []

        for i, hyper_layer in enumerate(self.hyper_layers):
            # 为当前层动态生成权重
            # 注意：这里的权重是带batch维度的，即为每个样本都生成了独立的权重
            # W: [batch_size, in, out], b: [batch_size, 1, out]
            weights_logits = hyper_layer(current_input)

            # 储存每一个门控的 logits
            gating_logits_list.append(weights_logits['logits'])

            # 使用生成的权重进行计算
            # (b, 1, in) @ (b, in, out) -> (b, 1, out)
            current_input = torch.bmm(current_input.unsqueeze(1), weights_logits['W']) + weights_logits['b']

            # 移除多余的维度
            current_input = current_input.squeeze(1)

            # 应用激活函数 (除了最后一层)
            if i < len(self.hyper_layers) - 1:
                current_input = self.relu(current_input)

        return current_input, gating_logits_list


def calculate_load_balancing_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    计算负载均衡损失 (Load Balancing Loss)。

    参数:
    - logits: 门控网络的输出, shape: [batch_size, num_experts]

    返回:
    - loss: 一个标量 (scalar) 损失值
    """

    # 检查logits是否至少是2D张量
    if logits.dim() < 2:
        raise ValueError("Logits tensor must be at least 2-dimensional (batch_size, num_experts)")

    # 从logits的最后一个维度获取专家数量
    num_experts = logits.shape[-1]

    # 如果只有一个专家，负载均衡没有意义，损失为0
    if num_experts == 1:
        return torch.tensor(0.0, device=logits.device)

    # 1. 对 logits 应用 softmax 得到每个样本对每个专家的选择概率 (P)
    # P 的形状: [batch_size, num_experts]
    probs = F.softmax(logits, dim=1)

    # 2. 计算每个专家的“重要性”(f_j)
    # importance_per_expert 的形状: [num_experts]
    importance_per_expert = torch.mean(probs, dim=0)

    # 3. 计算负载均衡损失
    # 公式: num_experts * sum(importance_per_expert^2)
    # 这鼓励 importance_per_expert 的分布尽可能均匀
    loss = num_experts * torch.sum(importance_per_expert.pow(2))

    return loss


def train_moe_model(model, X, y, epochs=10000, lr=0.001, alpha=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    X = X.to(device)
    y = y.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # --- 前向传播 ---
        # MoE 模型 forward 方法会同时返回主网络输出和门控的 logits
        # 这是一种好的实践，方便计算负载均衡损失
        y_pred, gating_logits_list = model(X)

        # --- 计算主损失 ---
        main_loss = criterion(y_pred, y)

        # --- 计算负载均衡损失 ---
        total_load_balancing_loss = 0
        num_layers_with_moe = len(model.hyper_layers)  # 假设所有层都是 MoE

        # 遍历所有 MoE 层的 gating_logits，并累加它们的负载均衡损失
        for gating_logits in gating_logits_list:
            lb_loss = calculate_load_balancing_loss(gating_logits)
            total_load_balancing_loss += lb_loss

        # 对多层的负载均衡损失求平均
        avg_load_balancing_loss = total_load_balancing_loss / num_layers_with_moe

        # --- 合并总损失 ---
        total_loss = main_loss + alpha * avg_load_balancing_loss

        # --- 反向传播和优化 ---
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss.item():.6f}, '
                  f'Main Loss: {main_loss.item():.6f}, LB Loss: {avg_load_balancing_loss.item():.6f}')

    print("Training finished.")
    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # --- 1. 准备数据 (与之前相同) ---
    torch.manual_seed(42)  # 设置随机种子以保证结果可复现
    key_points_num = 200
    frequency = 4
    X_train = torch.linspace(-np.pi, np.pi, key_points_num).unsqueeze(1)
    y_train = torch.sin(frequency * X_train) + torch.randn(key_points_num, 1) * 0.1  # 加入一些噪声

    # --- 2. 定义模型尺寸和超参数 ---
    # 主网络尺寸
    MAINNET_LAYER_SIZES = [1, 64, 64, 1]

    # HyperNetV3 (MoE) 的超参数
    RANK = 8
    NUM_EXPERTS = 5
    TOP_K = 2
    HYPER_HIDDEN_DIM = 32  # MoE内部网络的隐藏维度
    ALPHA = 0.01  # 负载均衡损失的权重

    # --- 3. 实例化 HyperNetV3 ---
    print("\n--- Instantiating HyperNetV3 (MoE) ---")
    hyper_model_v3 = HyperNetV3(
        layer_sizes=MAINNET_LAYER_SIZES,
        rank=RANK,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hyper_hidden_dim=HYPER_HIDDEN_DIM
    )

    # 计算参数量
    hyper_v3_params = sum(p.numel() for p in hyper_model_v3.parameters() if p.requires_grad)
    print(f"HyperNetV3 (MoE) Trainable Parameters: {hyper_v3_params:,}")
    print("-" * 30)

    # --- 4. 训练 HyperNetV3 ---
    print("\n--- Training HyperNetV3 (MoE) ---")
    hyper_model_v3 = train_moe_model(
        hyper_model_v3,
        X_train,
        y_train,
        epochs=15000,
        lr=0.001,  # MoE模型可能对学习率更敏感，可以从稍高处开始尝试
        alpha=ALPHA
    )

    # --- 5. 评估和可视化 ---
    hyper_model_v3.eval()

    device = next(hyper_model_v3.parameters()).device  # 获取模型所在的设备
    X_train_gpu = X_train.to(device)

    with torch.no_grad():
        # 注意: hyper_model_v3 返回一个元组 (prediction, logits_list)
        predicted_hyper_v3, _ = hyper_model_v3(X_train_gpu)

    plt.figure(figsize=(12, 6))  # 可以适当调整尺寸
    # --- 修改点: 更新图表标题 ---
    plt.title("HyperNetV3 (MoE) Fit on Sine Wave Data")

    # 绘制基础数据
    # .cpu() 对于已经在CPU上的张量没有影响，但写上更具鲁棒性
    plt.plot(X_train.cpu().numpy(), y_train.cpu().numpy(), 'ro', label='Original Data (noisy)', markersize=3, alpha=0.5)
    plt.plot(X_train.cpu().numpy(), torch.sin(frequency * X_train).numpy(), 'k-', label='True Function', linewidth=3)

    # 绘制 HyperNetV3 的拟合曲线
    # 将GPU上的预测结果移回CPU再转为numpy
    plt.plot(X_train.cpu().numpy(), predicted_hyper_v3.cpu().numpy(), 'g--',
             label=f'HyperNetV3 (MoE) Fit ({hyper_v3_params:,} params)', linewidth=2.5)

    plt.legend()
    plt.xlabel("Input (X)")
    plt.ylabel("Output (Y)")
    plt.grid(True)
    plt.ylim(-1.5, 1.5)
    plt.show()

    # (可选) 保存模型，以便下次直接加载
    # torch.save(hyper_model_v3.state_dict(), "hyper_model_v3_state.pth")