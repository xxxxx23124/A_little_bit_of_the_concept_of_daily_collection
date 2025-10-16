import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from experiment.Transformer.rmsNorm import RMSNorm

class LoRAHyperGenerator(nn.Module):
    """为MainNet的单层生成权重的超网络"""

    def __init__(self, input_dim, output_dim, hidden_dim, rank, ratio_dim):
        super(LoRAHyperGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # 阶段一：两个子网络用于压缩输入特征
        self.compressor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # 阶段二：使用两个独立的、职责单一的生成器
        self.a_generator = nn.Linear(hidden_dim, input_dim * rank)
        self.b_generator = nn.Linear(hidden_dim, rank * output_dim)

        # 第二个子网络从compressor的输出生成偏置b
        self.bias_generator = nn.Linear(hidden_dim, output_dim)

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
            nn.Linear(hidden_dim, ratio_dim), # 极小的中间层
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

class HybridLinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rank, ratio_dim):
        super().__init__()

        self.hyper_parameter = LoRAHyperGenerator(input_dim, output_dim, hidden_dim, rank, ratio_dim)
        self.static_parameter = nn.Linear(input_dim, input_dim)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.static_parameter(x)

        # 为当前层动态生成分解后的权重矩阵 A, B
        # A: [b, in, r], B: [b, r, out], b: [b, 1, out], ratio: [b, 1, 1]
        params = self.hyper_parameter(x)
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
        # (b, out)
        y = y.squeeze(1)

        return y

class HybridSwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dynamic_dim, rank, ratio_dim):
        super().__init__()
        # 1. 混合门控 (Static Gate)
        self.hybrid_gate = HybridLinear(input_dim, hidden_dim, dynamic_dim, rank, ratio_dim)
        self.swish = nn.SiLU()

        # 2. 混合内容 (Dynamic Content)
        self.hybrid_up = HybridLinear(input_dim, hidden_dim, dynamic_dim, rank, ratio_dim)

        # 3. 混合降维 (Static Down-projection)
        self.hybrid_down = HybridLinear(hidden_dim, output_dim, dynamic_dim, rank, ratio_dim)

        # 4. RMSNorm
        self.rmsnorm = RMSNorm(input_dim)

    def forward(self, x):
        x = self.rmsnorm(x)
        # 混合门控
        hybrid_gate = self.hybrid_gate(x)

        # 混合内容
        hybrid_content = self.hybrid_up(x)
        gated_hidden = self.swish(hybrid_gate) * hybrid_content

        # 混合降维，整合信息并输出
        output = self.hybrid_down(gated_hidden)

        return output

class HyperNetV6(nn.Module):
    def __init__(self, layer_sizes, hidden_dim, dynamic_dim, rank, ratio_dim):
        super(HyperNetV6, self).__init__()
        self.layer_sizes = layer_sizes

        self.hybrid_ffn_list = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            self.hybrid_ffn_list.append(
                HybridSwiGLU(input_dim, output_dim, hidden_dim, dynamic_dim, rank, ratio_dim)
            )

    def forward(self, x):
        # x的形状: [batch_size, input_dim]

        for i, hyper_ffn_layer in enumerate(self.hybrid_ffn_list):
            x = hyper_ffn_layer(x)

        return x


# --- 3. 训练和评估函数---
def train_model(model, X, y, epochs=10000, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    X = X.to(device)
    y = y.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Model: {model.__class__.__name__}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    print("Training finished.")
    return model


# --- 4. 实例化和训练新模型 ---
# 定义主网络结构
# 1 -> 64 -> 64 -> 1
if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    MAINNET_LAYER_SIZES = [1, 32, 32, 1]
    HIDDEN_DIM = 64
    GATE_HIDDEN_DIM = 8
    RANK=12
    RATIO_DIM = 4

    # --- 1. 准备数据 (与之前相同) ---
    X_train = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
    y_train = torch.sin(4 * X_train) + torch.randn(200, 1) * 0.1


    print("\n--- Training HyperNetV2 ---")
    hyper_model_v5 = HyperNetV6(MAINNET_LAYER_SIZES, HIDDEN_DIM, GATE_HIDDEN_DIM, RANK, RATIO_DIM)
    hyper_model_v5 = train_model(hyper_model_v5, X_train, y_train, epochs=15000, lr=0.0005)  # 稍微调整训练参数

    # --- 6. 可视化结果 ---
    hyper_model_v5.eval()
    # 加载之前训练的StaticNet用于对比
    # (如果你没有运行之前的代码，可以注释掉这部分)
    try:
        static_model = torch.load("static_model.pth")
        static_model.eval()
        with torch.no_grad():
            predicted_static = static_model(X_train)
    except FileNotFoundError:
        print("Static model_test not found, skipping comparison.")
        predicted_static = None

    with torch.no_grad():
        predicted_hyper_v2 = hyper_model_v5(X_train)

    plt.figure(figsize=(12, 6))
    plt.title("Function Fitting Comparison (HyperNetV4)")
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data (noisy)', markersize=3)
    plt.plot(X_train.numpy(), torch.sin(4 * X_train).numpy(), 'k-', label='True Function', linewidth=2)
    if predicted_static is not None:
        plt.plot(X_train.numpy(), predicted_static.numpy(), 'b-', label='StaticNet Fit', alpha=0.5)
    plt.plot(X_train.numpy(), predicted_hyper_v2.numpy(), 'g--', label='HyperNetV4 Fit', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()

    # (可选) 保存模型用于后续对比
    # torch.save(static_model, "static_model.pth")

    # 参数量分析
    hyper_v4_params = sum(p.numel() for p in hyper_model_v5.parameters() if p.requires_grad)
    print(f"HyperNetV4 trainable parameters: {hyper_v4_params}")
