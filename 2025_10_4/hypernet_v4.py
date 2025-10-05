import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 2. 定义高级超网络 (HyperNetV2) ---
class HyperLayer(nn.Module):
    """为MainNet的单层生成权重的超网络"""

    def __init__(self, input_dim, output_dim, hyper_hidden_dim):
        super(HyperLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 阶段一：两个子网络用于压缩输入特征
        self.compressor = nn.Sequential(nn.Linear(input_dim, hyper_hidden_dim), nn.Tanh())

        # 阶段二：剩下的子网络用于生成权重和偏置
        # 第一个子网络从compressor1的输出生成权重W
        self.weight_generator = nn.Linear(hyper_hidden_dim, input_dim * output_dim)
        # 第二个子网络从compressor2的输出生成偏置b
        self.bias_generator = nn.Linear(hyper_hidden_dim, output_dim)

    def forward(self, x):
        # 注意：这里的输入x是MainNet每一层的输入

        # 阶段一：压缩
        compressed_feat = self.compressor(x)

        # 阶段二：生成
        # 生成的权重需要根据batch_size进行调整
        # [batch_size, input_dim * output_dim] -> [batch_size, input_dim, output_dim]
        W = self.weight_generator(compressed_feat).view(-1, self.input_dim, self.output_dim)

        # [batch_size, output_dim] -> [batch_size, 1, output_dim]
        b = self.bias_generator(compressed_feat).unsqueeze(1)

        return {'W': W, 'b': b}

class HyperGate(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        # 一个小型的 HyperLayer 动态生成门控信号的参数
        self.hyper_layer = HyperLayer(input_dim=input_dim, output_dim=output_dim, hyper_hidden_dim=hyper_hidden_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 为当前层动态生成权重
        # 注意：这里的权重是带batch维度的，即为每个样本都生成了独立的权重
        # W: [batch_size, in, out], b: [batch_size, 1, out]
        weights = self.hyper_layer(x)

        # 使用生成的权重进行计算
        # (b, 1, in) @ (b, in, out) -> (b, 1, out)

        x = torch.bmm(x.unsqueeze(1), weights['W']) + weights['b']
        # 移除多余的维度
        x = x.squeeze(1)

        return self.sigmoid(x)

class HybridSwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, gate_hidden_dim):
        super().__init__()
        # 1. 静态升维 (Static Up-projection)
        self.static_up_proj = nn.Linear(input_dim, hidden_dim, bias=False) # SwiGLU通常不需要偏置

        # 2. 动态门控 (Dynamic Gating)
        # 注意：门控作用于升维后的hidden_dim
        self.dynamic_gate_generator = HyperGate(input_dim, hidden_dim, gate_hidden_dim)
        self.swish = nn.SiLU()

        # 3. 静态降维 (Static Down-projection)
        self.static_down_proj = nn.Linear(hidden_dim, output_dim) # 输出维度通常等于输入维度

    def forward(self, x):
        # 静态升维，获取高维表示
        up_projected = self.static_up_proj(x)

        # 动态生成门控，并应用SwiGLU的核心思想
        # 门控是根据原始输入x生成的，这保留了上下文信息
        gate = self.dynamic_gate_generator(x)
        gated_hidden = self.swish(up_projected) * gate

        # 静态降维，整合信息并输出
        output = self.static_down_proj(gated_hidden)

        return output

class HyperNetV4(nn.Module):
    def __init__(self, layer_sizes, hidden_dim, gate_hidden_dim):
        super(HyperNetV4, self).__init__()
        self.layer_sizes = layer_sizes

        self.hybrid_ffn_list = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            self.hybrid_ffn_list.append(
                HybridSwiGLU(input_dim, output_dim, hidden_dim, gate_hidden_dim)
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
    MAINNET_LAYER_SIZES = [1, 32, 32, 1]
    HIDDEN_DIM = 64
    GATE_HIDDEN_DIM = 8

    # --- 1. 准备数据 (与之前相同) ---
    X_train = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
    y_train = torch.sin(4 * X_train) + torch.randn(200, 1) * 0.1


    print("\n--- Training HyperNetV2 ---")
    hyper_model_v4 = HyperNetV4(MAINNET_LAYER_SIZES, HIDDEN_DIM, GATE_HIDDEN_DIM)
    hyper_model_v4 = train_model(hyper_model_v4, X_train, y_train, epochs=15000, lr=0.0005)  # 稍微调整训练参数

    # --- 6. 可视化结果 ---
    hyper_model_v4.eval()
    # 加载之前训练的StaticNet用于对比
    # (如果你没有运行之前的代码，可以注释掉这部分)
    try:
        static_model = torch.load("static_model.pth")
        static_model.eval()
        with torch.no_grad():
            predicted_static = static_model(X_train)
    except FileNotFoundError:
        print("Static model not found, skipping comparison.")
        predicted_static = None

    with torch.no_grad():
        predicted_hyper_v2 = hyper_model_v4(X_train)

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
    hyper_v4_params = sum(p.numel() for p in hyper_model_v4.parameters() if p.requires_grad)
    print(f"HyperNetV4 trainable parameters: {hyper_v4_params}")
