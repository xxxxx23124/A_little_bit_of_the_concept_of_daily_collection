import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 准备数据 (与之前相同) ---
X_train = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
y_train = torch.sin(4 * X_train) + torch.randn(200, 1) * 0.1


# --- 2. 主网络 (与之前相同，无参数) ---
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x, weights_list):
        # weights_list 是一个包含各层权重字典的列表
        out = x
        for weights in weights_list:
            # 手动实现全连接层操作
            out = torch.matmul(out, weights['W']) + weights['b']
            out = self.relu(out)
        return out


# --- 3. 定义高级超网络 (HyperNetV2) ---
class LayerHyperNet(nn.Module):
    """为MainNet的单层生成权重的超网络"""

    def __init__(self, input_dim, output_dim, hyper_hidden_dim=16):
        super(LayerHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 阶段一：两个子网络用于压缩输入特征
        self.compressor1 = nn.Sequential(nn.Linear(input_dim, hyper_hidden_dim), nn.Tanh())
        self.compressor2 = nn.Sequential(nn.Linear(input_dim, hyper_hidden_dim), nn.Tanh())

        # 阶段二：剩下的子网络用于生成权重和偏置
        # 第一个子网络从compressor1的输出生成权重W
        self.weight_generator = nn.Linear(hyper_hidden_dim, input_dim * output_dim)
        # 第二个子网络从compressor2的输出生成偏置b
        self.bias_generator = nn.Linear(hyper_hidden_dim, output_dim)

    def forward(self, x):
        # 注意：这里的输入x是MainNet每一层的输入

        # 阶段一：压缩
        compressed_feat1 = self.compressor1(x)
        compressed_feat2 = self.compressor2(x)

        # 阶段二：生成
        # 生成的权重需要根据batch_size进行调整
        # [batch_size, input_dim * output_dim] -> [batch_size, input_dim, output_dim]
        W = self.weight_generator(compressed_feat1).view(-1, self.input_dim, self.output_dim)

        # [batch_size, output_dim] -> [batch_size, 1, output_dim]
        b = self.bias_generator(compressed_feat2).unsqueeze(1)

        return {'W': W, 'b': b}


class HyperNetV2(nn.Module):
    def __init__(self, layer_sizes, hyper_hidden_dim=16):
        super(HyperNetV2, self).__init__()
        self.layer_sizes = layer_sizes
        self.relu = nn.ReLU()

        # 创建一个分层的超网络列表
        self.hyper_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            self.hyper_layers.append(
                LayerHyperNet(input_dim, output_dim, hyper_hidden_dim)
            )

    def forward(self, x):
        # x的形状: [batch_size, input_dim]
        current_input = x

        for i, hyper_layer in enumerate(self.hyper_layers):
            # 为当前层动态生成权重
            # 注意：这里的权重是带batch维度的，即为每个样本都生成了独立的权重
            # W: [batch_size, in, out], b: [batch_size, 1, out]
            weights = hyper_layer(current_input)

            # 使用生成的权重进行计算
            # (b, 1, in) @ (b, in, out) -> (b, 1, out)
            current_input = torch.bmm(current_input.unsqueeze(1), weights['W']) + weights['b']

            # 移除多余的维度
            current_input = current_input.squeeze(1)

            # 应用激活函数 (除了最后一层)
            if i < len(self.hyper_layers) - 1:
                current_input = self.relu(current_input)

        return current_input


# --- 4. 训练和评估函数 (与之前类似，但需要实例化新模型) ---
def train_model(model, X, y, epochs=10000, lr=0.001):
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


# --- 5. 实例化和训练新模型 ---
# 定义主网络结构
# 1 -> 64 -> 64 -> 1
if __name__=='__main__':
    MAINNET_LAYER_SIZES = [1, 64, 64, 1]
    HYPER_HIDDEN_DIM = 16  # 每个分层超网络的隐藏层大小

    print("\n--- Training HyperNetV2 ---")
    hyper_model_v2 = HyperNetV2(MAINNET_LAYER_SIZES, HYPER_HIDDEN_DIM)
    hyper_model_v2 = train_model(hyper_model_v2, X_train, y_train, epochs=15000, lr=0.0005)  # 稍微调整训练参数

    # --- 6. 可视化结果 ---
    hyper_model_v2.eval()
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
        predicted_hyper_v2 = hyper_model_v2(X_train)

    plt.figure(figsize=(12, 6))
    plt.title("Function Fitting Comparison (HyperNetV2)")
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data (noisy)', markersize=3)
    plt.plot(X_train.numpy(), torch.sin(4 * X_train).numpy(), 'k-', label='True Function', linewidth=2)
    if predicted_static is not None:
        plt.plot(X_train.numpy(), predicted_static.numpy(), 'b-', label='StaticNet Fit', alpha=0.5)
    plt.plot(X_train.numpy(), predicted_hyper_v2.numpy(), 'g--', label='HyperNetV2 Fit', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()

    # (可选) 保存模型用于后续对比
    # torch.save(static_model, "static_model.pth")

    # 参数量分析
    hyper_v2_params = sum(p.numel() for p in hyper_model_v2.parameters() if p.requires_grad)
    print(f"HyperNetV2 trainable parameters: {hyper_v2_params}")
