import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


try:
    from hypernet_v3 import HyperNetV3, train_moe_model
    from hypernet_v2 import train_model
except ImportError:
    print("错误：无法找到 'hypernet_v2.py'或'hypernet_v3' 文件。")
    exit()

# --- 1. 准备数据 ---
torch.manual_seed(42) # 设置随机种子以保证结果可复现

def fractal_noise_function(x, octaves=6, frequency=1.0, persistence=0.5, lacunarity=2.0):
    """
    通过叠加多层噪声（正弦波）来生成一维分形噪声。
    - x: 输入的 PyTorch Tensor
    - octaves: 叠加的层数。层数越多，细节越丰富，分形特征越明显。
    - frequency: 基础频率。
    - persistence: 持续度/增益。每次迭代后振幅的衰减率，通常为0.5。
    - lacunarity: 间隙度。每次迭代后频率的增长率，通常为2.0。
    """
    y = torch.zeros_like(x)
    amplitude = 1.0

    for i in range(octaves):
        # 使用sin和cos的组合来避免在原点处总是为0，增加随机性
        # 为每个octave添加一个小的相位偏移，使其看起来更不规则
        phase_shift = i * 2.3
        y += amplitude * torch.sin((x + phase_shift) * frequency)

        # 更新下一次迭代的振幅和频率
        amplitude *= persistence
        frequency *= lacunarity

    # 为了让整体形状更有趣，可以再用一个平滑函数进行调制
    # 例如，用一个宽大的高斯函数来包络它
    envelope = torch.exp(-x ** 2 / 12.0)

    return y * envelope


# --- 生成和可视化数据 ---
key_points_num = 75000
x_range = [-2 * np.pi, 2 * np.pi]
X_train = torch.linspace(x_range[0], x_range[1], key_points_num).unsqueeze(1)

# 生成分形函数数据
# 你可以调整 octaves, persistence, lacunarity 来改变分形的样貌
y_true = fractal_noise_function(
    X_train,
    octaves=12,      # 极高的细节层次
    frequency=18.0,
    persistence=0.5,
    lacunarity=2.0
)
y_train = y_true
# 加入噪声
# y_train = y_true + torch.randn(key_points_num, 1) * 0.05  # 分形函数细节多，噪声可以小一点

plt.figure(figsize=(12, 7))
plt.plot(X_train.numpy(), y_true.numpy(), 'r-', linewidth=1.5, label='True Fractal Function')
plt.scatter(X_train.numpy(), y_train.numpy(), s=5, alpha=0.4, label='Training Data (Fractal)')
plt.title("Fractal Noise Function for Model Training", fontsize=16)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 2. 定义一个强大的 StaticNetLarge ---
class StaticNetLarge(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, output_size):
        super(StaticNetLarge, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, h3_size),
            nn.ReLU(),
            nn.Linear(h3_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

if __name__=='__main__':
    # --- 定义模型尺寸和超参数 ---
    # 主网络尺寸
    MAINNET_LAYER_SIZES = [1, 64, 64, 1]

    # HyperNetV3 (MoE) 的超参数
    RANK = 10
    NUM_EXPERTS = 3
    TOP_K = 2
    HYPER_HIDDEN_DIM = 10  # MoE内部网络的隐藏维度
    ALPHA = 0.01  # 负载均衡损失的权重


    # StaticNetLarge 的尺寸，使其参数量与HyperNetV3接近
    H_LARGE = 220

    # --- 3 实例化 ---
    # 实例化 HyperNetV3
    print("\n--- Instantiating HyperNetV3 (MoE) ---")
    hyper_model_v3 = HyperNetV3(
        layer_sizes=MAINNET_LAYER_SIZES,
        rank=RANK,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hyper_hidden_dim=HYPER_HIDDEN_DIM
    )

    hyper_v3_params = sum(p.numel() for p in hyper_model_v3.parameters() if p.requires_grad)

    # 实例化 StaticNetLarge
    static_model_large = StaticNetLarge(1, H_LARGE, H_LARGE, H_LARGE, 1)
    static_large_params = sum(p.numel() for p in static_model_large.parameters() if p.requires_grad)

    print("--- 模型参数量对比 ---")
    print(f"HyperNetV3 Trainable Parameters:     {hyper_v3_params:,}")
    print(f"StaticNetLarge Trainable Parameters: {static_large_params:,}")
    print("-" * 30)

    # 3.2 训练 HyperNetV3
    print("\n--- Training HyperNetV3 (MoE) ---")
    hyper_model_v3 = train_moe_model(
        hyper_model_v3,
        X_train,
        y_train,
        epochs=30000,
        lr=0.0005,
        alpha=ALPHA
    )

    # 3.3 训练 StaticNetLarge
    print("\n--- Training StaticNetLarge ---")
    # 使用相同的训练函数和超参数
    static_model_large = train_model(static_model_large, X_train, y_train, epochs=30000, lr=0.0005)


    # --- 4. 可视化最终对比结果 ---
    hyper_model_v3.eval()
    static_model_large.eval()

    with torch.no_grad():
        device = next(hyper_model_v3.parameters()).device  # 获取模型所在的设备
        X_train_gpu = X_train.to(device)
        predicted_hyper_v3, _ = hyper_model_v3(X_train_gpu)
        predicted_static_large = static_model_large(X_train_gpu)

    plt.figure(figsize=(14, 7))
    plt.title("Fair Comparison: HyperNetV3 vs. StaticNetLarge")
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data (noisy)', markersize=3, alpha=0.6)
    plt.plot(X_train.numpy(), y_true.numpy(), 'k-', label='True Function', linewidth=3)
    plt.plot(X_train.numpy(), predicted_static_large.cpu().numpy(), 'b-', label=f'StaticNetLarge Fit ({static_large_params:,} params)', linewidth=2)
    plt.plot(X_train.numpy(), predicted_hyper_v3.cpu().numpy(), 'g--', label=f'HyperNetV3 Fit ({hyper_v3_params:,} params)', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.5, 1.5) # 固定y轴范围，防止过拟合导致图像变形
    plt.show()