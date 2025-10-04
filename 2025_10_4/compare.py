import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 从之前的脚本中导入HyperNetV2模型
# 确保 hypernet_v2.py 文件在同一个目录下
try:
    from hypernet_v2 import HyperNetV2, train_model
except ImportError:
    print("错误：无法找到 'hypernet_v2.py' 文件。")
    print("请确保将之前包含 HyperNetV2 的代码保存为 'hypernet_v2.py' 且与此文件在同一目录。")
    exit()

# --- 1. 准备数据 (与之前相同) ---
torch.manual_seed(42) # 设置随机种子以保证结果可复现
key_points_num = 200
# key_points_num = 6000
frequency = 4
# frequency = 32
X_train = torch.linspace(-np.pi, np.pi, key_points_num).unsqueeze(1)
y_train = torch.sin(frequency * X_train) + torch.randn(key_points_num, 1) * 0.1 # 加入一些噪声

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
    # --- 3. 实例化和训练模型 ---

    # 3.1 定义模型尺寸
    # HyperNetV2 的尺寸
    MAINNET_LAYER_SIZES = [1, 64, 64, 1]
    HYPER_HIDDEN_DIM = 16

    # StaticNetLarge 的尺寸，使其参数量与HyperNetV2接近
    H_LARGE = 200

    # 实例化 HyperNetV2
    hyper_model_v2 = HyperNetV2(MAINNET_LAYER_SIZES, HYPER_HIDDEN_DIM)
    hyper_v2_params = sum(p.numel() for p in hyper_model_v2.parameters() if p.requires_grad)

    # 实例化 StaticNetLarge
    static_model_large = StaticNetLarge(1, H_LARGE, H_LARGE, H_LARGE, 1)
    static_large_params = sum(p.numel() for p in static_model_large.parameters() if p.requires_grad)

    print("--- 模型参数量对比 ---")
    print(f"HyperNetV2 Trainable Parameters:     {hyper_v2_params:,}")
    print(f"StaticNetLarge Trainable Parameters: {static_large_params:,}")
    print("-" * 30)

    # 3.2 训练 HyperNetV2
    print("\n--- Training HyperNetV2 ---")
    # 注意：train_model 函数是从 hypernet_v2.py 导入的
    hyper_model_v2 = train_model(hyper_model_v2, X_train, y_train, epochs=15000, lr=0.0005)

    # 3.3 训练 StaticNetLarge
    print("\n--- Training StaticNetLarge ---")
    # 使用相同的训练函数和超参数
    static_model_large = train_model(static_model_large, X_train, y_train, epochs=15000, lr=0.0005)


    # --- 4. 可视化最终对比结果 ---
    hyper_model_v2.eval()
    static_model_large.eval()

    with torch.no_grad():
        predicted_hyper_v2 = hyper_model_v2(X_train)
        predicted_static_large = static_model_large(X_train)

    plt.figure(figsize=(14, 7))
    plt.title("Fair Comparison: HyperNetV2 vs. StaticNetLarge")
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data (noisy)', markersize=3, alpha=0.6)
    plt.plot(X_train.numpy(), torch.sin(frequency * X_train).numpy(), 'k-', label='True Function', linewidth=3)
    plt.plot(X_train.numpy(), predicted_static_large.numpy(), 'b-', label=f'StaticNetLarge Fit ({static_large_params:,} params)', linewidth=2)
    plt.plot(X_train.numpy(), predicted_hyper_v2.numpy(), 'g--', label=f'HyperNetV2 Fit ({hyper_v2_params:,} params)', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.5, 1.5) # 固定y轴范围，防止过拟合导致图像变形
    plt.show()