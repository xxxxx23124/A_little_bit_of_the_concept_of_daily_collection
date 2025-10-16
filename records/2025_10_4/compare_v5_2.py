import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 从之前的脚本中导入HyperNetV2模型
# 确保 hypernet_v2.py 文件在同一个目录下
try:
    from hypernet_v5 import HyperNetV5, train_model
except ImportError:
    print("错误：无法找到 'hypernet_v7.py' 文件。")
    exit()

# --- 1. 准备数据 (与之前相同) ---
torch.manual_seed(42) # 设置随机种子以保证结果可复现
# key_points_num = 200
# key_points_num = 6000
# frequency = 4
# frequency = 32
# X_train = torch.linspace(-np.pi, np.pi, key_points_num).unsqueeze(1)
# y_train = torch.sin(frequency * X_train) + torch.randn(key_points_num, 1) * 0.1 # 加入一些噪声

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


# --- 2. 定义 SwiGLU 模块和使用它的静态网络 ---

class SwiGLU(nn.Module):
    """
    标准的 SwiGLU (Swish-Gated Linear Unit) 实现。
    FFN(x) = (SiLU(x @ W_up) * (x @ W_gate)) @ W_down
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 通常 hidden_dim 是 input_dim 的倍数，例如 2/3 * 4 * input_dim
        # 这里为了公平对比，我们让 hidden_dim 可配置
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = nn.SiLU()  # SiLU (Swish-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, input_dim]
        gated = self.activation(self.gate_proj(x))
        up = self.sigmoid(self.up_proj(x))
        fused = gated * up  # 逐元素相乘
        output = self.down_proj(fused)
        return output


class StaticNetSwiGLU(nn.Module):
    """
    一个使用 SwiGLU 模块构建的深度静态网络。
    """

    def __init__(self, layer_sizes, hidden_dim_multiplier):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            # 计算这一层SwiGLU的中间隐藏维度
            # 通常 SwiGLU 的 hidden_dim 会设置得比较大
            hidden_dim = int(input_dim * hidden_dim_multiplier)

            self.layers.append(SwiGLU(input_dim, hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            # 注意：这里和传统的MLP不同，没有在层之间加激活函数
            # 因为激活函数已经包含在SwiGLU模块内部了。
            # 但为了增加非线性，可以在层之间添加残差连接或LayerNorm，
            # 不过为了保持简单和公平对比，我们这里直接串联。
            x = layer(x)
        return x


# (此处省略之前的模型定义，包括 HyperNetV4, HybridSwiGLU, HyperGate,
#  SwiGLU, 和 StaticNetSwiGLU)

if __name__ == '__main__':
    # --- 3. 实例化和训练模型 ---

    # 3.1 定义模型尺寸并进行公平的参数量对齐

    MAINNET_LAYER_SIZES = [1, 64, 64, 1]

    HIDDEN_DIM = 96
    GATE_HIDDEN_DIM = 16
    RANK = 12
    RATIO_DIM = 10
    # 实例化 HyperNetV5
    hyper_model_v5 = HyperNetV5(MAINNET_LAYER_SIZES, HIDDEN_DIM, GATE_HIDDEN_DIM, RANK, RATIO_DIM)
    hyper_v5_params = sum(p.numel() for p in hyper_model_v5.parameters() if p.requires_grad)

    # StaticNetSwiGLU 的配置
    # 我们定义一个基础结构，然后通过调整 `HIDDEN_DIM_MULTIPLIER` 来匹配参数量
    SWIGLU_LAYER_SIZES = [1, 64, 64, 1]  # 使用和HyperNet主网络相同的层级结构

    # --- 参数量对齐的计算 ---
    # 目标参数量: hyper_v4_params
    # SwiGLU参数量公式 (对于三层网络 [i, j, k, l]):
    # P = (i*h1 + i*h1 + h1*j) + (j*h2 + j*h2 + h2*k) + (k*h3 + k*h3 + h3*l)
    # 其中 h1=i*M, h2=j*M, h3=k*M (M是乘数)
    # P = M*(2*i^2+i*j) + M*(2*j^2+j*k) + M*(2*k^2+k*l)
    # P = M * [ (2*1*1 + 1*48) + (2*48*48 + 48*48) + (2*48*48 + 48*1) ]
    # P = M * [ 50 + 6912 + 4656 ] = M * 11618
    # 所以 M = hyper_v4_params / 11618

    # 为了避免浮点数问题，直接计算
    params_per_multiplier = ((2 * SWIGLU_LAYER_SIZES[0] ** 2 + SWIGLU_LAYER_SIZES[0] * SWIGLU_LAYER_SIZES[1]) +
                             (2 * SWIGLU_LAYER_SIZES[1] ** 2 + SWIGLU_LAYER_SIZES[1] * SWIGLU_LAYER_SIZES[2]) +
                             (2 * SWIGLU_LAYER_SIZES[2] ** 2 + SWIGLU_LAYER_SIZES[2] * SWIGLU_LAYER_SIZES[3]))

    HIDDEN_DIM_MULTIPLIER = hyper_v5_params / params_per_multiplier

    static_model_swiglu = StaticNetSwiGLU(SWIGLU_LAYER_SIZES, HIDDEN_DIM_MULTIPLIER)
    static_swiglu_params = sum(p.numel() for p in static_model_swiglu.parameters() if p.requires_grad)
    # --------------------------------------------------------------------

    print("--- 模型参数量对比 ---")
    print(f"HyperNetV4 Trainable Parameters:       {hyper_v5_params:,}")
    print(f"StaticNet-SwiGLU Trainable Parameters: {static_swiglu_params:,} (Multiplier: {HIDDEN_DIM_MULTIPLIER:.4f})")
    print("-" * 40)

    # --- 3.2 训练模型 ---
    EPOCHS = 30000
    LR = 0.0005

    print("\n--- Training HyperNetV4 ---")
    hyper_model_v4 = train_model(hyper_model_v5, X_train, y_train, epochs=EPOCHS, lr=LR)

    print("\n--- Training StaticNet-SwiGLU ---")
    static_model_swiglu = train_model(static_model_swiglu, X_train, y_train, epochs=EPOCHS, lr=LR)

    # --- 4. 可视化最终对比结果 ---
    hyper_model_v4.eval()
    static_model_swiglu.eval()

    with torch.no_grad():
        device = next(hyper_model_v4.parameters()).device  # 获取模型所在的设备
        X_train_gpu = X_train.to(device)
        predicted_hyper_v4 = hyper_model_v4(X_train_gpu)
        predicted_static_swiglu = static_model_swiglu(X_train_gpu)

    plt.figure(figsize=(16, 8))
    plt.title("Fair Comparison: Dynamic Gating (HyperNetV5) vs. Static Gating (SwiGLU)")
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data', markersize=2, alpha=0.3)
    plt.plot(X_train.numpy(), y_true.numpy(), 'k-', label='True Function', linewidth=2.5)
    plt.plot(X_train.numpy(), predicted_static_swiglu.cpu().numpy(), 'm-.',
             label=f'StaticNet-SwiGLU Fit ({static_swiglu_params:,} params)', linewidth=2)
    plt.plot(X_train.numpy(), predicted_hyper_v4.cpu().numpy(), 'g--',
             label=f'HyperNetV5 Fit ({hyper_v5_params:,} params)', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.5, 1.5)
    plt.show()
