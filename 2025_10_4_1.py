import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 准备数据 ---
# 我们要拟合的目标函数: y = sin(4*x)
# 生成一些训练数据
X_train = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
y_train = torch.sin(4 * X_train) + torch.randn(200, 1) * 0.1  # 加入一些噪声


# --- 2. 定义标准MLP模型 (0阶动态性) ---
class StaticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StaticNet, self).__init__()
        # 权重是固定的、可训练的参数
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# --- 3. 定义超网络和主网络 (2阶动态性) ---
# 3.1 主网络 (权重将被动态生成，所以没有nn.Linear)
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x, weights):
        # weights 是一个包含 W1, b1, W2, b2, ... 的字典
        # 手动实现全连接层操作
        out = torch.matmul(x, weights['W1']) + weights['b1']
        out = self.relu(out)
        out = torch.matmul(out, weights['W2']) + weights['b2']
        out = self.relu(out)
        out = torch.matmul(out, weights['W3']) + weights['b3']
        return out


# 3.2 超网络 (生成主网络的权重)
class HyperNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hyper_hidden_size=32):
        super(HyperNet, self).__init__()
        # 定义主网络的结构尺寸
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 超网络本身是一个MLP，它的参数是可训练的
        # 它的任务是输出主网络的所有权重
        self.total_weights_size = (
                (input_size * hidden_size + hidden_size) +  # Layer 1: W1, b1
                (hidden_size * hidden_size + hidden_size) +  # Layer 2: W2, b2
                (hidden_size * output_size + output_size)  # Layer 3: W3, b3
        )

        self.hyper_network = nn.Sequential(
            nn.Linear(1, hyper_hidden_size),  # 输入一个全局上下文信号，这里简化为常数1
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, self.total_weights_size)
        )

        # 主网络实例
        self.main_net = MainNet()

    def forward(self, x):
        # 1. 生成主网络的权重
        # 为了简化，我们给超网络一个固定的输入，比如一个常数向量。
        # 这意味着它会为整个训练过程生成一套固定的权重，但这些权重本身是通过网络动态计算的。
        # 更高级的做法是让这个输入依赖于x，但这里为了验证核心机制，我们简化它。
        context_input = torch.ones(1, 1, device=x.device)
        generated_params = self.hyper_network(context_input)

        # 2. 将生成的长向量解析为主网络的权重和偏置
        weights = {}
        start = 0
        # Layer 1
        end = start + self.input_size * self.hidden_size
        weights['W1'] = generated_params[:, start:end].view(self.input_size, self.hidden_size)
        start = end
        end = start + self.hidden_size
        weights['b1'] = generated_params[:, start:end].view(self.hidden_size)
        # Layer 2
        start = end
        end = start + self.hidden_size * self.hidden_size
        weights['W2'] = generated_params[:, start:end].view(self.hidden_size, self.hidden_size)
        start = end
        end = start + self.hidden_size
        weights['b2'] = generated_params[:, start:end].view(self.hidden_size)
        # Layer 3
        start = end
        end = start + self.hidden_size * self.output_size
        weights['W3'] = generated_params[:, start:end].view(self.hidden_size, self.output_size)
        start = end
        end = start + self.output_size
        weights['b3'] = generated_params[:, start:end].view(self.output_size)

        # 3. 使用生成的权重在主网络中进行前向传播
        return self.main_net(x, weights)


# --- 4. 训练函数 ---
def train_model(model, X, y, epochs=10000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Model: {model.__class__.__name__}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    print("Training finished.")
    return model


# --- 5. 实例化和训练模型 ---
# 定义模型尺寸
INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
HYPER_HIDDEN_SIZE = 32  # 超网络隐藏层大小

# 训练标准模型
print("--- Training StaticNet ---")
static_model = StaticNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
static_model = train_model(static_model, X_train, y_train)

# 训练超网络模型
print("\n--- Training HyperNet ---")
hyper_model = HyperNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, HYPER_HIDDEN_SIZE)
hyper_model = train_model(hyper_model, X_train, y_train)

# --- 6. 可视化结果 ---
static_model.eval()
hyper_model.eval()

with torch.no_grad():
    predicted_static = static_model(X_train)
    predicted_hyper = hyper_model(X_train)

plt.figure(figsize=(12, 6))
plt.title("Function Fitting Comparison")
plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original Data (noisy)', markersize=3)
plt.plot(X_train.numpy(), torch.sin(4 * X_train).numpy(), 'k-', label='True Function', linewidth=2)
plt.plot(X_train.numpy(), predicted_static.numpy(), 'b-', label='StaticNet Fit')
plt.plot(X_train.numpy(), predicted_hyper.numpy(), 'g--', label='HyperNet Fit')
plt.legend()
plt.grid(True)
plt.show()

# 比较两个模型的参数数量
static_params = sum(p.numel() for p in static_model.parameters() if p.requires_grad)
hyper_params = sum(p.numel() for p in hyper_model.parameters() if p.requires_grad)
print(f"StaticNet trainable parameters: {static_params}")
print(f"HyperNet trainable parameters: {hyper_params}")

main_net_params = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) + (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) + (
            HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)
print(f"MainNet would have {main_net_params} parameters if trained directly.")