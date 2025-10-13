import torch
import torch.nn as nn
import torch.optim as optim
from experiment.Transformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridSwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dynamic_dim):
        super().__init__()
        # 1. 静态门控 (Static Gate)
        self.static_gate = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()
        # 2. 动态内容 (Dynamic Content)
        self.dynamic_up = HyperMoMixLinear(input_dim,hidden_dim,dynamic_dim,2)
        self.sigmoid = nn.Sigmoid()

        # 3. 静态降维 (Static Down-projection)
        self.static_down = nn.Linear(hidden_dim, output_dim) # 输出维度通常等于输入维度

    def forward(self, x):
        # 静态门控
        static_gate = self.static_gate(x)

        # 动态内容
        dynamic_content = self.sigmoid(self.dynamic_up(x))
        gated_hidden = self.swish(static_gate) * dynamic_content

        # 静态降维，整合信息并输出
        output = self.static_down(gated_hidden)

        return output

class HyperNetV7(nn.Module):
    def __init__(self, layer_sizes, hidden_dim, dynamic_dim):
        super(HyperNetV7, self).__init__()
        self.layer_sizes = layer_sizes

        self.hybrid_ffn_list = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            self.hybrid_ffn_list.append(
                HybridSwiGLU(input_dim, output_dim, hidden_dim, dynamic_dim)
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
        # 计算主任务损失
        main_loss = criterion(outputs, y)
        # 递归收集所有子模块的辅助损失
        total_aux_loss = 0
        # model.modules() 会递归地返回模型中的所有模块 (包括它自己)
        for module in model.modules():
            # 检查模块是否是我们想要收集损失的类型
            if isinstance(module, HyperMoMixLinear):
                # 累加这个模块在前向传播中记录的所有损失
                total_aux_loss += sum(module.auxiliary_losses)
                module.clear_auxiliary_losses()
        # 将主损失和辅助损失相加
        loss = main_loss + total_aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Model: {model.__class__.__name__}, Epoch [{epoch + 1}/{epochs}], Loss: {main_loss.item():.6f}')

    print("Training finished.")
    return model


# --- 4. 实例化和训练新模型 ---
# 定义主网络结构
# 1 -> 64 -> 64 -> 1
if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    MAINNET_LAYER_SIZES = [1, 64, 64, 1]
    HIDDEN_DIM = 81
    GATE_HIDDEN_DIM = 16

    # --- 1. 准备数据 (与之前相同) ---
    X_train = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
    y_train = torch.sin(4 * X_train) + torch.randn(200, 1) * 0.1


    print("\n--- Training HyperNetV2 ---")
    hyper_model_v5 = HyperNetV7(MAINNET_LAYER_SIZES, HIDDEN_DIM, GATE_HIDDEN_DIM)
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
        print("Static model not found, skipping comparison.")
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
