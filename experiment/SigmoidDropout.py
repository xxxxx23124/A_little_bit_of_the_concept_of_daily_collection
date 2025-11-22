import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SigmoidDropout(nn.Module):
    def forward(self, x):
        if not self.training:
            return x
        prob_keep = 1 - torch.sigmoid(x)
        mask = torch.bernoulli(prob_keep)
        return mask * x / (prob_keep + 1e-6)  # 补偿保持期望为 x（训练更稳）

dim = 8192
net = nn.Sequential(
    nn.Linear(1, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, dim), SigmoidDropout(), nn.LayerNorm(dim),
    nn.Linear(dim, 1)
).to(device)

x = torch.linspace(0, 1, 2000, device=device).unsqueeze(1)
y = torch.sin(20 * x) * torch.tanh(40 * (x - 0.5)) + 0.6 * torch.sign(x - 0.73)
y = y.to(device)

opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

print("开始训练")
for epoch in range(3000):
    opt.zero_grad()
    pred = net(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    opt.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.8f}")

print("训练完成！开始画图")

# 推理 + 可视化
net.eval()
with torch.no_grad():
    pred = net(x).cpu()
    x_cpu = x.cpu()
    y_cpu = y.cpu()

plt.figure(figsize=(12, 7))
plt.plot(x_cpu, y_cpu, 'k-', lw=2.5, label='Target')
plt.plot(x_cpu, pred, 'r--', lw=2, label='SigmoidDropout-Net 拟合结果')
plt.legend(fontsize=16)
plt.title("SigmoidDropout-Net", fontsize=16)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()