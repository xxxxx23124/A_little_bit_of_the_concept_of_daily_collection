"""
# 伪代码
class HyperAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 三个独立的、非线性的“权重生成器”
        # 它们的结构可以类似我们实验中的HyperNetV2
        self.hyper_q = HyperNet_for_W_Q(...)
        self.hyper_k = HyperNet_for_W_K(...)
        self.hyper_v = HyperNet_for_W_V(...)

    def forward(self, x):
        # x 的形状: [batch_size, seq_len, embed_dim]

        # 1. 动态生成 W_Q, W_K, W_V
        # 超网络的输入可以是x的全局表示（如[CLS] token或平均池化）
        # 甚至可以是每个token自身的表示，生成token-wise的权重
        global_context = x.mean(dim=1)  # 举个例子

        W_q_dynamic = self.hyper_q(global_context)  # 生成的W_q
        W_k_dynamic = self.hyper_k(global_context)  # 生成的W_k
        W_v_dynamic = self.hyper_v(global_context)  # 生成的W_v

        # W_q_dynamic 的形状可能是 [batch_size, embed_dim, d_k]

        # 2. 使用动态生成的权重进行投影
        # 需要使用批处理矩阵乘法 (bmm)
        # (b, N, d) @ (b, d, d_k) -> (b, N, d_k)
        Q = torch.bmm(x, W_q_dynamic)
        K = torch.bmm(x, W_k_dynamic)
        V = torch.bmm(x, W_v_dynamic)

        # 3. 执行标准的注意力计算
        # ... 计算 softmax(Q @ K.T) @ V

        return output
"""