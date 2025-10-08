import torch.nn as nn
from hyperTransformer.linear.hyperMoMixLinear import HyperMoMixLinear

class HybridSwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, up_proj_dim, compressed_feature_dim):
        super().__init__()
        # 1. 静态门控 (Static Gate)
        self.static_gate = nn.Linear(input_dim, up_proj_dim)
        self.swish = nn.SiLU()

        # 2. 动态内容 (Dynamic Content)
        self.dynamic_up = HyperMoMixLinear(input_dim, up_proj_dim,compressed_feature_dim,num_monarchs=2)

        # 3. 静态降维 (Static Down-projection)
        self.static_down = nn.Linear(up_proj_dim, output_dim) # 输出维度通常等于输入维度

    def forward(self, x):
        # 静态门控
        static_gate = self.static_gate(x)

        # 动态内容
        dynamic_content = self.dynamic_up(x)
        gated_hidden = self.swish(static_gate) * dynamic_content

        # 静态降维，整合信息并输出
        output = self.static_down(gated_hidden)

        return output