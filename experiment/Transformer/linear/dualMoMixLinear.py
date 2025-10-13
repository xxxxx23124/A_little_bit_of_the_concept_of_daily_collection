import torch
import torch.nn as nn

from experiment.Transformer.linear.hyperMoMixLinear import HyperMoMixLinear


class DualMoMixLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 compressed_feature_dim,
                 num_monarchs,
                 reg_strength=1e-2,
                 dropout_rate=0.1,
                 use_checkpointing:bool=False
                 ):
        super().__init__()
        self.dynamic = HyperMoMixLinear(
            in_features=in_features,
            out_features=out_features,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            reg_strength=reg_strength,
            use_checkpointing=use_checkpointing
        )
        self.static = nn.Linear(in_features, out_features)

        self.compressor = nn.Linear(2*out_features, out_features)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_dynamic = self.dynamic(x)
        x_static = self.static(x)
        # --- 拼接与融合 ---
        # 1. 在最后一个维度（特征维度）上拼接两个路径的输出
        # 形状: [batch_size, seq_len, out_features] + [batch_size, seq_len, out_features]
        # -> [batch_size, seq_len, 2 * out_features]
        # 使用dropout强迫两个模块都学习，两个模块的信息需要组合
        combined_output = self.dropout(torch.cat([x_static, x_dynamic], dim=-1))

        # 2. 使用静态的线性层（compressor）将拼接后的特征降维回原始维度
        # 形状: [batch_size, seq_len, 2 * out_features] -> [batch_size, seq_len, out_features]
        final_output = self.compressor(combined_output)
        return final_output

