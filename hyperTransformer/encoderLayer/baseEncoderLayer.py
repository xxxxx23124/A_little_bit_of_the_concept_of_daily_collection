import torch.nn as nn
from abc import ABC, abstractmethod

from hyperTransformer.rmsNorm import RMSNorm
from hyperTransformer.rotaryEmbedding import RotaryEmbedding

class BaseEncoderLayer(nn.Module, ABC):
    """
    Transformer 编码器层的抽象基类，采用了 Pre-Norm 结构。

    这个基类定义了编码器层的通用架构：
    1. 第一个子层块：(Norm -> Self-Attention -> Dropout) + Residual Connection
    2. 第二个子层块：(Norm -> FFN -> Dropout) + Residual Connection

    子类需要通过调用 super().__init__() 并实现 _init_sublayers 方法
    来定义具体的 self-attention 和 FFN 模块。
    """
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        """
        Args:
            d_model (int): 模型的维度。
            dropout_rate (float): 子层输出后的 dropout 比率。
            **kwargs: 传递给 _init_sublayers 的额外参数。
        """
        super().__init__()

        # 1. 定义两个子层块共有的模块
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = RMSNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        # 2. 初始化具体的 attention 和 ffn 模块为 None，由子类负责定义
        self.attention = None
        self.ffn = None

        # 3. 调用抽象方法，强制子类实现子模块的初始化
        self._init_sublayers(d_model=d_model, **kwargs)

        # 4. 确保子类已经正确初始化了子模块
        assert self.attention is not None, "self.attention must be initialized in _init_sublayers"
        assert self.ffn is not None, "self.ffn must be initialized in _init_sublayers"

    @abstractmethod
    def _init_sublayers(self, d_model, **kwargs):
        """
        抽象方法，子类必须实现此方法来初始化 self.attention 和 self.ffn 模块。

        例如:
            self.attention = MyCustomAttention(...)
            self.ffn = MyCustomFFN(...)
        """
        raise NotImplementedError

    def forward(self, x, rotary_emb:RotaryEmbedding|None):
        """
        通用的前向传播逻辑，遵循 Pre-Norm 结构。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            rotary_emb (RotaryEmbedding | None): 旋转位置编码模块。

        Returns:
            torch.Tensor: 编码器层的输出，形状与输入 x 相同。
        """
        # --- 第一个子层：多头自注意力 ---
        # 残差连接的 "输入" (原始的 x) + Pre-Norm
        residual_1 = x
        x_norm1 = self.norm1(x)

        # 将归一化后的数据送入注意力层
        # 典型的Pre-Norm实现会将归一化的x_norm1传递给Q,K,V的计算。我们遵循这个标准实践。
        attention_output = self.attention(x_norm1, rotary_emb, kv_cache=None, use_causal_mask=False)

        # 应用Dropout和残差连接
        x = residual_1 + self.dropout1(attention_output)

        # --- 第二个子层：前馈神经网络 ---
        # 第二个残差连接的 "输入" + Pre-Norm
        residual_2 = x
        x_norm2 = self.norm2(x)

        # 将归一化后的数据送入FFN
        ffn_output = self.ffn(x_norm2)

        # 应用Dropout和残差连接
        x = residual_2 + self.dropout2(ffn_output)

        return x