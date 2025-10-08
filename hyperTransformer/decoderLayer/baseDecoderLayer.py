import torch
import torch.nn as nn
from abc import ABC, abstractmethod



from hyperTransformer.rmsNorm import RMSNorm
from hyperTransformer.rotaryEmbedding import RotaryEmbedding
from hyperTransformer.kvCache import KVCache # 假设 KVCache 的类型定义


class BaseDecoderLayer(nn.Module, ABC):
    """
    Transformer 解码器层的抽象基类，采用了 Pre-Norm 结构。

    这个基类定义了解码器层的通用架构：
    1. 第一个子层块 (带因果掩码的自注意力):
       (Norm -> Self-Attention -> Dropout) + Residual Connection
    2. 第二个子层块 (交叉注意力):
       (Norm -> Cross-Attention -> Dropout) + Residual Connection
    3. 第三个子层块 (前馈网络):
       (Norm -> FFN -> Dropout) + Residual Connection

    子类需要通过调用 super().__init__() 并实现 _init_sublayers 方法
    来定义具体的自注意力、交叉注意力和 FFN 模块。
    """

    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        """
        Args:
            d_model (int): 模型的维度。
            dropout_rate (float): 子层输出后的 dropout 比率。
            **kwargs: 传递给 _init_sublayers 的额外参数。
        """
        super().__init__()

        # 1. 定义三个子层块共有的模块
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = RMSNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm3 = RMSNorm(d_model)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. 初始化具体的 attention 和 ffn 模块为 None，由子类负责定义
        self.self_attention = None
        self.cross_attention = None
        self.ffn = None

        # 3. 调用抽象方法，强制子类实现子模块的初始化
        self._init_sublayers(d_model=d_model, **kwargs)

        # 4. 确保子类已经正确初始化了所有子模块
        assert self.self_attention is not None, "self.self_attention must be initialized in _init_sublayers"
        assert self.cross_attention is not None, "self.cross_attention must be initialized in _init_sublayers"
        assert self.ffn is not None, "self.ffn must be initialized in _init_sublayers"

    @abstractmethod
    def _init_sublayers(self, d_model, **kwargs):
        """
        抽象方法，子类必须实现此方法来初始化 self.self_attention,
        self.cross_attention 和 self.ffn 模块。

        例如:
            self.self_attention = MyCustomSelfAttention(...)
            self.cross_attention = MyCustomCrossAttention(...)
            self.ffn = MyCustomFFN(...)
        """
        raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor,
                rotary_emb: RotaryEmbedding | None,
                self_attn_kv_cache: KVCache | None,
                cross_attn_kv_cache: KVCache | None
                ) -> torch.Tensor:
        """
        通用的前向传播逻辑，遵循 Pre-Norm 结构。

        Args:
            x (torch.Tensor): 输入张量 (来自前一个解码器层)，形状 (batch, seq_len, d_model)。
            context (torch.Tensor): 编码器的输出，形状 (batch, context_len, d_model)。
            rotary_emb (RotaryEmbedding | None): 旋转位置编码模块。
            self_attn_kv_cache (KVCache | None): 自注意力的KV缓存。
            cross_attn_kv_cache (KVCache | None): 交叉注意力的KV缓存。

        Returns:
            torch.Tensor: 解码器层的输出，形状与输入 x 相同。
        """
        # --- 第一个子层：带因果掩码的自注意力 ---
        residual_1 = x
        x_norm1 = self.norm1(x)

        # 自注意力总是使用因果掩码
        self_attn_output = self.self_attention(
            x_norm1,
            rotary_emb=rotary_emb,
            kv_cache=self_attn_kv_cache,
            use_causal_mask=True  # 关键点：解码器自注意力必须使用因果掩码
        )

        x = residual_1 + self.dropout1(self_attn_output)

        # --- 第二个子层：交叉注意力 ---
        residual_2 = x
        x_norm2 = self.norm2(x)

        # 交叉注意力层，查询(Q)来自解码器，键(K)和值(V)来自编码器的输出(context)
        cross_attn_output = self.cross_attention(
            x=x_norm2,
            context=context,
            kv_cache=cross_attn_kv_cache
        )

        x = residual_2 + self.dropout2(cross_attn_output)

        # --- 第三个子层：前馈神经网络 ---
        residual_3 = x
        x_norm3 = self.norm3(x)

        ffn_output = self.ffn(x_norm3)

        x = residual_3 + self.dropout3(ffn_output)

        return x