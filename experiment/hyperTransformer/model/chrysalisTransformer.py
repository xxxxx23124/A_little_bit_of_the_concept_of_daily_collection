import torch.nn as nn
from torch import Tensor
import math

from experiment.hyperTransformer.encoderLayer.hybridEncoderLayer import HybridEncoderLayer
from experiment.hyperTransformer.encoderLayer.halfHybridEncoderLayer import HalfHybridEncoderLayer
from experiment.hyperTransformer.encoderLayer.dualEncoderLayer import DualEncoderLayer
from experiment.hyperTransformer.encoderLayer.halfDualEncoderLayer import HalfDualEncoderLayer
from experiment.hyperTransformer.encoderLayer.halfHyperEncoderLayer import HalfHyperEncoderLayer
from experiment.hyperTransformer.encoder import Encoder
from experiment.hyperTransformer.rotaryEmbedding import RotaryEmbedding


# ==============================================================================
# The Main Model: ChrysalisTransformer - ForSequenceClassification
# ==============================================================================

class ChrysalisTransformer(nn.Module):
    """
    一个完整的、自包含的、用于序列分类任务的Encoder-Only模型。
    它使用一个“配方驱动”的Chrysalis Encoder作为其主干。

    该模型封装了从token ID输入到最终分类logits输出的完整流程。
    """

    def __init__(self,
                 # --- 模型架构参数 ---
                 vocab_size: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 compressed_feature_dim:int,
                 num_monarchs: int = 4,
                 # --- 位置编码参数 ---
                 max_seq_len: int = 512,
                 rope_base: int = 10000,
                 # --- 其他配置 ---
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 use_checkpointing:bool = False
                 ):
        """
        Args:
            vocab_size (int): 词汇表大小。
            num_layers (int): Encoder的总层数。必须大于1。
            d_model (int): 模型的维度 (必须是完全平方数)。
            num_heads (int): 注意力头的数量。
            d_ff (int): FFN的中间层维度 (最好是完全平方数)。
            compressed_feature_dim(int): 动态生成参数的‘控制信号’的维度。
            num_monarchs (int): 混合专家模块中的专家数量。
            max_seq_len (int): 模型能处理的最大序列长度。
            rope_base (int): RoPE的基数。
            num_classes (int): 输出分类的数量 (例如, IMDb为2)。
            dropout_rate (float): Dropout比率。
            use_checkpointing(bool): 是否使用checkpoint
        """
        super().__init__()
        assert num_layers > 1, "num_layers must be greater than 1 to use the specified recipe."
        assert math.isqrt(d_model) ** 2 == d_model, "d_model must be a perfect square for Chrysalis."

        # --- 1. 输入层 (Input Layer) ---
        # Token Embedding: 将 token ID 映射为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Rotary Positional Embedding (RoPE)
        assert d_model % num_heads == 0, "d_model % num_heads must be 0"
        head_dim = d_model // num_heads
        assert head_dim % 2 == 0, "Rotary Positional Embedding (RoPE)'s head_dim must be even"
        self.rotary_emb = RotaryEmbedding(base=rope_base, head_dim=head_dim, max_seq_len=max_seq_len)

        # --- 2. 主干：Chrysalis Encoder ---
        # 定义我们的特殊“配方”
        # 第一层是 HalfHybridEncoderLayer，其余层是 HybridEncoderLayer
        # layer_recipe = [HalfHybridEncoderLayer] + [HybridEncoderLayer] * (num_layers - 1)
        # layer_recipe = [HalfDualEncoderLayer] + [DualEncoderLayer] * (num_layers - 1)
        # layer_recipe = [HalfHybridEncoderLayer] * num_layers
        # layer_recipe = [HalfDualEncoderLayer] * num_layers
        layer_recipe = [HalfHyperEncoderLayer] * num_layers
        # 创建Encoder实例，传入配方和共享的层参数
        self.encoder = Encoder(
            layer_recipe=layer_recipe,
            d_model=d_model,
            # 以下所有参数将通过 **layer_kwargs 传递给每一层的构造函数
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            compressed_feature_dim=compressed_feature_dim,
            num_monarchs=num_monarchs,
            use_checkpointing=use_checkpointing
        )

        # --- 3. 输出层 (Classification Head) ---
        # 我们使用 [CLS] token (即序列的第一个token) 的输出来进行分类
        self.classification_head = nn.Linear(d_model, num_classes)

        # --- 4. 保存配置 ---
        # 将关键配置保存下来，方便后续使用
        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor | None = None) -> Tensor:
        """
        模型的前向传播。

        Args:
            input_ids (Tensor): 输入的token ID，形状 (batch_size, seq_len)。
            attention_mask (Tensor, optional):
                attention mask。形状 (batch_size, seq_len)。
                值为1表示token有效，值为0表示是padding。

        Returns:
            Tensor: 最终的分类logits，形状 (batch_size, num_classes)。
        """
        # 1. 创建注意力掩码
        # F.scaled_dot_product_attention 需要的掩码是布尔类型，True表示被屏蔽
        # 我们的输入 attention_mask 是 1/0 类型，需要转换
        if attention_mask is not None:
            # (B, S) -> (B, 1, 1, S)
            # F.sdpa期望的形状可以被广播到 (B, H, S_q, S_k)
            # 我们将其反转 (0 -> True, 1 -> False)
            attn_mask_for_pytorch = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
        else:
            attn_mask_for_pytorch = None

        # 2. 计算输入嵌入
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)

        # 3. 通过Encoder主干
        encoder_output = self.encoder(
            x,
            rotary_emb=self.rotary_emb,
            attention_mask=attn_mask_for_pytorch
        )

        # 4. 池化 (Pooling)
        # 我们只取序列第一个 token ([CLS] token) 的输出来代表整个序列的语义
        cls_token_output = encoder_output[:, 0, :]

        # 5. 通过分类头得到Logits
        logits = self.classification_head(cls_token_output)

        return logits


if __name__ == '__main__':
    import torch
    # --- 为IMDb任务推荐的超小型配置 (测试参数效率) ---
    config = {
        'vocab_size': 20000,
        'num_layers': 4,
        'd_model': 64,  # 8*8
        'num_heads': 4,
        'd_ff': 256,  # 16*16
        'compressed_feature_dim':32,
        'num_monarchs': 3,
        'max_seq_len': 512,
        'num_classes': 2,
    }

    print("--- 实例化 ChrysalisTransformer ---")
    model = ChrysalisTransformer(**config)
    print("模型实例化成功！")

    # 打印模型结构
    # print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")  # 转换为百万 (M)

    # --- 模拟一次前向传播 ---
    batch_size = 4
    seq_len = 128

    # 随机生成输入数据
    dummy_input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    # 假设最后20个token是padding
    dummy_attention_mask[:, -20:] = 0

    print("\n--- 测试一次前向传播 ---")
    print(f"输入 input_ids 形状: {dummy_input_ids.shape}")
    print(f"输入 attention_mask 形状: {dummy_attention_mask.shape}")

    # 运行模型
    logits = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

    print(f"输出 logits 形状: {logits.shape}")
    assert logits.shape == (batch_size, config['num_classes'])
    print("前向传播成功！输出形状正确。")

    # 检查辅助损失
    """
    total_aux_loss = 0
    # model.modules() 会递归地返回模型中的所有模块 (包括它自己)
    for module in model.modules():
        # 检查模块是否是我们想要收集损失的类型
        if isinstance(module, HyperMoMixLinear):
            # 累加这个模块在前向传播中记录的所有损失
            total_aux_loss += sum(module.auxiliary_losses)
    # 将主损失和辅助损失相加
    loss = main_loss + total_aux_loss
    """