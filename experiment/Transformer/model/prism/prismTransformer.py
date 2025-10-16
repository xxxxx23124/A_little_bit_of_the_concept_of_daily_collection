import torch.nn as nn
from torch import Tensor

from experiment.Transformer.encoderLayer.staticCompositeEncoderLayer import StaticCompositeEncoderLayer
from experiment.Transformer.encoderLayer.halfStaticCompositeEncoderLayer import HalfStaticCompositeEncoderLayer
from experiment.Transformer.encoder import Encoder

from experiment.Transformer.decoderLayer.halfStaticCompositeDecoderLayer import HalfStaticCompositeDecoderLayer
from experiment.Transformer.decoderLayer.staticCompositeDecoderLayer import StaticCompositeDecoderLayer
from experiment.Transformer.decoder import Decoder

from experiment.Transformer.rotaryEmbedding import RotaryEmbedding
from experiment.Transformer.kvCache import KVCache

class PrismTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_linears: int = 4,
                 max_seq_len: int = 512,
                 rope_base: int = 10000,
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 use_checkpointing:bool = False
                 ):

        super().__init__()
        assert num_layers > 1, "num_layers must be greater than 1 to use the specified recipe."

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        assert d_model % num_heads == 0, "d_model % num_heads must be 0"
        head_dim = d_model // num_heads
        assert head_dim % 2 == 0, "Rotary Positional Embedding (RoPE)'s head_dim must be even"
        self.rotary_emb = RotaryEmbedding(base=rope_base, head_dim=head_dim, max_seq_len=max_seq_len)

        encoder_layer_recipe = [HalfStaticCompositeEncoderLayer] + [StaticCompositeEncoderLayer] * (num_layers - 1)
        decoder_layer_recipe = [HalfStaticCompositeDecoderLayer] + [StaticCompositeDecoderLayer] * (num_layers - 1)

        self.encoder = Encoder(
            layer_recipe=encoder_layer_recipe,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            num_linears=num_linears,
            use_checkpointing=use_checkpointing
        )

        self.decoder = Decoder(
            layer_recipe=decoder_layer_recipe,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            num_linears=num_linears,
            use_checkpointing=use_checkpointing
        )

        self.debedding = nn.Linear(d_model, vocab_size)
        self.debedding.weight = self.token_embedding.weight

        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}

    def forward(self,
                encoder_input_ids: Tensor,
                decoder_input_ids: Tensor,
                encoder_attention_mask: Tensor | None = None,
                decoder_attention_mask: Tensor | None = None,
                decoder_kv_caches: list[tuple[KVCache, KVCache]] | None = None,
                ) -> Tensor:
        """
        前向传播方法，处理编码器和解码器的输入，生成预测 logits。

        Args:
            encoder_input_ids (Tensor): 编码器输入的 token IDs，形状 (batch_size, encoder_seq_len)。
            decoder_input_ids (Tensor): 解码器输入的 token IDs，形状 (batch_size, decoder_seq_len)。
            encoder_attention_mask (Tensor | None, optional): 编码器注意力掩码，形状 (batch_size, encoder_seq_len)，值为 0 表示 padding 位置。默认为 None。
            decoder_attention_mask (Tensor | None, optional): 解码器注意力掩码，形状 (batch_size, decoder_seq_len)，值为 0 表示 padding 位置。默认为 None。
            decoder_kv_caches (list[tuple[KVCache, KVCache]] | None, optional): 解码器各层的 KV 缓存列表，用于自回归推理。每个元素是一个元组 (self_attn_kv_cache, cross_attn_kv_cache)。默认为 None。

        Returns:
            Tensor: 预测 logits，形状 (batch_size, decoder_seq_len, vocab_size)。

        Notes:
            - 在训练模式下，通常不使用 KV 缓存 (decoder_kv_caches 为 None)，处理整个序列。
            - 在推理模式下，提供 KV 缓存以支持增量生成（例如，一次生成一个 token）。
            - 掩码假设：输入掩码中 0 表示 padding 位置，将在内部转换为注意力屏蔽掩码。
        """

        encoder_output = self.encoder(
            x=self.embedding_dropout(self.token_embedding(encoder_input_ids)),
            rotary_emb=self.rotary_emb,
            self_attention_mask=encoder_attention_mask
        )

        decoder_output = self.decoder(
            x=self.embedding_dropout(self.token_embedding(decoder_input_ids)),
            context=encoder_output,
            rotary_emb=self.rotary_emb,
            all_kv_caches=decoder_kv_caches,
            self_attention_mask=decoder_attention_mask,
            cross_attention_mask=encoder_attention_mask
        )

        prediction_logits = self.debedding(decoder_output)

        return prediction_logits