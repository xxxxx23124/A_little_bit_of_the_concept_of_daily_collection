import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
from datasets import load_dataset  # Hugging Face Datasets
import random
from tqdm import tqdm

from experiment.Transformer.model_test.prism.prismTransformer import PrismTransformer


class MaskedReconstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len=512, mask_prob=0.15, mask_span_len=3):
        """
        自定义数据集：加载原始文本，生成masked版本作为encoder输入，原始作为decoder目标。

        Args:
            dataset: Hugging Face Dataset对象（e.g., text column）。
            tokenizer: BertTokenizer。
            max_seq_len: 最大序列长度。
            mask_prob: 掩码概率（整体比例）。
            mask_span_len: 平均span长度（模拟连续词缺失）。
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_span_len = mask_span_len
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']  # 假设数据集有'text'列
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_seq_len - 2]  # 预留CLS/SEP

        # 添加CLS和SEP
        input_tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        seq_len = len(input_tokens)

        # 生成masked版本（encoder输入）：随机mask spans
        masked_tokens = input_tokens.copy()
        i = 1  # 从CLS后开始
        while i < seq_len - 1:  # 跳过CLS/SEP
            if random.random() < self.mask_prob:
                span_len = random.randint(1, self.mask_span_len)
                masked_tokens[i:i + span_len] = [self.mask_token_id] * span_len
                i += span_len
            else:
                i += 1

        # Padding
        pad_len = self.max_seq_len - seq_len
        encoder_input_ids = masked_tokens + [self.pad_token_id] * pad_len
        decoder_input_ids = [self.cls_token_id] + tokens  # Decoder: teacher-forcing，右移一位（以CLS开头，原tokens）
        decoder_target_ids = tokens + [self.sep_token_id] + [self.pad_token_id] * (
                    pad_len - 1)  # 目标：原tokens + SEP + pad

        # Masks: 0 for pad, 1 for content
        encoder_attention_mask = [1] * seq_len + [0] * pad_len
        decoder_attention_mask = [1] * (seq_len - 1) + [0] * (pad_len + 1)  # Decoder seq_len稍短

        return {
            'encoder_input_ids': torch.tensor(encoder_input_ids),
            'decoder_input_ids': torch.tensor(decoder_input_ids),
            'encoder_attention_mask': torch.tensor(encoder_attention_mask),
            'decoder_attention_mask': torch.tensor(decoder_attention_mask),
            'labels': torch.tensor(decoder_target_ids)  # 用于loss计算
        }


def train(model, dataloader, optimizer, device, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=model.token_embedding.padding_idx)  # 忽略pad

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            logits = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                decoder_kv_caches=None  # 预训练不用缓存
            )

            # Loss: 展平logits和labels
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    # Tokenizer加载
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    dataset = load_dataset('thu-coai/lccc', 'base', split='train', trust_remote_code=True)

    # 你的自定义数据集类（MaskedReconstructionDataset）可以直接使用这个 IterableDataset
    custom_dataset = MaskedReconstructionDataset(dataset, tokenizer, max_seq_len=512, mask_prob=0.20, mask_span_len=5)

    # DataLoader 支持 IterableDataset，设置 batch_size 等
    dataloader = DataLoader(custom_dataset, batch_size=16, num_workers=4)
    # 无需 shuffle=True，因为 streaming 数据集不支持随机访问，但你可以添加 dataset.shuffle()

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrismTransformer(
        vocab_size=tokenizer.vocab_size,
        num_layers=8,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        num_linears=3,
        max_seq_len=512,
        dropout_rate=0.1,
        use_checkpointing=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 开始训练
    train(model, dataloader, optimizer, device, epochs=5)  # 调整epochs