import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from tqdm import tqdm
from datasets import load_dataset  # 使用Hugging Face datasets加载IMDb，简单且跨平台

from hyperTransformer.model.chrysalisTransformer import ChrysalisTransformer
from hyperTransformer.linear.hyperMoMixLinear import HyperMoMixLinear

# ==============================================================================
# 数据集和Tokenizer
# ==============================================================================

class SimpleTokenizer:
    def __init__(self, vocab_size=20000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[CLS]']
        self.pad_token = self.special_tokens[0]
        self.unk_token = self.special_tokens[1]
        self.cls_token = self.special_tokens[2]
        self.word_to_id = {}
        self.id_to_word = {}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            words = re.findall(r'\w+', text.lower())  # 简单分词：字母数字单词
            counter.update(words)

        # 最常见的词
        most_common = counter.most_common(self.vocab_size - len(self.special_tokens))
        vocab = self.special_tokens + [word for word, _ in most_common]

        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        self.pad_id = self.word_to_id[self.pad_token]
        self.unk_id = self.word_to_id[self.unk_token]
        self.cls_id = self.word_to_id[self.cls_token]
        self.vocab_size = len(self.word_to_id)  # 更新实际大小

    def encode(self, text, max_length=512):
        words = re.findall(r'\w+', text.lower())
        tokens = [self.cls_id] + [self.word_to_id.get(word, self.unk_id) for word in words]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        padding_length = max_length - len(tokens)
        tokens += [self.pad_id] * padding_length
        return tokens

    def get_vocab_size(self):
        return self.vocab_size


class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        input_ids = self.tokenizer.encode(text, self.max_length)
        attention_mask = [1 if token != self.tokenizer.pad_id else 0 for token in input_ids]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(label)
        }


# ==============================================================================
# 训练函数
# ==============================================================================

def train(model, train_loader, optimizer, criterion, device, accumulation_steps=64):
    """
    使用梯度累积进行模型训练的函数。

    Args:
        model (torch.nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.functional): 损失函数。
        device (torch.device): 运行设备 (e.g., 'cuda' or 'cpu')。
        accumulation_steps (int): 梯度累积的步数。
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 在进入循环前，先清零一次梯度，这是一个好的习惯
    optimizer.zero_grad()

    # 使用 enumerate 获取批次索引，方便进行条件判断
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 1. 前向传播
        logits = model(input_ids, attention_mask)

        # 2. 计算主损失
        loss = criterion(logits, labels)

        print(f"loss: {loss.item():.4f}")

        # 3. 累加辅助损失
        total_aux_loss = 0
        for module in model.modules():
            if isinstance(module, HyperMoMixLinear):
                total_aux_loss += sum(module.auxiliary_losses)

        # 将主损失和辅助损失相加
        loss += total_aux_loss

        # 4. 【关键】损失归一化
        # 为了使每次累积的梯度量级与正常训练保持一致，需要将损失除以累积步数
        loss = loss / accumulation_steps

        # 5. 【关键】反向传播，计算并累积梯度
        loss.backward()

        # 6. 【关键】根据累积步数，条件性地更新模型参数
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度已经累积了 accumulation_steps 次，现在执行参数更新
            optimizer.step()
            # 更新后，清零梯度，为下一轮累积做准备
            optimizer.zero_grad()

        # --- 以下是用于统计和日志记录的代码，不受梯度累积影响 ---
        # 注意：这里我们累加的是归一化后的 loss.item()
        # 为了得到真实的平均损失，最后需要乘以 accumulation_steps
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # 【关键】处理最后一个批次
    # 如果总批次数量不是 accumulation_steps 的整数倍，
    # 那么最后一个累积周期是不完整的，需要手动更新一次。
    if (len(train_loader)) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # 计算平均损失时，乘以 accumulation_steps 还原真实的损失值
    avg_loss = (total_loss / len(train_loader)) * accumulation_steps
    avg_acc = correct / total

    return avg_loss, avg_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(test_loader), correct / total


# ==============================================================================
# 主脚本
# ==============================================================================

if __name__ == '__main__':
    # 配置
    config = {
        'vocab_size': 20000,
        'num_layers': 8,
        'd_model': 576,
        'num_heads': 8,
        'd_ff': 2304,
        'compressed_feature_dim': 72,
        'num_experts': 4,
        'max_seq_len': 512,
        'num_classes': 2,
        'use_checkpointing': True
    }

    # 加载IMDb数据集
    print("加载IMDb数据集...")
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']

    # 构建tokenizer
    print("构建词汇表...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab([example['text'] for example in train_data])
    config['vocab_size'] = tokenizer.get_vocab_size()  # 更新vocab_size以匹配实际

    # 创建数据集
    train_dataset = IMDbDataset(train_data, tokenizer, config['max_seq_len'])
    test_dataset = IMDbDataset(test_data, tokenizer, config['max_seq_len'])

    # DataLoader
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 实例化模型
    print("实例化模型...")
    model = ChrysalisTransformer(**config)
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")  # 转换为百万 (M)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("训练完成！")