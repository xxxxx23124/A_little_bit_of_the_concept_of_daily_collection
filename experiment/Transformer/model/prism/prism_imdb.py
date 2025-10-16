import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import re
from tqdm import tqdm
from datasets import load_dataset
import math

from experiment.Transformer.model.prism.prismTransformer import PrismTransformer
from experiment.Transformer.linear.staticCompositeLinear import StaticCompositeLinear

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
            words = re.findall(r'\w+', text.lower())
            counter.update(words)
        most_common = counter.most_common(self.vocab_size - len(self.special_tokens))
        vocab = self.special_tokens + [word for word, _ in most_common]
        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.pad_id = self.word_to_id[self.pad_token]
        self.unk_id = self.word_to_id[self.unk_token]
        self.cls_id = self.word_to_id[self.cls_token]
        self.vocab_size = len(self.word_to_id)

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

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(model, train_loader, optimizer, scheduler, criterion, device,
          accumulation_steps, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_aux_loss = 0
        for module in model.modules():
            if isinstance(module, StaticCompositeLinear):
                total_aux_loss += sum(module.auxiliary_losses)
                module.clear_auxiliary_losses()

        # 将主损失和辅助损失相加
        total_batch_loss = loss + total_aux_loss

        # 梯度累积
        total_batch_loss = total_batch_loss / accumulation_steps
        total_batch_loss.backward()

        total_loss += total_batch_loss.item() * accumulation_steps  # 还原真实损失用于统计
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 使用 set_postfix 更新进度条后缀
        pbar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Acc': f'{(predicted == labels).sum().item() / labels.size(0):.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.8f}',
        })

    # 处理 epoch 结束时剩余的未更新的梯度
    if len(train_loader) % accumulation_steps != 0:
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    pbar.close()
    avg_loss = total_loss / len(train_loader)
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

if __name__ == '__main__':
    config = {
        'vocab_size': 20000,
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'num_linears': 3,
        'max_seq_len': 512,
        'num_classes': 2,
        'dropout_rate': 0.1,
        'use_checkpointing': False
    }
    accumulation_steps = 2
    batch_size = 16

    # --- 学习率配置 ---
    lr = 1e-4  # 统一学习率

    print("加载IMDb数据集...")
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']

    print("构建词汇表...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab([example['text'] for example in train_data])
    config['vocab_size'] = tokenizer.get_vocab_size()

    train_dataset = IMDbDataset(train_data, tokenizer, config['max_seq_len'])
    test_dataset = IMDbDataset(test_data, tokenizer, config['max_seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("实例化模型...")
    model = PrismTransformer(**config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("创建优化器...")
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, criterion, device,
                                      accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("训练完成！")