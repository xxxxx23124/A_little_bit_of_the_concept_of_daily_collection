import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from datasets import load_dataset
import math
from transformers import AutoTokenizer

from experiment.Transformer.model.prism.prismTransformer import PrismTransformer
from experiment.Transformer.linear.staticCompositeLinear import StaticCompositeLinear


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
        # 用 transformers 的 encode_plus
        encoding = self.tokenizer.encode_plus(
            text.lower(),  # 保持 lower，如果用 uncased
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True  # 自动加 [CLS], [SEP] 等
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉 batch 维
            'attention_mask': encoding['attention_mask'].squeeze(0),
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
            'Acc': f'{correct / total:.4f}',
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
        'vocab_size': 20000,  # 这个会更新为 tokenizer 的实际大小
        'num_layers': 4,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'num_linears': 2,
        'max_seq_len': 512,
        'num_classes': 2,
        'dropout_rate': 0.2
    }
    accumulation_steps = 1
    batch_size = 64

    # --- 学习率配置 ---
    lr = 1e-4  # 统一学习率

    print("加载IMDb数据集...")
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']

    print("构建词汇表...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 推荐 BERT 的 tokenizer，英文 uncased
    config['vocab_size'] = tokenizer.vocab_size  # 更新 vocab_size，通常是 30522

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

    num_epochs = 60
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