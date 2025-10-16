from collections import defaultdict, Counter, OrderedDict
import re
from tqdm import tqdm
import pickle

class BPETokenizer:
    def __init__(self, vocab_size=20000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[CLS]']
        self.word_to_id = {}
        self.id_to_word = {}
        self.merges = OrderedDict()  # 使用 OrderedDict 以保持合并顺序
        self.pad_id = None
        self.unk_id = None
        self.cls_id = None

    def train(self, texts):
        # 步骤1: 构建初始字符词汇表
        all_chars = set()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text.lower())  # 改进: 保留标点作为单独token
            for word in words:
                all_chars.update(word)
        char_vocab = list(all_chars)
        char_vocab.sort()

        # 添加special tokens到vocab
        vocab = self.special_tokens + char_vocab
        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.pad_id = self.word_to_id['[PAD]']
        self.unk_id = self.word_to_id['[UNK]']
        self.cls_id = self.word_to_id['[CLS]']

        # 步骤2: 准备训练数据 (词频计数，词以' '分隔字符)
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_freq.update(words)

        # 初始化词作为字符列表 (e.g., 'hello' -> ['h', 'e', 'l', 'l', 'o'])
        corpus = {}
        for word, freq in word_freq.items():
            corpus[' '.join(list(word)) + ' </w>'] = freq  # 添加</w>作为词尾标记

        # 步骤3: 训练BPE - 反复合并高频对
        current_vocab_size = len(self.word_to_id)
        pbar = tqdm(total=self.vocab_size - current_vocab_size, desc="BPE Training")  # 添加进度条
        while current_vocab_size < self.vocab_size:
            pair_freq = self._get_pair_freq(corpus)
            if not pair_freq:
                break
            best_pair = max(pair_freq, key=pair_freq.get)
            merged_token = ''.join(best_pair).replace(' ', '')  # e.g., ('h', 'e') -> 'he'

            # 添加新token到vocab
            new_id = len(self.word_to_id)
            self.word_to_id[merged_token] = new_id
            self.id_to_word[new_id] = merged_token
            self.merges[best_pair] = merged_token

            # 更新corpus: 替换所有出现的best_pair
            self._update_corpus(corpus, best_pair, merged_token)
            current_vocab_size += 1
            pbar.update(1)  # 更新进度条

        pbar.close()
        self.vocab_size = len(self.word_to_id)

    def _get_pair_freq(self, corpus):
        pair_freq = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freq[pair] += freq
        return pair_freq

    def _update_corpus(self, corpus, pair, merged):
        new_corpus = {}
        for word, freq in corpus.items():
            symbols = word.split()
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word = ' '.join(new_symbols)
            new_corpus[new_word] = freq
        corpus.clear()
        corpus.update(new_corpus)

    def encode(self, text, max_length=512):
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = [self.cls_id]
        for word in words:
            subwords = self._bpe_encode_word(word)
            tokens.extend(subwords)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        padding_length = max_length - len(tokens)
        tokens += [self.pad_id] * padding_length
        return tokens

    def _bpe_encode_word(self, word):
        symbols = list(word) + ['</w>']  # 添加词尾
        # 按合并规则的顺序依次应用每个规则
        for pair, merged in self.merges.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        # 移除</w>，映射到ID
        ids = []
        for sym in symbols:
            if sym == '</w>':
                continue
            ids.append(self.word_to_id.get(sym, self.unk_id))
        return ids

    def get_vocab_size(self):
        return self.vocab_size

    def save(self, path):
        """保存tokenizer到文件"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'merges': self.merges,
                'pad_id': self.pad_id,
                'unk_id': self.unk_id,
                'cls_id': self.cls_id
            }, f)

    @classmethod
    def load(cls, path):
        """从文件加载tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls(vocab_size=data['vocab_size'], special_tokens=data['special_tokens'])
        tokenizer.word_to_id = data['word_to_id']
        tokenizer.id_to_word = data['id_to_word']
        tokenizer.merges = data['merges']
        tokenizer.pad_id = data['pad_id']
        tokenizer.unk_id = data['unk_id']
        tokenizer.cls_id = data['cls_id']
        tokenizer.vocab_size = len(tokenizer.word_to_id)  # 确保vocab_size更新
        return tokenizer