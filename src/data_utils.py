import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os


class CharDataset(Dataset):
    """字符级文本数据集"""

    def __init__(self, text, block_size):
        if text is None or len(text) == 0:
            raise ValueError("文本数据不能为空")

        self.text = text
        self.block_size = block_size
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        print(f"数据集信息: 总字符数={len(text):,}, 词汇表大小={self.vocab_size}, 块大小={block_size}")

    def __len__(self):
        return len(self.text) - self.block_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError(f"索引 {idx} 超出范围 [0, {len(self) - 1}]")

        chunk = self.text[idx:idx + self.block_size + 1]
        x = torch.tensor([self.stoi[c] for c in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in chunk[1:]], dtype=torch.long)
        return x, y

    def encode(self, text):
        if text is None or len(text) == 0:
            raise ValueError("编码文本不能为空")
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens):
        if tokens is None:
            raise ValueError("解码tokens不能为None")

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([self.itos[i] for i in tokens])


def download_tiny_shakespeare():
    """下载Tiny Shakespeare数据集"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs('data', exist_ok=True)
    filepath = 'data/tiny_shakespeare.txt'

    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare dataset...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download completed!")
        except Exception as e:
            print(f"下载失败: {e}")
            # 创建示例数据
            example_text = "ROMEO: Hello world! This is a sample text.\nJULIET: Welcome to Transformer implementation.\n"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(example_text)
            print("创建示例数据")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if len(text) == 0:
            raise ValueError("数据集文件为空")

        return text
    except Exception as e:
        print(f"读取数据失败: {e}")
        # 返回默认数据
        return "ROMEO: Hello world! This is a sample text.\nJULIET: Welcome to Transformer implementation.\n" * 100


def get_data_loader(batch_size, block_size, split_ratio=0.9):
    """获取数据加载器"""
    text = download_tiny_shakespeare()
    n = int(split_ratio * len(text))

    train_dataset = CharDataset(text[:n], block_size)
    val_dataset = CharDataset(text[n:], block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.vocab_size, train_dataset


def create_mask(batch_size, seq_len, device):
    """创建因果mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(device)