import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class TextGenerator:
    """文本生成器"""

    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=None):
        """生成文本"""
        self.model.eval()

        # 编码提示文本
        tokens = self.dataset.encode(prompt).unsqueeze(0).to(self.device)
        generated = tokens.clone()

        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Generating text"):
                logits = self.model(generated)[0, -1, :]

                # 应用温度调节
                logits = logits / temperature

                # Top-k采样
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # 如果生成了结束符，可以提前停止
                # 这里我们使用简单的长度控制

        return self.dataset.decode(generated[0].cpu())

    def generate_multiple_samples(self, prompt, num_samples=3, max_length=100, temperature=0.8):
        """生成多个样本用于比较"""
        samples = []
        for i in range(num_samples):
            sample = self.generate_text(prompt, max_length, temperature)
            samples.append(sample)
            print(f"Sample {i + 1}: {sample}\n")

        return samples

    def calculate_perplexity(self, text):
        """计算文本的困惑度"""
        self.model.eval()

        tokens = self.dataset.encode(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tokens)[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens[0, 1:].view(-1)
            )
            perplexity = torch.exp(loss)

        return perplexity.item()


def analyze_attention_patterns(model, text, dataset, layer_idx=0, head_idx=0):
    """分析注意力模式"""
    model.eval()

    tokens = dataset.encode(text).unsqueeze(0)

    # 钩子函数来获取注意力权重
    attention_weights = []

    def hook_fn(module, input, output):
        attn_weights = output[1]  # (batch_size, n_heads, seq_len, seq_len)
        attention_weights.append(attn_weights.detach())

    # 注册钩子
    hook = model.decoder.layers[layer_idx].self_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(tokens)

    # 移除钩子
    hook.remove()

    # 获取特定头和层的注意力权重
    attn = attention_weights[0][0, head_idx]  # (seq_len, seq_len)

    # 绘制注意力热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(attn.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # 设置刻度标签
    chars = [dataset.itos[i] for i in tokens[0].cpu().numpy()]
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.yticks(range(len(chars)), chars)

    plt.tight_layout()
    plt.savefig('results/attention_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

    return attn


def evaluate_model_complexity(model, dataset):
    """评估模型复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算推理速度
    import time
    test_input = torch.randint(0, dataset.vocab_size, (1, 100)).to(next(model.parameters()).device)

    # Warmup
    for _ in range(10):
        _ = model(test_input)

    # 测速
    start_time = time.time()
    for _ in range(100):
        _ = model(test_input)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100

    complexity_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'average_inference_time_ms': avg_inference_time * 1000,
        'vocab_size': dataset.vocab_size,
        'sequence_length': test_input.size(1)
    }

    print("Model Complexity Analysis:")
    for key, value in complexity_info.items():
        print(f"{key:25}: {value:>15,}")

    return complexity_info