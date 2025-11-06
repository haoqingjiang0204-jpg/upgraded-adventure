#!/usr/bin/env python3
"""
Transformer从零实现 - 主运行脚本
"""

# !/usr/bin/env python3
"""
Transformer从零实现 - 主运行脚本
"""

import argparse
import torch
import argparse
import torch
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config, save_config, load_config, setup_experiment
from model import TransformerLM
from data_utils import get_data_loader
from train import Trainer, print_model_info
from generation import TextGenerator, evaluate_model_complexity
from experiments import run_complete_ablation_study

def main():
    parser = argparse.ArgumentParser(description='Transformer从零实现')
    parser.add_argument('--config', type=str, default='D:/AI/project/transformer/configs/base.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'ablation', 'generate', 'eval'],
                        help='运行模式: train, ablation, generate, eval')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                        help='生成文本的提示')
    parser.add_argument('--checkpoint', type=str,
                        help='用于生成或评估的检查点路径')

    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
        save_config(config, args.config)
        print(f"Created default config at {args.config}")

    # 设置实验环境
    setup_experiment(config)
    device = torch.device(config['experiment']['device'])

    if args.mode == 'train':
        train_model(config, device)
    elif args.mode == 'ablation':
        run_ablation_study(config)
    elif args.mode == 'generate':
        generate_text(config, device, args.prompt, args.checkpoint)
    elif args.mode == 'eval':
        evaluate_model(config, device, args.checkpoint)


def train_model(config, device):
    """训练模型"""
    print("=== Training Mode ===")

    # 获取数据
    train_loader, val_loader, vocab_size, dataset = get_data_loader(
        config['training']['batch_size'],
        config['training']['block_size'],
        config['data']['split_ratio']
    )

    # 更新词汇表大小
    config['model']['vocab_size'] = vocab_size

    # 创建模型
    model = TransformerLM(**config['model'])
    print_model_info(model)

    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, config['training'], device)
    trainer.train()

    print("Training completed!")


def run_ablation_study(config):
    """运行消融实验"""
    print("=== Ablation Study Mode ===")

    results = run_complete_ablation_study(config)

    print("Ablation study completed!")
    print("Results saved in results/ directory")


def generate_text(config, device, prompt, checkpoint_path):
    """生成文本"""
    print("=== Text Generation Mode ===")

    # 获取数据（用于词汇表）
    _, _, vocab_size, dataset = get_data_loader(
        config['training']['batch_size'],
        config['training']['block_size']
    )

    # 创建模型
    config['model']['vocab_size'] = vocab_size
    model = TransformerLM(**config['model']).to(device)

    # 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint provided or checkpoint not found. Using untrained model.")

    # 生成文本
    generator = TextGenerator(model, dataset, device)

    print(f"Prompt: {prompt}")
    print("Generated text:")

    generated = generator.generate_text(
        prompt,
        max_length=200,
        temperature=0.8,
        top_k=5
    )

    print(generated)

    # 计算困惑度
    ppl = generator.calculate_perplexity(generated[:100])  # 只计算前100个字符
    print(f"\nPerplexity of generated text: {ppl:.2f}")


def evaluate_model(config, device, checkpoint_path):
    """评估模型"""
    print("=== Model Evaluation Mode ===")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Please provide a valid checkpoint path for evaluation")
        return

    # 获取数据
    _, val_loader, vocab_size, dataset = get_data_loader(
        config['training']['batch_size'],
        config['training']['block_size']
    )

    # 创建模型并加载检查点
    config['model']['vocab_size'] = vocab_size
    model = TransformerLM(**config['model']).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 评估模型复杂度
    complexity_info = evaluate_model_complexity(model, dataset)

    # 计算验证损失
    from train import Trainer
    trainer = Trainer(model, None, val_loader, config['training'], device)
    val_loss = trainer.validate()

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")


if __name__ == "__main__":
    main()