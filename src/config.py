import yaml
import torch


def get_default_config():
    """返回默认配置"""
    return {
        'model': {
            'vocab_size': 65,  # Tiny Shakespeare字符数
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 512,
            'max_seq_len': 256,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'block_size': 256,
            'num_epochs': 50,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'save_interval': 10
        },
        'data': {
            'dataset': 'tiny_shakespeare',
            'split_ratio': 0.9
        },
        'experiment': {
            'name': 'transformer_baseline',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42
        }
    }


def save_config(config, filepath):
    """保存配置到YAML文件"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath):
    """从YAML文件加载配置"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config):
    """设置实验环境"""
    import random
    import numpy as np

    # 设置随机种子
    seed = config['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 创建结果目录
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    print(f"Experiment setup completed. Using device: {config['experiment']['device']}")
