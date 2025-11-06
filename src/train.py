import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        self.criterion = nn.CrossEntropyLoss()

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 创建检查点目录
        os.makedirs('checkpoints', exist_ok=True)

    def validate_batch(self, x, y):
        """验证批次数据"""
        if x is None or y is None:
            raise ValueError("训练数据包含None")

        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise ValueError("训练数据不是张量")

        if x.numel() == 0 or y.numel() == 0:
            raise ValueError("训练数据为空张量")

        return True

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            try:
                # 验证输入数据
                self.validate_batch(x, y)

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                logits = self.model(x)

                if logits is None:
                    print(f"批次 {batch_idx}: 模型输出为None，跳过")
                    continue

                # 计算损失
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"批次 {batch_idx}: 损失为NaN或inf，跳过")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )

                self.optimizer.step()

                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{math.exp(loss.item()):.2f}'
                })

            except Exception as e:
                print(f"批次 {batch_idx} 训练错误: {e}")
                continue

        if num_batches == 0:
            print("警告: 没有成功训练的批次")
            return float('inf')

        avg_loss = total_loss / total_tokens
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    self.validate_batch(x, y)
                    x, y = x.to(self.device), y.to(self.device)

                    logits = self.model(x)

                    if logits is None:
                        print(f"验证批次 {batch_idx}: 模型输出为None，跳过")
                        continue

                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item() * x.numel()
                        total_tokens += x.numel()
                        num_batches += 1

                except Exception as e:
                    print(f"验证批次 {batch_idx} 错误: {e}")
                    continue

        if num_batches == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return avg_loss

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            epoch_time = time.time() - start_time

            # 打印epoch结果
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_ppl={math.exp(train_loss):.2f}, val_ppl={math.exp(val_loss):.2f}, "
                  f"lr={current_lr:.2e}, time={epoch_time:.2f}s")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)

            # 绘制训练曲线
            self.plot_training_curves()

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        if is_best:
            filename = f"checkpoints/best_model.pth"
        else:
            filename = f"checkpoints/checkpoint_epoch_{epoch + 1}.pth"

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # 困惑度曲线
        plt.subplot(1, 2, 2)
        train_ppl = [math.exp(loss) for loss in self.train_losses]
        val_ppl = [math.exp(loss) for loss in self.val_losses]
        plt.plot(train_ppl, label='Train PPL')
        plt.plot(val_ppl, label='Val PPL')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.title('Training and Validation Perplexity')

        plt.tight_layout()
        plt.savefig('results/encoder_decoder_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """打印模型信息"""
    print("Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")

    # 打印各层参数分布
    print("\nParameter distribution by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} | {param.numel():8,} | {param.shape}")