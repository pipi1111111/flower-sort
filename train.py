import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from config.config import Config
from data.dataset import FlowerDataset, get_data_transforms
from models.convnext import FlowerConvNeXt
from utils.logger import TrainLogger
from utils.metrics import AverageMeter, accuracy
import timm.optim.optim_factory as optim_factory
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Union, Optional
from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_training_progress(train_accs, val_accs, train_losses, val_losses, output_dir):
    """绘制训练过程中的精度和损失变化图"""
    epochs = range(1, len(train_accs) + 1)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制准确率
    ax1.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax1.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax2.set_title('Model Loss During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # 保存图表
    plt.savefig(output_dir / 'training_progress.png')
    plt.close()

def main():
    config = Config()
    # 创建输出目录
    output_dir = Path('outputs') / datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志记录器
    logger = TrainLogger(output_dir)
    logger.log_hyperparameters(config)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 数据加载
    train_transform, val_transform = get_data_transforms(config)
    
    # 使用Path处理路径
    data_dir = Path(config.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}")
    
    train_dataset = FlowerDataset(
        train_dir,
        transform=train_transform,
        train=True
    )
    
    val_dataset = FlowerDataset(
        val_dir,
        transform=val_transform,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = FlowerConvNeXt(config).to(device)
    logger.log_model_info(model)
    
    # 记录最佳精度
    best_acc = 0.0
    
    # 记录训练过程中的指标
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 使用带预热的余弦退火学习率调度
    warmup_epochs = config.warmup_epochs
    total_epochs = config.epochs
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # 混合精度训练
    scaler = GradScaler() if config.mixed_precision else None
    
    # 训练循环
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, epoch, scaler, config, logger
        )
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, config, logger)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        # 更新最佳精度和保存模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # 保存检查点
        if is_best or (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            logger.save_model(model, epoch, val_acc, is_best)
        
        # 记录epoch总结
        logger.log_epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc, best_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 检查早停条件
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # 绘制训练过程中的指标变化图
    plot_training_progress(train_accuracies, val_accuracies, train_losses, val_losses, output_dir)
    logger.info(f"训练指标变化图已保存到: {output_dir / 'training_progress.png'}")
    
    # 关闭日志记录器
    logger.close()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler, config, logger):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    for i, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        
        # 混合精度训练
        with autocast(enabled=config.mixed_precision):
            output = model(images)
            loss = criterion(output, target)
            # 如果使用梯度累积，需要除以累积步数
            loss = loss / config.gradient_accumulation_steps
        
        # 计算准确率
        acc1 = accuracy(output, target)[0]
        losses.update(loss.item() * config.gradient_accumulation_steps, images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            if (i + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad()
        
        # 记录训练步骤
        if (i + 1) % config.gradient_accumulation_steps == 0:
            logger.log_training_step(
                epoch, i // config.gradient_accumulation_steps, len(loader) // config.gradient_accumulation_steps,
                losses.avg, top1.avg,
                optimizer.param_groups[0]['lr']
            )
    
    # 处理最后一个不完整的累积步骤
    if (i + 1) % config.gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
        optimizer.zero_grad()
    
    return losses.avg, top1.avg

def validate(model, loader, criterion, device, epoch, config, logger):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    # 记录验证结果
    logger.log_validation_results(epoch, losses.avg, top1.avg)
    
    return losses.avg, top1.avg

if __name__ == '__main__':
    main() 