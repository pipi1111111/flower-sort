# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from tqdm import tqdm
import argparse
import os
import logging
from pathlib import Path
from torchvision import transforms, datasets
from PIL import Image
from models.convnext import FlowerConvNeXt
from config.config import Config
from utils.seed import set_seed
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_config():
    """获取配置"""
    config = Config()
    # 不再修改预训练模型路径，使用配置文件中的默认值
    return config

def get_val_loader(config):
    """获取验证数据加载器"""
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(config.data_dir, 'val'),
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    return val_loader

def create_model(config):
    """创建模型"""
    model = FlowerConvNeXt(config)
    return model

class FlowerPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = Config()
        
        # 加载类别名称
        self.class_names = self._load_class_names()
        
        # 创建模型
        print(f"正在创建模型...")
        self.model = FlowerConvNeXt(self.config).to(self.device)
        
        # 加载训练好的权重
        print(f"正在加载权重: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 移除module.前缀（如果存在）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} keys")
            
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
        
        self.model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_class_names(self):
        # 加载花卉类别名称（需要创建这个JSON文件）
        class_names_path = Path('scripts/datasets/flower_names.json')
        if class_names_path.exists():
            with open(class_names_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {str(i): f"Flower_{i}" for i in range(102)}  # 默认名称
    
    def predict_image(self, image_path, top_k=5):
        """预测单张图片"""
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取top-k预测结果
            top_prob, top_class = torch.topk(probabilities, k=top_k)
            
            results = []
            for i in range(top_k):
                class_idx = top_class[0][i].item()
                results.append({
                    'class_name': self.class_names[str(class_idx)],
                    'probability': top_prob[0][i].item() * 100
                })
            
            return results
    
    def visualize_prediction(self, image_path, results):
        """可视化预测结果"""
        # 显示原图
        image = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # 显示预测结果条形图
        plt.subplot(1, 2, 2)
        names = [r['class_name'] for r in results]
        probs = [r['probability'] for r in results]
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, names)
        plt.xlabel('Probability (%)')
        plt.title('Top-5 Predictions')
        
        plt.tight_layout()
        # 保存图片而不是显示
        save_path = Path(image_path).parent / f"{Path(image_path).stem}_prediction.png"
        plt.savefig(save_path)
        plt.close()
        print(f"预测结果已保存到: {save_path}")

def evaluate_model(predictor, data_dir, split='val'):
    """评估模型性能"""
    eval_dir = Path(data_dir) / split
    
    # 添加预测结果保存
    predictions = []
    total_samples = 0
    correct_predictions = 0
    
    # 添加混淆矩阵
    confusion_matrix = np.zeros((102, 102), dtype=int)
    
    for class_dir in tqdm(list(eval_dir.iterdir())):
        if class_dir.is_dir():
            true_class = int(class_dir.name)
            for img_path in class_dir.glob('*.jpg'):
                total_samples += 1
                results = predictor.predict_image(img_path)
                pred_class = int(results[0]['class_name'].split('_')[1])
                confidence = results[0]['probability']
                is_correct = (pred_class == true_class)
                
                # 更新混淆矩阵
                confusion_matrix[true_class-1, pred_class-1] += 1
                
                if is_correct:
                    correct_predictions += 1
                predictions.append({
                    'true': true_class,
                    'pred': pred_class,
                    'confidence': confidence,
                    'correct': is_correct,
                    'image_path': str(img_path)
                })
    
    # 计算准确率
    accuracy = (correct_predictions / total_samples) * 100
    
    # 分析错误案例
    errors = [p for p in predictions if not p['correct']]
    print("\n=== 评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"错误预测数: {len(errors)}")
    print(f"准确率: {accuracy:.2f}%")
    
    print("\n=== 错误分析 ===")
    print(f"错误数量: {len(errors)}")
    
    # 置信度分析
    confidences = [p['confidence'] for p in predictions]
    error_confidences = [p['confidence'] for p in errors]
    print("\n置信度分布:")
    print(f"平均置信度: {np.mean(confidences):.2f}%")
    print(f"最高置信度: {np.max(confidences):.2f}%")
    print(f"最低置信度: {np.min(confidences):.2f}%")
    
    print("\n错误案例的置信度分布:")
    print(f"平均置信度: {np.mean(error_confidences):.2f}%")
    print(f"最高置信度: {np.max(error_confidences):.2f}%")
    print(f"最低置信度: {np.min(error_confidences):.2f}%")
    
    # 按类别统计错误
    class_errors = {}
    for p in errors:
        true_class = p['true']
        if true_class not in class_errors:
            class_errors[true_class] = 0
        class_errors[true_class] += 1
    
    print("\n各类别错误数量:")
    for class_id, error_count in sorted(class_errors.items()):
        print(f"类别 {class_id}: {error_count} 个错误")
    
    # 分析最常混淆的类别对
    print("\n最常混淆的类别对:")
    confusion_pairs = []
    for i in range(102):
        for j in range(102):
            if i != j and confusion_matrix[i, j] > 0:
                confusion_pairs.append((i+1, j+1, confusion_matrix[i, j]))
    
    # 按混淆次数排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_class, pred_class, count in confusion_pairs[:10]:
        print(f"类别 {true_class} 被错误分类为类别 {pred_class}: {count} 次")
    
    # 保存错误案例
    error_cases = [p for p in predictions if not p['correct']]
    error_cases.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("\n=== 高置信度错误案例 ===")
    for case in error_cases[:5]:
        print(f"真实类别: {case['true']}, 预测类别: {case['pred']}, 置信度: {case['confidence']:.2f}%, 图片: {case['image_path']}")
    
    return accuracy

def validate_model(model_path):
    """验证模型文件"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("\n=== 模型详细信息 ===")
        print(f"1. 模型文件大小: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"2. Checkpoint包含的键: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"3. 模型参数数量: {len(state_dict)}")
            print("\n4. 关键层权重形状:")
            for k, v in state_dict.items():
                if 'classifier' in k or k.endswith('weight'):
                    print(f"   {k}: {v.shape}")
        
        if 'config' in checkpoint:
            print("\n5. 训练配置:", checkpoint['config'])
            
        return True
    except Exception as e:
        print(f"模型验证失败: {str(e)}")
        return False

def analyze_predictions(predictor, image_path, true_label):
    """分析单个预测结果"""
    # 获取预测结果
    results = predictor.predict_image(image_path)
    pred_class = int(results[0]['class_name'].split('_')[1])
    
    # 获取特征图
    image = Image.open(image_path).convert('RGB')
    image_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)
    
    with torch.no_grad():
        features = predictor.model.backbone(image_tensor)
        print(f"\n特征统计:")
        print(f"特征范围: [{features.min().item():.3f}, {features.max().item():.3f}]")
        print(f"特征均值: {features.mean().item():.3f}")
        print(f"特征标准差: {features.std().item():.3f}")
    
    return pred_class == true_label

def get_best_model_path():
    """获取最佳模型路径"""
    # 使用指定的最佳模型
    model_path = os.path.join("outputs", "2025.04.08_09:36:51", "model_best.pth")
    if os.path.exists(model_path):
        logger.info(f"Using best model: {model_path}")
        return model_path
    
    logger.error(f"Best model not found at {model_path}")
    return None

def evaluate_model(model, val_loader, device, criterion):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            # 更新统计信息
            total_samples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # 计算准确率
    accuracy = correct_predictions / total_samples
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算每个类别的准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # 输出评估结果
    logger.info(f"\nValidation Results:")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Correct Predictions: {correct_predictions}")
    logger.info(f"Error Predictions: {total_samples - correct_predictions}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nPer-class Accuracy:\n{class_acc}")
    
    return accuracy, avg_loss, cm, class_acc

def main():
    # 设置随机种子
    set_seed(42)
    
    # 获取最佳模型路径
    model_path = get_best_model_path()
    if model_path is None:
        logger.error("No valid model path found")
        return
        
    # 加载配置
    config = get_config()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据
    val_loader = get_val_loader(config)
    
    # 创建模型
    model = create_model(config)
    
    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
                
            # 如果检查点中包含配置信息，使用它
            if "config" in checkpoint:
                model.config = checkpoint["config"]
        else:
            state_dict = checkpoint
            
        # 处理权重键名
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除可能的前缀
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('backbone.'):
                k = k[9:]
            new_state_dict[k] = v
            
        # 非严格加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
            
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    model = model.to(device)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    accuracy, avg_loss, cm, class_acc = evaluate_model(model, val_loader, device, criterion)
    
    # 保存评估结果
    save_dir = "evaluation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    
    # 保存每个类别的准确率
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_acc)), class_acc)
    plt.title("Per-class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, "per_class_accuracy.png"))
    plt.close()
    
    # 保存评估指标
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": class_acc.tolist()
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation results saved to {save_dir}")

if __name__ == "__main__":
    main() 