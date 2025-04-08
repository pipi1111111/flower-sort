#!/bin/bash

# 设置错误时退出
set -e

# 检查Python版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$python_version" != "3.10" ]]; then
    echo "错误: 需要Python 3.10.x版本，当前版本为 $python_version"
    exit 1
fi

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到CUDA环境..."
else
    echo "警告: 未检测到CUDA环境，将使用CPU训练"
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3.10 -m venv venv
fi

source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install --upgrade pip
pip install --upgrade setuptools wheel

# 清理已有的torch相关包
pip uninstall -y torch torchvision torchaudio

# 安装依赖
pip install -r requirements.txt

# 验证torch安装
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available())"

# 创建必要的目录
mkdir -p outputs scripts/datasets

# 检查数据集是否存在
if [ ! -d "scripts/datasets/oxford_flowers102" ]; then
    echo "下载并准备数据集..."
    python scripts/download_dataset.py
else
    echo "数据集已存在，跳过下载..."
fi

# 检查预训练模型是否存在
if [ ! -d "scripts/pretrained_models" ] || [ ! -f "scripts/pretrained_models/convnext_tiny_pretrained.pth" ]; then
    echo "下载预训练模型..."
    python scripts/download_model.py
else
    echo "预训练模型已存在，跳过下载..."
fi

# 开始训练
echo "开始训练..."
python train.py

echo "完成！" 