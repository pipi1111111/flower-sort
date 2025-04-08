#!/bin/bash

# 设置错误时退出
set -e

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建新的conda环境
conda create -n flower_classify python=3.10 -y
conda activate flower_classify

# 安装PyTorch (使用conda避免依赖问题)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 安装其他依赖
pip install timm==0.9.2 pillow==10.0.0 numpy==1.24.3 tqdm==4.65.0 tensorboard==2.13.0
pip install opencv-python==4.8.0.74 requests==2.31.0 scipy==1.10.1

# 验证安装
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available())"

# 创建目录结构
mkdir -p outputs scripts

# 下载并准备数据集
echo "准备数据集..."
python scripts/download_dataset.py

# 下载预训练模型
echo "下载预训练模型..."
python scripts/download_model.py

# 运行训练脚本
echo "开始训练..."
python train.py