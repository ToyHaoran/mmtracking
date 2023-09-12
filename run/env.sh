#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
alias ll='ls -al'  # 快捷键
module load anaconda3-2022.10  # 加载conda
module load cuda-11.3  # 加载cuda
module load gcc-11 
# conda create -n mmtracking  python=3.8 # 创建conda环境 (需要更新清华源或阿里源，自行解决)
source activate mmtracking  # 激活环境

# 安装pytorch，确保nvcc -V显示11.3，可能会安装失败，可以在本地结点安装。
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装依赖环境(需要GPU编译的一定要使用sbatch命令执行)
#pip uninstall mmcv -y
#pip uninstall mmcls -y
#pip uninstall mmengine -y
#pip uninstall mmdetection -y
#pip uninstall mmtracking -y
#
#pip install -U openmim
#mim install mmcv==2.0.0
#mim install mmpretrain==1.0.2
#mim install mmengine==0.8.4


cd ~/pro/mmdetection
#pip install -r requirements.txt
pip install -v -e .

# 安装mmtracking
#cd ~/pro/mmtracking
#pip install -r requirements.txt
#pip install -v -e .

pip list
