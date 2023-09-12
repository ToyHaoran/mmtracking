#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
module load anaconda3-2022.10  # 加载conda
module load cuda-11.3  # 加载cuda
module load gcc-11 
source activate mmtracking  # 激活环境

cd /mnt/nfs/data/home/1120220334/pro/mmtracking  # 打开项目
#python ./tools/train.py ./configs/vid/selsa/selsa_fpn_r50.py
#python ./tools/train.py ./configs/vid/selsa/selsa_r50-dc5_8xb1-7e_imagenetvid.py
#python ./tools/train.py ./configs/vid/temporal_roi_align/selsa-troialign_faster-rcnn_x101-dc5_8xb1-7e_imagenetvid.py
#python ./tools/train.py ./configs/vid/selsa/selsa_r50_dc5_sampler.py
python ./tools/train.py ./configs/vid/selsa/selsa_swin.py