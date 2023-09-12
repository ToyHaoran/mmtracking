#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
module load anaconda3-2022.10  # 加载conda
module load cuda-11.3  # 加载cuda
module load gcc-11 
source activate mmtracking  # 激活环境

cd /mnt/nfs/data/home/1120220334/pro/mmtracking  # 打开项目
python ./tools/test.py ./configs/vid/selsa/selsa_r50_dc5_sampler.py --checkpoint /mnt/nfs/data/home/1120220334/pro/mmtracking/work_dirs/selsa_r50_dc5_sampler/epoch_5.pth
