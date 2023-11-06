#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH --nodelist=aiwkr2

module load anaconda3-2022.10  # 加载conda
module load cuda-11.3  # 加载cuda
module load gcc-11 
source activate mmtracking  # 激活环境

cd /mnt/nfs/data/home/1120220334/pro/mmtracking  # 打开项目

# 多程序运行，后面加&，最后加wait
#python ./tools/test.py ./configs/vid/selsa/selsa_r50_demo.py --checkpoint /mnt/nfs/data/home/1120220334/pro/mmtracking/work_dirs/selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid/epoch_5.pth
python ./tools/test.py ./configs/vid/selsa/selsa_r50_my_aggregator.py --checkpoint /mnt/nfs/data/home/1120220334/pro/mmtracking/work_dirs/temp/epoch_5.pth

#wait
