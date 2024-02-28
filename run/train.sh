#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1

#不指定结点SBATCH --nodelist=aiwkr3

module load anaconda3-2022.10  # 加载conda
module load cuda-11.3  # 加载cuda
module load gcc-11 
source activate mmtracking  # 激活环境

# 超算平台有时会导致多个用户的程序运行在同一张显卡上，看运行时间就能看出来。
nvidia-smi  # 看看哪个显卡空闲，然后指定显卡号。CUDA_VISIBLE_DEVICES=7，但是建议不要配置，因为会被自动分配抢占GPU

# nohup xxx命令 >> log.txt 2>&1 &

cd /mnt/nfs/data/home/1120220334/pro/mmtracking  # 打开项目
python ./tools/train.py ./configs/vid/selsa/selsa_r50_demo.py
#python ./tools/train.py ./configs/vid/selsa/selsa_fpn_r50.py
#python ./tools/train.py ./configs/vid/selsa/selsa_r50-dc5_8xb1-7e_imagenetvid.py
#python ./tools/train.py ./configs/vid/selsa/selsa_r50_my_aggregator.py
#python ./tools/train.py ./configs/vid/selsa/selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py
#python ./tools/train.py ./configs/vid/selsa/selsa_r50_dc5_sampler.py
#python ./tools/train.py ./configs/vid/selsa/selsa_swin.py # --resume
#python ./tools/train.py ./configs/vid/temporal_roi_align/TROI_swinB.py

# nohup python ./tools/train.py ./configs/vid/selsa/selsa_r50_my_aggregator.py >> log1007.txt 2>&1 &