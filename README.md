
# 视频目标检测
MMTracking是一款基于PyTorch的视频目标感知开源工具箱，是[OpenMMLab](http://openmmlab.org/) 项目的一部分。

本项目基于mmtracking，删除了与VID无关的部分，并修改了大量的代码。

支持的算法:

- [x] [DFF](configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa/fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa/selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align/selsa-troialign_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (AAAI 2021)
- [x] [BoxMask](configs/vid/selsa/selsa_boxmask_r50.py) (WACV 2023) 
- [x] [PTSEFormer](configs/vid/PTSEFormer) (ECCV 2022) 官方代码无法运行，做了些修改。
- [ ] [TransVOD++] (TPAMI 2022)

支持的数据集：

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)

