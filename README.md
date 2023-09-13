# 视频目标检测

MMTracking是一款基于PyTorch的视频目标感知开源工具箱，是[OpenMMLab](http://openmmlab.org/) 项目的一部分。

本项目基于mmtracking，删除了与VID无关的部分，并修改了大量的代码。

支持的算法:

- [X]  [DFF](configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (CVPR 2017)
- [X]  [FGFA](configs/vid/fgfa/fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (ICCV 2017)
- [X]  [SELSA](configs/vid/selsa/selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (ICCV 2019)
- [X]  [Temporal RoI Align](configs/vid/temporal_roi_align/selsa-troialign_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py) (AAAI 2021)
- [X]  [BoxMask](configs/vid/selsa/selsa_boxmask_r50.py) (WACV 2023)
- [ ]  [PTSEFormer](configs/vid/PTSEFormer) (ECCV 2022) 官方代码无法运行，做了些修改。
- [ ]  [TransVOD++] (TPAMI 2022)

支持的数据集：

- [X]  [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)

# 最新版环境安装

删除了与VID无关的代码，无需编译mmcv，并且即将mmengine升级到了0.8.4，mmdetection升级到了3.1.0版本。

```shell
# (创建环境)
conda create -n mmengine python=3.8
conda activate mmengine
# (更新pip) (使用清华源)
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# (安装pytorch，确保nvcc -V显示11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装mmlab环境
pip install -U openmim
mim install mmcv==2.0.0
mim install mmpretrain==1.0.2
mim install mmengine==0.8.4
# mim install mmcls==1.0.0rc6
# (先安装依赖，然后自己编译安装)
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
# (先安装依赖，然后自己编译安装)
cd mmtracking
pip install -r requirements.txt
pip install -v -e .
```

其他问题，遇不到更好：
(BUG：linux无法识别反斜杠\\问题：FileNotFoundError: [Errno 2] No such file or directory:xxx\\000000.JPEG。
原因：windows的目录中可以用撇“/”，双捺“\\”来隔断文件夹。但Linux固定使用撇“/”。
解决：把json中的\\全部替换为/

(BUG：error: [Errno 2] No such file or directory: '/usr/local/cuda/bin/bin/nvcc'
多了一个bin，修改~/.bashrc中最后面的CUDAHOME，把最终的bin删掉。)

(BUG：KeyError: 'classes'。解决：将mmdet/evaluation/metrics/coco_metric.py中的classes改为大写)

(测试) python ./tools/train.py configs/vid/selsa/selsa_r50_demo.py

# 数据集
## 数据集格式(yolo coco voc)

[参考](https://zhuanlan.zhihu.com/p/29393415)

坐标区别：
voc(X0,Y0,X1,Y1)，其中X0,Y0是左上角的坐标，X1,Y1是右下角的坐标。
YOLO(X,Y,W,H)，其中X,Y是中心点的坐标(比值)。
coco(X,Y,W,H)，其中X,Y是左上角的坐标。

格式转换见
[voc2coco.py](tools/dataset_converters/voc2coco.py)
[yolo2coco.py](tools/dataset_converters/yolo2coco.py)

## 修改数据集路径
使用软链接(必须是绝对路径)(推荐)：
ln -s /root/data/coco /root/pro/mmdetection/data/coco 

## ImageNetVID格式转为coco

原数据集格式：
├── data
│   ├── ILSVRC
│   │   ├── Data
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
|   │   │   │   ├── test
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
|   │   │   │   ├── test
│   │   ├── Annotations
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── Lists

转换数据集格式：
ImageNet DET和VID  (需要[Lists文件](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets))
转换VID数据格式：
nohup python ./tools/dataset_converters/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations >> log.txt 2>&1 &
转换DET数据格式：
nohup python ./tools/dataset_converters/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations >> log.txt 2>&1 &

运行完结构如下图所示。
│   ├── ILSVRC
│   │   ├── Data
│   │   ├── Annotations (the official annotation files)
│   │   ├── Lists
│   │   ├── annotations (转换后的注解)

annotation内容如下：
1 imagenet_det_30plus1cls.json：包含ImageNet DET数据集中训练集标注信息的JSON文件。
30表示VID数据集中重叠的30个类别，1cls将DET 中的其他170个类别作为一个类别other_categeries。
2 imagenet_vid_train.json：VID中训练集标注信息。
3 imagenet_vid_val.json：VID中验证集标注信息。

# 训练和评估

单机单卡：
`[CUDA_VISIBLE_DEVICES=2] python tools/train.py或test.py ${CONFIG_FILE} [optional arguments]`
如`python tools/train.py configs/vid/xxx.py`

单机多卡：假设当前机器有7张显卡
`bash ./tools/dist_train.sh或dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]`

指定显卡的编号，例如使用第0123卡，可以设置环境变量：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE_1} 4
CUDA_VISIBLE_DEVICES=4,5,6 PORT=29550 ./tools/dist_train.sh ${CONFIG_FILE_2} 3
```

# 可视化loss曲线

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/log/xxx.json --keys loss loss_cls loss_bbox --out losses.pdf
# 必须python运行，不能在pycharm中运行，否则会出错：manager_pyplot_show = vars(manager_class).get("pyplot_show") TypeError: vars() argument must have __dict__ attribute
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/log/20230503_112734.json --keys loss
```
