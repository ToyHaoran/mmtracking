# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict

import mmengine
from tqdm import tqdm

"""
30个类别如下：
n02691156 1 airplane
n02419796 2 antelope
n02131653 3 bear
n02834778 4 bicycle
n01503061 5 bird
n02924116 6 bus
n02958343 7 car
n02402425 8 cattle
n02084071 9 dog
n02121808 10 domestic_cat
n02503517 11 elephant
n02118333 12 fox
n02510455 13 giant_panda
n02342885 14 hamster
n02374451 15 horse
n02129165 16 lion 
n01674464 17 lizard 
n02484322 18 monkey
n03790512 19 motorcycle
n02324045 20 rabbit
n02509815 21 red_panda
n02411705 22 sheep
n01726692 23 snake
n02355227 24 squirrel
n02129604 25 tiger
n04468005 26 train
n01662784 27 turtle
n04530566 28 watercraft
n02062744 29 whale
n02391049 30 zebra
"""

CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
           'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
           'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
           'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
           'train', 'turtle', 'watercraft', 'whale', 'zebra')
# 这个是上面类对应的id
CLASSES_ENCODES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
                   'n01503061', 'n02924116', 'n02958343', 'n02402425',
                   'n02084071', 'n02121808', 'n02503517', 'n02118333',
                   'n02510455', 'n02342885', 'n02374451', 'n02129165',
                   'n01674464', 'n02484322', 'n03790512', 'n02324045',
                   'n02509815', 'n02411705', 'n01726692', 'n02355227',
                   'n02129604', 'n04468005', 'n01662784', 'n04530566',
                   'n02062744', 'n02391049')

# 将难记的id转为从1到30
cats_id_maps = {}
for k, v in enumerate(CLASSES_ENCODES, 1):
    cats_id_maps[v] = k

def parse_args():
    parser = argparse.ArgumentParser(
        description='ImageNet VID to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='ImageNet VID数据集的根目录，该目录下有Data, Annotations, ImageSets三个文件夹',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def parse_train_list(ann_dir):
    """Parse the txt file of ImageNet VID train dataset."""
    img_list = osp.join(ann_dir, 'Lists/VID_train_15frames.txt')
    """该文件的含义就是每个视频只取其中的15帧，因为有的视频有600多帧，计算不过来
    每一行 train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001002 1 12 55
    表示帧的名称，无意义，帧序号，视频长度(帧总数)。
    """
    img_list = mmengine.list_from_file(img_list)
    train_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')  # ['train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000', '1', '10', '300']
        if info[0] not in train_infos:
            train_infos[info[0]] = dict(vid_train_frames=[int(info[2]) - 1], num_frames=int(info[-1]))
        else:
            train_infos[info[0]]['vid_train_frames'].append(int(info[2]) - 1)
        # train_infos[info[0]]的内容：{'vid_train_frames': [9, 29, 49, ...], 'num_frames': 300}
    return train_infos


def parse_val_list(ann_dir):
    """Parse the txt file of ImageNet VID val dataset."""
    img_list = osp.join(ann_dir, 'Lists/VID_val_videos.txt')
    """其中一行的含义：val/ILSVRC2015_val_00000000 1 0 464 感觉就是一个序号"""
    img_list = mmengine.list_from_file(img_list)
    val_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')  # info:['val/ILSVRC2015_val_00006000', '4855', '0', '60']
        val_infos[info[0]] = dict(num_frames=int(info[-1]))
        # val_infos[info[0]]内容：{'num_frames': 464}
    return val_infos


def convert_vid(VID, ann_dir, save_dir, mode='train'):
    """Convert ImageNet VID dataset in COCO style.

    Args:
        VID (dict): The converted COCO style annotations.
        ann_dir (str): The path of ImageNet VID dataset.
        save_dir (str): The path to save `VID`.
        mode (str): Convert train dataset or validation dataset. Options are
            'train', 'val'. Default: 'train'.
    """
    assert mode in ['train', 'val']
    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_train_frames=0,
        num_no_objects=0)
    obj_num_classes = dict()
    xml_dir = osp.join(ann_dir, 'Annotations/VID/')
    if mode == 'train':
        vid_infos = parse_train_list(ann_dir)
        # vid_infos["xxx_train_000001"]的内容：{'vid_train_frames': [9, 29, 49, ...], 'num_frames': 300}
    else:
        vid_infos = parse_val_list(ann_dir)
    # 处理所有视频，遍历每个视频
    for vid_info in tqdm(vid_infos):  # tqdm是个进度条，提示信息
        # vid_info:'train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000'
        instance_id_maps = dict()
        # 每个视频的训练帧长度，这里选了15帧
        vid_train_frames = vid_infos[vid_info].get('vid_train_frames', [])
        records['num_vid_train_frames'] += len(vid_train_frames)  # 统计所有的训练帧数
        # video: {'id': 1, 'name': 'train/xxx/xx_00000000', 'vid_train_frames': [9, ..., 229, 249, 269, 289]}
        video = dict(
            id=records['vid_id'],
            name=vid_info,
            vid_train_frames=vid_train_frames)
        # 在最终的json中加入该视频属性
        VID['videos'].append(video)
        num_frames = vid_infos[vid_info]['num_frames']  # 每个视频的总帧数
        # 处理整个视频，遍历每一帧
        for frame_id in range(num_frames):
            is_vid_train_frame = True if frame_id in vid_train_frames else False
            # 图片路径的前缀
            img_prefix = osp.join(vid_info, '%06d' % frame_id)
            # 对应标注信息xml的路径
            xml_name = osp.join(xml_dir, f'{img_prefix}.xml')
            # 读取并解析xml标注文件
            tree = ET.parse(xml_name)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image = dict(
                file_name=f'{img_prefix}.JPEG',
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=is_vid_train_frame)
            # 将信息保存到最后的json中
            VID['images'].append(image)
            # 处理xml中的目标信息
            if root.findall('object') == []:
                print(xml_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in cats_id_maps:
                    continue
                category_id = cats_id_maps[name]
                bnd_box = obj.find('bndbox')
                x1, y1, x2, y2 = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                w = x2 - x1
                h = y2 - y1
                track_id = obj.find('trackid').text
                if track_id in instance_id_maps:
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id
                occluded = obj.find('occluded').text
                generated = obj.find('generated').text
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=[x1, y1, w, h],  # xy为左上角坐标，wh为宽高。
                    # boxmask会用到，用来计算mask。每两个是一个坐标。从左上角逆时针回到左上角，是一个回路。
                    segmentation=[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1],
                    area=w * h,
                    iscrowd=False,
                    occluded=occluded == '1',
                    generated=generated == '1')
                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    # 在win下对大小写不敏感，但在Linux下对大小写敏感
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmengine.dump(VID, osp.join(save_dir, f'imagenet_vid_{mode}.json'))
    print(f'-----ImageNet VID {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["num_vid_train_frames"]} train frames for video detection')
    print(f'{records["num_no_objects"]} images have no objects')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------')
    # 打印每个类有多少个目标(没有的默认为0)
    for i in range(1, len(CLASSES) + 1):
        # print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')
        print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes.get(i, 0)} objects.')
    """  以下为VID完整数据集的信息。
    -----ImageNet VID train------                             -----ImageNet VID val------
    3862 videos                                               555 videos
    1122397 images                                            176126 images
    57834 train frames for video detection                    0 train frames for video detection
    36265 images have no objects                              4046 images have no objects
    1731913 objects                                           273505 objects
    -----------------------                                   -----------------------
    Class 1 airplane has 86067 objects.                       Class 1 airplane has 26387 objects.
    Class 2 antelope has 59402 objects.                       Class 2 antelope has 7968 objects.
    Class 3 bear has 51903 objects.                           Class 3 bear has 9780 objects.
    Class 4 bicycle has 34897 objects.                        Class 4 bicycle has 8854 objects.
    Class 5 bird has 128943 objects.                          Class 5 bird has 9862 objects.
    Class 6 bus has 30186 objects.                            Class 6 bus has 6313 objects.
    Class 7 car has 114571 objects.                           Class 7 car has 26690 objects.
    Class 8 cattle has 53043 objects.                         Class 8 cattle has 11365 objects.
    Class 9 dog has 129650 objects.                           Class 9 dog has 20237 objects.
    Class 10 domestic_cat has 58899 objects.                  Class 10 domestic_cat has 11629 objects.
    Class 11 elephant has 84078 objects.                      Class 11 elephant has 11934 objects.
    Class 12 fox has 37186 objects.                           Class 12 fox has 9092 objects.
    Class 13 giant_panda has 52984 objects.                   Class 13 giant_panda has 7593 objects.
    Class 14 hamster has 38586 objects.                       Class 14 hamster has 5476 objects.
    Class 15 horse has 54314 objects.                         Class 15 horse has 7939 objects.
    Class 16 lion has 32196 objects.                          Class 16 lion has 2840 objects.
    Class 17 lizard has 31817 objects.                        Class 17 lizard has 5972 objects.
    Class 18 monkey has 70809 objects.                        Class 18 monkey has 10654 objects.
    Class 19 motorcycle has 34456 objects.                    Class 19 motorcycle has 1139 objects.
    Class 20 rabbit has 39177 objects.                        Class 20 rabbit has 5898 objects.
    Class 21 red_panda has 47935 objects.                     Class 21 red_panda has 1149 objects.
    Class 22 sheep has 35905 objects.                         Class 22 sheep has 4405 objects.
    Class 23 snake has 32128 objects.                         Class 23 snake has 6393 objects.
    Class 24 squirrel has 46891 objects.                      Class 24 squirrel has 12088 objects.
    Class 25 tiger has 21062 objects.                         Class 25 tiger has 2614 objects.
    Class 26 train has 105390 objects.                        Class 26 train has 13095 objects.
    Class 27 turtle has 44573 objects.                        Class 27 turtle has 5761 objects.
    Class 28 watercraft has 58774 objects.                    Class 28 watercraft has 6089 objects.
    Class 29 whale has 39150 objects.                         Class 29 whale has 7379 objects.
    Class 30 zebra has 76941 objects.                         Class 30 zebra has 6910 objects.
    """

def main():
    args = parse_args()

    # 构造json中的categories属性，如{'id': 1, 'name': 'airplane', 'encode_name': 'n02691156'}
    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v, encode_name=CLASSES_ENCODES[k - 1]))

    VID_train = defaultdict(list)
    VID_train['categories'] = categories
    convert_vid(VID_train, args.input, args.output, 'train')

    VID_val = defaultdict(list)
    VID_val['categories'] = categories
    convert_vid(VID_val, args.input, args.output, 'val')


# python ./tools/dataset_converters/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations
# python ./tools/dataset_converters/ilsvrc/imagenet2coco_vid.py -i ./data/VID_small -o ./data/VID_small/annotations
if __name__ == '__main__':
    main()
