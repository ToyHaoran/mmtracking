"""
将VOC格式的数据集(XML等)转为coco格式，用来检测，如yolo
VOC数据集的格式：
data/VOC
    -Annotations
        -a.xml
        -b.xml
        -...
    -ImageSet
        -Main
            -val.txt  # 所有的图片名称
    -JPEGimages  # 对应的图片
        -a.png
        -b.jpg
        -...
    -labels.txt  # 就是80个类的名称，即使你只有5个类也要用80个，避免id溢出错误。注意类别名称与数据集xml中一致。以空格划分，如果类名中有空格需要修改代码。

运行voc2coco.py进行数据格式转换，注意修改注解和数据路径。

数据集转换后的COCO格式为：
data/coco
    -annotations
        -val.json
    -val  # 数据集，对应上面的JPEGimages
        -a.png
        -b.jpg
        -...

在命令行运行 python tools/analysis_tools/browse_coco_json.py，注意修改注解路径。
大致查看目标框是否有问题。

修改以下文件避免id溢出问题：mmdet/evaluation/metrics/coco_metric.py:412
将cat_ids改为从数据集(你自定义的labels.txt)获取id，而不是使用代码中的classes。
        # handle lazy init
        if self.cat_ids is None:
            # 这里的classes大小写会导致错误，有的使用小写，有的使用大写，这里忽略大小写。
            classes = 'classes' if 'classes' in self.dataset_meta.keys() else 'CLASSES'
            # self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta[classes])
            self.cat_ids = self._coco_api.get_cat_ids()  # 从数据集获取id


显示每个类的AP值：在数据集配置文件中添加classwise=True, 如下示例：
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

修改自定义的模型，然后运行即可。
"""

import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
import re

img_id = 0

def next_id():
    global img_id
    img_id += 1
    return img_id

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().strip().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    
    # 这里不能从文件中提取ID，因为有重复的，在外部自定义一个全局变量，每次+1
    if extract_num_from_imgid:  # 用于顺序排列，无重复的图片
        img_name = os.path.basename(filename)
        img_id = os.path.splitext(img_name)[0]
        if extract_num_from_imgid and isinstance(img_id, str):
            img_id = int(re.findall(r'\d+', img_id)[0])
    else:
        img_id = next_id()  # 在外部自定义一个全局变量，每次+1

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin'))
    ymin = int(bndbox.findtext('ymin'))
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?

    for a_path in annotation_paths:
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
    print('Convert successfully !')


def main():
    parser = argparse.ArgumentParser(description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default='./data/RTTS/Annotations')
    parser.add_argument('--ann_ids', type=str, default='./data/RTTS/ImageSets/Main/val.txt')  # train val test
    parser.add_argument('--ann_paths_list', type=str, default=None)
    parser.add_argument('--labels', type=str, default='./data/RTTS/labels.txt')
    parser.add_argument('--output', type=str, default='./data/coco/annotations/val.json')  # 这里修改 train val test 一共修改三次
    parser.add_argument('--ext', type=str, default='xml')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=False  # 不要从文件名称中取ID，有重复的数字
    )


if __name__ == '__main__':
    if not os.path.exists('./data/coco/annotations'):
        os.makedirs('./data/coco/annotations')
    main()
