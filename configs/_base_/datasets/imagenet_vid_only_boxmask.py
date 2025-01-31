# dataset settings  仅使用vid数据集
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(type='mmdet.Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            # 使用旧的代码实现boxmask失败
        ]),
    dict(type='PackTrackInputs', meta_keys=('gt_masks', 'segmentation'))
]
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.Resize', scale=(1000, 600), keep_ratio=True),
        ]),
    dict(type='PackTrackInputs', pack_single_img=False)
]

# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/imagenet_vid_train.json',
        data_prefix=dict(img_path='Data/VID'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=2,
            frame_range=9,
            filter_key_img=True,
            method='bilateral_uniform')))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/imagenet_vid_val.json',
        data_prefix=dict(img_path='Data/VID'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'),
        test_mode=True))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='CocoVideoMetric',
    ann_file=data_root + 'annotations/imagenet_vid_val.json',
    metric='bbox')
test_evaluator = val_evaluator
