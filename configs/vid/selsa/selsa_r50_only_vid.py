_base_ = [
    # dc5提取特征时输出1个(3,512,38,50)的特征图
    # fpn提取特征时输出4个(3,512,x,x)的特征图。
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    # 这里仅使用VID数据集(完整)，而不使用DET数据集
    '../../_base_/datasets/imagenet_vid_only.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        # r50和r101的主要区别
        backbone=dict(
                    depth=101,
                    init_cfg=dict(
                        type='Pretrained', checkpoint='torchvision://resnet101')),
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)),
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]))))

# dataset settings
val_dataloader = dict(
    dataset=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')))
test_dataloader = val_dataloader

# training schedule
# 在TITAN上运行，一个epoch大约12H，显存占用8G。
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=7,
        by_epoch=True,
        milestones=[2, 5],
        gamma=0.1)
]

visualizer = dict(type='DetLocalVisualizer')