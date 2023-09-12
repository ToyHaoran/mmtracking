_base_ = [
    '../../_base_/models/faster-rcnn_ConvNeXt_backbone.py',  # 更换了backbone
    # '../../_base_/datasets/imagenet_vid_demo.py',  # demo数据集测试
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',  # 混合数据集 official
    '../../_base_/default_runtime.py'
]

model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2)),
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)),
            )))

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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.00025, weight_decay=0.0001),
    optimizer=dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# 学习率：2.5e-4持续3个epoch，然后2.5e-5一个，2.5e-6一个，可以达到80.4%的map。
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=1.0 / 3,
    #     by_epoch=False,
    #     begin=0,
    #     end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=5,
        by_epoch=True,
        milestones=[3, 4],
        gamma=0.1)
]

visualizer = dict(type='DetLocalVisualizer')
