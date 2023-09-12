_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    # '../../_base_/datasets/imagenet_vid_fgfa_style.py',   # 全数据集
    '../../_base_/datasets/imagenet_vid_demo.py',   # demo数据集
    '../../_base_/default_runtime.py'
]

model = dict(
    type='SELSA',
    detector=dict(
        # 使用RetinaHead替换RPN网络
        # neck=dict(
        #     start_level=1,
        #     add_extra_convs='on_output',  # 使用 P5
        #     relu_before_extra_convs=True),
        rpn_head=dict(
            _delete_=True,  # 忽略未使用的旧设置
            type='RetinaRPNHead',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            stacked_convs=4,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),  # FPN为 [4, 8, 16, 32, 64]), 与featmap_strides对应  [8, 16, 32, 64, 128]
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

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
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,  # 这里从512改为256
                # featmap_strides的更新取决于于颈部的步伐 从[16]改为[8, 16, 32, 64, 128]
                featmap_strides=[4, 8, 16, 32]))))

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
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=5)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)  # 测试用
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    optimizer=dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.001/4,  # 原来1.0/3
    #     by_epoch=False,
    #     begin=0,
    #     end=500),  # 慢慢增加 lr，否则损失变成 NAN， 从500改为1000
    dict(
        type='MultiStepLR',
        begin=0,
        end=7,
        by_epoch=True,
        milestones=[3, 4],
        gamma=0.1)
]

visualizer = dict(type='DetLocalVisualizer')
