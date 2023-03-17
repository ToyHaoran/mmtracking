_base_ = [
    './faster-rcnn_r50_fpn.py',
    './imagenet_vid_demo.py',
    './default_runtime.py'
]

# custom_imports = dict(
#     imports=['configs.vid.selsa_diffusion.DiffusionDet.diffusiondet'], allow_failed_imports=False)

model = dict(
    type='SELSA',
    detector=dict(
        # r50和r101的主要区别
        backbone=dict(
            depth=101,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
        # 使用FCOSHead替换RPN网络
        neck=dict(
            start_level=1,
            add_extra_convs='on_output',  # 使用 P5
            relu_before_extra_convs=True),
        rpn_head=dict(
            _delete_=True,  # 忽略未使用的旧设置
            type='FCOSHead',
            num_classes=1,  # 对于 rpn, num_classes = 1，如果 num_classes > 1，它将在 TwoStageDetector 中自动设置为1
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        ),

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
        start_factor=0.001,  # 原来1.0/3
        by_epoch=False,
        begin=0,
        end=1000),  # 慢慢增加 lr，否则损失变成 NAN， 原来500
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,  # 原来7
        by_epoch=True,
        milestones=[8, 11],  # 原来[2,5]
        gamma=0.1)
]

visualizer = dict(type='DetLocalVisualizer')
