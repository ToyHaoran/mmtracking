_base_ = [
    './faster-rcnn_r50_fpn.py',
    './imagenet_vid_demo.py',
    './default_runtime.py'
]

custom_imports = dict(
    imports=['configs.vid.selsa_diffusion.DiffusionDet.diffusiondet'], allow_failed_imports=False)

model = dict(
    type='SELSA',
    detector=dict(
        # 换为r101
        backbone=dict(
            depth=101,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
        # 使用DiffusionNet替换RPN网络，但是梯度优化有问题？？
        neck=dict(
            num_outs=4,
        ),
        rpn_head=dict(
            _delete_=True,  # 忽略未使用的旧设置
            type='DynamicDiffusionDetHead',
            num_classes=80,  # 原来为80类，这里只需要分辨一个目标和背景就行
            feat_channels=256,
            num_proposals=100,  # TODO 原来500导致内存溢出，改为200才能跑起来
            num_heads=6,
            deep_supervision=True,
            prior_prob=0.01,
            snr_scale=2.0,
            sampling_timesteps=1,
            ddim_sampling_eta=1.0,
            single_head=dict(
                type='SingleDiffusionDetHead',
                num_cls_convs=1,
                num_reg_convs=3,
                dim_feedforward=2048,
                num_heads=8,
                dropout=0.0,
                act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
            roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            # criterion
            criterion=dict(
                type='DiffusionDetCriterion',
                num_classes=80,  # 这里也改为1
                assigner=dict(
                    type='DiffusionDetMatcher',
                    match_costs=[
                        dict(
                            type='FocalLossCost',
                            alpha=0.25,
                            gamma=2.0,
                            weight=2.0,
                            eps=1e-8),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ],
                    center_radius=2.5,
                    candidate_topk=5),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    alpha=0.25,
                    gamma=2.0,
                    reduction='sum',
                    loss_weight=2.0),
                loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
                loss_giou=dict(type='GIoULoss', reduction='sum',
                               loss_weight=2.0))
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
