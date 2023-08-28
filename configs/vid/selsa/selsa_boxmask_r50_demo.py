_base_ = [
    '../../_base_/models/faster-rcnn_r50_dc5_for_boxmask.py',
    # '../../_base_/datasets/imagenet_vid_boxmask.py',  # 使用混合数据集，官方数据集
    # '../../_base_/datasets/imagenet_vid_only_boxmask.py',  # 仅使用vid数据集
    '../../_base_/datasets/imagenet_vid_demo_boxmask.py',  # 仅使用demo数据集，非常小
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_head=dict(
                type='mmtrack.SelsaBBoxHead',
                num_shared_fcs=2,  # TROI中这里为3
                aggregator=dict(
                    type='mmtrack.SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)),
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),

            # 以下是增加的maskhead
            mask_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                # 这里就是第三个分支，输出为14*14，boxmask的sampling_ratio为2.
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=1,  # 这里只需要一个卷积就行，多了反而效果不好。
                in_channels=512,
                conv_out_channels=512,
                num_classes=30,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=0.2))  # 论文中的超参数λ
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
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
