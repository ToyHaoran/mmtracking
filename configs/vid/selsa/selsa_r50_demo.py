_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',  # dc5提取特征时输出1个(3,512,38,50)的特征图
    # '../../_base_/models/faster-rcnn_r50_fpn.py',  # fpn提取特征时输出4个(3,512,x,x)的特征图。
    # '../../_base_/models/faster-rcnn_r50_fpn_neck.py',
    # '../../_base_/models/faster-rcnn_r50_YOLOF_neck.py',
    # '../../_base_/models/faster-rcnn_r50-dc5_sampler.py',
    # '../../_base_/datasets/imagenet_vid_demo.py',  # demo数据集，很小，便于调试。
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='mmtrack.SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='mmtrack.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            ),
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
            method='test_with_adaptive_stride',
            # method='test_with_adaptive_stride_and_random',
        )))
test_dataloader = val_dataloader

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
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
