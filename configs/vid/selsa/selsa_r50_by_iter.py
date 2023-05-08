_base_ = [
    '../../_base_/models/faster-rcnn_r50-dc5.py',
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
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
train_cfg = dict(type='IterBasedTrainLoop', max_iters=220000, val_interval=220000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=70000, max_keep_ckpts=7),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00025/4, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

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
        end=220000,
        by_epoch=False,
        milestones=[110000, 165000],
        gamma=0.1)
]

visualizer = dict(type='DetLocalVisualizer')
