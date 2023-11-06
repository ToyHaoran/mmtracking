# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=16),
    detector=dict(
        type='FasterRCNN',
        _scope_='mmdet',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),  # 最后一层有膨胀卷积
            frozen_stages=1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,
            kernel_size=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                # 每个特征映射回原图生成的anchor个数为length(scales)*length(anchor_ratios)
                scales=[4, 8, 16, 32],  # 将anchor，这里加个2会导致大目标精度下降，小目标提升很小。
                ratios=[0.5, 1.0, 2.0],  # 原来 0.5, 1.0, 2.0
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16],  # 需要映射的featmap尺度， 若为FPN的P2-P5层，则[4, 8, 16, 32]
            ),
            bbox_head=dict(  # 对roi操作之后的proposals进行后续的卷积、全连接、以及预测
                type='Shared2FCBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=30,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
        # detector training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps2D'),
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,  # 关键帧的提议数量
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=6000,  # 对每一层在nms之前都取scores分值最高得前6000个
                max_per_img=600,  # 参考帧的提议数量
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps2D'),
                    pos_iou_thr=0.5,  # 对2000个proposals的每一个，都找到与所有gt最大的iou，这个iou>pos_iou_thrde的为正例
                    neg_iou_thr=0.5,  # 对2000个proposals的每一个，都找到与所有gt最大的iou，这个0<iou<neg_iou_thr的为正例
                    min_pos_iou=0.5,  # 假设有10个gt，对每一个gt都从2000个proposals中找到与之iou最大的，并且iou>min_pos_iou的为正例
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,  # 从assigner之后proposals中，随机选取256个
                    pos_fraction=0.25,  # 正例的比例占0.25，其余为负例，正例不足用负例补齐
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),  # 将gt添加到当前的proposals中
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
