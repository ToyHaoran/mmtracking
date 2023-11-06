# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmdet.models import ConvFCBBoxHead
from torch import Tensor
import torch

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType


@MODELS.register_module()
class SelsaBBoxHead(ConvFCBBoxHead):
    """Selsa bbox head.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        aggregator (ConfigType): Configuration of aggregator.
    """

    def __init__(self, aggregator: ConfigType, *args, **kwargs):
        super(SelsaBBoxHead, self).__init__(*args, **kwargs)
        self.aggregator = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            self.aggregator.append(MODELS.build(aggregator))  # 用于关键帧和参考帧提议的聚合。
        self.self_aggregator = MODELS.build(aggregator)  # 用于参考帧提议的自我聚合。
        self.self_linear = nn.Linear(self.in_channels * 49, self.shared_out_channels, bias=True)
        self.inplace_false_relu = nn.ReLU(inplace=False)  # 把负值变为0值。

        self.proposal_fetures_memory = []  # 用于记忆模块

    def forward(self, x: Tensor, ref_x: Tensor) -> Tuple:
        """Computing the `cls_score` and `bbox_pred` of the features `x` of key
        frame proposals.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of key frame proposals.
            ref_x (Tensor): of shape [M, C, H, W]. M is the number of reference frame proposals.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * 4.
        """
        # 使用共享卷积降低维度，默认输出(N,256,7,7)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
                ref_x = conv(ref_x)

        if self.num_shared_fcs > 0:
            # 使用全局平均池化降低维度，平均池化，缓解卷积层对位置的过度敏感性。
            if self.with_avg_pool:
                x = self.avg_pool(x)
                ref_x = self.avg_pool(ref_x)

            # TODO 在这里进行一个记忆模块读取，作为参考帧提议的一部分，用来融合，但是怎么保存呢？
            # 想要保存必须在检测之后，把好的检测结果保存下来。
            self.proposal_fetures_memory = []

            # flatten之后就是降维了，破坏了特征，如果要保存好的提议特征需要在这里进行检测并保存。
            x = x.flatten(1)  # 从(256,512,7,7)变为(256,25088)。
            ref_x = ref_x.flatten(1)  # 从(600,512,7,7)变为(600,25088)。

            # # 增加batch-size,令其能使用nn.MultiheadAttention，不用自己写了
            # x = x.unsqueeze(0)
            # ref_x = ref_x.unsqueeze(0)

            # # 对关键帧进行自注意力增强(效果下降)
            # x = self.shared_fcs[0](x)  # 从25088到1024，降维
            # x = x + self.self_aggregator(x, x)
            # x = self.inplace_false_relu(x)

            # # 对参考帧进行自注意力增强(略微提升效果map803)
            # ref_x = self.shared_fcs[0](ref_x)  # 从25088到1024，降维
            # ref_x = ref_x + self.self_aggregator(ref_x, ref_x)
            # ref_x = self.inplace_false_relu(ref_x)

            # 聚合参考帧的提议特征 第一次聚合
            x = self.shared_fcs[0](x)  # 从25088到1024，降维
            ref_x = self.shared_fcs[0](ref_x)  # 如果前面参考帧降维了，这里就必须注释掉。
            x = x + self.aggregator[0](x, ref_x)
            ref_x = self.inplace_false_relu(ref_x)  # 把负值变为0值。
            x = self.inplace_false_relu(x)

            # # 参考帧二次提纯
            # ref_x = ref_x + self.self_aggregator(ref_x, ref_x)
            # ref_x = self.inplace_false_relu(ref_x)

            # 第二次聚合(map802)
            x = self.shared_fcs[1](x)  # 从1024到1024
            ref_x = self.shared_fcs[1](ref_x)
            x = x + self.aggregator[1](x, ref_x)
            ref_x = self.inplace_false_relu(ref_x)
            x = self.inplace_false_relu(x)

            # # 第三次聚合
            # x = self.shared_fcs[2](x)  # 从1024到1024
            # ref_x = self.shared_fcs[2](ref_x)
            # x = x + self.aggregator[2](x, ref_x)
            # ref_x = self.inplace_false_relu(ref_x)
            # x = self.inplace_false_relu(x)

            if len(x.shape) == 3:  # 删除batch-size
                x = x.squeeze(0)

        # 这里得到的x是最终特征，接下来就将其送入检测头进行检测了。
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # 预测的分类分数(256,31)，或者说置信度？
        cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # # 把预测分数大的提议特征放入记忆模块，用于同一视频之后的聚合。(没什么用)
        # mask = torch.any(cls_score > 0.5, dim=1)  # 过滤出大于0.5的行。
        # cls_store = cls_score[mask]

        # 预测的边界框(256,120)
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def forward_old(self, x: Tensor, ref_x: Tensor) -> Tuple:
        """Computing the `cls_score` and `bbox_pred` of the features `x` of key
        frame proposals.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of key frame proposals.
            ref_x (Tensor): of shape [M, C, H, W]. M is the number of reference frame proposals.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
                ref_x = conv(ref_x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
                ref_x = self.avg_pool(ref_x)
            x = x.flatten(1)  # 从(256,512,7,7)变为(256,25088)。
            ref_x = ref_x.flatten(1)  # 从(600,512,7,7)变为(600,25088)。
            # shared_fcs有两层，从25088到1024，然后从1024到1024

            for i, fc in enumerate(self.shared_fcs):
                x = fc(x)  # 从25088到1024，降维，避免数据量太大。  # 这里共享了权重fc
                ref_x = fc(ref_x)
                # 将关键帧和参考帧的特征进行聚合，得到增强后的特征。
                x = x + self.aggregator[i](x, ref_x)  # 聚合了两次，有没有必要？
                ref_x = self.inplace_false_relu(ref_x)  # 把负值变为0值。
                x = self.inplace_false_relu(x)

        # 这里得到的x是最终特征，接下来就将其送入检测头进行检测了。
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
