# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


@MODELS.register_module()
class SelsaAggregator(BaseModule):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int, optional): The number of channels of the features of
            proposal.
        num_attention_blocks (int, optional): The number of attention blocks
            used in selsa aggregator module. Defaults to 16.
        init_cfg (OptConfigType, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 num_attention_blocks: int = 16,
                 init_cfg: OptConfigType = None):
        super(SelsaAggregator, self).__init__(init_cfg)
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks
        # SelsaAggregator(
        #     (fc_embed): Linear(in_features=1024, out_features=1024, bias=True)
        #     (ref_fc_embed): Linear(in_features=1024, out_features=1024, bias=True)
        #     (fc): Linear(in_features=1024, out_features=1024, bias=True)
        #     (ref_fc): Linear(in_features=1024, out_features=1024, bias=True)
        # )

    def forward(self, x: Tensor, ref_x: Tensor) -> Tensor:
        """Aggregate the features `ref_x` of reference proposals.
        将参考提议 ref_x 的特征进行加权求和，得到聚合后的特征表示 x_new。

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].
        """
        # 直接在这里打断点，就可以看到最深层的调用关系。
        roi_n, C = x.shape  # 176*1024
        ref_roi_n, _ = ref_x.shape  # 373*1024
        num_c_per_att_block = C // self.num_attention_blocks

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               num_c_per_att_block).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       num_c_per_att_block).permute(1, 2, 0)

        # 计算输入张量 x_embed 和参考张量 ref_x_embed 之间的自注意力矩阵/余弦相似度矩阵,类似QK/V
        # bmm表示批量矩阵乘法(Tensor(16,187,64)*(16,64,374)=(16,187,374))，表示16个注意力块，187个roi，374个参考帧roi
        # 除法是为了进行缩放，以确保点积的结果不会因为向量维度较大而过大。这是在实现自注意力机制时常用的技巧。
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)  # 得到一个形状与weights相同的概率分布张量。

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   num_c_per_att_block).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        # 用于将注意力权重 weights 和参考特征 ref_x_new 进行加权求和，得到聚合后的特征表示 x_new。
        # 使用 contiguous() 函数将张量变为连续的内存块，以便后续的全连接层操作。
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        x_new = self.fc(x_new.view(roi_n, C))  # (187,1024)
        return x_new
