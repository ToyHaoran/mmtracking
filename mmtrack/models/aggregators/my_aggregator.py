# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType
# from mmpretrain.models.utils.attention import MultiheadAttention
from mmcv.cnn.bricks.transformer import MultiheadAttention

@MODELS.register_module()
class MyAggregator(BaseModule):
    """ 自定义聚合器.

    Args:
        in_channels (int, optional): 提议特征的通道数
        num_attention_blocks (int, optional): 用于聚合器的注意力块数量，默认16
        init_cfg (OptConfigType, optional): 初始化参数，默认None
    """

    def __init__(self,
                 in_channels: int = 1024,  # 默认1024
                 num_attention_blocks: int = 16,
                 init_cfg: OptConfigType = None):
        super(MyAggregator, self).__init__(init_cfg)
        self.fc_embed = nn.Linear(in_channels, in_channels)  # Linear(in_features=1024, out_features=1024, bias=True)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

        # 使用官方的多头自注意力
        self.attn = nn.MultiheadAttention(in_channels, num_attention_blocks, batch_first=True)  # 多头自注意力
        self.layer_norm = nn.LayerNorm(in_channels)  # 层标准化
        self.batch_norm = nn.BatchNorm2d(in_channels)

    # 参考 https://zhuanlan.zhihu.com/p/366592542

    # def ScaledDotProductAttention(self, query, key, value, dropout_p=0, scale=None, mask=None) -> Tensor:
    #     """输入和输出的shape都为(bs,num,d)"""
    #
    #     # Q表示关键帧，KV表示参考帧。shape都为(batch_size, roi_n个数, d维度)，
    #     bs, roi_n, d = query.shape
    #
    #     scale = scale or d ** 0.5
    #     # transpose是交换两个维度，permute()是交换多个维度。
    #     weights = torch.bmm(query, key.transpose(1, 2)) / scale  # Q*K shape=(16, 256, 600)
    #     if mask is not None:
    #         weights = weights.masked_fill(mask, -np.inf)  # Mask
    #     weights = torch.softmax(weights, dim=-1)
    #     weights = torch.dropout(weights, dropout_p, True)
    #
    #     x_new = torch.bmm(weights, value).contiguous()  # *V
    #
    #     return x_new

    def forward(self, x: Tensor, ref_x: Tensor) -> Tensor:
        """将参考提议的特征进行加权求和，得到x_new

        1. 用多头注意力计算 x 和 ref_x 之间的权重。
        2. 归一化权重并对参考提议进行加权求和。

        Args:
            x (Tensor): of shape [bs, N, C]. 表示N个关键帧提议，每个提议有1024个特征表示，从512*7*7=25088被压缩到1024了。
            ref_x (Tensor): of shape [bs, M, C]. 表示M个参考帧提议，同上。

        Returns:
            Tensor: 聚合后的关键帧提议，with shape [N, C].
        """
        x = self.fc_embed(x)
        ref_x = self.ref_fc_embed(ref_x)
        x, x_weight = self.attn(x, ref_x, ref_x)
        x = self.fc(x)
        # ref_x = self.ref_fc(ref_x)
        return x

    def forward_old(self, x: Tensor, ref_x: Tensor) -> Tensor:
        """将参考提议的特征进行加权求和，得到x_new

        1. 用多头注意力计算 x 和 ref_x 之间的权重。
        2. 归一化权重并对参考提议进行加权求和。

        Args:
            x (Tensor): of shape [N, C]. 表示N个关键帧提议，每个提议有1024个特征表示，从512*7*7=25088被压缩到1024了。
            ref_x (Tensor): of shape [M, C]. 表示M个参考帧提议，同上。

        Returns:
            Tensor: 聚合后的关键帧提议，with shape [N, C].
        """
        # 直接在这里打断点，就可以看到最深层的调用关系。
        roi_n, C = x.shape  # roi_n表示roi的数量(如256)，C表示维度或提议特征=1024。
        ref_roi_n, _ = ref_x.shape  # 600*1024
        num_c_per_att_block = C // self.num_attention_blocks  # 每个注意力块(共16个)中的特征数量(1024/16=64)。

        x_embed = self.fc_embed(x)  # 线性变换引入了一组可学习的权重参数，模型可以根据训练数据自动学习出适合任务的特征表示。
        x_embed = x_embed.view(roi_n, self.num_attention_blocks, num_c_per_att_block).permute(1, 0, 2)
        # 经过上面一步，x_embed的shape=[16, 256, 64]=[注意力块, 提议数量, 每个注意块要处理的特征数]，相当于把1024个特征分成了16份。

        ref_x_embed = self.ref_fc_embed(ref_x)
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks, num_c_per_att_block).permute(1, 2, 0)
        # 经过上面一步，ref_x_embed的shape=[16, 64, 600]，便于两个矩阵相乘。

        """
        计算输入张量 x_embed 和参考张量 ref_x_embed 之间的自注意力矩阵/余弦相似度矩阵,类似QK/d
        bmm表示批量矩阵乘法(Tensor(16,256,64)*(16,64,600)=(16,256,600))
        除法是为了进行缩放，以确保点积的结果不会因为向量维度较大而过大。
        对于余弦相似度，值靠近1表示高度相似，靠近-1表示高度不相似，0表示垂直，也是不相似。
        """
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)  # 得到一个形状与weights相同的概率分布张量。 # 每个16x600的切片(维度2)应用softmax函数。
        # weights = torch.dropout(weights, p=0.2, train=self.training)  # 新加的
        # 这里的权重是(16, 256, 600)，批次表示1024中的一段特征，换成8个注意力块效果反而变差了。

        ref_x_new = self.ref_fc(ref_x)
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks, num_c_per_att_block).permute(1, 0, 2)

        # 用于将注意力权重 weights 和参考特征 ref_x_new 进行加权求和，得到聚合后的特征表示 x_new。
        # 使用 contiguous() 函数将张量变为连续的内存块，以便后续的全连接层操作。
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        x_new = self.fc(x_new.view(roi_n, C))  # (256,1024)
        return x_new
