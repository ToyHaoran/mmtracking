# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import mmcv
import torch
from addict import Dict
from mmengine.structures import InstanceData
from torch import Tensor

from mmengine.visualization import Visualizer
from mmtrack.registry import MODELS
from mmtrack.utils import (ConfigType, OptConfigType, SampleList,
                           convert_data_sample_type)
from .base import BaseVideoDetector


@MODELS.register_module()
class SELSA(BaseVideoDetector):
    """Sequence Level Semantics Aggregation for Video Object Detection.

    This video object detector is the implementation of `SELSA
    <https://arxiv.org/abs/1907.06390>`_.
    """

    def __init__(self,
                 detector: ConfigType,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(SELSA, self).__init__(data_preprocessor, init_cfg)
        self.detector = MODELS.build(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def loss(self, inputs: dict, data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in SELSA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = inputs['img']  # 关键帧 Tensor:(1,1,3,608,800) T=1表示1个关键帧
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, 'SELSA video detector only has 1 key image per batch.'
        img = img[0]

        ref_img = inputs['ref_img']  # 参考帧 Tensor:(1,2,3,608,800) T=2表示2个参考帧
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(0) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'
        ref_img = ref_img[0]

        assert len(data_samples) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'

        all_imgs = torch.cat((img, ref_img), dim=0)  # 把关键帧和参考帧 0维连接 Tensor:(3,3,608,800)、
        # 提取特征(进入mmdet)，detector是FasterRCNN，经过backbone(ResNet)，经过neck(ChannelMapper)，得到特征。
        all_x = self.detector.extract_feat(all_imgs)  # Tensor:(3,512,38,50)

        if False:  # 可视化特征图
            image = mmcv.imread(data_samples[0].get('img_path'), channel_order='rgb')  # 原始图片 HWC格式
            featmap = all_x[4][1].squeeze(dim=0)  # 特征图 CHW格式
            visualizer = Visualizer()
            # drawn_img = visualizer.draw_featmap(featmap, image, channel_reduction="squeeze_mean")
            drawn_img = visualizer.draw_featmap(featmap, image, channel_reduction="select_max")
            # drawn_img = visualizer.draw_featmap(featmap, image, channel_reduction=None, topk=8, arrangement=(4, 2))
            visualizer.show(drawn_img)

        # 就是把关键帧和参考帧一起提取特征，最后再分开
        x = []
        ref_x = []
        for i in range(len(all_x)):
            x.append(all_x[i][[0]])
            ref_x.append(all_x[i][1:])

        losses = dict()
        ref_data_samples, _ = convert_data_sample_type(data_samples[0], num_ref_imgs=len(ref_img))

        # 进行帧级别的特征聚合


        # 提取完特征，经过RPN获得提议，为后面的SELSA聚合准备数据。
        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get('rpn_proposal', self.detector.test_cfg.rpn)
            rpn_data_samples = deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)
            # 详见RPNHead：rpn_losses包括cls和bbox损失。proposal_list关键帧的提议框列表，bboxes、labels、scores
            (rpn_losses,  proposal_list) = \
                self.detector.rpn_head.loss_and_predict(x, rpn_data_samples, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            # 参考帧的提议框列表，bboxes、labels、scores
            ref_proposals_list = self.detector.rpn_head.predict(ref_x, ref_data_samples)
        else:
            proposal_list, ref_proposals_list = [], []
            for i in range(len(data_samples)):
                proposal, ref_proposals = InstanceData(), InstanceData()
                proposal.bboxes = data_samples[i].proposals
                proposal_list.append(proposal)
                ref_proposals.bboxes = data_samples[i].ref_proposals
                ref_proposals_list.append(ref_proposals)

        # 内部调用关系复杂，可在aggregator处打断点理解调用关系
        roi_losses = self.detector.roi_head.loss(x, ref_x, proposal_list, ref_proposals_list, data_samples, **kwargs)

        losses.update(roi_losses)

        return losses

    def extract_feats(self, img: Tensor, img_metas: dict,
                      ref_img: Optional[Tensor],
                      ref_img_metas: Optional[dict]) -> Tuple:
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (dict): list of image information dict where each
                dict may has: 'img_id', 'img_path',
                'ori_shape', 'img_shape', 'scale_factor','flip',
                'flip_direction', 'frame_id', 'is_video_data', 'video_id',
                'video_length', 'instances'.

            ref_img (Tensor | None): of shape (1, N, C, H, W) encoding input
                reference images. Typically these should be mean centered and
                std scaled. N denotes the number of reference images. There
                may be no reference images in some cases.

            ref_img_metas (list[dict] | None): The list contains image
                information dict where each dict may has: 'img_id', 'img_path',
                'ori_shape', 'img_shape', 'scale_factor','flip',
                'flip_direction', 'frame_id', 'is_video_data', 'video_id',
                'video_length', 'instances'.

        Returns:
            tuple(x, img_metas, ref_x, ref_img_metas): x is the multi level
                feature maps of `img`, ref_x is the multi level feature maps
                of `ref_img`.
        """
        frame_id = img_metas.get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas.get('num_left_ref_imgs', -1)
        frame_stride = img_metas.get('frame_stride', -1)

        # 默认方法为自适应采样
        if self.test_cfg is not None:
            method = self.test_cfg.get('ref_samper_method', 'test_with_adaptive_stride')  # 获取采样方法
        else:
            method = 'test_with_adaptive_stride'

        # 原来判断采样方法如下：
        # if frame_stride < 1: 为test with adaptive stride
        # else:  为test with fixed stride

        if method == 'bilateral_power':
            # 处理参考帧。如果不保存重复帧的话会导致大量重复计算，但是保存的话会占用大量的内存，只能作为一种参考，不如自适应采样。
            ref_x = self.detector.extract_feat(ref_img)
            # 处理关键帧：
            x = self.detector.extract_feat(img)

        elif method == 'test_with_fix_stride':
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]
                ref_x = self.detector.extract_feat(ref_img)
                # 'tuple' object (e.g. the output of FPN) does not support item assignment
                self.memo.feats = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
                    x.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x = []
                ref_x = self.detector.extract_feat(ref_img)
                for i in range(len(ref_x)):
                    self.memo.feats[i] = torch.cat(
                        (self.memo.feats[i], ref_x[i]), dim=0)[1:]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img_metas.extend(ref_img_metas[0])
                self.memo.img_metas = self.memo.img_metas[1:]
            else:
                assert ref_img is None
                x = self.detector.extract_feat(img)

            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i][num_left_ref_imgs] = x[i]
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas[num_left_ref_imgs] = img_metas

        else:  # 默认方法，自适应采样
            # 只有每个视频的第0帧才提取参考帧的特征，并作为所有视频的参考帧。
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas
                ref_x = self.detector.extract_feat(ref_img)  # 提取参考帧特征，每个视频提取一次。
                # 'tuple' object (e.g. the output of FPN) does not support item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
            # 视频的其他帧的处理方式：直接复制第0帧的参考帧特征，避免重复提取特征。
            x = self.detector.extract_feat(img)  # 提取关键帧的特征
            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i] = torch.cat((ref_x[i], x[i]), dim=0)
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas.append(img_metas)

        return x, img_metas, ref_x, ref_img_metas

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Test without augmentation.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in SELSA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor, Optional): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            list[obj:`TrackDataSample`]: Tracking results of the
            input images. Each TrackDataSample usually contains
            ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, 'SELSA video detector only has 1 key image per batch.'
        img = img[0]

        # 注意只有每个视频的第0帧才有参考帧，其他关键帧没有参考帧。
        if 'ref_img' in inputs:
            ref_img = inputs['ref_img']
            assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
            assert ref_img.size(0) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'
            ref_img = ref_img[0]
        else:
            ref_img = None

        assert len(data_samples) == 1, 'SELSA video detectors only support 1 batch size per gpu for now.'

        data_sample = data_samples[0]
        img_metas = data_sample.metainfo

        if ref_img is not None:
            _, ref_img_metas = convert_data_sample_type(data_sample, num_ref_imgs=len(ref_img))
        else:
            ref_img_metas = None

        x, img_metas, ref_x, ref_img_metas = self.extract_feats(img, img_metas, ref_img, ref_img_metas)

        ref_data_samples = [deepcopy(data_sample) for _ in range(len(ref_img_metas))]
        for i in range(len(ref_img_metas)):
            ref_data_samples[i].set_metainfo(ref_img_metas[i])

        if data_samples[0].get('proposals', None) is None:
            proposal_list = self.detector.rpn_head.predict(x, data_samples)
            ref_proposals_list = self.detector.rpn_head.predict(ref_x, ref_data_samples)
        else:
            assert hasattr(data_samples[0], 'ref_proposals')
            proposal_list = data_samples[0].proposals
            ref_proposals_list = data_samples[0].ref_proposals

        results_list = self.detector.roi_head.predict(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            data_samples,
            rescale=rescale)

        track_data_sample = deepcopy(data_samples[0])
        track_data_sample.pred_det_instances = results_list[0]
        return [track_data_sample]

    def aug_test(self,
                 inputs: dict,
                 data_samples: SampleList,
                 rescale: bool = True,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
