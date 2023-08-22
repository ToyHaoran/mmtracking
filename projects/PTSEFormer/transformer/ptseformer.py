# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple, Union
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet.models.utils import samplelist_boxtype2tensor
from mmtrack.registry import MODELS
from mmtrack.models import BaseVideoDetector
from mmdet.structures import OptSampleList, SampleList

from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)
from mmtrack.utils import convert_data_sample_type
from .decoder import SimpleDecoderV2, OursDecoderV2


@MODELS.register_module()
class PTSEFormer(BaseVideoDetector, metaclass=ABCMeta):
    r"""Base class for Video Detection Transformer. 是从DetectionTransformer修改而来。

    In Detection Transformer, an encoder is used to process output features of
    neck, then several queries interact with the encoder features using a
    decoder and do the regression and classification with the bounding box
    head.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,

                 backbone: ConfigType = None,
                 neck: OptConfigType = None,
                 d_encoder: OptConfigType = None,
                 d_decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,

                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 TFAM_num_layers: int = 2,
                 STAM_num_layers: int = 2,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (d_decoder['num_layers'] + 1) \
                if self.as_two_stage else d_decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage

        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_queries = num_queries
        self.embed_dims = 256

        num_feats = positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. Found {self.embed_dims} and {num_feats}.'

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

        # 初始化 backbone, neck and bbox_head
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)

        # 初始化位置编码、DeformableDETR
        self.positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.d_encoder = DeformableDetrTransformerEncoder(**d_encoder)  # DeformableTransformerEncoder
        self.d_decoder = DeformableDetrTransformerDecoder(**d_decoder)  # DeformableTransformerDecoder
        self.TFAM = SimpleDecoderV2(num_layers=TFAM_num_layers)  # 黄色块 时间特征聚合模块
        self.STAM = OursDecoderV2(num_layers=STAM_num_layers)  # 绿色块 空间转换感知模块
        self.QAM = nn.Linear(self.embed_dims, self.embed_dims * 2)
        nn.init.constant_(self.QAM.bias, 0)


    def loss(self, inputs: Tensor,
             data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict[Tensor]): of shape (N=bs, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in SELSA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components
        """
        # 对视频数据集进行处理，将关键帧和参考帧放在一起。
        img = inputs['img']  # 关键帧 Tensor:(bs=1,1,3,608,800) T=1表示1个关键帧
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, 'video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, 'video detector only has 1 key image per batch.'
        img = img[0]
        ref_img = inputs['ref_img']  # 参考帧 Tensor:(1,2,3,608,800) T=2表示2个参考帧
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(0) == 1, 'video detectors only support 1 batch size per gpu for now.'
        ref_img = ref_img[0]

        assert len(data_samples) == 1, 'video detectors only support 1 batch size per gpu for now.'

        all_imgs = torch.cat((img, ref_img), dim=0)  # 把关键帧和参考帧 0维连接 Tensor:(T=3,C=3,H=608,W=800)、

        # 把关键帧和参考帧放在一起提取特征：经过backbone(ResNet)，经过neck(ChannelMapper)，提取特征。后面在pre_transformer会分开处理。
        img_feats = self.extract_feat(all_imgs)  # 4个level的Tensor:(3,256,38,50)

        # 进入PTSEFormer流程：直接在这里面全部运行了，不调来调去了。
        head_inputs_dict = self.forward_transformer(img_feats, data_samples)

        # 计算loss
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=data_samples)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

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
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, 'video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, 'video detector only has 1 key image per batch.'
        img = img[0]

        assert len(data_samples) == 1, 'detectors only support 1 batch size per gpu for now.'

        if 'ref_img' in inputs:
            ref_img = inputs['ref_img']
            assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
            assert ref_img.size(0) == 1, ' video detectors only support 1 batch size per gpu for now.'
            ref_img = ref_img[0]
            all_imgs = torch.cat((img, ref_img), dim=0)  # 把关键帧和参考帧 0维连接 Tensor:(T=3,C=3,H=608,W=800)、
        else:
            ref_img = None
            all_imgs = img

        img_feats = self.extract_feat(all_imgs)
        head_inputs_dict = self.forward_transformer(img_feats, data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=data_samples)
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_det_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples


    def _forward(
            self,
            inputs: Tensor,
            data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        # 对视频数据集进行处理，将关键帧和参考帧放在一起。
        img = inputs['img']  # 关键帧 Tensor:(bs=1,1,3,608,800) T=1表示1个关键帧
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, 'video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, 'video detector only has 1 key image per batch.'
        img = img[0]
        ref_img = inputs['ref_img']  # 参考帧 Tensor:(1,2,3,608,800) T=2表示2个参考帧
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(0) == 1, 'video detectors only support 1 batch size per gpu for now.'
        ref_img = ref_img[0]

        assert len(data_samples) == 1, 'video detectors only support 1 batch size per gpu for now.'

        all_imgs = torch.cat((img, ref_img), dim=0)  # 把关键帧和参考帧 0维连接 Tensor:(T=3,C=3,H=608,W=800)、

        img_feats = self.extract_feat(all_imgs)
        head_inputs_dict = self.forward_transformer(img_feats, data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:

        # ======1st stage---最左侧的梯形淡蓝色框--经过DeformableDETR
        mlvl_feats = img_feats
        batch_size = mlvl_feats[0].size(0)  # 注意这里的batch_size只有第一个是关键帧，剩下的是参考帧。

        # construct binary masks for the transformer.
        # assert batch_data_samples is not None
        assert len(batch_data_samples) == 1  # batch_data_samples就一个元素
        batch_input_shape = batch_data_samples[0].batch_input_shape

        # img_shape_list = [sample.img_shape for sample in batch_data_samples]  # 改一下，取得每张图片的实际shape
        key_img_shape = batch_data_samples[0].get("img_shape")
        ref_img_shape = batch_data_samples[0].get("ref_img_shape")
        img_shape_list = [key_img_shape, *ref_img_shape]
        if isinstance(ref_img_shape, Tuple):  # (demo数据集样本太少导致只有一个参考帧)
            img_shape_list = [key_img_shape, ref_img_shape]

        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []  # 多尺度特征图对应的mask，每个mask的shape为[bs, H_feat, W_feat]
        mlvl_pos_embeds = []  # 多尺度特征图对应的位置编码，每个位置编码对应的shape为[bs, 256, H_feat, W_feat]
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]  即长宽相乘，得到特征点数量。
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points=h*w, dim) 如(2, 19947, 256)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points) 如(2, 19947), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # 调用DeformableDetrTransformerEncoder，对特征进行编码
        memory = self.d_encoder(
            query=feat_flatten,  # 输入query，是碾平后的多尺度特征图[bs, num_feat_points, 256]，自注意力中K V是由Q算出来的。
            query_pos=lvl_pos_embed_flatten,  # 输入query的位置编码[bs, num_feat_points, 256]
            key_padding_mask=mask_flatten,  # for self_attn # 特征图的有效padding mask [bs, num_feat_points]
            spatial_shapes=spatial_shapes,  # 每层level/scale特征的w,h的列表 [num_levels, bs]
            level_start_index=level_start_index,  # 每层特征图碾平后的第一个元素的位置索引[num_levels]
            valid_ratios=valid_ratios)  # 每层特征图对应的mask中有效的宽高比 [bs, num_levels, 2]

        # 将encoder之后的特征还原回未碾平的状态，4个level，每个Tensor为(3,h*w,256)
        memory_level_list = []
        mask_level_list = []
        for i in range(len(level_start_index) - 1):
            memory_l = memory[:, level_start_index[i]:level_start_index[i + 1], :]
            memory_level_list.append(memory_l)
            mask_l = mask_flatten[:, level_start_index[i]:level_start_index[i + 1]]
            mask_level_list.append(mask_l)
        memory_level_list.append(memory[:, level_start_index[-1]:, :])
        mask_level_list.append(mask_flatten[:, level_start_index[-1]:])

        # 将当前帧和参考帧的feature/memory、mask分离，得到M_t和M_t+i
        mem_cur_stg1, mem_ref_stg1_list = self.separate_cur_and_ref_frames(memory_level_list)
        mask_cur_stg1, mask_ref_stg1_list = self.separate_cur_and_ref_frames(mask_level_list)

        # ======2nd stage=========中间的黄色块和绿色块==================================
        # 中间下方的绿色块，产生一堆f_t
        mem_ref_stg2_list = []
        for mem_ref, mask_ref in zip(mem_ref_stg1_list, mask_ref_stg1_list):  # 遍历两个参考帧
            mem_ref_stg2_level_list = []
            for mem_ref_level, mask_ref_level, mem_cur_level, mask_cur_level in \
                    zip(mem_ref, mask_ref, mem_cur_stg1, mask_cur_stg1):  # 对参考帧的每个level进行遍历
                # 参考帧的特征M_t+i作为Q，当前帧的特征M_t作为KV输入STAM。
                mem_ref_stg2 = self.STAM(tgt=mem_ref_level, memory=mem_cur_level).squeeze(0)
                mem_ref_stg2_level_list.append(mem_ref_stg2)
            mem_ref_stg2_list.append(mem_ref_stg2_level_list)

        # ======中间上方的黄色块，即第一个黄色块。产生h_t。
        # BUG: 源代码把参考帧的特征和mask汇总cat到一起，但是会导致QKV维度不一致。
        # 创新点：修改为关键帧与每个参考帧分别进行注意力机制。
        # 遍历并用每个参考帧增强关键帧的特征，mask没有用到(删掉)
        mem_cur_stg2 = mem_cur_stg1
        num_level = len(mem_cur_stg1)  # 判断level数
        for mem_ref_stg1 in mem_ref_stg1_list:
            for mem_ref_level, mem_cur_level in zip(mem_ref_stg1, mem_cur_stg2):
                # 当前帧特征M_t作为Q，参考帧特征M_t+i作为KV输入TFAM模块。注意 seq_len必须相同，否则出错。
                mem_cur_stg2_level = self.TFAM(tgt=mem_cur_level, memory=mem_ref_level).squeeze(0)
                mem_cur_stg2.append(mem_cur_stg2_level)
            mem_cur_stg2 = mem_cur_stg2[-num_level:]  # 取增强后的特征

        # ======3rd stage----右侧第1个黄色块，即上边最右边那个--------------------------------
        # mem_cur_stg2表示第二阶段产生的h_t, mem_ref_stg2_list表示第二阶段产生的一些f_t
        mem_cur_stg3 = mem_cur_stg2
        for mem_ref_stg2 in mem_ref_stg2_list:
            for mem_cur_level, mem_ref_level in zip(mem_cur_stg3, mem_ref_stg2):
                mem_final_level = self.TFAM(tgt=mem_cur_level, memory=mem_ref_level).squeeze(0)
                mem_cur_stg3.append(mem_final_level)
            mem_cur_stg3 = mem_cur_stg3[-num_level:]  # 取增强后的特征

        # ======4th stage----右侧第2个绿色块--------------------------------
        # mem_cur_stg1表示第一阶段的M_t，mem_cur_stg3表示第三阶段的E_t。
        mem_cur_stg4 = []
        for mem_ref, mem_cur in zip(mem_cur_stg3, mem_cur_stg1):
            mem_level = self.STAM(tgt=mem_ref, memory=mem_cur, memory_mask=None).squeeze(0)
            mem_cur_stg4.append(mem_level)

        mem_cur_stg4_cat = torch.cat(mem_cur_stg4, dim=1)

        # ======5th stage--------QAM块，用来聚合查询----------------------------
        query_embeds = self.query_embedding.weight
        query_ref_list = []
        for i, mem_ref in enumerate(mem_ref_stg1_list):
            mem_ref_cat = torch.cat(mem_ref, dim=1)
            outputs_dict, _ = self.pre_and_decoder(
                mem_ref_cat,
                query_embeds,
                (spatial_shapes, level_start_index, valid_ratios[[i+1]], mask_flatten[[i+1]]))
            hs = outputs_dict.get("hidden_states")
            query_ref = self.QAM(hs[-1])  # SD块
            query_ref_list.append(query_ref)

        bs, _, _ = mem_cur_stg4_cat.shape
        query_embed = query_embeds.expand(bs, -1, -1)
        query_ref_cat = torch.cat(query_ref_list, dim=1)
        query_mix = torch.cat((query_embed, query_ref_cat), dim=1)
        query_mix = torch.squeeze(query_mix, dim=0)

        # ======final stage--淡蓝色块----------------------------------
        decoder_outputs_dict, head_inputs_dict = self.pre_and_decoder(
            mem_cur_stg4_cat,
            query_mix,
            (spatial_shapes, level_start_index, valid_ratios[[0]], mask_flatten[[0]]))
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_and_decoder(self, memory: Tensor, query_embed: Tensor, dec_utils: Tuple):
        # 准备decoder的输入，详见DeformableDETR.pre_decoder
        # memory经过自注意力后的多尺度特征图，如[bs=2, num_feat_points=19947, dim=256]
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        batch_size, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord = None, None
        # query_embed = self.query_embedding.weight
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points_fc(query_pos).sigmoid()

        # 进入decoder，详见DeformableDETR.forward_decoder
        inter_states, inter_references = self.d_decoder(
            query=query,  # (2,100,256)
            value=memory,  # 经过encoder输出的特征图，作为V输入corss attention
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        # inter_states: 经过交叉注意力并包含每一层decoder输出的object query，shape为[num_layer=6, bs=2, num_query=100, dim=256]
        # inter_references: 每一层layer的reference point，shape为[num_layer=6, bs=2, num_query=100, 2]
        references = [reference_points, *inter_references]  # 保存最初的和中间参考点=1+num_layer=7个元素
        decoder_outputs_dict = dict(hidden_states=inter_states, references=references)

        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()

        return decoder_outputs_dict, head_inputs_dict

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_mask[:,
                                        _cur:(_cur + H * W)].view(bs, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    @staticmethod
    def get_proposal_pos_embed(proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def separate_cur_and_ref_frames(self, img_feats):
        """
        因为是放在一起进行提取特征和处理的，所以要分离关键帧和参考帧。
        Args:
            img_feats: (bs, dim, h, w)，这里的bs只有第一个是关键帧，其他为参考帧。

        Returns: Tuple(关键帧x, 参考帧ref_x)
        """
        x = []
        num_ref = img_feats[0].shape[0]-1
        # ref_x = [[]]*num_ref  # 错误：列表中的元素都指向同一个空列表对象，因此当你访问ref[1]时，它实际上引用的是同一个空列表。
        ref_x = [[] for _ in range(num_ref)]  # 正确：独立的空列表
        for i in range(len(img_feats)):  # 4个level，循环4次
            x.append(img_feats[i][[0]])  # 只有第一个是关键帧，关键帧的4个level特征图
            for j in range(1, num_ref+1):
                ref_x[j-1].append(img_feats[i][[j]])  # 剩下的是参考帧，参考帧的4个level特征图
        return x, ref_x

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
