# model settings
_base_ = 'faster-rcnn_swinT_backbone.py'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'
# pretrained = '/root/pro/mmtracking/work_dirs/swinb_90.0/checkpoint0005.pth'
pretrained = '/mnt/nfs/data/home/1120220334/pro/mmtracking/work_dirs/swinb_90.0/checkpoint0005.pth'
# pretrained = '/mnt/nfs/data/home/1120220334/pro/mmtracking/work_dirs/selsa_swin/epoch_5.pth'
model = dict(
    detector=dict(
        backbone=dict(
            type='SwinTransformer',
            embed_dims=128,  # 改
            depths=[2, 2, 18, 2],  # Swin-B的配置
            num_heads=[4, 8, 16, 32],  # 改
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            # convert_weights=True,   # 这里为True表示官方模型转为mmtracking的模型。 resume失败，权重加载有问题。
            # init_cfg=dict(type='Pretrained', convert_transvod_plusplus=False, checkpoint=pretrained),
            convert_weights=False,  # 下面为True表示将transvod_plusplus中的模型转为mmtracking的模型。
            init_cfg=dict(type='Pretrained', convert_transvod_plusplus=True, checkpoint=pretrained),
            ),
        neck=dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],  # Swin-B的输出维度。
            out_channels=256,
            num_outs=5),
    ))
