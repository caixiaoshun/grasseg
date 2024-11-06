_base_ = [
    "../_base_/models/upernet_swin.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

norm_cfg = dict(type="BN", requires_grad=True)
data_preprocessor = dict(
    size=(256, 256),
    type="SegDataPreProcessor",
    mean=[0.4417, 0.5110, 0.3178],
    std=[0.2330, 0.2358, 0.2247],
    bgr_to_rgb=False,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=3,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=6),
    auxiliary_head=dict(in_channels=384, num_classes=6),
)