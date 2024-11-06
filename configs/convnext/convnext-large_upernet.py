_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/grass.py', '../_base_/default_runtime.py',
    '../_base_/schedules/grass_schedule.py'
]

data_preprocessor = dict(size=(256,256))
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=6,
    ),
    auxiliary_head=dict(in_channels=768, num_classes=6),
    test_cfg=dict(mode='whole'),
)
