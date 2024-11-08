# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=True, begin=0, end=50),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=50,
        end=500,
        by_epoch=True,
    ),
]
# training schedule for 320k
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=500, val_begin=1,val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=23, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1,save_best=['mIoU'],rule=['greater'],max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
