# dataset config
_base_ = [
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/models/dinov2_mask2former.py",
    "../_base_/schedules/grass_schedule.py",
]
model = dict(
    decode_head=dict(sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000))
)
