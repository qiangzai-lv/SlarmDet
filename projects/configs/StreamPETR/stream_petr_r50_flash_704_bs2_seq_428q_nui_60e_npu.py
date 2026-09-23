"""Ascend NPU variant of the StreamPETR nuImages/nuScenes configuration.

The model, data pipeline, and optimization schedule are inherited unchanged.
Only runtime/device settings and the CUDA-only flash-attention implementation
are overridden.  ``torch_npu`` must be installed on the target host before
launching this configuration.
"""

_base_ = ['./stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py']

# MMEngine uses this value when moving the model and data to the accelerator.
device = 'npu'

# Ascend distributed training uses HCCL.  Keeping one process per device also
# makes the same config usable for a single-NPU smoke test.
num_gpus = 1
batch_size = 2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 60
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='hccl'))

# flash-attn is a CUDA extension.  The regular PETR attention has the same
# residual/positional-encoding contract and is supported by the NPU backend.
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            batch_first=True),
                    ])))))

# ``AmpOptimWrapper`` is device-neutral in MMEngine and torch_npu provides
# the corresponding autocast/scaler implementation.  The conservative scale
# is retained because the temporal decoder is numerically sensitive.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale=dict(init_scale=128, growth_interval=2000))

# Recompute all iteration-based schedules after changing the device count.
total_iters = num_epochs * num_iters_per_epoch
runner = dict(type='IterBasedRunner', max_iters=total_iters)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=total_iters,
    val_interval=num_iters_per_epoch)
evaluation = dict(interval=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=total_iters,
        eta_min=4e-7,
        begin=0,
        end=total_iters,
        by_epoch=False)
]
