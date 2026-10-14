"""Random-initialized, single-frame six-camera SLARM + PETR experiment."""
_base_ = ['../../PETR/configs/petr_vovnet_gridmask_p4_800x320.py']
custom_imports = dict(imports=['projects.PETR.petr', 'projects.SLARMPETR'])

model = dict(
    type='SLARMPETR', use_grid_mask=False,
    # RGB [0,255], then backbone /255 and SLARM normalization exactly once.
    data_preprocessor=dict(_delete_=True, type='Det3DDataPreprocessor',
                           bgr_to_rgb=True, pad_size_divisor=32),
    img_backbone=dict(
        _delete_=True,
        type='SLARMFeatureBackbone',
        embed_dim=768,
        depth=12,
        num_heads=16,
        patch_size=8,
        encoder_size=None,
        checkpoint=None,
        frozen=False,
        unfreeze_last_blocks=None,
        use_bf16=True,
        expand_affine_tokens=False),
    img_neck=dict(_delete_=True, type='SLARMFeatureAdapter',
                  in_channels=1536, out_channels=256, downsample=2),
    pts_bbox_head=dict(type='SLARMPETRHead'))

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for transform in train_pipeline:
    if transform['type'] == 'ResizeCropFlipImage':
        transform['type'] = 'SLARMResizeCropFlipImage'
for transform in test_pipeline:
    if transform['type'] == 'ResizeCropFlipImage':
        transform['type'] = 'SLARMResizeCropFlipImage'

# Restore scene-disjoint training; do not inherit the baseline's overfit split.
train_dataloader = dict(dataset=dict(ann_file='v1.0-mini_infos_train.pkl',
                                     pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_evaluator = dict(modality=dict(use_camera=True, use_lidar=False))
test_evaluator = dict(modality=dict(use_camera=True, use_lidar=False))

# Train the randomly initialized encoder instead of freezing random features.
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'img_backbone': dict(lr_mult=1.0)}))
load_from = None
resume = False
work_dir = './work_dirs/slarm_petr_singleframe'
