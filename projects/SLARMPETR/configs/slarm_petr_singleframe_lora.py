"""SLARM attention LoRA + trainable affine tokens + PETR detection.

Set img_backbone.checkpoint to the matching pretrained SLARM checkpoint,
or use load_from/resume for a complete detector checkpoint. Architecture
parameters are inherited and must match the chosen pretrained model.
"""
_base_ = ['./slarm_petr_singleframe.py']

model = dict(
    img_backbone=dict(
        frozen=False,
        unfreeze_last_blocks=None,
        expand_affine_tokens=True,
        lora_cfg=dict(
            enabled=True,
            block_indices=[8, 9, 10, 11],
            branches=['frame_blocks', 'global_blocks'],
            target_modules=['attn.qkv', 'attn.proj'],
            rank=8,
            alpha=16,
            dropout=0.0,
            train_affine_tokens=True,
            train_feature_norm=False,
            gradient_checkpointing=True,
            # Affine tokens need gradients through the entire aggregator.
            checkpoint_start_block=0)),
    img_neck=dict(in_channels=2 * _base_.model.img_backbone.embed_dim))

# Keep the inherited schedule and optimizer; adapters and affine tokens use
# the base detection learning rate. Frozen base parameters receive no updates.
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'img_backbone': dict(lr_mult=1.0)}))
work_dir = './work_dirs/slarm_petr_singleframe_lora'
