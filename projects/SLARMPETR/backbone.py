import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS
from .slarm.aggregator import Aggregator
from .slarm_lora import (LoRALinear, configure_slarm_lora,
                         enable_lora_parameters, lora_parameter_name)


PETR_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
SLARM_CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT')


@MODELS.register_module()
class SLARMFeatureBackbone(BaseModule):
    """Single-frame, calibrated six-camera SLARM features, in PETR order.

    Input images are RGB floats in [0, 255], before SLARM normalization.
    ``encoder_size`` optionally resizes internally; detection metadata stays
    in the original image coordinates. Camera names are taken from img_path.
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=16, patch_size=8,
                 encoder_size=(160, 288), checkpoint=None,
                 frozen=False, unfreeze_last_blocks=None, use_bf16=True,
                 expand_affine_tokens=False, aggregator_cfg=None, lora_cfg=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.encoder_size = tuple(encoder_size) if encoder_size else None
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.use_bf16 = use_bf16
        self.expand_affine_tokens = expand_affine_tokens
        self.unfreeze_last_blocks = unfreeze_last_blocks
        if frozen and unfreeze_last_blocks not in (None, 0):
            raise ValueError('Use frozen=False for partial fine-tuning')
        if unfreeze_last_blocks is not None and not 0 <= unfreeze_last_blocks <= depth:
            raise ValueError('unfreeze_last_blocks must be between 0 and depth')
        cfg = dict(img_size=self.encoder_size or (320, 800), num_cams=6,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   patch_size=patch_size, patch_embed='conv',
                   decoder_type='dummy', num_motion_tokens=0,
                   use_sky_token=True, use_affine_token=True,
                   use_time_token=False, concat_plucker_embed=False,
                   add_patch_plucker_embed=True, add_camera_embed=False)
        cfg.update(aggregator_cfg or {})
        for name, value in dict(num_cams=6, embed_dim=embed_dim, depth=depth,
                                patch_size=patch_size, num_heads=num_heads).items():
            if cfg[name] != value:
                raise ValueError(f'Set {name} at backbone level, not aggregator_cfg')
        if cfg['patch_embed'] != 'conv':
            raise ValueError('Initial integration supports Conv patch checkpoints only')
        self.aggregator = Aggregator(**cfg)
        self.feature_norm = nn.LayerNorm(2 * embed_dim)
        self.lora_cfg, self.lora_modules = configure_slarm_lora(self.aggregator, lora_cfg)
        self.lora_enabled = self.lora_cfg['enabled']
        if self.lora_enabled and (frozen or unfreeze_last_blocks is not None):
            raise ValueError('LoRA requires frozen=False and unfreeze_last_blocks=None')
        self._set_trainability()
        if self.lora_enabled:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            MMLogger.get_current_instance().info(
                f'SLARM LoRA: config={self.lora_cfg}, '
                f'modules={self.lora_modules}, trainable_backbone_parameters={trainable}')

    def _set_trainability(self):
        if self.lora_enabled:
            enable_lora_parameters(self.aggregator, self.lora_cfg['train_affine_tokens'])
            self.feature_norm.requires_grad_(self.lora_cfg['train_feature_norm'])
            return
        count = self.unfreeze_last_blocks
        train_all = not self.frozen and count is None
        self.aggregator.requires_grad_(train_all)
        self.feature_norm.requires_grad_(train_all or (not self.frozen and bool(count)))
        if not self.frozen and count:
            for blocks in (self.aggregator.frame_blocks, self.aggregator.global_blocks):
                for block in blocks[-count:]:
                    block.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.lora_enabled:
            self._set_trainability()
            return self
        if self.frozen or self.unfreeze_last_blocks == 0:
            self.aggregator.eval()
            self.feature_norm.eval()
        elif self.unfreeze_last_blocks is not None:
            self.aggregator.eval()
            for blocks in (self.aggregator.frame_blocks, self.aggregator.global_blocks):
                for block in blocks[-self.unfreeze_last_blocks:]:
                    block.train(mode)
        return self

    def init_weights(self):
        if self._is_init:
            return
        # Match SLARM's Linear initialization; Conv/LayerNorm retain defaults.
        adapters = [m for m in self.modules() if isinstance(m, LoRALinear)]
        adapter_children = {id(child) for m in adapters for child in (m.lora_a, m.lora_b)}
        for module in self.modules():
            if (isinstance(module, (nn.Linear, LoRALinear))
                    and id(module) not in adapter_children):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in adapters:
            module.reset_lora_parameters()
        if self.checkpoint:
            self.load_slarm_checkpoint(self.checkpoint)
        elif self.lora_enabled:
            MMLogger.get_current_instance().warning(
                'LoRA initialized without a SLARM checkpoint. Supply pretrained '
                'weights via img_backbone.checkpoint or a full detector checkpoint '
                'via load_from/resume; otherwise the frozen base remains random.')
        self._is_init = True

    def load_slarm_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('model', state.get('state_dict', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        selected = {k[len('aggregator.'):]: v for k, v in state.items()
                    if k.startswith('aggregator.')}
        if not selected:
            raise ValueError('Checkpoint must contain SLARM aggregator.* keys')
        target = self.aggregator.state_dict()
        derived = {'plucker_embedder.x', 'plucker_embedder.y'}
        adapter_keys = {k for k in target if lora_parameter_name(k)}
        # A base SLARM checkpoint has no adapters; a checkpoint containing
        # adapters must contain the complete set with matching shapes.
        if not any(lora_parameter_name(k) for k in selected):
            selected.update({k: target[k] for k in adapter_keys})
        token = 'affine_token'
        if (token in selected and token in target
                and selected[token].shape != target[token].shape):
            old, new = selected[token], target[token]
            if not self.expand_affine_tokens or old.shape != (1, 3, self.embed_dim):
                raise ValueError('Affine token shape mismatch; explicitly enable '
                                 'expand_affine_tokens for 3-camera checkpoints')
            selected[token] = old.repeat(1, 2, 1)
            MMLogger.get_current_instance().warning('Expanded affine tokens from 3 to 6')
        errors = [k for k, v in target.items() if k not in derived
                  and (k not in selected or selected[k].shape != v.shape)]
        unexpected = set(selected) - set(target)
        norm = {k[len('aggregated_last_tokens_norm.'):]: v for k, v in state.items()
                if k.startswith('aggregated_last_tokens_norm.')}
        norm_target = self.feature_norm.state_dict()
        norm_mismatch = (set(norm) != set(norm_target) or any(
            norm[k].shape != v.shape for k, v in norm_target.items() if k in norm))
        if errors or unexpected or norm_mismatch:
            raise ValueError(f'Incompatible SLARM checkpoint: missing/mismatched={errors}, '
                             f'unexpected={sorted(unexpected)}, norm_keys={list(norm)}')
        selected.update({k: target[k] for k in derived})
        self.aggregator.load_state_dict(selected, strict=True)
        self.feature_norm.load_state_dict(norm, strict=True)
        MMLogger.get_current_instance().info(f'Loaded SLARM encoder from {path}')

    @staticmethod
    def camera_names(meta):
        names = meta.get('camera_names')
        if names is None:
            paths = meta.get('img_path', meta.get('filename'))
            if paths is None:
                raise ValueError('Camera names or per-view img_path are required')
            names = [Path(p).parent.name for p in paths]
        if len(names) != 6 or set(names) != set(SLARM_CAMERAS):
            raise ValueError(f'Expected six unique nuScenes cameras, got {names}')
        return tuple(names)

    def prepare_inputs(self, images, metas):
        if images.ndim != 5 or images.shape[1:3] != (6, 3):
            raise ValueError('Expected images [B,6,3,H,W]')
        b, _, _, h, w = images.shape
        if len(metas) != b:
            raise ValueError('Metadata batch size does not match images')
        if not torch.isfinite(images).all() or torch.any(images < 0) or torch.any(images > 255):
            raise ValueError('Expected finite RGB image values in [0,255] before normalization')
        if any(tuple(m['pad_shape'][:2]) != (h, w) for m in metas):
            raise ValueError('pad_shape must match input images')
        # Dense SLARM attention has no padding mask. Do not silently include
        # padded views; deterministic sizes divisible by 32 avoid padding.
        for meta in metas:
            shapes = meta['img_shape']
            if len(shapes) == 6 and isinstance(shapes[0], (tuple, list)):
                shapes = [tuple(s[:2]) for s in shapes]
            else:
                shapes = [tuple(shapes[:2])] * 6
            if any(s != (h, w) for s in shapes):
                raise ValueError('Initial SLARM encoder requires unpadded equal-size views')
        ordered_images, intrinsics, poses, inverse_orders = [], [], [], []
        axis = images.new_tensor([[0, 0, 1, 0], [-1, 0, 0, 0],
                                  [0, -1, 0, 0], [0, 0, 0, 1]])
        for i, meta in enumerate(metas):
            names = self.camera_names(meta)
            order = [names.index(name) for name in SLARM_CAMERAS]
            inverse_orders.append([SLARM_CAMERAS.index(name) for name in names])
            k = torch.as_tensor(np.asarray(meta['cam2img']), device=images.device,
                                dtype=torch.float32)[:, :3, :3].clone()
            e = torch.as_tensor(np.asarray(meta['lidar2cam']), device=images.device,
                                dtype=torch.float32)
            if e.shape != (6, 4, 4) or k.shape != (6, 3, 3):
                raise ValueError('Expected six 4x4 extrinsics and 3x3/4x4 intrinsics')
            if not torch.isfinite(k).all() or not torch.isfinite(e).all():
                raise ValueError('Non-finite camera calibration')
            if torch.any(k[:, 0, 0] <= 0) or torch.any(k[:, 1, 1] <= 0):
                raise ValueError('Initial integration requires positive focal lengths '
                                 '(disable horizontal flip)')
            local = axis.float() @ e[names.index('CAM_FRONT')]
            poses.append((local @ torch.linalg.inv(e))[order])
            intrinsics.append(k[order])
            ordered_images.append(images[i, order].float() / 255.0)
        image = torch.stack(ordered_images)
        k = torch.stack(intrinsics)
        eh, ew = self.encoder_size or (h, w)
        if eh % self.patch_size or ew % self.patch_size:
            raise ValueError('Encoder image size must be divisible by patch_size')
        if (eh, ew) != (h, w):
            image = F.interpolate(image.reshape(b * 6, 3, h, w), size=(eh, ew),
                                  mode='bilinear', align_corners=False,
                                  antialias=True).reshape(b, 6, 3, eh, ew)
            k[..., 0, :] *= ew / w
            k[..., 1, :] *= eh / h
        return dict(context_image=image[:, None],
                    context_intrinsics=k[:, None],
                    context_camtoworlds=torch.stack(poses)[:, None],
                    context_time=image.new_zeros(b, 1, 6)), inverse_orders

    def forward(self, images, metas):
        inputs, inverse_orders = self.prepare_inputs(images, metas)
        frozen = not any(p.requires_grad for p in self.parameters())
        grad_context = torch.no_grad() if frozen else contextlib.nullcontext()
        use_amp = self.use_bf16 and images.device.type == 'cuda'

        if use_amp and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Set use_bf16=False on GPUs without BF16 support')

        amp_context = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if use_amp else contextlib.nullcontext()
        )

        with grad_context, amp_context:
            outputs, start = self.aggregator(inputs, mode='full')
            tokens = self.feature_norm(outputs[-1][:, :, start:].float())
        eh, ew = inputs['context_image'].shape[-2:]
        hp, wp = eh // self.patch_size, ew // self.patch_size
        if tokens.shape[2] != hp * wp:
            raise RuntimeError('Patch token count does not match encoder image grid')
        feature = tokens.float().reshape(images.shape[0], 6, hp, wp,
                                         2 * self.embed_dim).permute(0, 1, 4, 2, 3)
        # Restore detection camera order without mixing batch elements.
        return torch.stack([f[order] for f, order in zip(feature, inverse_orders)])
