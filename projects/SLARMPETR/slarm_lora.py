"""Attention LoRA, following OGC-Det's vggt_lora implementation.

Original weight/bias names are preserved for SLARM checkpoint compatibility.
Adapters are constructed before the optimizer, never during forward/init_weights.
"""
import math

from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout=0.0):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        factory = dict(device=self.weight.device, dtype=self.weight.dtype)
        self.lora_a = nn.Linear(self.in_features, rank, bias=False, **factory)
        self.lora_b = nn.Linear(rank, self.out_features, bias=False, **factory)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        return (F.linear(inputs, self.weight, self.bias)
                + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling)


DEFAULT_LORA_CONFIG = dict(
    enabled=False, block_indices=(), branches=('frame_blocks', 'global_blocks'),
    target_modules=('attn.qkv', 'attn.proj'), rank=8, alpha=16,
    dropout=0.0, train_affine_tokens=True, train_feature_norm=False,
    gradient_checkpointing=True, checkpoint_start_block=0)


def configure_slarm_lora(aggregator, config):
    """Validate before mutation, then inject configured adapters."""
    values = dict(DEFAULT_LORA_CONFIG)
    unknown = set(config or {}) - set(values)
    if unknown:
        raise ValueError(f'Unknown lora_cfg keys: {sorted(unknown)}')
    values.update(config or {})
    for name in ('enabled', 'train_affine_tokens', 'train_feature_norm',
                 'gradient_checkpointing'):
        if not isinstance(values[name], bool):
            raise TypeError(f'{name} must be boolean')
    if not values['enabled']:
        return values, ()
    rank = values['rank']
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError('rank must be a positive integer')
    for name in ('alpha', 'dropout'):
        value = values[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError(f'{name} must be a finite number')
    if values['alpha'] <= 0 or not 0 <= values['dropout'] < 1:
        raise ValueError('alpha must be positive and dropout must be in [0, 1)')
    depth = aggregator.depth
    start = values['checkpoint_start_block']
    if (not isinstance(start, int) or isinstance(start, bool)
            or not 0 <= start < depth):
        raise ValueError('checkpoint_start_block must be in [0, depth)')
    for name in ('block_indices', 'branches', 'target_modules'):
        sequence = values[name]
        if not isinstance(sequence, (tuple, list)) or not sequence:
            raise ValueError(f'{name} must be a nonempty list or tuple')
        if len(set(sequence)) != len(sequence):
            raise ValueError(f'{name} contains duplicates')
        values[name] = tuple(sequence)
    if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < depth
           for i in values['block_indices']):
        raise ValueError('block_indices must contain integers in [0, depth)')
    if not set(values['branches']).issubset(('frame_blocks', 'global_blocks')):
        raise ValueError('branches must be frame_blocks and/or global_blocks')
    if not set(values['target_modules']).issubset(('attn.qkv', 'attn.proj')):
        raise ValueError('target_modules must be attn.qkv and/or attn.proj')
    if values['train_affine_tokens'] and not hasattr(aggregator, 'affine_token'):
        raise ValueError('train_affine_tokens requires use_affine_token=True')
    selected = []
    for branch in values['branches']:
        for index in values['block_indices']:
            for target in values['target_modules']:
                parent = getattr(aggregator, branch)[index].attn
                attr = target.split('.')[1]
                layer = getattr(parent, attr)
                if not isinstance(layer, nn.Linear):
                    raise TypeError(f'{branch}.{index}.{target} must be nn.Linear')
                selected.append((parent, attr, layer, f'{branch}.{index}.{target}'))
    for parent, attr, layer, _ in selected:
        setattr(parent, attr, LoRALinear(
            layer, rank, values['alpha'], values['dropout']))
    aggregator.grad_checkpointing = values['gradient_checkpointing']
    aggregator.checkpoint_start_block = start
    return values, tuple(item[3] for item in selected)


def lora_parameter_name(name):
    return '.lora_a.' in name or '.lora_b.' in name


def enable_lora_parameters(aggregator, train_affine_tokens):
    aggregator.requires_grad_(False)
    for module in aggregator.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    if train_affine_tokens:
        aggregator.affine_token.requires_grad_(True)
