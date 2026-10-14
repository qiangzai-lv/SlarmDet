"""Bounded real-data optimizer/evaluation check; no training checkpoints.

Single GPU: python .../smoke_test.py --device cuda
Two CPU ranks: torchrun --standalone --nproc_per_node=2 .../smoke_test.py --ddp
"""
import argparse
import json
import logging
import os
from pathlib import Path
import time

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.logging import MMLogger

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--small', action='store_true', help='Use the bounded test encoder on GPU too')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    rank = int(os.getenv('RANK', '0'))
    if args.ddp:
        if args.device != 'cpu':
            raise ValueError('DDP smoke test uses CPU/Gloo to avoid sharing GPU ranks')
        torch.distributed.init_process_group('gloo')
    try:
        register_all_modules()
        MMLogger.get_current_instance().setLevel(logging.ERROR)
        root = Path(__file__).resolve().parents[3]
        config_name = 'slarm_petr_singleframe_lora.py' if args.lora else 'slarm_petr_singleframe.py'
        cfg = Config.fromfile(root/'projects/SLARMPETR/configs'/config_name)
        if args.device == 'cpu' or args.small:
            cfg.model.img_backbone.update(embed_dim=32, num_heads=4,
                                          encoder_size=(32,64),
                                          use_bf16=args.device == 'cuda')
            cfg.model.img_neck.in_channels = 64
            cfg.model.pts_bbox_head.num_query = 20
            cfg.model.pts_bbox_head.bbox_coder.max_num = 20
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        torch.manual_seed(1)
        model = MODELS.build(cfg.model)
        model.init_weights()
        model.to(args.device).train()
        affine_before = model.img_backbone.aggregator.affine_token.detach().clone()
        frozen_before = {n: p.detach().clone() for n, p in model.img_backbone.named_parameters()
                         if not p.requires_grad} if args.lora else {}
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        wrapped = torch.nn.parallel.DistributedDataParallel(model) if args.ddp else model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=2e-4, weight_decay=.01)
        start = time.perf_counter()
        losses_seen = []
        for step in range(2 if args.ddp or args.lora else 1):
            batch = model.data_preprocessor(
                pseudo_collate([dataset[rank + step]]), training=True)
            optimizer.zero_grad(set_to_none=True)
            losses = wrapped(batch['inputs'], batch['data_samples'], mode='loss')
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise AssertionError('Non-finite detection loss')
            loss.backward()
            missing = [n for n,p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            if missing:
                raise AssertionError(f'Unused trainable parameters: {missing}')
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f'Non-finite gradient: {name}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()
            losses_seen.append(float(loss.detach()))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if args.lora:
            for name, parameter in model.img_backbone.named_parameters():
                if name in frozen_before:
                    torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)
            if torch.equal(affine_before, model.img_backbone.aggregator.affine_token):
                raise AssertionError('Affine tokens did not update')
            if args.ddp:
                for value in (model.img_backbone.aggregator.affine_token,
                              model.img_backbone.aggregator.global_blocks[-1].attn.qkv.lora_b.weight):
                    replicas = [torch.empty_like(value) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(replicas, value.detach())
                    for replica in replicas[1:]:
                        torch.testing.assert_close(replicas[0], replica, rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            predictions = model.predict(batch['inputs'], batch['data_samples'])
            features = model.extract_img_feat(
                batch['inputs']['imgs'], [s.metainfo for s in batch['data_samples']])
        report = dict(device=args.device, ddp=args.ddp, rank=rank, lora=args.lora,
                      optimizer_steps=len(losses_seen), losses=losses_seen,
                      feature_shape=list(features[0].shape),
                      prediction_count=len(predictions[0].pred_instances_3d.bboxes_3d),
                      train_step_seconds=elapsed,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      unused_trainable_parameters=0)
        if args.lora:
            report.update(affine_tokens_updated=True, frozen_backbone_unchanged=True,
                          lora_modules=len(model.img_backbone.lora_modules))
        if args.device == 'cuda':
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            report['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
        if rank == 0:
            print(json.dumps(report, indent=2))
            if args.output:
                Path(args.output).write_text(json.dumps(report, indent=2))
    finally:
        if args.ddp:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
