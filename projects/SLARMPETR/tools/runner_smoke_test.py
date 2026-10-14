"""Two MMEngine training iterations and a two-image format-only validation.

Run with CUDA_VISIBLE_DEVICES='' for a bounded CPU configuration check.
Artifacts/checkpoints are temporary; no official AP is reported for a subset.
"""
import json
from pathlib import Path
import tempfile

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[3]
    cfg = Config.fromfile(root/'projects/SLARMPETR/configs/slarm_petr_singleframe.py')
    cfg.model.img_backbone.update(embed_dim=32, num_heads=4,
                                  encoder_size=(32,64), use_bf16=False)
    cfg.model.img_neck.in_channels = 64
    cfg.model.pts_bbox_head.num_query = 20
    cfg.model.pts_bbox_head.bbox_coder.max_num = 20
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=2, val_interval=2)
    cfg.param_scheduler = [dict(type='LinearLR', start_factor=1.,
                                by_epoch=False, begin=0, end=2)]
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        cfg[key].num_workers = 0
        cfg[key].persistent_workers = False
        cfg[key].dataset.indices = [0, 1]
    cfg.default_hooks.checkpoint.update(interval=-1, save_last=False)
    cfg.log_processor.by_epoch = False
    cfg.log_level = 'ERROR'
    cfg.launcher = 'none'
    with tempfile.TemporaryDirectory() as directory:
        cfg.work_dir = directory
        cfg.val_evaluator.update(format_only=True, jsonfile_prefix=directory+'/results')
        runner = Runner.from_cfg(cfg)
        runner.train()
        result = json.loads(Path(directory+'/results/pred_instances_3d/results_nusc.json').read_text())
        assert runner.iter == 2
        assert len(result['results']) == 2
        assert result['meta']['use_camera'] is True
        print(json.dumps(dict(mmengine_training_steps=runner.iter,
                              formatted_validation_samples=len(result['results']),
                              official_ap_evaluated=False), indent=2))


if __name__ == '__main__':
    main()
