import numpy as np
import torch

from mmdet3d.registry import MODELS
from projects.PETR.petr.petr import PETR


@MODELS.register_module()
class SLARMPETR(PETR):
    """PETR detection losses/decoder with a calibrated SLARM image encoder."""

    def __init__(self, **kwargs):
        if kwargs.get('use_grid_mask', False):
            raise ValueError('Initial SLARMPETR integration disables GridMask')
        super().__init__(**kwargs)

    def add_lidar2img(self, img, batch_input_metas):
        # Unlike original PETR, preserve batch/view axes and all shape metadata.
        b, v, _, h, w = img.shape
        if len(batch_input_metas) != b or v != 6:
            raise ValueError('Expected one metadata dict per six-camera frame')
        for meta in batch_input_metas:
            shape = meta['img_shape']
            if len(shape) == v and isinstance(shape[0], (tuple, list)):
                meta['img_shape'] = [tuple(s[:2]) for s in shape]
            else:
                meta['img_shape'] = [tuple(shape[:2])] * v
            projections = []
            for k, e in zip(meta['cam2img'], meta['lidar2cam']):
                k = np.asarray(k)
                padded = np.eye(4)
                padded[:k.shape[0], :k.shape[1]] = k
                projections.append(padded @ np.asarray(e))
            meta['lidar2img'] = projections
            meta['input_shape'] = (h, w)
        return batch_input_metas

    def extract_img_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img)
        feature = self.img_backbone(img, img_metas)
        return self.img_neck(feature)

    def _forward(self, inputs, data_samples=None, **kwargs):
        metas = [sample.metainfo for sample in data_samples]
        self.add_lidar2img(inputs['imgs'], metas)
        features = self.extract_img_feat(inputs['imgs'], metas)
        return self.pts_bbox_head(features, metas)
