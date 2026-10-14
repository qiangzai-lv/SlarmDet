import torch

from mmdet3d.registry import MODELS
from projects.PETR.petr.petr_head import PETRHead


@MODELS.register_module()
class SLARMPETRHead(PETRHead):
    """Original PETR head with an empty-3D-target compatibility guard."""

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes,
                           gt_bboxes_ignore=None):
        if gt_bboxes.shape[0]:
            return super()._get_target_single(cls_score, bbox_pred, gt_labels,
                                              gt_bboxes, gt_bboxes_ignore)
        assert gt_bboxes_ignore is None
        count = bbox_pred.shape[0]
        labels = bbox_pred.new_full((count,), self.num_classes, dtype=torch.long)
        weights = bbox_pred.new_ones(count)
        targets = bbox_pred.new_zeros((count, gt_bboxes.shape[-1]))
        box_weights = torch.zeros_like(bbox_pred)
        positives = bbox_pred.new_empty((0,), dtype=torch.long)
        negatives = torch.arange(count, device=bbox_pred.device)
        return labels, weights, targets, box_weights, positives, negatives
