from mmdet3d.registry import TRANSFORMS
from projects.PETR.petr.transforms_3d import ResizeCropFlipImage


@TRANSFORMS.register_module()
class SLARMResizeCropFlipImage(ResizeCropFlipImage):
    """Keep actual per-image shape and camera identity after fixed resizing."""

    def __init__(self, data_aug_conf, training=False):
        if training:
            raise ValueError('Initial integration uses deterministic image transforms')
        super().__init__(data_aug_conf=data_aug_conf, training=False)

    def transform(self, results):
        results = super().transform(results)
        results['img_shape'] = results['img'][0].shape[:2]
        results['pad_shape'] = results['img_shape']
        return results
