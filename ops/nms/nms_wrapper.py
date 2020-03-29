import numpy as np
import torch

from .gpu_nms import gpu_nms
from .cpu_nms import cpu_nms


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        if dets.is_cuda:
            device_id = dets.get_device()
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    if dets_np.shape[0] == 0:
        inds = []
    else:
        inds = (gpu_nms(dets_np, iou_thr, device_id=device_id)
                if device_id is not None else cpu_nms(dets_np, iou_thr))

    if is_tensor:
        inds = dets.new_tensor(inds, dtype=torch.long)
    else:
        inds = np.array(inds, dtype=np.int64)
    return dets[inds, :], inds

