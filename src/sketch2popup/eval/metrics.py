"""분할 평가 지표: 마스크 IoU, 경계 F1."""
import numpy as np
from scipy import ndimage


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return mask ^ ndimage.binary_erosion(mask)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    """tol 픽셀 이내에 상대 경계가 있으면 맞춘 것으로 본다."""
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dist_to_gt = ndimage.distance_transform_edt(~bg)
    dist_to_pred = ndimage.distance_transform_edt(~bp)
    precision = (dist_to_gt[bp] <= tol).mean()
    recall = (dist_to_pred[bg] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
