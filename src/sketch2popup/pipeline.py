"""config 하나로 A~D 실험을 조립한다."""
from sketch2popup.preprocess.illumination import correct_illumination
from sketch2popup.segment.baseline import segment
from sketch2popup.segment.postprocess import refine


def run(img, bbox, cfg):
    x = correct_illumination(img, cfg["illumination"]) if cfg["illumination"]["enabled"] else img
    mask = segment(x, bbox, cfg["segment"])
    if cfg["postprocess"]["enabled"]:
        mask = refine(mask, x, cfg["postprocess"])
    return mask
