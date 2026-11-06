import numpy as np
from sketch2popup.eval.metrics import iou, boundary_f1


def square(x0, y0, s, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_identical():
    m = square(10, 10, 20)
    assert iou(m, m) == 1.0
    assert boundary_f1(m, m) == 1.0


def test_disjoint():
    assert iou(square(0, 0, 10), square(40, 40, 10)) == 0.0


def test_small_shift_within_tolerance():
    assert boundary_f1(square(10, 10, 20), square(11, 10, 20), tol=2) == 1.0
