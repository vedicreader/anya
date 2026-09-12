"""Combine Diffuse2Seg with rembg: a learned foreground matte to gate masks and snap contours.

rembg does salient-object matting (U^2-Net): one crisp foreground alpha, not instances. Used here to
(1) drop Diffuse2Seg masks that sit in the background and (2) supply a matte-quality outer contour for
the subject. Best on images with a clear subject; on frame-filling scenes the foreground is ~all.
"""
import numpy as np
from PIL import Image

_SESSION = None


def foreground_alpha(img, size, model="u2net"):
    "rembg salient-object alpha in [0,1] at sizexsize."
    global _SESSION
    from rembg import remove, new_session
    if _SESSION is None: _SESSION = new_session(model)
    m = remove(img.convert("RGB").resize((size, size), Image.BICUBIC), session=_SESSION, only_mask=True)
    return np.asarray(m, np.float32) / 255.0


def gate_masks(masks, alpha, thr=0.5, cover=0.5):
    "Keep masks whose area is mostly inside the foreground (alpha>thr)."
    fg = alpha > thr
    return [m for m in masks if (m & fg).sum() / (m.sum() + 1e-9) >= cover]


def alpha_contour(alpha, thr=0.5):
    "Boolean outline of the foreground matte."
    from skimage.segmentation import find_boundaries
    return find_boundaries(alpha > thr, mode="outer")


def snap_to_alpha(mask, alpha, thr=0.5):
    "Clip a subject-level mask to the matte and keep its largest component."
    from skimage.measure import label
    m = mask & (alpha > thr)
    lab = label(m)
    if lab.max() == 0: return m
    return lab == (1 + np.bincount(lab.ravel())[1:].argmax())
