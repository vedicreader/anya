"""Optional edge-aware mask refinement (the paper's CascadePSP slot, training-free)."""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import label as cc_label

try:
    import cv2
except ImportError:                                              # pragma: no cover
    cv2 = None


def _need_cv2():
    if cv2 is None: raise ImportError("refinement needs opencv: pip install opencv-contrib-python-headless")


def _largest_cc(mask):
    "Keep only the largest connected component (drops refinement spray)."
    lab = cc_label(mask)
    if lab.max() == 0: return mask
    return lab == (1 + np.bincount(lab.ravel())[1:].argmax())


def guided_refine(rgb, mask, radius=12, eps=1e-3):
    "Snap a mask to image edges with a guided filter, re-threshold, keep the largest CC."
    _need_cv2()
    g = cv2.ximgproc.guidedFilter(rgb, mask.astype(np.float32), radius, eps * 255 ** 2)
    return _largest_cc(g > 0.5)


def grabcut_refine(rgb, mask, pad=12, iters=3, min_area=64):
    "Refine a mask by GrabCut on its padded bbox crop, seeded by the mask as probable-FG."
    _need_cv2()
    if mask.sum() < min_area: return mask
    ys, xs = np.where(mask); H, W = mask.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    sub = rgb[y0:y1, x0:x1]; m = mask[y0:y1, x0:x1]
    gc = np.where(m, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[:pad // 2] = gc[-pad // 2:] = gc[:, :pad // 2] = gc[:, -pad // 2:] = cv2.GC_BGD  # frame is background
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try: cv2.grabCut(np.ascontiguousarray(sub), gc, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error: return mask
    out = np.zeros_like(mask); out[y0:y1, x0:x1] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    ref = _largest_cc(out)
    return ref if ref.sum() >= min_area else mask


def refine_masks(img, masks, method="guided", workers=4, **kw):
    "Refine every mask in parallel (cv2 releases the GIL); returns refined masks, order kept."
    if not masks: return masks
    from PIL import Image
    sz = masks[0].shape[0]
    rgb = np.asarray(img.convert("RGB").resize((sz, sz), Image.BICUBIC))
    fn = {"guided": guided_refine, "grabcut": grabcut_refine}[method]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda m: fn(rgb, m, **kw), masks))
