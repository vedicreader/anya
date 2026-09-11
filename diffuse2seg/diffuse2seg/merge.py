"""Step 2b: merge soft maps into multi-granular masks (Eq. 3 KL clustering + NMS)."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from skimage.transform import resize
from skimage.measure import label as cc_label


def kl_distance(f):
    "Symmetric-KL distance matrix (K x K) over prompt maps normalized to distributions."
    p = f / (f.sum(0, keepdims=True) + 1e-12)                     # N x K prob columns
    lp = np.log(p + 1e-12)
    cross = -(p.T @ lp)                                           # cross[k,k'] = -sum_i p_k log p_k'
    ent = np.diag(cross)
    d = 0.5 * ((cross - ent[:, None]) + (cross - ent[None, :]))   # symmetric KL
    d = np.clip(d, 0, None); np.fill_diagonal(d, 0.0)
    return 0.5 * (d + d.T)


def level_thresholds(L=6, hmin=0.186, hmax=2.99):
    "L log-spaced dendrogram cut heights, small h = fine, large h = coarse."
    return np.logspace(np.log10(hmin), np.log10(hmax), L)


def masks_at(f, Z, up, h, a_min):
    "Instance masks at one dendrogram cut h: cluster-average, argmax partition, split into CCs."
    cl = fcluster(Z, t=h, criterion="distance")
    cmaps = np.stack([up[cl == c].mean(0) for c in np.unique(cl)])   # C x H x W
    seg = cmaps.argmax(0)
    return [inst for c in range(cmaps.shape[0]) for inst in _components(seg == c, a_min)]


def _upsampled(f, Hf, Wf, H, W):
    "Normalize prompt maps to distributions and bilinearly upsample to image size."
    p = f / (f.sum(0, keepdims=True) + 1e-12)
    return resize(p.T.reshape(-1, Hf, Wf), (f.shape[1], H, W), order=1, mode="edge", anti_aliasing=False)


def merge_levels(f, Hf, Wf, H, W, L=6, hmin=0.186, hmax=2.99, a_min=100):
    "Return {h: instance masks} at each granularity level."
    Z = linkage(squareform(kl_distance(f), checks=False), method="average")
    up = _upsampled(f, Hf, Wf, H, W)
    return {float(h): masks_at(f, Z, up, h, a_min) for h in level_thresholds(L, hmin, hmax)}


def merge_masks(f, Hf, Wf, H, W, L=6, hmin=0.186, hmax=2.99, a_min=100):
    "Pool multi-granularity instance masks across all L levels (paper's candidate set)."
    return [m for masks in merge_levels(f, Hf, Wf, H, W, L, hmin, hmax, a_min).values() for m in masks]


def _components(mask, a_min):
    "Connected components of a boolean mask, dropping specks below a_min pixels."
    lab = cc_label(mask)
    return [lab == i for i in range(1, lab.max() + 1) if (lab == i).sum() >= a_min]


def area_nms(masks, tau_iou=0.9, n_max=1000, ds=4):
    "Area-descending NMS; IoU tested on ds-downsampled flattened masks, full-res masks returned."
    if not masks: return []
    V = np.stack([m[::ds, ::ds].reshape(-1) for m in masks]).astype(np.float32)  # n x d
    areas = V.sum(1)
    order = np.argsort(-areas)
    keep, kept_V, kept_a = [], [], []
    for i in order:
        if len(keep) >= n_max: break
        v, a = V[i], areas[i]
        if kept_V:
            inter = np.asarray(kept_V) @ v
            iou = inter / (np.array(kept_a) + a - inter + 1e-9)
            if (iou > tau_iou).any(): continue
        keep.append(masks[i]); kept_V.append(v); kept_a.append(a)
    return keep
