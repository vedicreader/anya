"""Diffuse2Seg pipeline: image -> multi-granular instance masks, no supervision."""
from PIL import Image
from .attn import load_pipe, extract_affinity
from .propagate import grid_prompts, propagate, sparsify
from .merge import merge_masks, merge_levels, area_nms


class Diffuse2Seg:
    "Training-free mask generator over a frozen SD2 UNet's self-attention."
    def __init__(self, device="cpu", model=None):
        from .attn import SD2
        self.pipe = load_pipe(model or SD2, device)

    def generate(self, img, size=512, t=100, w=(0.85, 0.15), k=48, s=6, p=1.6, lam=1e-5,
                 tau_prop=1e-4, max_iter=200, L=6, hmin=0.186, hmax=2.99, a_min=100,
                 tau_iou=0.9, n_max=1000, out_size=None):
        "Return the pooled multi-granularity instance masks (paper's candidate set)."
        H = W = out_size or size
        f, (Hf, Wf) = self._soft_maps(img, size, t, w, k, s, p, lam, tau_prop, max_iter)
        pool = merge_masks(f, Hf, Wf, H, W, L, hmin, hmax, a_min)
        return area_nms(pool, tau_iou, n_max)

    def levels(self, img, size=512, t=100, w=(0.85, 0.15), k=48, s=6, p=1.6, lam=1e-5,
               tau_prop=1e-4, max_iter=200, L=6, hmin=0.186, hmax=2.99, a_min=200,
               tau_iou=0.9, out_size=None):
        "Return {h: masks} at each granularity, NMS-deduplicated within each level."
        H = W = out_size or size
        f, (Hf, Wf) = self._soft_maps(img, size, t, w, k, s, p, lam, tau_prop, max_iter)
        lv = merge_levels(f, Hf, Wf, H, W, L, hmin, hmax, a_min)
        return {h: area_nms(m, tau_iou) for h, m in lv.items()}

    def _soft_maps(self, img, size, t, w, k, s, p, lam, tau_prop, max_iter):
        "Extract affinity, sparsify, and p-Laplacian-propagate the grid prompts."
        if isinstance(img, str): img = Image.open(img)
        A, (Hf, Wf) = extract_affinity(self.pipe, img, size, t, w)
        f = propagate(sparsify(A, k), grid_prompts(Hf, Wf, s), p, lam, tau_prop, max_iter)
        return f, (Hf, Wf)
