"""Compare Diffuse2Seg (SD2 attention) vs DINOv2 features through the identical pipeline.

DINOv3 weights are license-gated on HF and the timm path needs a torchvision build that is
ABI-incompatible with this container's torch, so DINOv2 (same DINO lineage) stands in. Only the
affinity source changes; sparsify -> p-Laplacian propagate -> KL-merge -> NMS is shared, so the
comparison isolates the backbone.
"""
import argparse, numpy as np, torch
from PIL import Image
from diffuse2seg import Diffuse2Seg
from diffuse2seg.propagate import grid_prompts, propagate, sparsify
from diffuse2seg.merge import merge_levels, area_nms
from diffuse2seg.viz import overlay, _row

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def dino_affinity(img, size=896, model="facebook/dinov2-base"):
    "Cosine-similarity affinity over DINOv2 patch tokens; grid is size/14 per side."
    from transformers import Dinov2Model
    net = Dinov2Model.from_pretrained(model).eval()
    ps = net.config.patch_size; g = size // ps; size = g * ps
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), np.float32) / 255
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        out = net(torch.from_numpy(x), interpolate_pos_encoding=True).last_hidden_state[0]
    feat = out[1:1 + g * g]                                       # drop CLS, keep patch tokens
    feat = torch.nn.functional.normalize(feat, dim=-1)
    A = (feat @ feat.T).clamp(min=0).numpy().astype(np.float32)   # cosine, negatives -> 0
    return A, (g, g)


def dino_levels(img, size=896, k=48, s=6, a_min=300, L=10, hmin=0.2, hmax=4.0, tau_iou=0.9, out=512):
    "Run the shared pipeline on a DINOv2 affinity."
    A, (Hf, Wf) = dino_affinity(img, size)
    f = propagate(sparsify(A, k), grid_prompts(Hf, Wf, s), p=1.6, max_iter=200)
    lv = merge_levels(f, Hf, Wf, out, out, L, hmin, hmax, a_min)
    return {h: area_nms(m, tau_iou) for h, m in lv.items()}


def pick(levels, target):
    "Level whose cut height is nearest target."
    h = min(levels, key=lambda x: abs(x - target)); return h, levels[h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="outputs/compare_dino.png")
    ap.add_argument("--h", type=float, default=2.06, help="object-level cut to display")
    a = ap.parse_args()
    img = Image.open(a.image)
    sd2 = Diffuse2Seg().levels(a.image, size=512, s=6, k=48, a_min=300, L=10, hmin=0.2, hmax=4.0, out_size=512)
    dino = dino_levels(img, size=896, out=512)
    hs, ms = pick(sd2, a.h); hd, md = pick(dino, a.h)
    _row([img.convert("RGB"), overlay(img, ms), overlay(img, md)],
         ["input", f"Diffuse2Seg SD2  h={hs:.2f}  {len(ms)} masks",
          f"DINOv2  h={hd:.2f}  {len(md)} masks"], tile=520).save(a.out)
    print("saved", a.out, "sd2", len(ms), "dino", len(md))


if __name__ == "__main__":
    main()
