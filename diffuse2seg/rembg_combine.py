"""Diffuse2Seg x rembg: foreground-gated instances with a matte-quality subject contour."""
import argparse, numpy as np
from PIL import Image
from diffuse2seg import Diffuse2Seg
from diffuse2seg.viz import overlay, _row
from diffuse2seg.rembg_seg import foreground_alpha, gate_masks, alpha_contour


def pick(levels, target):
    h = min(levels, key=lambda x: abs(x - target)); return levels[h]


def draw_contour(img, contour, size, color=(255, 255, 0)):
    "Input image with the rembg foreground outline overlaid."
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC)).copy()
    a[contour] = color
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--h", type=float, default=1.72, help="object-level cut to display")
    ap.add_argument("--out", default="outputs/rembg_combine.png")
    a = ap.parse_args()
    img = Image.open(a.image)
    alpha = foreground_alpha(img, a.size)
    masks = pick(Diffuse2Seg().levels(a.image, size=a.size, a_min=300, L=10, hmin=0.2, hmax=4.0,
                                      out_size=a.size), a.h)
    gated = gate_masks(masks, alpha)
    contour = alpha_contour(alpha)
    fg = (np.stack([alpha] * 3, -1) * 255).astype(np.uint8)
    _row([draw_contour(img, contour, a.size), Image.fromarray(fg),
          overlay(img, masks), overlay(img, gated)],
         ["input + rembg contour", "rembg foreground alpha",
          f"Diffuse2Seg  {len(masks)} masks", f"foreground-gated  {len(gated)} masks"],
         tile=430).save(a.out)
    print("saved", a.out, "masks", len(masks), "-> gated", len(gated))


if __name__ == "__main__":
    main()
