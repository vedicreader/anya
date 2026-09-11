"""Overlay generated masks on the source image."""
import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries


def overlay(img, masks, alpha=0.5, seed=0):
    "Colour each mask, blend over the image (resized to mask res), draw white boundaries."
    sz = masks[0].shape[0] if masks else img.size[0]
    rng = np.random.default_rng(seed)
    out = np.asarray(img.convert("RGB").resize((sz, sz), Image.BICUBIC), np.float32)
    for m in masks:
        c = rng.integers(60, 256, 3).astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
    for m in masks: out[find_boundaries(m, mode="outer")] = (255, 255, 255)
    return Image.fromarray(out.astype(np.uint8))


def side_by_side(img, masks, tile=512):
    "Input image next to its overlay."
    return _row([img.convert("RGB"), overlay(img, masks)], ["input", f"{len(masks)} masks"], tile)


def levels_panel(img, levels, tile=320):
    "Input plus one overlay per granularity level, coarse to fine, labelled by mask count."
    items = sorted(levels.items(), reverse=True)                 # large h (coarse) first
    tiles = [img.convert("RGB")] + [overlay(img, m) for _, m in items]
    labels = ["input"] + [f"h={h:.2f}  {len(m)} masks" for h, m in items]
    return _row(tiles, labels, tile)


def instance_map(img, masks, area=(0.003, 0.12), alpha=0.55, seed=1):
    "Paint pool masks whose area fraction is in `area` as distinct instances, large first."
    sz = masks[0].shape[0] if masks else img.size[0]
    tot = sz * sz
    sel = [m for m in masks if area[0] * tot <= m.sum() <= area[1] * tot]
    sel.sort(key=lambda m: -m.sum())                            # large first, small painted on top
    return overlay(img, sel, alpha=alpha, seed=seed), len(sel)


def _row(tiles, labels, tile, pad=8, bar=22):
    "Thumbnail tiles to `tile` px, lay out horizontally with a caption bar under each."
    tiles = [t.resize((tile, tile), Image.BICUBIC) for t in tiles]
    n = len(tiles); W = n * tile + (n - 1) * pad
    canvas = Image.new("RGB", (W, tile + bar), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x = i * (tile + pad)
        canvas.paste(t, (x, 0)); d.text((x + 4, tile + 4), lab, fill=(0, 0, 0))
    return canvas
