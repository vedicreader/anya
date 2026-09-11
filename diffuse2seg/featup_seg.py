"""Standalone segmentation on FeatUp-upsampled features. No Diffuse2Seg pipeline.

Tests whether FeatUp's high-resolution features segment on their own: upsample a backbone's feature map
with FeatUp, then cluster the per-pixel feature vectors directly (KMeans) into regions. Runs in a
torch<=2.4 venv because FeatUp needs a torchvision the main venv's torch cannot load.
"""
import argparse, numpy as np, torch
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label as cc_label

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(path, size=224):
    "Resize to a FeatUp-friendly square and normalize to ImageNet stats."
    from torchvision import transforms as T
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return img, tf(img)[None]


CKPT = "/home/user/.cache/featup/dinov2_jbu_stack_cocostuff.ckpt"


def _curl_downloads():
    "Route torch.hub downloads through curl and skip its GitHub-API validation (proxy rejects urllib)."
    import subprocess
    def dl(url, dst, hash_prefix=None, progress=True):
        subprocess.run(["curl", "-sSL", "-o", dst, url], check=True)
    torch.hub.download_url_to_file = dl                           # weights via curl
    _orig = torch.hub.load
    def load(repo, model, *a, **k):                               # dinov2 from a local clone
        if repo == "facebookresearch/dinov2":
            for key in ("force_reload", "trust_repo", "skip_validation"): k.pop(key, None)
            return _orig("/home/user/dinov2_repo", model, *a, source="local", **k)
        return _orig(repo, model, *a, **k)
    torch.hub.load = load


def load_featup(model="dinov2", use_norm=True, ckpt=CKPT):
    "Build FeatUp's UpsampledBackbone and load the pre-fetched JBU checkpoint from disk."
    _curl_downloads()
    import sys; sys.path.insert(0, "/home/user/FeatUp")
    from hubconf import UpsampledBackbone
    up = UpsampledBackbone(model, use_norm)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k: v for k, v in sd.items() if "scale_net" not in k and "downsampler" not in k}
    up.load_state_dict(sd, strict=False)
    return up.eval()


def featup_hr(x, model="dinov2", use_norm=True):
    "Return FeatUp high-res features (1,C,H,W) and the raw low-res features for comparison."
    up = load_featup(model, use_norm)
    with torch.no_grad():
        hr = up(x)                                               # JBU-upsampled features
        lr = up.model(x)                                         # backbone low-res features
    return hr, lr


def segment(feats, k=25, min_area=80):
    "KMeans over per-pixel features, then split label regions into connected components."
    C, H, W = feats.shape[1:]
    X = feats[0].reshape(C, H * W).T.numpy()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(X).reshape(H, W)
    masks = []
    for c in range(k):
        cc = cc_label(lab == c)
        masks += [cc == i for i in range(1, cc.max() + 1) if (cc == i).sum() >= min_area]
    return masks, (H, W)


def overlay(img, masks, hw, alpha=0.5, seed=0):
    "Colour each region over the resized image with white boundaries."
    from skimage.segmentation import find_boundaries
    H, W = hw; rng = np.random.default_rng(seed)
    base = np.asarray(img.resize((W, H), Image.BICUBIC), np.float32)
    for m in masks:
        base[m] = (1 - alpha) * base[m] + alpha * rng.integers(60, 256, 3)
    for m in masks: base[find_boundaries(m, mode="outer")] = (255, 255, 255)
    return Image.fromarray(base.astype(np.uint8))


def row(tiles, labels, tile=460):
    from PIL import ImageDraw
    tiles = [t.resize((tile, tile), Image.BICUBIC) for t in tiles]
    n = len(tiles); c = Image.new("RGB", (n * tile + (n - 1) * 8, tile + 22), "white")
    d = ImageDraw.Draw(c)
    for i, (t, l) in enumerate(zip(tiles, labels)):
        c.paste(t, (i * (tile + 8), 0)); d.text((i * (tile + 8) + 4, tile + 4), l, fill="black")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--model", default="dinov2")
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", default="outputs/featup_seg.png")
    a = ap.parse_args()
    img, x = load_image(a.image, a.size)
    hr, lr = featup_hr(x, a.model)
    print("lr feats", tuple(lr.shape), "-> featup hr", tuple(hr.shape))
    mh, hwh = segment(hr, a.k); ml, hwl = segment(lr, a.k)
    row([img, overlay(img, ml, hwl), overlay(img, mh, hwh)],
        ["input", f"{a.model} low-res  {len(ml)} regions", f"FeatUp {a.model} hi-res  {len(mh)} regions"]
        ).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
