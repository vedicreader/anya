"""Run Diffuse2Seg on images; save a multi-granularity panel per image. CPU-friendly."""
import argparse, glob, os, time
from PIL import Image
from diffuse2seg import Diffuse2Seg, levels_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image files or globs")
    ap.add_argument("--size", type=int, default=512, help="working resolution (paper uses 1120)")
    ap.add_argument("--t", type=int, default=100, help="denoise timestep for attention")
    ap.add_argument("--s", type=int, default=6, help="prompt grid spacing on latent")
    ap.add_argument("--k", type=int, default=48, help="top-k affinity neighbours per token")
    ap.add_argument("--a_min", type=int, default=300, help="min mask area in pixels")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    paths = [p for g in a.images for p in glob.glob(g)]
    os.makedirs(a.out, exist_ok=True)
    seg = Diffuse2Seg()
    for path in paths:
        t = time.time()
        lv = seg.levels(path, size=a.size, t=a.t, s=a.s, k=a.k, a_min=a.a_min)
        name = os.path.splitext(os.path.basename(path))[0]
        levels_panel(Image.open(path), lv, tile=a.size // 2 + 96).save(
            os.path.join(a.out, name + "_levels.png"))
        n = sum(len(m) for m in lv.values())
        print(f"{os.path.basename(path)}: {n} masks over {len(lv)} levels in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
