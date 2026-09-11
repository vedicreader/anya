# Diffuse2Seg (training-free reimplementation)

Reimplements the label generator of [Diffuse2Seg: Diffusion Models Can Segment Anything Without
Supervision](https://arxiv.org/abs/2609.06491) (arXiv:2609.06491). A frozen text-to-image diffusion
model segments arbitrary objects at multiple granularities. Nothing is trained: the masks come out of
the pretrained UNet's self-attention.

## What this is

The paper has three steps. Steps 1-2 turn a pretrained diffusion model into an unsupervised
multi-granular mask generator. Step 3 trains a Mask2Former detector on those masks. This repo
implements Steps 1-2 (the novel, training-free part) and runs them on wild images. No detector is
trained; the overlays are the raw output of the frozen diffusion model.

The paper uses the gated `stabilityai/stable-diffusion-2-base`. That checkpoint now needs a license
token, so this uses the ungated `friedrichor/stable-diffusion-2-1-realistic`, whose UNet is the
identical SD2 architecture.

## Pipeline

1. **Self-attention extraction** (`attn.py`). VAE-encode the image, add noise at timestep `t`, run one
   UNet step with an empty prompt, and cache the self-attention of the first two attention blocks of
   the highest-resolution decoder block (`up_blocks[-1]`). Aggregate over heads and combine the two
   layers with weights `w=(0.85, 0.15)` into one `N x N` affinity `A` (Eq. 1).
2. **Prompt propagation** (`propagate.py`). Keep each token's top-`k` affinities (sparse, edge-aware
   graph), place an equidistant grid of one-hot seeds, and propagate each over `A` by non-linear
   `p`-Laplacian smoothing (Eq. 2) solved with Gauss-Jacobi iterations. Smoothing flows inside objects
   and stops at attention discontinuities (edges), giving soft object maps.
3. **Map merging** (`merge.py`). Normalize maps to distributions, build a symmetric-KL distance
   matrix (Eq. 3), average-link cluster the prompts, cut the dendrogram at `L` log-spaced heights,
   argmax each level into a partition, split into connected components, then area-descending NMS to a
   complementary mask set. Small heights isolate parts, large heights isolate whole objects.

## Departures from the paper

- Runs at `size=512` (latent 64x64) instead of `1120` for CPU feasibility. Everything is a flag.
- `p`-Laplacian sharpening is done by a top-`k` sparse affinity (`k=48`) rather than the paper's
  `tau_att` threshold, whose formula the paper does not give. Same effect: suppress weak edges, keep
  the graph connected.
- No CascadePSP refinement (the paper marks it optional) and no Step-3 detector training.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run.py 'wild/*.jpg' --size 512      # writes outputs/<name>_levels.png
```

```python
from PIL import Image
from diffuse2seg import Diffuse2Seg, levels_panel
seg = Diffuse2Seg()                        # loads frozen SD2.1, no training
levels = seg.levels("wild/picsum_1080.jpg")   # {cut height: [masks]}
levels_panel(Image.open("wild/picsum_1080.jpg"), levels).save("panel.png")
```

`seg.generate(...)` returns the pooled multi-granularity candidate set (the paper's up-to-1000 masks);
`seg.levels(...)` returns one clean partition per granularity, which is what the panels show.

## Files

| file | role |
|---|---|
| `diffuse2seg/attn.py` | SD2 self-attention extraction via single-step denoising |
| `diffuse2seg/propagate.py` | top-k sparse affinity + p-Laplacian Gauss-Jacobi propagation |
| `diffuse2seg/merge.py` | KL clustering, multi-granular partitioning, area-NMS |
| `diffuse2seg/generate.py` | end-to-end `Diffuse2Seg` |
| `diffuse2seg/viz.py` | mask overlays and level panels |
| `run.py` | CLI over image globs |
