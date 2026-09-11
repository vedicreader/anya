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

## Edge-aware refinement (optional, `refine.py`)

The masks live on the latent grid, so their boundaries are soft. `refine=` snaps each mask to image
edges, filling the slot the paper gives CascadePSP, training-free. `guided` runs an edge-preserving
guided filter (fast, safe); `grabcut` runs GrabCut on each mask's bbox crop (crisper instance
boundaries). Refinement is parallel across masks (`cv2` releases the GIL), ~2s for ~120 masks.

```python
seg.levels("img.jpg", refine="grabcut")     # or refine="guided"
```

## High resolution

The self-attention grab is query-tiled (`attn._Grab`), so `N=19600` at the paper's 1120px fits in
~12GB RAM on CPU without materializing the full `heads x N x N` score tensor. `--size 1120` runs the
paper's productive resolution; 512-768 is the fast CPU default.

## Departures from the paper

- Default `size=512` (latent 64x64) for speed; `--size 1120` reaches the paper's resolution.
- `p`-Laplacian sharpening is done by a top-`k` sparse affinity (`k=48`) rather than the paper's
  `tau_att` threshold, whose formula the paper does not give. Same effect: suppress weak edges, keep
  the graph connected.
- Refinement is guided-filter/GrabCut instead of the learned CascadePSP. No Step-3 detector training
  (needs GPUs); the overlays are the raw generator output, optionally refined.

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

## Comparison vs DINO (`compare_dino.py`)

Swap the affinity source, keep the pipeline: `dino_affinity` builds a cosine-similarity graph over
DINO patch tokens and runs the same `sparsify -> propagate -> merge -> NMS`, so any difference is the
backbone.

DINOv3 could not be loaded here: the official weights are license-gated on HF (401), and the one
accessible mirror (`timm/vit_base_patch16_dinov3.lvd_1689m`) needs a `torchvision` whose compiled ops
are ABI-incompatible with this container's `torch 2.14.0+cpu` (importing torchvision raises
`operator torchvision::nms does not exist`). So the comparison uses **DINOv2** (same Meta
self-supervised lineage, ungated, loads via `transformers` with no torchvision).

At matched settings (same 64x64 grid, same cut height), SD2 self-attention (Diffuse2Seg) produces
cleaner object-coherent masks; DINOv2 patch-cosine over-fragments: 213 vs 136 masks on the
strawberries and 134 vs 66 on the coffee scene at the same height, and DINOv2's affinity is so
uniformly high that even its coarsest cut cannot merge below ~189 clusters. This matches the paper's
claim that diffusion self-attention already encodes object structure. Caveats: this is DINOv2 not v3,
and raw cosine-affinity is not DINO's intended segmentation recipe (it is usually paired with a head
or spectral method), so read it as a controlled backbone swap, not a ceiling on DINO.

```bash
python compare_dino.py wild/picsum_1080.jpg --h 2.06   # -> outputs/compare_dino.png
```

## Files

| file | role |
|---|---|
| `diffuse2seg/attn.py` | SD2 self-attention extraction via single-step denoising |
| `diffuse2seg/propagate.py` | top-k sparse affinity + p-Laplacian Gauss-Jacobi propagation |
| `diffuse2seg/merge.py` | KL clustering, multi-granular partitioning, area-NMS |
| `diffuse2seg/refine.py` | optional guided-filter / GrabCut edge-aware refinement |
| `compare_dino.py` | DINOv2 affinity through the same pipeline, backbone comparison |
| `diffuse2seg/generate.py` | end-to-end `Diffuse2Seg` |
| `diffuse2seg/viz.py` | mask overlays and level panels |
| `run.py` | CLI over image globs |
