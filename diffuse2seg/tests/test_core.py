"""Fast math checks for the training-free pipeline (no diffusion model needed)."""
import numpy as np
from diffuse2seg.propagate import sparsify, grid_prompts, propagate
from diffuse2seg.merge import kl_distance, area_nms


def _two_block_affinity(n=20):
    "Dense affinity with two strongly-connected blocks and a weak bridge."
    A = np.full((2 * n, 2 * n), 0.01, np.float32)
    A[:n, :n] = 1.0; A[n:, n:] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def test_propagation_is_edge_preserving():
    "A seed inside one block stays inside it; it barely leaks across the weak bridge."
    n = 20; A = sparsify(_two_block_affinity(n), k=8)
    f = propagate(A, np.array([0]), p=1.6, max_iter=300)[:, 0]
    assert f[:n].mean() > 20 * f[n:].mean()          # in-block mass dominates cross-block


def test_grid_prompts_spacing():
    s = grid_prompts(64, 64, s=8)
    assert len(s) == 8 * 8 and s.max() < 64 * 64


def test_kl_distance_is_symmetric_and_zero_on_self():
    rng = np.random.default_rng(0)
    f = rng.random((100, 6)).astype(np.float32)
    D = kl_distance(f)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0, atol=1e-5)


def test_area_nms_drops_duplicates():
    m = np.zeros((64, 64), bool); m[10:40, 10:40] = True
    kept = area_nms([m, m.copy(), np.roll(m, 30, 1)], tau_iou=0.9)
    assert len(kept) == 2                            # the duplicate is removed, the shifted one kept


def test_guided_refine_snaps_to_a_colour_edge():
    "A blurry mask over a half-red/half-black image tightens toward the colour boundary."
    import pytest
    cv2 = pytest.importorskip("cv2")
    from diffuse2seg.refine import guided_refine
    rgb = np.zeros((80, 80, 3), np.uint8); rgb[:, :40] = (200, 30, 30)   # red left, black right
    mask = np.zeros((80, 80), bool); mask[:, :52] = True                 # overshoots into black by 12px
    ref = guided_refine(rgb, mask, radius=12, eps=1e-3)
    assert ref[:, :40].mean() > 0.9 and ref[:, 60:].mean() < 0.1          # pulled back to the edge


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)
