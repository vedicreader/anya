"""Step 2a: edge-preserving p-Laplacian prompt propagation (Eq. 2, Gauss-Jacobi)."""
import numpy as np
from scipy import sparse


def sparsify(A, k=48):
    "Symmetric top-k affinity graph: keep each row's k strongest edges, then union with A^T."
    N = A.shape[0]
    idx = np.argpartition(-A, k, axis=1)[:, :k]                   # k largest per row
    rows = np.repeat(np.arange(N), k)
    cols = idx.ravel()
    S = sparse.csr_matrix((A[rows, cols], (rows, cols)), shape=(N, N))
    S = S.maximum(S.T)                                           # symmetrize by union
    S.setdiag(0); S.eliminate_zeros()
    return S


def grid_prompts(Hf, Wf, s=6):
    "Equidistant one-hot seed indices on the HfxWf latent token grid, spacing s."
    ys = np.arange(s // 2, Hf, s); xs = np.arange(s // 2, Wf, s)
    return np.array([y * Wf + x for y in ys for x in xs], np.int64)


def propagate(A, seeds, p=1.6, lam=1e-5, tau_prop=1e-4, max_iter=200):
    "Smooth one-hot seeds over sparse affinity A into soft object maps f (N x K)."
    N = A.shape[0]; K = len(seeds)
    f0 = np.zeros((N, K), np.float32); f0[seeds, np.arange(K)] = 1.0
    f = f0.copy()
    d = np.asarray(A.sum(1)).reshape(-1, 1)                      # node degrees (N,1)
    e = p / 2 - 1
    for _ in range(max_iter):
        Af = A @ f
        S = np.clip(A @ (f * f) - 2 * f * Af + (f * f) * d, 0, None)
        phi = np.power(S + 1e-8, e)                              # per-node p-Laplacian coeff
        num = phi * Af + A @ (phi * f)                           # Sum_j gamma_ij f_j
        den = phi * d + A @ phi                                  # Sum_j gamma_ij
        fn = (lam * f0 + num) / (lam + den + 1e-12)
        if ((fn - f) ** 2).sum() / K <= tau_prop: f = fn; break
        f = fn
    return f
