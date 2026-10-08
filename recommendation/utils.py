import math, numpy as np
from datetime import datetime, timezone
from .schemas import LAMBDA_DECAY, MMR_LAMBDA

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def days_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 86400.0

def decayed_weight(base: float, ts: datetime | None, lam: float = LAMBDA_DECAY) -> float:
    if ts is None:
        return base
    d = days_between(now_utc(), ts)
    return float(base * math.exp(-lam * d))

def normalize_rows_csr(X):
    from scipy.sparse import csr_matrix
    if not hasattr(X, "tocsr"):
        return X
    X = X.tocsr().astype(np.float32)
    norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1 + 1e-9
    inv = 1.0 / norms
    D = csr_matrix((inv, X.indices, X.indptr), shape=X.shape)
    X = X.multiply(inv.reshape(-1, 1))
    return X

def cosine_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    v = vec / (np.linalg.norm(vec) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return (M @ v)

def mmr_rerank(
    item_ids: list[str],
    relevance_scores: np.ndarray,
    item_vecs: np.ndarray,
    k: int,
    lamb: float = MMR_LAMBDA,
) -> list[str]:
    selected, pool = [], list(range(len(item_ids)))
    rel = (relevance_scores - relevance_scores.min()) / (relevance_scores.ptp() + 1e-9)

    while pool and len(selected) < k:
        best_idx, best_val = None, -1e9
        for ci in pool:
            if not selected:
                div_pen = 0.0
            else:
                sims = cosine_sim(item_vecs[ci], item_vecs[selected])
                div_pen = sims.max() if sims.size else 0.0
            val = lamb * rel[ci] - (1 - lamb) * div_pen
            if val > best_val:
                best_val, best_idx = val, ci
        selected.append(best_idx)
        pool.remove(best_idx)
    return [item_ids[i] for i in selected]
