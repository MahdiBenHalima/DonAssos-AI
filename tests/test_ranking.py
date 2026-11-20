"""Tests of actual DonAssos ranking utilities; no fabricated integration claims."""
import numpy as np
from scipy.sparse import csr_matrix

from recommendation.utils import mmr_rerank, normalize_rows_csr, decayed_weight


def test_mmr_empty_candidates():
    assert mmr_rerank([], np.array([]), np.empty((0, 2)), k=5) == []


def test_mmr_returns_unique_ranked_ids():
    ids = ["c1", "c2", "c3"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    ordered = mmr_rerank(ids, np.array([0.9, 0.8, 0.7]), vectors, k=3)
    assert len(ordered) == 3
    assert set(ordered) == set(ids)
    assert ordered[0] == "c1"


def test_normalize_rows_handles_zero_rows():
    matrix = csr_matrix(np.array([[3.0, 4.0], [0.0, 0.0]]))
    out = normalize_rows_csr(matrix).toarray()
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-6)


def test_decay_without_timestamp():
    assert decayed_weight(2.0, None) == 2.0
