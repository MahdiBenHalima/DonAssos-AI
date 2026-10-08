import numpy as np
from motor.motor_asyncio import AsyncIOMotorCollection
from .schemas import WEIGHTS
from .utils import decayed_weight, now_utc

async def user_profile_from_events(
    events_col: AsyncIOMotorCollection,
    content_index,
    user_id: str,
) -> np.ndarray:
    evs = await events_col.find({"user_id": user_id}).to_list(100_000)
    if not evs or content_index.item_matrix is None or content_index.item_matrix.shape[0] == 0:

        from scipy.sparse import csr_matrix
        return np.zeros((content_index.item_matrix.shape[1] if content_index.item_matrix is not None else 1), dtype=np.float32)

    num = np.zeros(content_index.item_matrix.shape[1], dtype=np.float32)
    den = 1e-9
    id2idx = {iid:i for i,iid in enumerate(content_index.item_ids)}

    for e in evs:
        base = WEIGHTS.get(e["type"], 0.0)
        if e["type"] == "view" and (e.get("duration_sec") or 0) < 5:
            continue
        w = decayed_weight(base, e.get("ts"))
        iidx = id2idx.get(e["item_id"])
        if iidx is None:
            continue
        vec = content_index.item_matrix[iidx].toarray().ravel()
        num += w * vec
        den += w
    return (num / den).astype(np.float32)

async def popularity_scores(
    events_col: AsyncIOMotorCollection,
    item_ids: list[str],
) -> dict[str, float]:
    # somme des dons (et vues filtrées) avec time-decay
    pipeline = [
        {"$match": {"item_id": {"$in": item_ids}, "type": {"$in": ["don","view"]}}},
        {"$project": {"item_id":1,"type":1,"ts":1,"duration_sec":1}},
    ]
    scores = {iid: 0.0 for iid in item_ids}
    async for e in events_col.aggregate(pipeline):
        base = WEIGHTS.get(e["type"], 0.0)
        if e["type"] == "view" and (e.get("duration_sec") or 0) < 5:
            continue
        scores[e["item_id"]] += decayed_weight(base, e.get("ts"))
    return scores
