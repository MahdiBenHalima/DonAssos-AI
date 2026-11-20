from __future__ import annotations
from fastapi import APIRouter, Query
from bson import ObjectId
import numpy as np
from .database import campaigns_collection, events_collection

from .schemas import EventIn, RecoQuery, ALPHA, BETA, GAMMA
from .content import ContentIndex
from .signals import user_profile_from_events, popularity_scores
from .cf import CFUserBased
from .utils import mmr_rerank, now_utc

router = APIRouter(prefix="/reco", tags=["reco"])

CONTENT = ContentIndex()
CF = CFUserBased()

@router.on_event("startup")
async def _startup():
    # passe l'objet collection
    await CONTENT.build(campaigns_collection)
    await CF.build(events_collection)

@router.post("/events")
async def log_event(ev: EventIn):
    doc = ev.model_dump()
    doc["ts"] = doc.get("ts") or now_utc()
    await events_collection.insert_one(doc)
    return {"ok": True}

@router.post("/reindex")
async def rebuild_index(lang: str | None = None):
    await CONTENT.build(campaigns_collection, lang=lang)
    await CF.build(events_collection)
    return {"ok": True, "lang": lang}

@router.get("/recommendations")
async def recommendations(
    user_id: str = Query(...),
    k: int = Query(20, ge=1, le=100),
    lang: str | None = Query(None),
    exclude_seen: bool = Query(True),
):
    # build index (par langue)
    await CONTENT.build(campaigns_collection, lang=lang)

    # profil utilisateur contenu
    profile = await user_profile_from_events(events_collection, CONTENT, user_id)

    # candidats
    q = {"approved": True}
    if lang:
        q["lang"] = lang
    projection = {"title": 1, "description": 1, "lang": 1, "tags": 1}
    docs = await campaigns_collection.find(q, projection=projection).to_list(200_000)

    # déjà vus
    seen_ids: set[str] = set()
    if exclude_seen:
        evs = await events_collection.find({"user_id": user_id}).to_list(100_000)
        seen_ids = {e.get("item_id") for e in evs if e.get("item_id")}

    item_ids: list[str] = []
    for d in docs:
        iid = str(d["_id"])
        if exclude_seen and iid in seen_ids:
            continue
        item_ids.append(iid)

    if not item_ids or CONTENT.item_matrix is None or CONTENT.item_matrix.shape[0] == 0:
        return {"items": []}

    # aligner avec l'index
    idxs = CONTENT.ids_to_indices(item_ids)
    item_ids = [CONTENT.item_ids[i] for i in idxs]
    item_mat = CONTENT.item_matrix[idxs]

    # contenu (cosine)
    prof = profile / (np.linalg.norm(profile) + 1e-9)
    item_norms = np.linalg.norm(item_mat, axis=1) + 1e-9
    sim_content = (item_mat @ prof) / item_norms
    sim_content = np.asarray(sim_content).ravel().astype(np.float32)

    # CF (user-based)
    sim_cf = np.array([CF.cf_score(user_id, iid) for iid in item_ids], dtype=np.float32)
    if sim_cf.max() > 0:
        sim_cf = sim_cf / (sim_cf.max() + 1e-9)

    # popularité
    pop_map = await popularity_scores(events_collection, item_ids)
    pop_vec = np.array([pop_map.get(iid, 0.0) for iid in item_ids], dtype=np.float32)
    if pop_vec.max() > 0:
        pop_vec = pop_vec / (pop_vec.max() + 1e-9)

    # score hybride
    score = ALPHA * sim_content + BETA * sim_cf + GAMMA * pop_vec

    # MMR
    item_vecs = item_mat.toarray().astype(np.float32)
    ranked_ids = mmr_rerank(item_ids, score, item_vecs, k=k)

    # retour
    docs2 = await campaigns_collection.find(
        {"_id": {"$in": [ObjectId(i) for i in ranked_ids]}},
        projection={"title": 1, "description": 1, "lang": 1, "tags": 1},
    ).to_list(200_000)

    byid = {str(d["_id"]): d for d in docs2}
    items = [{"id": iid, **{key: value for key, value in byid[iid].items() if key != "_id"}} for iid in ranked_ids if iid in byid]
    return {"items": items}
