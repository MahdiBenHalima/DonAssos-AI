import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from motor.motor_asyncio import AsyncIOMotorCollection
from .schemas import WEIGHTS
from .utils import decayed_weight, normalize_rows_csr, now_utc

class CFUserBased:
    def __init__(self):
        self.users: list[str] = []
        self.items: list[str] = []
        self.user_index: dict[str,int] = {}
        self.item_index: dict[str,int] = {}
        self.R: csr_matrix | None = None
        self.R_row_norm: csr_matrix | None = None

    async def build(self, events_col: AsyncIOMotorCollection):

        pipeline = [
            {"$match": {"type": {"$in": ["view","don","init_donation","favorite"]}}},
            {"$project": {"user_id":1,"item_id":1,"type":1,"ts":1,"duration_sec":1}},
        ]
        rows, cols, data = [], [], []
        users_set, items_set = set(), set()

        async for e in events_col.aggregate(pipeline):
            base = WEIGHTS.get(e["type"], 0.0)
            if e["type"] == "view" and (e.get("duration_sec") or 0) < 5:
                continue
            w = decayed_weight(base, e.get("ts"))
            u, i = e["user_id"], e["item_id"]
            users_set.add(u); items_set.add(i)
            rows.append(u); cols.append(i); data.append(w)

        self.users = sorted(users_set)
        self.items = sorted(items_set)
        self.user_index = {u: idx for idx, u in enumerate(self.users)}
        self.item_index = {i: idx for idx, i in enumerate(self.items)}

        if not data:
            # matrice vide
            self.R = csr_matrix((0,0), dtype=np.float32)
            self.R_row_norm = self.R
            return

        r = [self.user_index[u] for u in rows]
        c = [self.item_index[i] for i in cols]
        self.R = coo_matrix((data, (r, c)), shape=(len(self.users), len(self.items)), dtype=np.float32).tocsr()
        self.R_row_norm = normalize_rows_csr(self.R)

    def cf_score(self, user_id: str, item_id: str) -> float:

        if self.R_row_norm is None or self.R_row_norm.shape[0] == 0:
            return 0.0
        ui = self.user_index.get(user_id); ii = self.item_index.get(item_id)
        if ui is None or ii is None:
            return 0.0


        r_u = self.R_row_norm.getrow(ui).T
        sim_users = self.R_row_norm @ r_u
        if sim_users.shape[0] > ui:
            sim_users[ui, 0] = 0.0
        scores = (sim_users.T @ self.R).A1
        return float(scores[ii])
