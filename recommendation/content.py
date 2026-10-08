import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from motor.motor_asyncio import AsyncIOMotorCollection

class ContentIndex:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.item_ids: list[str] = []
        self.item_matrix = None
        self.lang_cached: str | None = None

    async def build(self, campaigns_col: AsyncIOMotorCollection, lang: str | None = None):
        if self.item_matrix is not None and self.lang_cached == (lang or "__all__"):
            return

        query = {"approved": True}
        if lang:
            query["lang"] = lang

        cursor = campaigns_col.find(query, projection={"title":1,"description":1,"tags":1})
        docs = await cursor.to_list(length=200_000)

        texts, self.item_ids = [], []
        for d in docs:
            txt = " ".join([
                str(d.get("title","")),
                str(d.get("description","")),
                " ".join(d.get("tags", []))
            ])
            texts.append(txt)
            self.item_ids.append(str(d["_id"]))

        if not texts:
            from scipy.sparse import csr_matrix
            self.vectorizer = TfidfVectorizer(max_features=10)
            self.item_matrix = csr_matrix((0, 10))
            self.lang_cached = lang or "__all__"
            return

        self.vectorizer = TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1,2),
            min_df=2
        )
        self.item_matrix = self.vectorizer.fit_transform(texts).astype(np.float32)
        self.lang_cached = lang or "__all__"

    def ids_to_indices(self, ids: list[str]) -> list[int]:
        pos = {iid: i for i, iid in enumerate(self.item_ids)}
        return [pos[i] for i in ids if i in pos]
