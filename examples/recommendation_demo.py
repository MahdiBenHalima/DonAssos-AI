import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Load sample campaigns
# ============================================================

with open("examples/sample_data/sample_campaigns.json", "r", encoding="utf-8") as f:
    campaigns = json.load(f)


# ============================================================
# 2. Content-Based Filtering
# ============================================================

texts = [
    f"{c['title']} {c['description']} {' '.join(c['tags'])}"
    for c in campaigns
]

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=1
)

tfidf_matrix = vectorizer.fit_transform(texts)


# ============================================================
# 3. Simulated user profile
# ============================================================

# Campaigns previously interacted with by the user
user_history = [0, 2]

user_profile = np.asarray(
    tfidf_matrix[user_history].mean(axis=0)
)


# ============================================================
# 4. Content similarity
# ============================================================

content_scores = cosine_similarity(
    user_profile,
    tfidf_matrix
).flatten()


# ============================================================
# 5. Simulated Collaborative Filtering
# ============================================================

# Rows = users
# Columns = campaigns
user_item_matrix = np.array([
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1]
], dtype=float)

target_user = 0

user_similarity = cosine_similarity(
    user_item_matrix[target_user].reshape(1, -1),
    user_item_matrix
).flatten()

# Ignore similarity with the user himself
user_similarity[target_user] = 0

cf_scores = user_similarity @ user_item_matrix


# Normalize CF scores
if cf_scores.max() > 0:
    cf_scores = cf_scores / cf_scores.max()


# ============================================================
# 6. Popularity score
# ============================================================

popularity_scores = np.array([
    0.90,
    0.70,
    1.00,
    0.50,
    0.80,
    0.40
])


# ============================================================
# 7. Hybrid Recommendation
# ============================================================

ALPHA = 0.45
BETA = 0.35
GAMMA = 0.20

hybrid_scores = (
    ALPHA * content_scores
    + BETA * cf_scores
    + GAMMA * popularity_scores
)


# ============================================================
# 8. Remove already seen campaigns
# ============================================================

for campaign_id in user_history:
    hybrid_scores[campaign_id] = -1


# ============================================================
# 9. MMR Reranking
# ============================================================

MMR_LAMBDA = 0.7
TOP_K = 3

selected = []

for _ in range(TOP_K):

    candidates = [
        i for i in range(len(campaigns))
        if i not in selected and hybrid_scores[i] >= 0
    ]

    if not candidates:
        break

    if not selected:
        best = max(
            candidates,
            key=lambda i: hybrid_scores[i]
        )

    else:
        mmr_scores = {}

        for candidate in candidates:

            relevance = hybrid_scores[candidate]

            diversity = max(
                cosine_similarity(
                    tfidf_matrix[candidate],
                    tfidf_matrix[selected]
                ).flatten()
            )

            mmr_scores[candidate] = (
                MMR_LAMBDA * relevance
                - (1 - MMR_LAMBDA) * diversity
            )

        best = max(
            mmr_scores,
            key=mmr_scores.get
        )

    selected.append(best)


# ============================================================
# 10. Display recommendations
# ============================================================

print("\n===== DONASSOS AI RECOMMENDATIONS =====\n")

for rank, index in enumerate(selected, start=1):

    campaign = campaigns[index]

    print(
        f"{rank}. {campaign['title']}"
    )

    print(
        f"   Score: {hybrid_scores[index]:.4f}"
    )

    print(
        f"   Tags: {', '.join(campaign['tags'])}"
    )

    print()