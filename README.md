# DonAssos AI
### Hybrid Recommendation System & Multilingual Neural Translation

**AI Engineering · Recommendation Systems · Natural Language Processing**

DonAssos is a donation platform designed to connect donors with associations and fundraising campaigns. During my internship at **IT-BAFA**, I developed the AI components supporting **personalized campaign discovery** and **multilingual accessibility**.

This repository presents the actual recommendation and translation modules derived from that work, with limited integration adaptations and documented compatibility fixes.

**Project focus:** Machine Learning and NLP components — not the full mobile or business application.

---

## At a glance

| AI capability | Implementation | Purpose |
| --- | --- | --- |
| Content-based recommendation | TF-IDF · Cosine similarity | Match campaigns with user interests |
| Collaborative filtering | User-based CF · Sparse interaction matrix | Learn from related users' interactions |
| Behavioral signals | Weighted events · Time decay · Popularity | Incorporate engagement and recency |
| Ranking diversity | Maximal Marginal Relevance (MMR) | Reduce redundancy in Top-N suggestions |
| Neural translation | Hugging Face Transformers · PyTorch · OPUS-MT | Translate text across French, English and Arabic |

**Tech stack:** Python · NumPy · SciPy · scikit-learn · PyTorch · Hugging Face Transformers · MongoDB / Motor · FastAPI

## 1. System architecture

### Personalized campaign recommendations

```text
    Approved Campaigns                      User Interactions
  (titles, descriptions, tags)          (views, donations, favorites)
              |                                      |
              v                                      v
       TF-IDF Vectorization                Weighted Events + Time Decay
              |                                      |
              +------------+-------------------------+
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
   Content Similarity   User-Based CF  Popularity Signals
          |                |                |
          +----------------+----------------+
                           |
                           v
                     Hybrid Scoring
                  (0.45 / 0.35 / 0.20)
                           |
                           v
                     MMR Reranking
                           |
                           v
                 Personalized Top-N List
```

The original implementation uses campaign documents and interaction events stored in **MongoDB**.

**Hybrid scoring**

```text
score(user, campaign)
    = 0.45 × content_similarity
    + 0.35 × collaborative_score
    + 0.20 × popularity_score
```

These weights come from the original project code; they are not claimed to be the outcome of a benchmark or optimization study.

### Multilingual translation

```text
Input text + source/target language
                 |
                 v
       Hugging Face Tokenizer
                 |
                 v
       Helsinki-NLP / OPUS-MT
                 |
                 v
          PyTorch Inference
                 |
                 v
           Translated Text
```

The translation module uses sequence-to-sequence pretrained models and selects CUDA, Apple MPS or CPU depending on availability. Checkpoint availability depends on the requested language pair and upstream models.

## 2. My technical contributions

- Implemented a **hybrid recommendation pipeline** combining semantic content, collaborative behavior and popularity signals.
- Built user–item interaction representations using **SciPy sparse matrices** and weighted event histories.
- Incorporated **time-decay weighting** and an **MMR reranker** to balance relevance with diversity.
- Integrated a **neural text translation module** based on pretrained Hugging Face sequence-to-sequence models.
- Exposed the AI modules through focused **FastAPI endpoints** backed by the actual project's MongoDB collection schema.

## 3. Source code structure

```text
DonAssos-AI/
├── recommendation/
│   ├── content.py        # TF-IDF indexing and content similarity
│   ├── cf.py             # User-based collaborative filtering
│   ├── signals.py        # User profiles and popularity signals
│   ├── schemas.py        # Interaction types and ranking weights
│   ├── utils.py          # Recency decay, normalization and MMR
│   ├── reco.py           # End-to-end hybrid ranking logic
│   └── database.py       # AI-only MongoDB collection adapter
├── translation/
│   └── i18n_router.py    # Neural translation inference
├── tests/
│   ├── test_ranking.py   # Ranking utility tests
│   └── test_mongodb_integration.py # Actual engine against temporary MongoDB
├── examples/
├── app.py                # Minimal AI-focused API integration
├── requirements.txt
└── README.md
```

The core recommendation components are taken from the original project's `Stage_DonAssos/recon/` directory. The translation inference originates from `Stage_DonAssos/routes/i18n_router.py`. Some integration and numerical-compatibility changes are documented below.

## 4. Running the project

**Requirements:** Python 3.10+; an authorized MongoDB instance with `campaigns` and `events` collections for recommendation requests; internet access and model downloads for first-time translation inference.

```bash
git clone https://github.com/MahdiBenHalima/DonAssos-AI.git
cd DonAssos-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set connection settings in your environment, using your **own authorized** database:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="donassos"
uvicorn app:app --reload
```

For PowerShell, use `$env:MONGODB_URI="mongodb://localhost:27017"` and `$env:MONGODB_DB_NAME="donassos"`.

The app deliberately **requires** `MONGODB_URI` at startup; it does not substitute artificial campaign records. For local testing, you may configure a database you own, but this repository does not distribute donor data.

### AI endpoints

| Method | Route | Function |
| --- | --- | --- |
| GET | `/reco/recommendations` | Rank campaigns for `user_id`, optional `k`, `lang`, `exclude_seen` |
| POST | `/reco/events` | Record an interaction event |
| POST | `/reco/reindex` | Refresh campaign and interaction indexes |
| POST | `/i18n/translate` | Run neural translation |
| GET | `/i18n/healthz` | Return model-device and language-code information |

Example translation request:

```json
{
  "text": "Bonjour",
  "src_lang": "fr",
  "tgt_lang": "en"
}
```

### Tests

```bash
pytest -q
```

The unit tests cover MMR ranking, sparse-matrix normalization and time-decay. The **MongoDB integration test exercises the real recommendation modules**, using an isolated temporary database with explicitly controlled fixtures. No scoring algorithm is reimplemented inside the test.

Run the integration test with `MONGODB_URI` pointing to an accessible **disposable** MongoDB server:

```bash
python -m pytest tests -q
```

GitHub Actions runs these tests against its own MongoDB service on pull requests and pushes to `main`. This validates software integration, **not recommendation accuracy on real donors** or live production infrastructure.

To request recommendations from a **running real API**, use:

```bash
python -m examples.live_recommendations YOUR_EXISTING_USER_ID
```

The earlier example with simulated collaborative interactions and manually assigned popularity values was removed. The live example invokes `/reco/recommendations`; it does not simulate ranking.

## 5. Implementation notes

The repository preserves the core project approach, with a small number of fixes for reproducibility and library compatibility:

- NumPy 2 compatibility for MMR score range calculation.
- Correct sparse-vector normalization and collaborative-filtering score conversion.
- TF-IDF support for small campaign collections.
- Explicit handling of empty ranking candidate sets and index refreshes.
- An isolated MongoDB adapter exposing only the collections needed by the AI pipeline.

**Current verification status:** The implementation has been compared with the original private source repository. The new integration test is designed to check the real engine with a disposable MongoDB instance in CI. Its pass status must be checked in GitHub Actions; it does not constitute validation against production donor records. Hugging Face translation model checkpoints also require separate runtime validation. No recommendation accuracy, latency or translation quality metric is asserted without measurement.

## 6. Scope and responsible use

This is an **AI engineering portfolio repository**. It does not include the Flutter application, authentication implementation, payments or the full business backend. There are no real donor records or database credentials provided.

The standalone AI endpoints do **not** implement the original application's authorization layer. They must not be exposed publicly without authentication, authorization, request limits and proper privacy controls.

---

**Context:** AI & Development Internship — **IT-BAFA**  
**Focus:** Recommendation Systems · NLP · Machine Learning Integration
