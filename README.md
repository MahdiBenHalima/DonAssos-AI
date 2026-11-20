# DonAssos AI — Original recommendation and translation modules

This repository contains the **real AI implementation** extracted from the private DonAssos internship project. It intentionally excludes the Flutter application and the general-purpose donation, authentication and payment backend.

## What is included

### Hybrid recommendation (original source: `Stage_DonAssos/recon/`)

- **Content-based filtering:** TF-IDF over approved campaign titles, descriptions and tags, with cosine similarity to a user's interaction-derived profile.
- **User-based collaborative filtering:** sparse user–campaign interaction matrix, weighted by interaction type and recency.
- **Popularity:** recent donation/view signals from MongoDB.
- **Hybrid score:** `0.45 * content + 0.35 * collaborative + 0.20 * popularity`.
- **MMR reranking:** balance relevance and diversity for the final Top-N campaigns.

`recommendation/reco.py` is adapted from the **actual** original recommendation router. The integration reads the `campaigns` and `events` collections of a MongoDB database. No simulated recommendations or fabricated performance results are included.

### Neural machine translation (original source: `Stage_DonAssos/routes/i18n_router.py`)

The real PyTorch/Hugging Face Transformers inference code uses Helsinki-NLP/OPUS-MT sequence-to-sequence models. It exposes `POST /i18n/translate` for French, English and Arabic language codes. A model is downloaded and loaded on first use; not every direct model pair may be available upstream, and a missing checkpoint produces a meaningful error. The original in-memory organization-demo CRUD was deliberately excluded.

## Repository layout

```text
recommendation/
    cf.py          # user-based collaborative filtering
    content.py     # TF-IDF content index
    signals.py     # profile and popularity from MongoDB events
    utils.py       # time-decay, normalization, MMR
    schemas.py     # event types, algorithm weights
    reco.py        # original hybrid ranking pipeline and endpoints
    database.py    # minimal MongoDB adapter for AI collections
translation/
    i18n_router.py # original neural translation inference
app.py             # minimal AI-only API entry point
tests/             # algorithm tests (no medical or donor data)
```

## Installation

Python 3.10+ is recommended. Create a virtual environment, then:

```bash
pip install -r requirements.txt
```

To use recommendations against **your own authorized DonAssos database**, set the environment variables shown in `.env.example` (in your shell or via your deployment secret manager; `app.py` does not load `.env` automatically):

```bash
export MONGODB_URI='mongodb://localhost:27017'
export MONGODB_DB_NAME='donassos'
uvicorn app:app --reload
```

Windows PowerShell:

```powershell
$env:MONGODB_URI='mongodb://localhost:27017'
$env:MONGODB_DB_NAME='donassos'
uvicorn app:app --reload
```

**The API will not start without `MONGODB_URI`**; it does not silently use fictional campaigns. Required MongoDB collections are `campaigns` (approved campaign documents) and `events` (interaction documents with user and item identifiers).

## API

- `GET /reco/recommendations?user_id=<id>&k=10`: rank approved campaigns from real stored interactions and campaign content
- `POST /reco/events`: write an interaction event to MongoDB
- `POST /reco/reindex`: rebuild the content and collaborative indexes
- `POST /i18n/translate`: run actual model inference
- `GET /i18n/healthz`: report supported language codes and chosen compute device

Example translation request:

```json
{"text":"Bonjour","src_lang":"fr","tgt_lang":"en"}
```

These are the AI-specific interfaces; **they are not an independently secured production service**. The original general backend handled authentication and authorization. Do not expose this app directly to the public Internet: protect the API with authentication, authorization, rate limits, and appropriate data minimization before deployment.

## Design notes and known limits

- Indexing and scoring follow the original project. The TF-IDF minimum document frequency is adjusted to support small datasets; the MMR NumPy 2 incompatibility and sparse vector norm/scoring issues are corrected.
- `POST /reco/reindex` should be called after new campaigns or events so both indices reflect the stored data. Content filtering also rebuilds when the requested language changes.
- The code still needs integration tests with a configured, authorized MongoDB database, and runtime tests using downloaded translation checkpoints. Neither production behavior nor benchmark accuracy is claimed here.
- No real donor records, MongoDB credentials, payment functions or Flutter code are distributed in this repository.
- Only publish code you are authorized to disclose.
