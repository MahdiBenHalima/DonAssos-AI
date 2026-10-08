# DonAssos AI

AI components developed for the DonAssos donation platform.

## AI Components

### 1. Hybrid Recommendation System

A hybrid recommendation system combining:

- Content-Based Filtering using TF-IDF
- User-Based Collaborative Filtering
- Popularity-based signals
- Time-decay weighting
- MMR (Maximal Marginal Relevance) reranking

### 2. Multilingual Translation

A multilingual translation module based on:

- PyTorch
- Hugging Face Transformers
- Helsinki-NLP / OPUS-MT
- French, English and Arabic

## Architecture

```text
User Interactions
       |
       +----------------------+
       |                      |
       v                      v
Content-Based          Collaborative Filtering
(TF-IDF)               (User-Based CF)
       |                      |
       +----------+-----------+
                  |
                  v
          Hybrid Recommendation
                  |
                  v
             MMR Reranking
                  |
                  v
          Top-N Recommendations