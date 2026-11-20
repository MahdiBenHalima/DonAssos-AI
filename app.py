"""Minimal AI-only FastAPI entry point; no DonAssos business backend."""
from fastapi import FastAPI
from recommendation.reco import router as recommendation_router
from translation.i18n_router import router as translation_router

app = FastAPI(title="DonAssos AI Services")
app.include_router(recommendation_router)
app.include_router(translation_router)
