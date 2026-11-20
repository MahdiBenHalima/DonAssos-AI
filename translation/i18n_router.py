# routes/i18n_router.py
from functools import lru_cache
from enum import Enum

import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

router = APIRouter(prefix="/i18n", tags=["i18n"])

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()

class LangEnum(str, Enum):
    fr = "fr"
    en = "en"
    ar = "ar"

SUPPORTED_LANGS = {x.value for x in LangEnum}
SUPPORTED_PAIRS = {(a.value, b.value) for a in LangEnum for b in LangEnum if a != b}

def marian_name(src: str, tgt: str) -> str:
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

@lru_cache(maxsize=12)
def load_pipeline(src: str, tgt: str):
    if (src, tgt) not in SUPPORTED_PAIRS:
        raise HTTPException(400, detail=f"Paire non supportée: {src}->{tgt}")
    name = marian_name(src, tgt)
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    except Exception as e:
        raise HTTPException(400, detail=f"Impossible de charger {name}: {e}")
    mdl.to(DEVICE)
    return tok, mdl

@torch.inference_mode()
def translate_text(text: str, src: str, tgt: str) -> str:
    tok, mdl = load_pipeline(src, tgt)
    enc = tok(text, return_tensors="pt", truncation=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = mdl.generate(
        **enc,
        num_beams=4,
        max_new_tokens=256,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]

class TranslateIn(BaseModel):
    text: str
    src_lang: LangEnum
    tgt_lang: LangEnum

class TranslateOut(BaseModel):
    translation: str

@router.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE), "langs": list(SUPPORTED_LANGS)}

@router.post("/translate", response_model=TranslateOut)
def translate_endpoint(body: TranslateIn):
    if body.src_lang == body.tgt_lang:
        return TranslateOut(translation=body.text)
    out = translate_text(body.text, body.src_lang.value, body.tgt_lang.value)
    return TranslateOut(translation=out)
