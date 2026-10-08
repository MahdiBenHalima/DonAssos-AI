from functools import lru_cache
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# Device
# ============================================================

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = pick_device()


# ============================================================
# Supported languages
# ============================================================

class LangEnum(str, Enum):
    fr = "fr"
    en = "en"
    ar = "ar"


SUPPORTED_LANGS = {x.value for x in LangEnum}

SUPPORTED_PAIRS = {
    (src.value, tgt.value)
    for src in LangEnum
    for tgt in LangEnum
    if src != tgt
}


# ============================================================
# Model
# ============================================================

def marian_name(src: str, tgt: str) -> str:
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"


@lru_cache(maxsize=12)
def load_pipeline(src: str, tgt: str):

    if (src, tgt) not in SUPPORTED_PAIRS:
        raise ValueError(
            f"Unsupported translation pair: {src} -> {tgt}"
        )

    model_name = marian_name(src, tgt)

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model.to(DEVICE)

    return tokenizer, model


# ============================================================
# Translation
# ============================================================

@torch.inference_mode()
def translate_text(
    text: str,
    src: str,
    tgt: str
) -> str:

    if not text.strip():
        return ""

    if src == tgt:
        return text

    if (src, tgt) not in SUPPORTED_PAIRS:
        raise ValueError(
            f"Unsupported translation pair: {src} -> {tgt}"
        )

    tokenizer, model = load_pipeline(src, tgt)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    output = model.generate(
        **encoded,
        num_beams=4,
        max_new_tokens=256,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    return tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )[0]