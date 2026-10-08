from pydantic import BaseModel, Field
from typing import Literal, Optional

EventType = Literal["view", "don", "init_donation", "favorite"]

class EventIn(BaseModel):
    user_id: str
    item_id: str
    type: EventType
    duration_sec: Optional[int] = None
    amount: Optional[float] = None
    source: Optional[str] = "web"

class RecoQuery(BaseModel):
    user_id: str
    k: int = 20
    lang: Optional[str] = None
    exclude_seen: bool = True


WEIGHTS: dict[EventType, float] = {
    "view": 0.2,
    "don": 2.0,
    "init_donation": 0.5,
    "favorite": 0.7,
}
LAMBDA_DECAY: float = 0.05


ALPHA = 0.45
BETA  = 0.35
GAMMA = 0.20


MMR_LAMBDA = 0.7