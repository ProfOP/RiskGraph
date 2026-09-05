from typing import Any
from pydantic import BaseModel, Field
class ScoreRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=80)
    prediction_time: str | None = None
class Signal(BaseModel):
    feature: str
    value: float | None = None
    contribution: float | None = None
    explanation: str
class RiskResponse(BaseModel):
    customer_id: str
    risk_probability: float
    risk_score: float
    risk_band: str
    expected_loss: float
    historical_exposure: float
    recommended_intervention: str
    top_risk_signals: list[Signal] = []
    protective_signals: list[Signal] = []
    behavioral_statistics: dict[str, Any] = {}
    network_statistics: dict[str, Any] = {}
    snapshot: str
    model_status: str
    source: str
