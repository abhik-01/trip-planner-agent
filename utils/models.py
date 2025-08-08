from __future__ import annotations
from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Dict, Any

class PlanStep(BaseModel, extra=Extra.allow):
    id: str
    step: str
    requires: List[str] = Field(default_factory=list)
    status: str = Field(default="pending")  # pending|running|done|skipped|error
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class FlightOffer(BaseModel, extra=Extra.ignore):
    price: float
    currency: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    airline: str

class WeatherSnapshot(BaseModel, extra=Extra.ignore):
    date: str
    summary: str

class ToolResult(BaseModel, extra=Extra.allow):
    ok: bool
    data: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
