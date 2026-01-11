from pydantic import BaseModel
from typing import List, Optional

class GeoLocation(BaseModel):
    lat: float
    lon: float

class InfrastructureData(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    operator: Optional[str] = None
    address: Optional[str] = None

class FireAlert(BaseModel):
    location: GeoLocation
    infrastructure: List[InfrastructureData]
    timestamp: str
    severity: str

class DetectionResponse(BaseModel):
    fire_detected: bool
    alerts: List[FireAlert]
