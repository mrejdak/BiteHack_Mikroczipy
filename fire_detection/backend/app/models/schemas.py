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
    annotated_image_base64: Optional[str] = None

class DetectionResponse(BaseModel):
    fire_detected: bool
    alerts: List[FireAlert]


# Fire Spread Simulation Schemas
class FireSpreadRequest(BaseModel):
    """Request to simulate fire spread from coordinates."""
    lat: float
    lon: float
    hours: int = 1  # Duration to simulate (default 1 hour)


class FireSpreadPoint(BaseModel):
    """A point affected by fire."""
    lat: float
    lon: float
    status: str  # "burning" or "burned"


class WeatherInfo(BaseModel):
    """Weather data used for simulation."""
    wind_speed_mph: float
    wind_direction_deg: float
    humidity: float


class FireSpreadResponse(BaseModel):
    """Response containing fire spread simulation results."""
    success: bool
    fire_origin: Optional[GeoLocation] = None
    affected_areas: List[FireSpreadPoint] = []
    simulation_duration_hours: int = 1
    weather: Optional[WeatherInfo] = None
    error: Optional[str] = None
