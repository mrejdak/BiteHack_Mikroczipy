"""
Fire spread simulation API router.
"""

from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    FireSpreadRequest,
    FireSpreadResponse,
    FireSpreadPoint,
    GeoLocation,
    WeatherInfo
)
from ..services.weather import WeatherService
from ..services.fire_simulation import FireSimulationService

router = APIRouter()
weather_service = WeatherService()
fire_simulation_service = FireSimulationService()


@router.post("/spread", response_model=FireSpreadResponse)
async def simulate_fire_spread(request: FireSpreadRequest):
    """
    Simulate fire spread from given coordinates.
    
    Uses current weather data from Open-Meteo API and SimFire library
    for fire spread modeling.
    
    Args:
        request.hours: Number of hours to simulate (default 1)
    
    Returns affected areas as lat/lon coordinates that can be displayed
    on a map (mark as red or with fire emoji 🔥).
    """
    try:
        # Get current weather
        weather = await weather_service.get_weather(request.lat, request.lon)
        
        if not weather:
            return FireSpreadResponse(
                success=False,
                error="Failed to fetch weather data"
            )
        
        # Convert humidity to moisture (0-100% to 0-1 scale, lower = drier)
        # Lower humidity = lower fuel moisture = faster fire spread
        moisture = max(0.001, weather.humidity / 1000.0)  # Scale down significantly
        
        # Run fire simulation
        result = await fire_simulation_service.simulate_fire(
            lat=request.lat,
            lon=request.lon,
            wind_speed=weather.wind_speed_mph,
            wind_direction=weather.wind_direction_deg,
            moisture=moisture,
            hours=request.hours
        )
        
        if not result:
            return FireSpreadResponse(
                success=False,
                error="Fire simulation failed"
            )
        
        # Convert result to response
        affected_areas = [
            FireSpreadPoint(
                lat=area["lat"],
                lon=area["lon"],
                status=area["status"]
            )
            for area in result.affected_areas
        ]
        
        return FireSpreadResponse(
            success=True,
            fire_origin=GeoLocation(lat=request.lat, lon=request.lon),
            affected_areas=affected_areas,
            simulation_duration_hours=result.simulation_duration_hours,
            weather=WeatherInfo(
                wind_speed_mph=weather.wind_speed_mph,
                wind_direction_deg=weather.wind_direction_deg,
                humidity=weather.humidity
            )
        )
        
    except Exception as e:
        return FireSpreadResponse(
            success=False,
            error=str(e)
        )
