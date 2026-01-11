"""
Weather service using Open-Meteo API for fire simulation.
Free API, no key required.
"""

import httpx
from typing import Optional
from dataclasses import dataclass


@dataclass
class WeatherData:
    """Weather data from Open-Meteo API."""
    wind_speed_mph: float
    wind_direction_deg: float
    humidity: float


class WeatherService:
    """Service to fetch weather data from Open-Meteo API."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    async def get_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        """
        Fetch current weather for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            WeatherData with wind and humidity info, or None on error
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "wind_speed_10m,wind_direction_10m,relative_humidity_2m"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                current = data.get("current", {})
                
                # Convert wind speed from km/h to mph (SimFire expects mph)
                wind_speed_kmh = current.get("wind_speed_10m", 10.0)
                wind_speed_mph = wind_speed_kmh * 0.621371
                
                return WeatherData(
                    wind_speed_mph=round(wind_speed_mph, 2),
                    wind_direction_deg=current.get("wind_direction_10m", 0.0),
                    humidity=current.get("relative_humidity_2m", 50.0)
                )
        except Exception as e:
            print(f"Weather API error: {e}")
            # Return default values on error
            return WeatherData(
                wind_speed_mph=10.0,
                wind_direction_deg=0.0,
                humidity=50.0
            )
