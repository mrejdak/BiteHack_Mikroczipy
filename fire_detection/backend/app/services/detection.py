import random
from typing import List
from datetime import datetime
from ..models.schemas import GeoLocation, FireAlert, InfrastructureData

class DetectionService:
    def __init__(self):
        # In a real scenario, load the model here
        pass

    async def detect(self, image_data: bytes) -> List[FireAlert]:
        """
        Mock detection logic.
        Ignores the image content and returns a random fire location or fixed test points.
        """
        # Simulate processing time
        # await asyncio.sleep(0.5)

        # Mock: 50% chance of detecting fire for demo purposes (or always true for debugging)
        # For this stage, let's always return a detection to make UI dev easier.
        
        # Coordinates near a known industrial area in Warsaw (for OSM testing)
        # Example: Siekierki Power Station area roughly 52.19, 21.08
        # mock_lat = 52.195 + (random.random() * 0.01 - 0.005)
        # mock_lon = 21.085 + (random.random() * 0.01 - 0.005)
        mock_lat = 52.688349
        mock_lon = 23.702295

        # 52.688349, 23.702295

        # Mock: 50% chance of detecting fire for demo purposes (or always true for debugging)
        # For this stage, let's always return a detection to make UI dev easier.
        
        timestamp = datetime.now().isoformat()

        # We will populate infrastructure later via the OSM service.
        # This service just finds the fire.
        
        alert = FireAlert(
            location=GeoLocation(lat=mock_lat, lon=mock_lon),
            infrastructure=[], # To be filled by OSM Service
            timestamp=timestamp,
            severity="high"
        )
        
        return [alert]
