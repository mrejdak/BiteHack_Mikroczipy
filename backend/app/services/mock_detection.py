"""
Mock detection service that uses YOLO on pre-loaded satellite images.
"""
import os
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from ..models.schemas import GeoLocation, FireAlert
from .yolo_detection import yolo_service


class MockDetectionService:
    def __init__(self):
        self.mock_images_dir = Path(__file__).parent.parent.parent.parent / "mock_images"
    
    def _load_coordinates(self, image_id: int) -> Optional[tuple]:
        """
        Load coordinates from lonlat{id}.txt
        
        Returns:
            ((lat_tl, lon_tl), (lat_br, lon_br)) or None if file not found
        """
        coord_file = self.mock_images_dir / f"lonlat{image_id}.txt"
        
        if not coord_file.exists():
            print(f"Coordinate file not found: {coord_file}")
            return None
        
        try:
            with open(coord_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    print(f"Invalid coordinate file format: {coord_file}")
                    return None
                
                # Parse top-left coordinates (line 1)
                tl_parts = lines[0].strip().split(',')
                lat_tl = float(tl_parts[0].strip())
                lon_tl = float(tl_parts[1].strip())
                
                # Parse bottom-right coordinates (line 2)
                br_parts = lines[1].strip().split(',')
                lat_br = float(br_parts[0].strip())
                lon_br = float(br_parts[1].strip())
                
                return ((lat_tl, lon_tl), (lat_br, lon_br))
                
        except Exception as e:
            print(f"Error reading coordinate file: {e}")
            return None
    
    def _normalize_to_latlon(
        self, 
        cx: float, 
        cy: float, 
        coords: tuple
    ) -> GeoLocation:
        """
        Convert normalized image coordinates (0-1) to lat/lon.
        
        Args:
            cx: normalized center x (0=left, 1=right)
            cy: normalized center y (0=top, 1=bottom)
            coords: ((lat_tl, lon_tl), (lat_br, lon_br))
        """
        (lat_tl, lon_tl), (lat_br, lon_br) = coords
        
        # Interpolate
        lat = lat_tl + (lat_br - lat_tl) * cy
        lon = lon_tl + (lon_br - lon_tl) * cx
        
        return GeoLocation(lat=lat, lon=lon)
    
    async def detect_from_mock(self, image_id: int) -> List[FireAlert]:
        """
        Run YOLO detection on a mock satellite image.
        
        Args:
            image_id: ID of the mock image (e.g., 1 for image1.png)
            
        Returns:
            List of FireAlert objects with detected fire locations
        """
        image_path = self.mock_images_dir / f"image{image_id}.png"
        
        if not image_path.exists():
            print(f"Mock image not found: {image_path}")
            return []
        
        # Load coordinates
        coords = self._load_coordinates(image_id)
        if not coords:
            return []
        
        # Run YOLO detection
        detections = yolo_service.detect_fires(str(image_path))
        
        if not detections:
            print("No fires detected in mock image")
            return []
        
        # Create annotated image
        annotated_base64 = yolo_service.annotate_image(str(image_path), detections)
        
        # Convert detections to alerts
        alerts = []
        timestamp = datetime.now().isoformat()
        
        for cx, cy, w, h, conf in detections:
            location = self._normalize_to_latlon(cx, cy, coords)
            
            # Determine severity based on confidence
            if conf >= 0.8:
                severity = "high"
            elif conf >= 0.5:
                severity = "medium"
            else:
                severity = "low"
            
            alert = FireAlert(
                location=location,
                infrastructure=[],  # Will be filled by OSM service
                timestamp=timestamp,
                severity=severity,
                annotated_image_base64=annotated_base64
            )
            alerts.append(alert)
        
        print(f"Created {len(alerts)} fire alerts from mock image {image_id}")
        return alerts


# Singleton instance
mock_detection_service = MockDetectionService()
