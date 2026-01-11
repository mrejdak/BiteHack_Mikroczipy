from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from ..services.detection import DetectionService
from ..services.mock_detection import mock_detection_service
from ..services.osm import OSMService
from ..models.schemas import DetectionResponse, FireAlert

router = APIRouter()
detection_service = DetectionService()
osm_service = OSMService()

@router.post("/detect", response_model=DetectionResponse)
async def detect_fire(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read file content
    contents = await file.read()
    
    # Run Detection (Mock or Real)
    alerts = await detection_service.detect(contents)
    
    if not alerts:
        return DetectionResponse(fire_detected=False, alerts=[])

    # Enrich alerts with OSM Data
    for alert in alerts:
        infra = await osm_service.check_infrastructure(
            alert.location.lat, 
            alert.location.lon
        )
        alert.infrastructure = infra

    return DetectionResponse(
        fire_detected=True,
        alerts=alerts
    )


@router.get("/detect/mock/{image_id}", response_model=DetectionResponse)
async def detect_mock_fire(image_id: int):
    """
    Run fire detection on a mock satellite image.
    
    Args:
        image_id: ID of the mock image (e.g., 1 for image1.png, lonlat1.txt)
    """
    alerts = await mock_detection_service.detect_from_mock(image_id)
    
    if not alerts:
        return DetectionResponse(fire_detected=False, alerts=[])
    
    # Enrich alerts with OSM Data
    for alert in alerts:
        infra = await osm_service.check_infrastructure(
            alert.location.lat, 
            alert.location.lon
        )
        alert.infrastructure = infra
    
    return DetectionResponse(
        fire_detected=True,
        alerts=alerts
    )

