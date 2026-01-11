"""
YOLO-based fire detection service.
Uses ultralytics YOLO to detect fires in satellite images.
"""
import os
import base64
from typing import List, Tuple, Optional
from pathlib import Path
from io import BytesIO

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics or cv2 not installed. YOLO detection disabled.")


class YoloDetectionService:
    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent.parent.parent.parent / "model" / "best.pt"
        
        if YOLO_AVAILABLE and self.model_path.exists():
            try:
                self.model = YOLO(str(self.model_path))
                print(f"YOLO model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            if not YOLO_AVAILABLE:
                print("YOLO not available - ultralytics not installed")
            else:
                print(f"YOLO model not found at {self.model_path}")

    def detect_fires(self, image_path: str) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect fires in an image using YOLO.
        
        Returns:
            List of (cx, cy, w, h, confidence) tuples, all normalized [0, 1]
        """
        if not self.model:
            print("YOLO model not loaded, returning empty detections")
            return []
        
        try:
            results = self.model(image_path, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get normalized xywh (center x, center y, width, height)
                        xywhn = box.xywhn[0].tolist()
                        conf = float(box.conf[0])
                        detections.append((xywhn[0], xywhn[1], xywhn[2], xywhn[3], conf))
            
            print(f"YOLO detected {len(detections)} fire(s)")
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def annotate_image(self, image_path: str, detections: List[Tuple[float, float, float, float, float]]) -> Optional[str]:
        """
        Draw bounding boxes on image and return as base64 PNG.
        
        Args:
            image_path: Path to the original image
            detections: List of (cx, cy, w, h, conf) normalized detections
            
        Returns:
            Base64 encoded PNG string, or None on error
        """
        if not YOLO_AVAILABLE:
            return None
            
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return None
            
            h, w = img.shape[:2]
            
            # Draw each detection
            for cx, cy, bw, bh, conf in detections:
                # Convert normalized to pixel coordinates
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                # Draw red rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Draw label
                label = f"Fire {conf:.2f}"
                font_scale = 0.7
                thickness = 2
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 0, 255), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', img)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            
            return base64_str
            
        except Exception as e:
            print(f"Image annotation error: {e}")
            return None


# Singleton instance
yolo_service = YoloDetectionService()
