"""
Fire simulation service using SimFire library.
Runs headless fire spread simulations and returns affected coordinates.
"""

import sys
import os
import tempfile
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add simfire to path
SIMFIRE_PATH = "/home/tommeh/Projects/hackathon/BiteHack_Mikroczipy/simfire"
if SIMFIRE_PATH not in sys.path:
    sys.path.insert(0, SIMFIRE_PATH)

import numpy as np
import yaml

from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus


@dataclass
class FireSpreadResult:
    """Result of fire spread simulation."""
    fire_origin: Tuple[float, float]  # (lat, lon)
    affected_areas: List[dict]  # [{"lat": float, "lon": float, "status": str}]
    simulation_duration_hours: int


class FireSimulationService:
    """Service to run fire spread simulations."""
    
    # Simulation grid settings
    BASE_GRID_SIZE = 100  # base pixels for 1 hour
    GRID_SIZE_PER_HOUR = 75  # additional pixels per hour (heuristic: fire spreads ~1-3 mph)
    PIXEL_SCALE = 50  # feet per pixel
    
    # Approximate conversion: 1 degree lat/lon ≈ 364,000 ft at equator
    # More accurate for mid-latitudes: ~300,000 ft
    FEET_PER_DEGREE = 300000.0
    
    def _calculate_grid_size(self, hours: int) -> int:
        """
        Calculate grid size based on simulation duration.
        
        Heuristic: Fire typically spreads 1-3 mph in moderate conditions.
        With 50 ft/pixel, we need ~100 pixels/hour of additional spread margin.
        """
        grid_size = self.BASE_GRID_SIZE + (hours * self.GRID_SIZE_PER_HOUR)
        # Cap at 400 to prevent excessive computation time
        return min(grid_size, 400)
    
    def _create_config(
        self,
        wind_speed: float,
        wind_direction: float,
        moisture: float,
        grid_size: int
    ) -> str:
        """
        Create a SimFire YAML config file.
        
        Args:
            wind_speed: Wind speed in mph
            wind_direction: Wind direction in degrees (0 = North)
            moisture: Fuel moisture (0.0 to 1.0, typically 0.03)
            grid_size: Size of simulation grid in pixels
            
        Returns:
            Path to temporary config file
        """
        config = {
            "area": {
                "screen_size": [grid_size, grid_size],
                "pixel_scale": self.PIXEL_SCALE
            },
            "display": {
                "fire_size": 2,
                "control_line_size": 2,
                "agent_size": 4,
                "rescale_factor": 1
            },
            "simulation": {
                "update_rate": 1,
                "runtime": "2h",
                "headless": True,
                "draw_spread_graph": False,
                "record": False,
                "save_data": False,
                "data_type": "npy",
                "sf_home": tempfile.gettempdir()
            },
            "mitigation": {
                "ros_attenuation": False
            },
            "operational": {
                "seed": None,
                "latitude": 38.422,
                "longitude": -118.266,
                "height": 1000,
                "width": 1000,
                "resolution": 30,
                "year": 2020
            },
            "terrain": {
                "topography": {
                    "type": "functional",
                    "functional": {
                        "function": "flat"
                    }
                },
                "fuel": {
                    "type": "functional",
                    "functional": {
                        "function": "chaparral",
                        "chaparral": {
                            "seed": 1113
                        }
                    }
                }
            },
            "fire": {
                "fire_initial_position": {
                    "type": "static",
                    "static": {
                        "position": f"({grid_size // 2}, {grid_size // 2})"
                    }
                },
                "max_fire_duration": 3,
                "diagonal_spread": True
            },
            "environment": {
                "moisture": max(0.001, min(moisture, 1.0))
            },
            "wind": {
                "function": "simple",
                "simple": {
                    "speed": max(1, wind_speed),
                    "direction": wind_direction
                }
            }
        }
        
        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return path
    
    def _pixel_to_latlon(
        self,
        pixel_x: int,
        pixel_y: int,
        center_lat: float,
        center_lon: float,
        grid_size: int
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            center_lat: Latitude of grid center
            center_lon: Longitude of grid center
            grid_size: Size of the simulation grid
            
        Returns:
            (lat, lon) tuple
        """
        center_pixel = grid_size // 2
        
        # Calculate offset in pixels from center
        dx = pixel_x - center_pixel
        dy = pixel_y - center_pixel
        
        # Convert pixels to feet
        dx_feet = dx * self.PIXEL_SCALE
        dy_feet = dy * self.PIXEL_SCALE
        
        # Convert feet to degrees (approximate)
        # Note: longitude degrees vary with latitude
        lat_offset = -dy_feet / self.FEET_PER_DEGREE  # Negative because y increases downward
        lon_offset = dx_feet / (self.FEET_PER_DEGREE * np.cos(np.radians(center_lat)))
        
        return (
            round(center_lat + lat_offset, 6),
            round(center_lon + lon_offset, 6)
        )
    
    async def simulate_fire(
        self,
        lat: float,
        lon: float,
        wind_speed: float = 10.0,
        wind_direction: float = 0.0,
        moisture: float = 0.03,
        hours: int = 1
    ) -> Optional[FireSpreadResult]:
        """
        Run fire spread simulation for specified hours.
        
        Args:
            lat: Fire origin latitude
            lon: Fire origin longitude
            wind_speed: Wind speed in mph
            wind_direction: Wind direction in degrees (0 = North)
            moisture: Fuel moisture level
            hours: Number of hours to simulate
            
        Returns:
            FireSpreadResult with affected coordinates, or None on error
        """
        config_path = None
        try:
            # Calculate dynamic grid size based on hours
            grid_size = self._calculate_grid_size(hours)
            
            # Create config with dynamic grid
            config_path = self._create_config(wind_speed, wind_direction, moisture, grid_size)
            
            # Initialize SimFire
            config = Config(config_path)
            sim = FireSimulation(config)
            
            # Run for specified hours
            fire_map, active = sim.run(f"{hours}h")
            
            # Extract affected areas
            affected_areas = []
            
            # Find all burning and burned pixels
            burning_coords = np.argwhere(fire_map == BurnStatus.BURNING)
            burned_coords = np.argwhere(fire_map == BurnStatus.BURNED)
            
            # Convert to lat/lon (limit to prevent huge responses)
            # Prioritize burned areas as they show actual spread, then add burning
            max_points = 1000  # Increased limit
            
            # Add burned areas first (primary spread indicator)
            for y, x in burned_coords[:max_points]:
                point_lat, point_lon = self._pixel_to_latlon(x, y, lat, lon, grid_size)
                affected_areas.append({
                    "lat": point_lat,
                    "lon": point_lon,
                    "status": "burned"
                })
            
            # Add burning areas (active fire front)
            remaining_slots = max_points - len(affected_areas)
            for y, x in burning_coords[:remaining_slots]:
                point_lat, point_lon = self._pixel_to_latlon(x, y, lat, lon, grid_size)
                affected_areas.append({
                    "lat": point_lat,
                    "lon": point_lon,
                    "status": "burning"
                })
            
            return FireSpreadResult(
                fire_origin=(lat, lon),
                affected_areas=affected_areas,
                simulation_duration_hours=hours
            )
            
        except Exception as e:
            print(f"Fire simulation error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Cleanup temp config
            if config_path and os.path.exists(config_path):
                try:
                    os.unlink(config_path)
                except:
                    pass
