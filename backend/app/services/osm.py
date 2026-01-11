import overpy
import asyncio
import math
from typing import List, Tuple
from ..models.schemas import InfrastructureData

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

class OSMService:
    def __init__(self):
        self.api = overpy.Overpass()

    async def check_infrastructure(self, lat: float, lon: float, radius: int = 500) -> List[InfrastructureData]:
        """
        Query Overpass API for industrial/commercial buildings around the coordinates.
        Returns the 5 closest infrastructure items.
        """
        query = f"""
        [out:json];
        (
          node["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
          way["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
          rel["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
          way["man_made"="works"](around:{radius},{lat},{lon});
          node["man_made"="works"](around:{radius},{lat},{lon});
        );
        out body center;
        """

        try:
            result = await asyncio.to_thread(self.api.query, query)
            
            print(f"OSM Results: {len(result.ways)} ways, {len(result.nodes)} nodes, {len(result.relations)} relations found.")
            
            infrastructure_with_distance: List[Tuple[float, InfrastructureData]] = []
            
            def process_element(element, element_lat: float, element_lon: float):
                name = element.tags.get("name", None)
                building_type = element.tags.get("building", element.tags.get("man_made", "Unknown"))
                operator = element.tags.get("operator", None)
                address = element.tags.get("addr:street", None)
                
                display_name = name if name else f"{building_type.capitalize()} Building"
                
                distance = haversine_distance(lat, lon, element_lat, element_lon)
                
                infra = InfrastructureData(
                    name=display_name,
                    type=building_type,
                    operator=operator,
                    address=address
                )
                return (distance, infra)

            for way in result.ways:
                # Use center_lat/center_lon for ways (requires 'out body center')
                if hasattr(way, 'center_lat') and way.center_lat is not None:
                    infrastructure_with_distance.append(process_element(way, float(way.center_lat), float(way.center_lon)))

            for node in result.nodes:
                if "building" in node.tags or "man_made" in node.tags or "name" in node.tags:
                    infrastructure_with_distance.append(process_element(node, float(node.lat), float(node.lon)))

            for rel in result.relations:
                if hasattr(rel, 'center_lat') and rel.center_lat is not None:
                    infrastructure_with_distance.append(process_element(rel, float(rel.center_lat), float(rel.center_lon)))
            
            # Sort by distance (closest first) and take top 5
            infrastructure_with_distance.sort(key=lambda x: x[0])
            top_5 = [infra for _, infra in infrastructure_with_distance[:5]]
            
            print(f"Returning {len(top_5)} closest infrastructure items.")
            return top_5

        except Exception as e:
            print(f"OSM Query Error: {e}")
            return []

