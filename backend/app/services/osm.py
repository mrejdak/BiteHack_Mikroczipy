import overpy
import asyncio
from typing import List
from ..models.schemas import InfrastructureData

class OSMService:
    def __init__(self):
        self.api = overpy.Overpass()

    async def check_infrastructure(self, lat: float, lon: float, radius: int = 500) -> List[InfrastructureData]:
        """
        Query Overpass API for industrial/commercial buildings around the coordinates.
        """
        # Overpass QL query
        # Looking for factories, industrial buildings, warehouses, etc.
        query = f"""
        [out:json];
        (
          way["building"="industrial"](around:{radius},{lat},{lon});
          way["building"="warehouse"](around:{radius},{lat},{lon});
          way["man_made"="works"](around:{radius},{lat},{lon});
          node["man_made"="works"](around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            # interacting with external API, run in thread pool to not block async loop if sync lib
            # overpy is synchronous
            result = await asyncio.to_thread(self.api.query, query)
            
            infrastructure = []
            
            for way in result.ways:
                name = way.tags.get("name", "Unknown Facility")
                building_type = way.tags.get("building", way.tags.get("man_made", "Unknown Type"))
                operator = way.tags.get("operator", None)
                
                # Simple extraction
                inf = InfrastructureData(
                    name=name,
                    type=building_type,
                    operator=operator,
                    address=way.tags.get("addr:street", None)
                )
                infrastructure.append(inf)

            for node in result.nodes:
                # Deduplicate if needed, but for now just add
                name = node.tags.get("name", "Unknown Facility")
                inf = InfrastructureData(
                    name=name,
                    type=node.tags.get("man_made", "Unknown Node"),
                    operator=node.tags.get("operator", None)
                )
                infrastructure.append(inf)
                
            return infrastructure

        except Exception as e:
            print(f"OSM Query Error: {e}")
            return []
