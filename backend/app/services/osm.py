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
        # Improved Overpass QL query using regex for building types
        # Also including relations (rel), common for large complexes.
        query = f"""
        [out:json];
        (
          node["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
          way["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
          rel["building"~"industrial|warehouse|hangar|storage|factory|retail|commercial|office"](around:{radius},{lat},{lon});
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
            
            print(f"OSM Results: {len(result.ways)} ways, {len(result.nodes)} nodes, {len(result.relations)} relations found.")
            
            infrastructure = []
            
            # Helper to extract from elements
            def process_element(element):
                name = element.tags.get("name", None)
                building_type = element.tags.get("building", element.tags.get("man_made", "Unknown"))
                operator = element.tags.get("operator", None)
                address = element.tags.get("addr:street", None)
                
                # If we don't have a name, maybe we can use the building type as a fallback name
                display_name = name if name else f"{building_type.capitalize()} Building"
                
                return InfrastructureData(
                    name=display_name,
                    type=building_type,
                    operator=operator,
                    address=address
                )

            for way in result.ways:
                infrastructure.append(process_element(way))

            for node in result.nodes:
                # Often nodes don't have tags if they are just parts of ways, 
                # but if they have building/man_made tags they are standalone.
                if "building" in node.tags or "man_made" in node.tags or "name" in node.tags:
                    infrastructure.append(process_element(node))

            for rel in result.relations:
                infrastructure.append(process_element(rel))
                
            return infrastructure

        except Exception as e:
            print(f"OSM Query Error: {e}")
            return []
