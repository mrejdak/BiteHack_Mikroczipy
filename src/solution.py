import time
import networkx as nx
from src.simulation.sky import Constellation
from src.viz.visualizer import Visualizer

def run_demo():
    print("Initializing Large Constellation...")
    # Scale up!
    num_orbits = 4
    num_sats = 20
    total_sats = num_orbits * num_sats
    
    constellation = Constellation(num_orbits, num_sats)
    print(f"Created {total_sats} satellites.")
    
    # Define Source and Destination
    src = 0
    dst = total_sats - 1 # Furthest point approx
    
    # Callback for clicks
    def on_click_recalculate(broken_nodes_set):
        print("Recalculating route...")
        # Get active subgraph
        active_subgraph = constellation.get_graph()
        try:
            path = nx.shortest_path(active_subgraph, src, dst)
            print(f"New Path calculated: {len(path)} hops.")
            return path
        except nx.NetworkXNoPath:
            print("CRITICAL: No path available!")
            return None

    # Initial Path
    print(f"Initial Routing from {src} to {dst}")
    active_subgraph = constellation.get_graph()
    initial_path = nx.shortest_path(active_subgraph, src, dst)
    
    # Setup Visualizer
    viz = Visualizer(constellation, on_click_callback=on_click_recalculate)
    viz.path = initial_path
    
    print("\n--- Interactive Demo Running ---")
    print("1. Earth is rendered as a blue sphere.")
    print("2. Satellites are orbiting (20x speed).")
    print("3. CLICK any satellite to BREAK/REPAIR it.")
    print("4. Close the window to stop.")
    
    viz.run_interactive()
    
    print("Demo finished.")
