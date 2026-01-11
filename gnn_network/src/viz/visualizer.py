import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
from matplotlib import cm

class Visualizer:
    def __init__(self, constellation, on_click_callback=None):
        self.constellation = constellation
        self.on_click_callback = on_click_callback
        
        # Setup Figure
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.start_time = time.time()
        
        # --- Earth Sphere (Better Rendering) ---
        # --- Earth Sphere (Better Rendering) ---
        # Load Texture
        try:
            # User provided specific texture
            print(f"Attempting to load texture from: earth_texture.jpeg")
            img = plt.imread('earth_texture.jpeg')
            print(f"Texture loaded successfully. Shape: {img.shape}")
            
            # Normalize if needed (jpg is 0-255)
            if img.max() > 1.0:
                img = img / 255.0
        except Exception as e:
            print(f"Texture loading failed: {e}")
            img = None

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        if img is not None:
             self.ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False)
        else:
             self.ax.plot_surface(x, y, z, cmap=cm.Blues, alpha=0.9, rstride=2, cstride=2, shade=True)
        
        # Add wireframe overlay for techy look
        self.ax.plot_wireframe(x, y, z, color="white", alpha=0.1, rstride=5, cstride=5)

        # Event Handling
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Initial Plot Handles
        self.node_scatter = None
        self.edge_lines = []
        self.sat_ids_in_plot = [] # To map pick index back to sat ID
        
        # State
        self.path = None
        self.broken_nodes = set()
        self.is_paused = False

    def on_pick(self, event):
        # Handle click on satellite
        if event.artist != self.node_scatter:
            return
            
        ind = event.ind[0]
        if ind < len(self.sat_ids_in_plot):
            sat_id = self.sat_ids_in_plot[ind]
            print(f"Clicked Satellite: {sat_id}")
            
            # Toggle Status
            if sat_id in self.broken_nodes:
                self.broken_nodes.remove(sat_id)
                self.constellation.set_satellite_status(sat_id, True)
                print(f"Satellite {sat_id} REPAIRED.")
            else:
                self.broken_nodes.add(sat_id)
                self.constellation.set_satellite_status(sat_id, False)
                print(f"Satellite {sat_id} BROKEN.")
            
            # Trigger Callback to re-calculate path
            if self.on_click_callback:
                self.path = self.on_click_callback(self.broken_nodes)
            
            # FORCE IMMEDIATE UPDATE for responsiveness
            # We don't wait for next loop, we update right away?
            # Actually, standard loop is fast enough if we reduce pause.
            # But let's try to update title or something.

    def update_plot(self, t=0):
        # 1. Update Satellite Positions
        positions = {}
        xs, ys, zs = [], [], []
        colors = []
        sizes = []
        self.sat_ids_in_plot = []
        
        for sat_id, sat in self.constellation.satellites.items():
            pos = sat.get_position(t)
            positions[sat_id] = pos
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            self.sat_ids_in_plot.append(sat_id)
            
            if sat_id in self.broken_nodes:
                colors.append('red')
                sizes.append(100) # Make broken ones bigger
            elif self.path and sat_id in self.path:
                colors.append('lime')
                sizes.append(80)
            else:
                colors.append('dodgerblue')
                sizes.append(40)

        # 2. Update Scatter Plot (remove old, add new)
        if self.node_scatter:
            self.node_scatter.remove()
        
        # Picker=10 makes it easier to click
        self.node_scatter = self.ax.scatter(xs, ys, zs, c=colors, s=sizes, picker=10, depthshade=False)

        # 3. Update Connections (ISLs)
        for line in self.edge_lines:
            line.remove()
        self.edge_lines = []

        G = self.constellation.graph
        
        # Draw dynamic links
        # Optimization: Pre-calculate edge list for standard grid? 
        # For now, just iteration is okay for 60 nodes (approx 240 edges).
        
        for u, v in G.edges():
            if u in self.broken_nodes or v in self.broken_nodes:
                continue
                
            p1 = positions[u]
            p2 = positions[v]
            
            # Path Logic
            is_path_edge = False
            if self.path:
                try:
                    # Optimized check?
                    if u in self.path and v in self.path:
                         # We can check connectivity in path
                         # But easier is:
                         is_path_edge = True
                         # We need to be careful not to draw path edges between non-adjacent path nodes 
                         # (e.g. if path loops back, unlikely here).
                         # Let's check strict adjacency in path list
                         u_idx = self.path.index(u)
                         v_idx = self.path.index(v)
                         if abs(u_idx - v_idx) != 1:
                             is_path_edge = False
                except:
                    pass

            if is_path_edge:
                line, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='lime', linewidth=3, zorder=10)
                self.edge_lines.append(line)
            else:
                # Optimized drawing: Draw connection if valid.
                line, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.4, linewidth=0.8)
                self.edge_lines.append(line)

        # View settings
        r_limit = 8000
        self.ax.set_xlim([-r_limit, r_limit])
        self.ax.set_ylim([-r_limit, r_limit])
        self.ax.set_zlim([-r_limit, r_limit])
        
        # Remove panes for cleaner look
        self.ax.set_axis_off()
        self.ax.set_title(f"Satellite Network (t={t:.0f}s) - Click to Break/Fix", color='white')
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')

    def run_interactive(self):
        plt.ion() # Interactive mode
        self.show()
        
        print("Visualizer running... Close window to exit.")
        while plt.fignum_exists(self.fig.number):
            # Faster animation speed
            current_sim_time = (time.time() - self.start_time) * 50 
            self.update_plot(current_sim_time)
            self.fig.canvas.flush_events()
            # Minimal pause for max FPS
            plt.pause(0.001)

    def show(self):
        plt.show()
