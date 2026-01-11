import networkx as nx
import numpy as np
from astropy import units as u
from src.simulation.satellite import Satellite

class Constellation:
    def __init__(self, num_orbits, num_sats_per_orbit):
        self.num_orbits = num_orbits
        self.num_sats_per_orbit = num_sats_per_orbit
        self.satellites = {} # ID -> Satellite object
        self.graph = nx.DiGraph()
        self.setup_constellation()

    def get_sat_id(self, orbit, phase):
        return orbit * self.num_sats_per_orbit + phase

    def setup_constellation(self):
        # Mixed Constellation Config
        # Shell 1: Polar (8 Orbits)
        # Shell 2: Inclined (Remaining Orbits)
        num_polar = 6
        
        self.visible_orbits = [] # List of list of [x,y,z]
        
        # Create Satellites
        for o in range(self.num_orbits):
            # Shell Logic
            if o < num_polar:
                # Shell 1: Polar
                inclination = 86.4
                num_in_shell = num_polar
                eff_o = o
                shell_id = 1
                raan = (360.0 / num_polar) * eff_o
            else:
                # Shell 2: Inclined (53 deg)
                inclination = 53.0
                num_inclined = self.num_orbits - num_polar
                eff_o = o - num_polar
                shell_id = 2
                raan = (360.0 / num_inclined) * eff_o
            
            # Use the first satellite of each orbit to trace the path
            trace_sat = None
            
            for p in range(self.num_sats_per_orbit):
                sat_id = self.get_sat_id(o, p)
                
                # Mean Anomaly spread over 360 degrees
                mean_anomaly = (360.0 / self.num_sats_per_orbit) * p
                
                sat = Satellite(
                    id=sat_id, 
                    orbit_id=o, 
                    phase_id=p, 
                    altitude=780, 
                    inclination=inclination, 
                    raan=raan, 
                    mean_anomaly=mean_anomaly
                )
                self.satellites[sat_id] = sat
                self.graph.add_node(sat_id, orbit=o, phase=p, active=True, shell=shell_id)
                
                if p == 0:
                    trace_sat = sat
            
            # Generate Trace
            if trace_sat:
                path_points = []
                period_seconds = trace_sat.orbit.period.to(u.s).value
                steps = 100
                for i in range(steps + 1):
                    t = (period_seconds / steps) * i
                    pos = trace_sat.get_position(t)
                    path_points.append(pos.tolist())
                self.visible_orbits.append(path_points)

        # Establish ISLs (Grid Topology)
        # Intra-orbit (UP/DOWN) - Universal
        for o in range(self.num_orbits):
            for p in range(self.num_sats_per_orbit):
                sat_id = self.get_sat_id(o, p)
                
                up_p = (p + 1) % self.num_sats_per_orbit
                up_id = self.get_sat_id(o, up_p)
                self.satellites[sat_id].add_neighbor('UP', up_id)
                self.graph.add_edge(sat_id, up_id, direction='UP', type='intra')

                down_p = (p - 1 + self.num_sats_per_orbit) % self.num_sats_per_orbit
                down_id = self.get_sat_id(o, down_p)
                self.satellites[sat_id].add_neighbor('DOWN', down_id)
                self.graph.add_edge(sat_id, down_id, direction='DOWN', type='intra')

        # Inter-orbit (LEFT/RIGHT/DIAG) - Per Shell
        for o in range(self.num_orbits):
            is_polar = (o < num_polar)
            
            if is_polar:
                base_o = 0
                count_o = num_polar
                eff_o = o
            else:
                base_o = num_polar
                count_o = self.num_orbits - num_polar
                eff_o = o - num_polar
                
            if count_o <= 1: continue

            for p in range(self.num_sats_per_orbit):
                sat_id = self.get_sat_id(o, p)
                
                # Right neighbor (next orbit in shell)
                right_eff = (eff_o + 1) % count_o
                right_o = base_o + right_eff
                right_id = self.get_sat_id(right_o, p)
                
                self.satellites[sat_id].add_neighbor('RIGHT', right_id)
                self.graph.add_edge(sat_id, right_id, direction='RIGHT', type='inter')

                # Left neighbor (prev orbit in shell)
                left_eff = (eff_o - 1 + count_o) % count_o
                left_o = base_o + left_eff
                left_id = self.get_sat_id(left_o, p)
                
                self.satellites[sat_id].add_neighbor('LEFT', left_id)
                self.graph.add_edge(sat_id, left_id, direction='LEFT', type='inter')

                # Diagonals (Enhanced Connectivity)
                # UP-RIGHT
                ur_p = (p + 1) % self.num_sats_per_orbit
                ur_id = self.get_sat_id(right_o, ur_p)
                self.satellites[sat_id].add_neighbor('UP_RIGHT', ur_id)
                self.graph.add_edge(sat_id, ur_id, direction='UP_RIGHT', type='inter')

                # UP-LEFT
                ul_p = (p + 1) % self.num_sats_per_orbit
                ul_id = self.get_sat_id(left_o, ul_p)
                self.satellites[sat_id].add_neighbor('UP_LEFT', ul_id)
                self.graph.add_edge(sat_id, ul_id, direction='UP_LEFT', type='inter')
                
                # DOWN-RIGHT
                dr_p = (p - 1 + self.num_sats_per_orbit) % self.num_sats_per_orbit
                dr_id = self.get_sat_id(right_o, dr_p)
                self.satellites[sat_id].add_neighbor('DOWN_RIGHT', dr_id)
                self.graph.add_edge(sat_id, dr_id, direction='DOWN_RIGHT', type='inter')

                # DOWN-LEFT
                dl_p = (p - 1 + self.num_sats_per_orbit) % self.num_sats_per_orbit
                dl_id = self.get_sat_id(left_o, dl_p)
                self.satellites[sat_id].add_neighbor('DOWN_LEFT', dl_id)
                self.graph.add_edge(sat_id, dl_id, direction='DOWN_LEFT', type='inter')
                
        # --- Cross-Shell Connectivity ---
        # Connect every Inclined Sat to every Polar Sat (Potentially)
        # We rely on dynamic LoS/Distance filtering to sparsify this
        if self.num_orbits > num_polar:
            for o_i in range(num_polar, self.num_orbits):
                for p_i in range(self.num_sats_per_orbit):
                    sat_u = self.get_sat_id(o_i, p_i)
                    
                    for o_p in range(num_polar):
                        for p_p in range(self.num_sats_per_orbit):
                             sat_v = self.get_sat_id(o_p, p_p)
                             
                             # Add edge u -> v and v -> u
                             self.graph.add_edge(sat_u, sat_v, direction='CROSS', type='cross')
                             self.graph.add_edge(sat_v, sat_u, direction='CROSS', type='cross')

        # --- Initialize Link States ---
        total_bw = 5000.0 # Mbps
        
        # Static Betweenness is expensive with cross-links (N^2 edges)
        # Skip bet_cen for speed or use approximation
        # bet_cen = nx.edge_betweenness_centrality(self.graph) 
        bet_cen = {}

        for n_u, n_v in self.graph.edges():
            self.graph.edges[n_u, n_v]['bw_total'] = total_bw
            self.graph.edges[n_u, n_v]['bw_occupied'] = 0.0 # Dynamic
            self.graph.edges[n_u, n_v]['betweenness'] = 0.0
            self.graph.edges[n_u, n_v]['quality'] = 1.0

    def set_satellite_status(self, sat_id, active):
        if sat_id in self.satellites:
            self.satellites[sat_id].set_active(active)
            # Update graph
            self.graph.nodes[sat_id]['active'] = active

    def calculate_los(self, r1, r2):
        # r1, r2: numpy arrays of position [x, y, z]
        # Check if segment intersects Earth sphere
        # Earth center at [0,0,0], radius R
        R = 6371.0
        ATMOSPHERE = 0.0 # Conservative Margin to ensure no visual clipping
        R_safe = R + ATMOSPHERE
        
        # Vector from p1 to p2
        d = r2 - r1
        f = r1 
        
        # Quadratic eq: |f + t*d|^2 = R^2
        # (f.f - R^2) + 2(f.d)t + (d.d)t^2 = 0
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - R_safe**2
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            # No intersection with sphere (line misses Earth)
            return True
        else:
            # Intersection with sphere/line
            # Check if intersection points are within segment t=[0,1]
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return False # Blocked
            
            return True

    def update_traffic(self, time, positions):
        # Simulate traffic based on DISTANCE (User Request)
        # load_factor proportional to link distance
        MAX_DIST = 8000.0 # Approx max ISL distance in km
        
        for u, v in self.graph.edges():
            # Enforce symmetry: only process if u < v
            if u > v: continue
            
            # Check if active (if not, load means nothing, but safe to calc)
            if u not in positions or v not in positions:
                continue

            pos_u = positions[u]
            pos_v = positions[v]
            
            dist = np.linalg.norm(pos_u - pos_v)
            
            # Normalize Load: 0.0 at 0km, 1.0 at MAX_DIST
            # Use a curve or linear? Linear is simplest.
            # load_factor = dist / MAX_DIST
            # But we want avoid > 1.0
            load_factor = min(1.0, dist / MAX_DIST)
            
            # Quality is usually good unless very far?
            quality = 1.0
            if dist > 4500: quality = 0.5 # Degradation at extreme range

            # Apply to Forward Link (u -> v)
            self.graph.edges[u, v]['bw_occupied'] = self.graph.edges[u, v]['bw_total'] * load_factor
            self.graph.edges[u, v]['quality'] = quality
            self.graph.edges[u, v]['distance'] = dist # Store distance for AI/Reward

            # Apply to Reverse Link (v -> u) if exists
            if self.graph.has_edge(v, u):
                self.graph.edges[v, u]['bw_occupied'] = self.graph.edges[v, u]['bw_total'] * load_factor
                self.graph.edges[v, u]['quality'] = quality
                self.graph.edges[v, u]['distance'] = dist

    def get_dynamic_graph(self, time):
        # 0. Calculate Positions (Needed for Traffic AND LoS)
        positions = {}
        for sat_id, sat in self.satellites.items():
            positions[sat_id] = sat.get_position(time)

        # 1. Update Traffic/Physics (Distance Based)
        self.update_traffic(time, positions)
        
        # 2. Build Graph based on Active + LoS
        # We need to transfer edge attributes (BW, load) to the new dynamic graph
        
        active_edges = []
        
        for u, v in self.graph.edges():
            # Node Status Check
            if not self.satellites[u].is_active or not self.satellites[v].is_active:
                continue
                
            # LoS Check
            if self.calculate_los(positions[u], positions[v]):
                # Copy attributes from main graph
                attrs = self.graph.edges[u, v].copy()
                active_edges.append((u, v, attrs))
                
        # Create temp graph
        G = nx.DiGraph()
        G.add_nodes_from(self.graph.nodes(data=True)) 
        G.add_edges_from(active_edges)
        
        # DEBUG
        # print(f"Dynamic Graph: {len(active_edges)} active edges (Total filtered from: {len(self.graph.edges)})")
        
        return G
