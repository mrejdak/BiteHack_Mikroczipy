from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import torch
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import asyncio

# Import existing simulation logic
# We need to make sure src is in python path or we move this file to root and adjust imports
# For now, let's assume we run uvicorn from root.
from src.simulation.sky import Constellation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
class SimulationState:
    def __init__(self):
        self.start_time = time.time()
        # Scale: 6x10
        num_orbits = 4
        num_sats = 20
        self.constellation = Constellation(num_orbits, num_sats)
        # Pick disjoint orbits for Src/Dst
        import random
        orbit_ids = list(range(num_orbits))
        o_src = random.choice(orbit_ids)
        if len(orbit_ids) > 1:
            orbit_ids.remove(o_src)
            o_dst = random.choice(orbit_ids)
        else:
            o_dst = o_src
            
        # Get random sat index in that orbit
        # Sat ID = orbit * num_sats + phase
        self.src = o_src * num_sats + random.randint(0, num_sats - 1)
        self.dst = o_dst * num_sats + random.randint(0, num_sats - 1)
        
        # Ensure distinct
        if self.src == self.dst:
             self.dst = (self.dst + 1) % (num_orbits * num_sats)

        self.broken_nodes = set()
        self.active_path = []
        
        # Pause State
        self.is_paused = False
        self.paused_time = 0.0 # Stores logical time when paused
        self.accumulated_pause_duration = 0.0 # Total time spent paused to subtract from real time
        self.last_pause_timestamp = 0.0
        self.ai_path = [] # Store AI path here

    def update_path_with_graph(self, current_time, dynamic_graph):
        # 1. Optimal Path (Weighted Dijkstra - Minimize Congestion)
        try:
            self.active_path = nx.shortest_path(dynamic_graph, self.src, self.dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.active_path = []
            
        # 2. AI Path (Inference)
        self.ai_path = calculate_ai_path(self.src, self.dst, dynamic_graph)
    
    def update_path(self, current_time):
        # Legacy/Fallback
        dg = self.constellation.get_dynamic_graph(current_time)
        self.update_path_with_graph(current_time, dg)

    def get_current_logical_time(self):
        if self.is_paused:
            return self.paused_time
        else:
            # Time since start - Total Pause Duration
            # Speed Factor 20
            real_elapsed = time.time() - self.start_time - self.accumulated_pause_duration
            return real_elapsed * 20

sim_state = SimulationState()
# sim_state.initialize() # __init__ now handles initial setup

# --- Pydantic Models ---
class SatelliteData(BaseModel):
    id: int
    orbit_id: int
    phase_id: int
    position: List[float] # [x, y, z]
    is_active: bool

# ... (omitted)

@app.get("/simulation/state")
def get_state():
    t = sim_state.get_current_logical_time()
    
    # 1. Dynamic Topology (Traffic + LoS)
    try:
        dynamic_graph = sim_state.constellation.get_dynamic_graph(t)
    except Exception as e:
        print(f"Error getting dynamic graph: {e}")
        # Fallback to static graph if LoS fails
        dynamic_graph = sim_state.constellation.graph
    
    # Update AI Logic using this graph
    if not sim_state.is_paused:
        try:
             sim_state.update_path_with_graph(t, dynamic_graph)
        except Exception as e:
            print(f"Error in AI Path update: {e}")
            # Ensure path is empty if failed
            sim_state.ai_path = []

    sats = []
    edges = []
    
    # DEBUG: Check if we have satellites
    # print(f"DEBUG: Satellites count: {len(sim_state.constellation.satellites)}")
    
    # 2. Edges from Dynamic Graph (LoS verified)
    for u, v in dynamic_graph.edges():
        data = dynamic_graph.edges[u, v]
        load = data.get('bw_occupied', 0) / data.get('bw_total', 1000)
        edges.append({"u": u, "v": v, "load": load})

    for sat_id, sat in sim_state.constellation.satellites.items():
        pos = sat.get_position(t)
        sats.append({
            "id": sat.id,
            "orbit_id": sat.orbit_id,
            "phase_id": sat.phase_id,
            "position": pos.tolist(),
            "is_active": sat.is_active
        })

    return {
        "time": t,
        "satellites": sats,
        "edges": edges,
        "path": sim_state.active_path,
        "ai_path": sim_state.ai_path,
        "src_id": sim_state.src,
        "dst_id": sim_state.dst,
        "is_paused": sim_state.is_paused,
        "orbits": sim_state.constellation.visible_orbits
    }
    phase_id: int
    position: List[float]
    is_active: bool
    is_path: bool

class SimulationResponse(BaseModel):
    time: float
    satellites: List[SatelliteData]
    path: List[int]
    edges: List[List[int]] # Simplified edge list for grid

# --- Endpoints ---

# --- Helper for AI Path Inference ---
def calculate_ai_path(src, dst, dynamic_graph):
    # Uses global training_mgr.agent (if available)
    # Be careful about thread safety if training is running!
    # Ideally use a snapshot/copy of the model.

    # Only if trained
    # We need access to internal objects (agent, gnn) which are inside the training loop closure in src/train?
    # NO, src/train returns them.
    # We need to store them in TrainingManager.

    if training_mgr.agent is None:
        return []

    path = [src]
    current = src
    max_hops = 30

    # Shared objects
    constellation = sim_state.constellation # Static features
    agent = training_mgr.agent
    gnn = training_mgr.gnn

    # 1. Embeddings
    num_nodes = len(constellation.satellites)
    adj = torch.eye(num_nodes)
    for u, v in dynamic_graph.edges():
        adj[u, v] = 1.0
    row_sum = adj.sum(dim=1, keepdim=True); row_sum[row_sum==0]=1; adj=adj/row_sum

    feats = training_mgr.get_features_fn(constellation)  # Need this helper available or import it

    with torch.no_grad():
        embeddings = gnn(feats, adj)

    for _ in range(max_hops):
        if current == dst:
            break

        if current not in dynamic_graph:
            break

        neighbors = list(dynamic_graph.successors(current))
        if not neighbors:
            break

        # Link Features
        link_feats_dict = {}
        for n in neighbors:
            attrs = dynamic_graph.edges[current, n]
            f1_avail = (attrs['bw_total'] - attrs['bw_occupied']) / attrs['bw_total']
            f2_occ = attrs['bw_occupied'] / attrs['bw_total']
            f3_bet = attrs['betweenness']
            
            # Use stored distance if available
            dist = attrs.get('distance', 1000.0)
            f5_dist = dist / 5000.0
            
            # Pad to 20 dims (n=20)
            lf_base = torch.tensor([f1_avail, f2_occ, f3_bet, 1.0, f5_dist], dtype=torch.float32)
            padding = torch.zeros(15, dtype=torch.float32)
            lf = torch.cat([lf_base, padding])
            
            link_feats_dict[n] = lf

        # Select Action (Greedy)
        # We need to temporarily set epsilon to 0? Or pass a flag?
        # The agent.select_action uses self.epsilon.
        old_eps = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(neighbors, embeddings, current, dst, link_feats_dict)
        agent.epsilon = old_eps

        if action is None:
            break

        current = action
        path.append(current)

    return path

@app.get("/simulation/state")
def get_state():
    current_time = sim_state.get_current_logical_time()

    # --- Dynamic Topology Update ---
    dynamic_graph = sim_state.constellation.get_dynamic_graph(current_time)

    # 1. Optimal Path (Weighted Dijkstra - Minimize Congestion)
    # Weight = 1 + (Occupied/Total)*10?
    # Let's verify NetworkX weighted path
    try:
        # Define weight function/attr
        # For simple path, weight=None (Hops). The paper minimizes Delay (Hops) + ensures Throughput.
        # So Hops are primary. But if congested?
        # Let's use Hops (BFS) for "Optimal Delay" path.
        sim_state.active_path = nx.shortest_path(dynamic_graph, sim_state.src, sim_state.dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        sim_state.active_path = []

    # 2. AI Path (Inference)
    ai_path = calculate_ai_path(sim_state.src, sim_state.dst, dynamic_graph)

    sats_data = []

    for sat_id, sat in sim_state.constellation.satellites.items():
        pos = sat.get_position(current_time)
        sats_data.append(SatelliteData(
            id=sat_id,
            orbit_id=sat.orbit_id,
            phase_id=sat.phase_id,
            position=pos.tolist(),
            is_active=sat_id not in sim_state.broken_nodes,
            is_path=(sat_id in sim_state.active_path)
        ))

    # Extract edges with Load Info
    edges = []
    for u, v in dynamic_graph.edges():
        attrs = dynamic_graph.edges[u, v]
        load = attrs.get('bw_occupied', 0) / attrs.get('bw_total', 1000)
        edges.append({'u': u, 'v': v, 'load': load})

    return {
        "time": current_time,
        "is_paused": sim_state.is_paused,
        "satellites": sats_data,
        "path": sim_state.active_path, # Green
        "ai_path": ai_path,            # Magenta
        "edges": edges,
        "orbits": sim_state.constellation.visible_orbits
    }

@app.post("/simulation/pause")
def pause_simulation():
    if not sim_state.is_paused:
        sim_state.is_paused = True
        sim_state.last_pause_timestamp = time.time()
        # Capture the exact logical time we paused at
        # Recalculate it manually to be safe or use getter?
        # Note: getter depends on is_paused flag.
        # We need to compute logical time BEFORE setting flag?
        # Actually logic in getter: if is_paused return paused_time.
        # So we must set paused_time first.

        # Re-calc current logic time
        real_elapsed = time.time() - sim_state.start_time - sim_state.accumulated_pause_duration
        sim_state.paused_time = real_elapsed * 50

    return {"status": "paused", "time": sim_state.paused_time}

@app.post("/simulation/resume")
def resume_simulation():
    if sim_state.is_paused:
        sim_state.is_paused = False
        # Calculate how long we were paused
        pause_duration = time.time() - sim_state.last_pause_timestamp
        sim_state.accumulated_pause_duration += pause_duration

    return {"status": "resumed"}

@app.post("/simulation/reset")
def reset_simulation():
    sim_state.initialize()
    return {"status": "reset", "node_count": len(sim_state.constellation.satellites)}

@app.post("/simulation/toggle/{sat_id}")
def toggle_satellite(sat_id: int):
    if sat_id not in sim_state.constellation.satellites:
        raise HTTPException(status_code=404, detail="Satellite not found")
    
    if sat_id in sim_state.broken_nodes:
        sim_state.broken_nodes.remove(sat_id)
        sim_state.constellation.set_satellite_status(sat_id, True)
    else:
        sim_state.broken_nodes.add(sat_id)
        sim_state.constellation.set_satellite_status(sat_id, False)
        
    sim_state.recalculate_path()
    
    return {
        "sat_id": sat_id, 
        "is_active": sat_id not in sim_state.broken_nodes,
        "new_path": sim_state.active_path
    }

@app.get("/simulation/edges")
def get_edges():
    # Static topology unless we want dynamic updates, but React app can filter by active status
    edges = []
    for u, v in sim_state.constellation.graph.edges():
        edges.append([u, v])
    return edges

# --- Training Integration ---
import threading
from src.train import train

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.current_episode = 0
        self.latest_reward = 0.0
        self.epsilon = 0.0
        self.history = []
        self.thread = None
        self.agent = None # Store model here
        self.gnn = None
        self.get_features_fn = None

    def _train_worker(self, num_episodes):
        print("DEBUG: _train_worker started")
        self.is_training = True
        self.history = []
        
        try:
            print("DEBUG: Importing train module...")
            # We need to grab get_node_features
            from src.train import get_node_features, train
            self.get_features_fn = get_node_features
            
            def callback(ep, reward, eps):
                self.current_episode = ep
                self.latest_reward = reward
                self.epsilon = eps
                self.history.append({'episode': ep, 'reward': reward})
                if ep % 5 == 0:
                    print(f"DEBUG: Episode {ep} completed. R={reward}")
                
            print(f"DEBUG: Starting training logic for {num_episodes} episodes...")
            # Run training
            agent, gnn, history = train(num_episodes=num_episodes, progress_callback=callback)
            print("DEBUG: Training finished successfully.")
            
            # Store result
            self.agent = agent
            self.gnn = gnn
            
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("DEBUG: _train_worker finished. Resetting is_training flag.")
            self.is_training = False
        # OR: We can't see the evolution LIVE unless we refactor train to update an external agent ref.
        # Let's keep it simple: Updated AFTER training. 
        # User asked "Training process", usually means charts.
        
        # Actually user wants "Show that it uses AI... see it struggle then snap".
        # This implies LIVE updates.
        # I need to refactor train to yield the agent or update a shared reference.
        # But `train` creates the agent inside.
        
        # Hack: The callback can fetch the agent from the local scope? No.
        # I will change logic: `train` returns agent at end.
        # But for live viz, I need the agent object accessible.
    def start_training(self, num_episodes=50):
        if self.is_training:
            return False
            
        self.thread = threading.Thread(target=self._train_worker, args=(num_episodes,))
        self.thread.start()
        return True

    def load_saved_model(self):
        # Helper to load model if exists (without running train)
        import os
        model_path = "model.pth"
        if os.path.exists(model_path):
            print("Found saved model. Loading...")
            try:
                from src.train import train
                # train() checks for existence and returns loaded model
                agent, gnn, _ = train(num_episodes=0) 
                self.agent = agent
                self.gnn = gnn
                self.get_features_fn = lambda c: torch.eye(len(c.satellites)) # Placeholder if import fails
                # Re-import proper feature fn
                from src.train import get_node_features
                self.get_features_fn = get_node_features
                print("Model loaded into TrainingManager.")
            except Exception as e:
                print(f"Failed to load model: {e}")

training_mgr = TrainingManager()
# Attempt load on startup
training_mgr.load_saved_model()

@app.post("/training/start")
def start_training():
    started = training_mgr.start_training(num_episodes=50)
    if not started:
        return {"status": "Already training"}
    return {"status": "Training started", "target_episodes": 50}

@app.get("/training/status")
def get_training_status():
    return {
        "is_training": training_mgr.is_training,
        "episode": training_mgr.current_episode,
        "reward": training_mgr.latest_reward,
        "epsilon": training_mgr.epsilon,
        "history": training_mgr.history[-50:] # Send last 50 points for chart
    }
