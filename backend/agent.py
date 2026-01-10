import requests
import time
import threading
import numpy as np
from stable_baselines3 import PPO

# Try to use local sim environment only for verifying shapes if needed, 
# but for inference we just need the model.
# from simulation.gym_env import OrbitalJanitorEnv

SERVER_URL = "http://localhost:5000"
MODEL_PATH = "models/janitor_final" # No .zip extension needed

class MothershipAgent:
    def __init__(self, sensor_radius=2000):
        self.id = None
        self.sensor_radius = sensor_radius
        self.running = False
        self.model = None
        self.fuel = 1000.0 # Match training environment
        
        # Load Model
        try:
            print(f"[Agent] Loading model from {MODEL_PATH}...")
            self.model = PPO.load(MODEL_PATH)
            print("[Agent] Model loaded successfully.")
        except Exception as e:
            print(f"[Agent] FAILED to load model: {e}")
            self.model = None

    def start(self):
        """Registers and starts the control loop in a thread."""
        if self.register():
            self.running = True
            self.thread = threading.Thread(target=self._control_loop)
            self.thread.start()
            
    def register(self):
        try:
            resp = requests.post(f"{SERVER_URL}/api/mothership/register")
            if resp.status_code == 200:
                self.id = resp.json()["id"]
                print(f"[Agent] Registered as {self.id}")
                return True
        except Exception as e:
            print(f"[Agent] Registration failed: {e}")
        return False

    def _control_loop(self):
        while self.running:
            try:
                # 0. Initial Sync (No action, just get state)
                state = self._sync_state(action=None)
                if not state:
                    time.sleep(1)
                    continue
                    
                # 1. Process State -> Observation
                obs = self._get_observation(state)
                
                # 2. Predict Action
                if self.model and obs is not None:
                    # Model outputs Thrust Vector [Fx, Fy, Fz] in LVLH frame (normalized -1 to 1)
                    # Training Env scales this by max_thrust=1000N
                    action_norm, _ = self.model.predict(obs, deterministic=True)
                    
                    # 3. Convert Action to Delta-V (Inertial Frame)
                    delta_v_inertial = self._process_action(action_norm, state)
                    
                    # 4. Send Action
                    if np.linalg.norm(delta_v_inertial) > 0:
                        self._sync_state(action={"delta_v": delta_v_inertial.tolist()})
                        # time.sleep(1.0) # Wait for action to take effect? Sim is 100x realtime?
                        # If sim is 100x, then 1s real = 100s sim.
                        # Agent step should match training dt (1s).
                        # So we should sleep 0.01s? No, network latency is >10ms.
                        # Let's run at ~1Hz real time for stability.
                
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"[{self.id}] Error: {e}")
                time.sleep(1)

    def _sync_state(self, action=None):
        payload = {
            "id": self.id,
            "sensor_radius": self.sensor_radius
        }
        if action:
            payload["action"] = action
            
        try:
            resp = requests.post(f"{SERVER_URL}/api/mothership/sync", json=payload)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def _get_observation(self, state):
        """
        Constructs the 7-dim observation vector:
        [rx, ry, rz, vx, vy, vz, fuel]
        All vectors are Relative in LVLH frame.
        """
        my_pos = np.array(state["self_state"]["pos"]) # km
        my_vel = np.array(state["self_state"].get("vel", [0,0,0])) # km/s
        
        nearby = state["nearby_objects"]
        if not nearby:
            # No target, can't form observation. Just drift or explore (not trained for exploration).
            return None
            
        # Strategy: Find Closest Target
        closest = min(nearby, key=lambda x: x['distance'])
        target_pos = np.array(closest["pos"]) # km
        target_vel = np.array(closest.get("vel", [0,0,0])) # km/s
        
        # --- COORDINATE TRANSFORMATION (Inertial -> LVLH) ---
        # 1. Basis Vectors
        # Inputs are in km/s. Convert to m/s for precision if needed? 
        # Actually Model inputs are normalized, but math should be consistent.
        # Let's use METERS for internal math to match training env exactly.
        
        r_vec = my_pos * 1000.0 # m
        v_vec = my_vel * 1000.0 # m/s
        
        tr_vec = target_pos * 1000.0
        tv_vec = target_vel * 1000.0
        
        r_mag = np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)
        
        if r_mag == 0 or h_mag == 0: return None
        
        i_unit = r_vec / r_mag
        k_unit = h_vec / h_mag
        j_unit = np.cross(k_unit, i_unit)
        
        Q = np.array([i_unit, j_unit, k_unit]) # Rotation Matrix
        
        # 2. Relative State (Inertial)
        delta_r = tr_vec - r_vec
        delta_v = tv_vec - v_vec
        
        # 3. Transport Theorem Correction (Omega x r)
        omega_vec = h_vec / (r_mag**2)
        cross_term = np.cross(omega_vec, delta_r)
        dv_rotating = delta_v - cross_term
        
        # 4. Rotate to LVLH
        dr_lvlh = Q @ delta_r # meters
        dv_lvlh = Q @ dv_rotating # m/s
        
        # --- NORMALIZATION (Must match gym_env.py) ---
        # Pos: Divide by 50km (position in meters, 50000m = 1.0)
        # Vel: Divide by 100m/s
        # Fuel: Divide by 1000
        obs_dr = dr_lvlh / 50000.0
        obs_dv = dv_lvlh / 100.0
        obs_fuel = self.fuel / 1000.0
        
        obs = np.concatenate([obs_dr, obs_dv, [obs_fuel]]).astype(np.float32)
        return obs

    def _process_action(self, action_norm, state):
        """
        Converts normalized Action (Thrust LVLH) -> Delta-V (Inertial km/s)
        """
        # 1. Scale Action
        # Match training environment: 5000N thrust
        max_thrust = 5000.0  # Newtons - matches training
        thrust_lvlh = action_norm * max_thrust # Newtons [Fx, Fy, Fz] in LVLH
        
        # 2. Rotate LVLH -> Inertial
        my_pos = np.array(state["self_state"]["pos"]) * 1000.0
        my_vel = np.array(state["self_state"]["vel"]) * 1000.0
        
        r_mag = np.linalg.norm(my_pos)
        h_vec = np.cross(my_pos, my_vel)
        
        i_unit = my_pos / r_mag
        k_unit = h_vec / np.linalg.norm(h_vec)
        j_unit = np.cross(k_unit, i_unit)
        
        Q_T = np.column_stack([i_unit, j_unit, k_unit]) # Transpose = Inverse
        
        thrust_inertial = Q_T @ thrust_lvlh # Newtons [Fx, Fy, Fz] Inertial
        
        # 3. Calculate Delta-V
        # F = ma -> a = F/m
        # dv = a * dt
        mass = 100.0 # kg
        dt = 1.0 # Training step size
        
        dv_meters_per_sec = (thrust_inertial / mass) * dt
        
        # 4. Convert to km/s
        dv_kms = dv_meters_per_sec / 1000.0
        
        # Update fuel tracking (approx)
        thrust_mag = np.linalg.norm(thrust_lvlh)
        self.fuel -= thrust_mag * dt * 0.001
        if self.fuel < 0: self.fuel = 0
            
        return dv_kms

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

if __name__ == "__main__":
    agent = MothershipAgent()
    agent.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
