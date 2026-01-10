import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Optional, Tuple

from .physics_engine import PhysicsEngine, OrbitalBody, G, M_EARTH, R_EARTH

class OrbitalJanitorEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_debris=5, max_steps=1000):
        super(OrbitalJanitorEnv, self).__init__()
        
        self.num_debris = num_debris
        self.max_steps = max_steps
        self.current_step = 0
        
        self.engine: Optional[PhysicsEngine] = None
        self.agent_id = "agent_007"
        
        # Observation Space: Relative Physics (LVLH)
        # 1. Relative Position to Target [rx, ry, rz] (3)
        # 2. Relative Velocity to Target [rvx, rvy, rvz] (3)
        # 3. Fuel Level (1)
        # Total = 7
        obs_size = 7
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action Space: Continuous Thrust [Fx, Fy, Fz]
        # Range [-1, 1], scaled to max_thrust
        self.max_thrust = 1000.0 # Newtons (Reduced for precision)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def generate_orbit(self, altitude_km: float) -> Tuple[np.ndarray, np.ndarray]:
        # Create a circular orbit at random inclination
        r = R_EARTH + (altitude_km * 1000.0)
        v_mag = math.sqrt(G * M_EARTH / r)
        
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        pos = np.array([x, y, z])
        
        random_vec = np.random.randn(3)
        tangent = np.cross(pos, random_vec)
        if np.linalg.norm(tangent) == 0:
            tangent = np.cross(pos, np.array([1, 0, 0]))
        
        tangent = tangent / np.linalg.norm(tangent)
        vel = tangent * v_mag
        
        return pos, vel

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.engine = PhysicsEngine(dt=1.0) # 1 sec per step

        # 1. Create Agent
        pos, vel = self.generate_orbit(altitude_km=400)
        agent = OrbitalBody(
            id=self.agent_id,
            position=pos,
            velocity=vel,
            mass=100.0,
            radius=1.0,
            type='janitor',
            fuel=1000.0
        )
        self.engine.add_body(agent)

        # 2. Create Target (Single Target for Curriculum)
        # Curriculum: Start relatively close (1km - 5km)
        # In Hill's frame, we just offset position and velocity slightly
        # But we need to convert that back to inertial for the engine
        
        # Curriculum: Start VERY close for tutorial mode (200m - 1000m)
        dist_offset = np.random.uniform(200, 1000) 
        pos_offset = np.random.randn(3)
        pos_offset /= np.linalg.norm(pos_offset)
        pos_offset *= dist_offset
        
        # Velocity offset: Small drift (0.1 to 2 m/s)
        vel_offset = np.random.randn(3) * np.random.uniform(0.1, 2.0)
        
        debris = OrbitalBody(
            id="target_debris",
            position=pos + pos_offset,
            velocity=vel + vel_offset,
            mass=10.0,
            radius=2.0,
            type='debris'
        )
        self.engine.add_body(debris)

        return self._get_obs(), {}

    def _get_obs(self):
        # We assume Single Target logic for the Pilot Agent
        rel_state = self.engine.get_relative_state_lvlh(self.agent_id, "target_debris")
        agent = self.engine.get_body(self.agent_id)
        
        if not rel_state or not agent:
             return np.zeros(7).astype(np.float32)

        # Normalize Inputs
        # Pos: Divide by 1km (so 1000m = 1.0) - More sensitive now
        # Vel: Divide by 10m/s (so 10m/s = 1.0)
        # Fuel: Divide by 1000
        
        dr = rel_state['dr'] / 1000.0 
        dv = rel_state['dv'] / 10.0
        fuel = agent.fuel / 1000.0
        
        return np.concatenate([dr, dv, [fuel]]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Execute Action
        thrust_vector = action * self.max_thrust
        
        # Thrust is in BODY frame (or LVLH?). Ideally Body. 
        # For simplicity, let's assume Agent can thrust in LVLH directions instantly (3-axis control).
        # We need to rotate this thrust vector from LVLH to Inertial before applying to PhysicsEngine.
        
        # ... Re-calculating Q (Rotation Matrix) strictly for thrust application
        agent = self.engine.get_body(self.agent_id)
        if not agent: return np.zeros(7), -1000, True, False, {}
        
        r_vec = agent.position
        v_vec = agent.velocity
        r_mag = np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        
        if r_mag > 0 and np.linalg.norm(h_vec) > 0:
            i_unit = r_vec / r_mag
            k_unit = h_vec / np.linalg.norm(h_vec)
            j_unit = np.cross(k_unit, i_unit)
            Q_T = np.column_stack([i_unit, j_unit, k_unit]) # Transpose of Q is inverse
            
            # inertial_thrust = Q_T @ action_lvlh
            thrust_inertial = Q_T @ thrust_vector
        else:
            thrust_inertial = thrust_vector # Fallback
            
        self.engine.step(thrust_vectors={self.agent_id: thrust_inertial})
        
        # 2. Reward Calculation
        rel_state = self.engine.get_relative_state_lvlh(self.agent_id, "target_debris")
        if not rel_state:
             return np.zeros(7), -1000.0, True, False, {}
             
        dist = rel_state['dist']
        dv_mag = np.linalg.norm(rel_state['dv'])
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # 2a. Constant Existence Penalty (Urgency)
        reward -= 0.01
        
        # 2b. Fuel Penalty
        reward -= 0.001 * np.linalg.norm(action)
        
        # 2c. Distance Reward (Dense)
        # Weight currently: -dist/1000. If 500m -> -0.5. 
        reward -= (dist / 1000.0)
        
        # 2d. Velocity Matching Bonus
        # If we are close (<100m) AND slow (<2m/s), give bonus
        if dist < 100.0 and dv_mag < 2.0:
            reward += 1.0
            
        # Velocity penalty is always active to encourage stability
        reward -= 0.1 * dv_mag
        
        # 2e. Terminal Success
        # Capture: Dist < 20m AND Relative Vel < 2 m/s
        if dist < 20.0:
            if dv_mag < 2.0:
                reward += 1000.0
                print(f"Captured! Dist: {dist:.2f}, dV: {dv_mag:.2f}")
                terminated = True
            else:
                # Close but too fast! Crash risk!
                reward -= 10.0 # Warning penalty
        
        # 2f. Crash/Lost
        if dist > 20000.0: # Lost track (>20km)
            reward -= 500.0
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

