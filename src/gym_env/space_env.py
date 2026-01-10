import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.physics import OrbitalPhysics
from src.core.entities import Mothership, Debris

class SpaceJanitorEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_motherships=3, num_debris=10, dt=10.0):
        super().__init__()
        self.num_motherships = num_motherships
        self.num_debris = num_debris
        self.dt = dt
        self.physics = OrbitalPhysics()
        
        # Action Space: MultiDiscrete
        self.action_space = spaces.MultiDiscrete([num_debris + 1] * num_motherships)
        
        # Observation Space
        # Global state
        obs_dim = (num_motherships * 5) + (num_debris * 7)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.motherships = []
        self.debris_list = []
        self.time_elapsed = 0.0
        
        self.reset()
        
    def _create_random_orbit(self, alt_km=1000.0, inc_deg=45.0):
        r_mag = (6371.0 + alt_km) * 1000.0
        v_mag = np.sqrt(3.986e14 / r_mag)
        
        Theta = np.random.uniform(0, 2*np.pi)
        # Random position on sphere
        u_r = np.random.randn(3)
        u_r /= np.linalg.norm(u_r)
        r_vec = u_r * r_mag
        
        # Random velocity perp
        u_temp = np.random.randn(3)
        u_perp = np.cross(u_r, u_temp)
        u_perp /= np.linalg.norm(u_perp)
        v_vec = u_perp * v_mag
        
        return self.physics.create_orbit_from_vectors(r_vec, v_vec)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.motherships = []
        self.debris_list = []
        self.time_elapsed = 0.0
        
        for i in range(self.num_motherships):
            orbit = self._create_random_orbit(alt_km=800)
            self.motherships.append(Mothership(i, orbit, self.physics))
            
        for i in range(self.num_debris):
            orbit = self._create_random_orbit(alt_km=1200 + np.random.uniform(-100, 100))
            self.debris_list.append(Debris(i, orbit, self.physics))
            
        return self._get_obs(), {}

    def step(self, action):
        rewards = 0.0
        
        # 1. Process Actions
        for i, ms in enumerate(self.motherships):
            target_idx = action[i]
            
            if ms.state == "TRANSFER":
                continue
                
            if target_idx < self.num_debris:
                target = self.debris_list[target_idx]
                if target.active:
                     # Check Range first
                     ms_pos = ms.position
                     dist_to_target = np.linalg.norm(ms_pos - target.position)
                     if dist_to_target > 2000000.0: # 2000 km range limit
                         rewards -= 1.0 # Out of range penalty
                     else:
                         dt_transfer = 900.0
                         future_orbit = self.physics.propagate(target.orbit, dt_transfer)
                         future_pos = self.physics.get_state_vector(future_orbit)[:3]
                         
                         req_v, cost = ms.plan_transfer(future_pos, dt_transfer)
                         
                         if req_v is not None:
                             success = ms.execute_transfer(req_v, cost, target.id, dt_transfer)
                             if success:
                                 rewards -= (cost * 0.01) 
                         else:
                             # Plan failed (Safety or Delta-V or Fuel)
                             rewards -= 10.0 # Heavy penalty for unsafe/impossible action
                else:
                    rewards -= 0.5
            else:
                 pass # IDLE

        # 2. Update Transfer
        for ms in self.motherships:
            if ms.state == "TRANSFER":
                 if ms.transfer_timer >= ms.transfer_duration:
                     ms.complete_transfer()
        
        # Propagate
        for ms in self.motherships:
            ms.update(self.dt)
        for d in self.debris_list:
            d.update(self.dt)
            
        self.time_elapsed += self.dt
        
        # 3. Check Captures
        for ms in self.motherships:
             ms_pos = ms.position
             for d in self.debris_list:
                 if d.active:
                     dist = np.linalg.norm(ms_pos - d.position)
                     if dist < 50000.0:
                         d.active = False
                         rewards += 100.0
                         
        terminated = all(not d.active for d in self.debris_list)
        truncated = False
        
        return self._get_obs(), rewards, terminated, truncated, {}

    def _get_obs(self):
        obs = []
        for ms in self.motherships:
            state_val = 1.0 if ms.state == "TRANSFER" else 0.0
            vel = ms.velocity
            obs.extend([ms.fuel, state_val, vel[0], vel[1], vel[2]])
            
        for d in self.debris_list:
            active_val = 1.0 if d.active else 0.0
            pos = d.position
            vel = d.velocity
            obs.extend([active_val, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
            
        return np.array(obs, dtype=np.float32)
