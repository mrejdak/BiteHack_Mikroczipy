import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from .physics import OrbitalPhysics
from .entities import Mothership, Debris

R_EARTH = 6371000.0
LEO_ALTITUDE = 1000000.0  # Increased to 1000km for better visibility vs Earth scale

class SpaceJanitorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, num_motherships=3, num_debris=10):
        self.physics = OrbitalPhysics()
        self.render_mode = render_mode
        self.num_motherships = num_motherships
        self.num_debris = num_debris
        
        self.motherships = []
        self.debris_list = []
        
        self.action_space = spaces.MultiDiscrete([num_debris + 1] * num_motherships)
        
        # Observation space placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        self.screen = None
        self.clock = None
        self.earth_img = None
        
        self.time_elapsed = 0
        self.time_elapsed = 0
        self.sim_step_size = 10.0 # 10 seconds per frame = slower, smoother
        self.rotation_angle = 0.0
        self.rotation_angle = 0.0 # For rotating the camera/earth

    def _init_orbit_vectors_3d(self, altitude):
        # Fully random 3D orbit
        r_mag = R_EARTH + altitude
        
        # Random Inclination and RAAN
        phi = np.random.uniform(0, np.pi) # Inclination
        theta = np.random.uniform(0, 2*np.pi) # Position on plane
        
        # Simple Spherical to Cartesian for position
        x = r_mag * np.sin(phi) * np.cos(theta)
        y = r_mag * np.sin(phi) * np.sin(theta)
        z = r_mag * np.cos(phi)
        r = np.array([x, y, z])
        
        # Velocity? 
        # For a circular orbit, V is perpendicular to R and lies in the orbital plane.
        # Speed magnitude
        v_mag = np.sqrt(3.986004418e14 / r_mag)
        
        # Create a random velocity vector perpendicular to position
        # 1. Random vector
        rand_vec = np.random.randn(3)
        # 2. Cross product with R to get perpendicular
        perp = np.cross(r, rand_vec)
        # 3. Normalize and scale
        perp = perp / np.linalg.norm(perp)
        v = perp * v_mag
        
        return r, v

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.motherships = []
        self.debris_list = []
        self.time_elapsed = 0
        
        for i in range(self.num_motherships):
            r, v = self._init_orbit_vectors_3d(LEO_ALTITUDE)
            ms = Mothership(i, r, v, self.physics)
            self.motherships.append(ms)
            
        for i in range(self.num_debris):
            # Vary altitude for 3D depth
            alt = LEO_ALTITUDE + np.random.uniform(200000, 1000000)
            r, v = self._init_orbit_vectors_3d(alt)
            d = Debris(i, r, v, self.physics)
            self.debris_list.append(d)
            
        return self._get_obs(), {}

    def step(self, action):
        # Action Logic (Simplified PPO adapter)
        # Check maneuvers (same as before but 3D vectors)
        
        # Simulation Step
        for obj in self.motherships + self.debris_list:
            obj.update(self.sim_step_size)
        
        # Collision Logic
        for ms in self.motherships:
            if ms.state_status == "TRANSFER":
               # Mock transfer logic
               ms.state_status = "IDLE" # Instant transfer for visual MVP

            # Collision/Capture Logic
            ms_pos = ms.position
            for debris in self.debris_list:
                if debris.active:
                    dist = np.linalg.norm(ms_pos - debris.position)
                    # Capture threshold: 100km (100,000 m) - large for visibility
                    if dist < 100000.0:
                        debris.active = False
                        print(f"Captured Debris {debris.id}!")
        
        self.time_elapsed += self.sim_step_size
        self.rotation_angle += 0.002 # Slow camera rotation
        
        # Human render mode now handled externally by Ursina App
        # if self.render_mode == "human":
        #     self.render()
            
        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        return np.zeros(1, dtype=np.float32)

    def project_3d(self, vec3, width, height, scale, rotation):
        """ Projects 3D world coordinates to 2D screen coordinates with rotation """
        x, y, z = vec3
        
        # Rotate around Y axis (Camera orbit)
        cos_a = math.cos(rotation)
        sin_a = math.sin(rotation)
        
        bx = x * cos_a - z * sin_a
        bz = x * sin_a + z * cos_a
        by = y
        
        # Perspective projection
        # Camera distance
        cam_dist = R_EARTH * 5.0
        fov = 500.0
        
        # Simple perspective division
        factor = fov / (cam_dist - bz)
        
        sx = bx * factor * scale + width / 2
        sy = -by * factor * scale + height / 2 # Y flip
        
        scale_factor = factor # For sizing dots based on depth
        
        return int(sx), int(sy), scale_factor, bz

    # Render methods removed/deprecated in favor of external visualizer
    def render(self):
        pass

    def close(self):
        pass
