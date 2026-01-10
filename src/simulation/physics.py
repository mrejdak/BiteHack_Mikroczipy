import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.iod import izzo

class OrbitalPhysics:
    def __init__(self):
        self.earth = Earth
        self.k = Earth.k # Gravitational parameter

    def create_orbit(self, r_vec, v_vec):
        """Creates a Poliastro Orbit object from vectors."""
        # Input vectors expected in meters and meters/second
        r = r_vec * u.m
        v = v_vec * u.m / u.s
        return Orbit.from_vectors(self.earth, r, v)

    def propagate(self, orbit, dt_seconds):
        """Propagates orbit by dt seconds."""
        dt = dt_seconds * u.s
        return orbit.propagate(dt)

    def solve_lambert(self, orbit_curr, r_target, dt_seconds):
        """
        Solves Lambert's problem to find the transfer velocity.
        Returns: required_velocity_vector (m/s), delta_v cost (m/s)
        """
        # Target position needs to be Quantity
        r_f = r_target * u.m
        tof = dt_seconds * u.s
        
        # We assume the maneuver starts NOW at the current orbit position
        # Poliastro's manuever.lambert solves the arc
        
        # Using fast izzo implementation from core
        try:
            (v0, v1), = izzo.lambert(self.k, orbit_curr.r, r_f, tof)
            
            # v0 is the velocity we NEED to have at start
            # current velocity is orbit_curr.v
            # delta_v is the difference
            
            required_v = v0.to(u.m/u.s).value
            current_v = orbit_curr.v.to(u.m/u.s).value
            
            delta_v_vec = required_v - current_v
            delta_v_mag = np.linalg.norm(delta_v_vec)
            
            return required_v, delta_v_mag
            
        except Exception as e:
            # Lambert solution failed (e.g. geometric constraints)
            return None, float('inf')

    def get_state(self, orbit):
        """Extracts raw numpy state [x, y, z, vx, vy, vz] from orbit."""
        r = orbit.r.to(u.m).value
        v = orbit.v.to(u.m/u.s).value
        # Assuming 2D/3D consistency, handling full 3D vectors
        return np.concatenate([r, v])
