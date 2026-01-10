import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo

class OrbitalPhysics:
    def __init__(self):
        self.k = Earth.k
        
    def create_orbit_from_vectors(self, r, v):
        """
        Create Orbit object.
        r: [x, y, z] in meters
        v: [vx, vy, vz] in m/s
        """
        r_vec = r * u.m
        v_vec = v * u.m / u.s
        return Orbit.from_vectors(Earth, r_vec, v_vec)
        
    def propagate(self, orbit, dt_seconds):
        """Returns new Orbit object after dt_seconds"""
        return orbit.propagate(dt_seconds * u.s)
        
    def solve_lambert(self, orbit_start, r_final, dt_seconds):
        """
        Solves for transfer velocity.
        orbit_start: Current Orbit object
        r_final: [x, y, z] Target position in meters
        dt: Time of flight in seconds
        
        Returns: (v_required_vector_m_s, delta_v_cost_m_s) or (None, Inf)
        """
        try:
            r0 = orbit_start.r
            rf = r_final * u.m
            tof = dt_seconds * u.s
            
            solutions = izzo.lambert(self.k, r0, rf, tof)
            # It seems izzo.lambert returns (v0, v1) directly in this version/context
            v0, v1 = solutions
            
            v_req = v0.to(u.m / u.s).value
            v_curr = orbit_start.v.to(u.m / u.s).value
            
            delta_v = np.linalg.norm(v_req - v_curr)
            return v_req, delta_v
            
        except Exception as e:
            # Lambert solver failure (geometry, hyperbola, etc.)
            print(f"Lambert Exception: {e}")
            import traceback
            traceback.print_exc()
            return None, float('inf')

    def get_state_vector(self, orbit):
        """Returns numpy array [x, y, z, vx, vy, vz] (m, m/s)"""
        r = orbit.r.to(u.m).value
        v = orbit.v.to(u.m / u.s).value
        return np.concatenate([r, v])

    def check_orbit_safety(self, orbit, min_alt_km=100.0):
        """
        Returns True if orbit perigee is greater than Earth Radius + min_alt.
        R_Earth approx 6371 km.
        """
        r_p = orbit.r_p.to(u.km).value
        safe_r = 6371.0 + min_alt_km
        if r_p < safe_r:
            return False
            
        # Also check apogee? No, apogee being high is fine usually, unless escape.
        # Check eccentricity < 1 (Elliptical)
        if orbit.ecc.value >= 1.0:
             # Hyperbolic/Parabolic - Leaving Earth system?
             # For this simulation, maybe we ban escape trajectories?
             return False
             
        return True
