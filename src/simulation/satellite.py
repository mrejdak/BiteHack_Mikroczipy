import math
import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class Satellite:
    def __init__(self, id, orbit_id, phase_id, altitude=550, inclination=53, raan=0, mean_anomaly=0):
        self.id = id
        self.orbit_id = orbit_id
        self.phase_id = phase_id
        self.is_active = True
        self.neighbors = {} # Direction -> Satellite ID
        
        # --- Poliastro Orbit Initialization (User Pattern) ---
        # "create_orbit" logic integrated into __init__
        alt_km = altitude
        inc_deg = inclination
        
        # Note: Orbit.circular assumes inc, but what about raan/mean_anomaly?
        # Orbit.circular(Earth, alt=..., inc=..., ...) creates a circular orbit.
        # But we need to specify RAAN (Omega) and Mean Anomaly (nu) for the constellation.
        # Orbit.circular takes: (attractor, alt, inc, raan, arglat)
        # arglat (Argument of Latitude) = arg_p + nu for circular orbits.
        
        # User snippet: return Orbit.circular(Earth, alt=alt_km * u.km, inc=inc_deg * u.deg)
        # But that misses RAAN/Anomaly.
        # I will use Orbit.from_classical BUT strictly with the patterns they requested where possible,
        # OR use Orbit.circular and then modify? No, Orbit is immutable.
        
        # Let's check Poliastro docs mental cache: Orbit.circular(..., raan=..., arglat=...) exists.
        # https://docs.poliastro.space/en/stable/api/safe/twobody/orbit.html#poliastro.twobody.orbit.Orbit.circular
        
        self.orbit = Orbit.circular(
            Earth, 
            alt=alt_km * u.km, 
            inc=inc_deg * u.deg, 
            raan=raan * u.deg, 
            arglat=mean_anomaly * u.deg # For circular, Mean Anomaly ~ Argument of Latitude (w+v)
        )
        
        # Pre-calc parameters for Fast Propagation (Circular)
        # Mean Motion (n) = sqrt(mu / a^3) for circular
        # Earth mu = 3.986004418e5 km^3/s^2
        mu = 3.986004418e5 
        a = (Earth.R.to(u.km).value + altitude) # semi-major axis in km
        self.n_rad_s = math.sqrt(mu / (a**3)) # rad/s
        
        self.raan_rad = math.radians(raan)
        self.inc_rad = math.radians(inclination)
        self.start_anomaly_rad = math.radians(mean_anomaly)
        self.curr_r = a
        
    def get_position(self, time_offset_seconds):
        """
        Fast Analytical Propagation for Circular Orbits.
        Returns (x, y, z) in km.
        """
        # 1. Update Mean Anomaly
        M = self.start_anomaly_rad + self.n_rad_s * time_offset_seconds
        
        # For circular, True Anomaly (nu) = Mean Anomaly (M)
        # Argument of Latitude (u) = arg_p + nu. Here arg_p=0, so u = M.
        u_lat = M
        
        # 2. Convert to Cartesian (Inertial Frame)
        # x = r * (cos(OMG)*cos(u) - sin(OMG)*sin(u)*cos(i))
        # y = r * (sin(OMG)*cos(u) + cos(OMG)*sin(u)*cos(i))
        # z = r * (sin(u)*sin(i))
        
        c_O = math.cos(self.raan_rad)
        s_O = math.sin(self.raan_rad)
        c_u = math.cos(u_lat)
        s_u = math.sin(u_lat)
        c_i = math.cos(self.inc_rad)
        s_i = math.sin(self.inc_rad)
        
        x = self.curr_r * (c_O * c_u - s_O * s_u * c_i)
        y = self.curr_r * (s_O * c_u + c_O * s_u * c_i)
        z = self.curr_r * (s_u * s_i)
        
        return np.array([x, y, z])

    def get_state_vector(self):
         # Not implemented in fast mode
         return np.zeros(3), np.zeros(3)

    def apply_impulse(self, delta_v_km_s):
         pass 

    def add_neighbor(self, direction, neighbor_id):
        """Neighbors directions: 'UP', 'DOWN', 'LEFT', 'RIGHT'"""
        self.neighbors[direction] = neighbor_id

    def set_active(self, active):
        self.is_active = active

    def __repr__(self):
        return f"Sat(ID={self.id}, Orbit={self.orbit_id}, Phase={self.phase_id})"
