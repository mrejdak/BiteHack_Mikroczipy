from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np

# Scale: 1 Ursina Unit = 1000 km
SCALE_FACTOR = 1 / 1000.0 

def create_orbit(alt_km=400, inc_deg=51.6, ecc=0.0):
    """
    Creates a circular orbit around Earth.
    """
    return Orbit.circular(Earth, alt=alt_km * u.km, inc=inc_deg * u.deg)

def get_position(orbit, time_offset_seconds):
    """
    Propagates the orbit by `time_offset_seconds` and returns the (x, y, z) position in km.
    """
    tof = time_offset_seconds * u.s
    # Propagate gives a new Orbit object at the new time
    new_orbit = orbit.propagate(tof)
    
    # Extract radius vector (x, y, z) with astropy units
    r_vec = new_orbit.r.to(u.km).value
    return r_vec

def get_state_vector(orbit):
    """Returns position (km) and velocity (km/s) vectors."""
    r = orbit.r.to(u.km).value
    v = orbit.v.to(u.km / u.s).value
    return r, v

def create_orbit_from_vectors(r_vec, v_vec):
    """Creates an orbit from position (km) and velocity (km/s) vectors."""
    return Orbit.from_vectors(Earth, r_vec * u.km, v_vec * u.km / u.s)

def apply_impulse(orbit, delta_v_km_s):
    """
    Applies an instantaneous velocity change to the orbit.
    Returns a NEW Orbit object.
    
    delta_v_km_s: List or array [dvx, dvy, dvz] in km/s
    """
    # Get current state
    r, v = get_state_vector(orbit)
    
    # Apply delta v
    v_new = [v[0] + delta_v_km_s[0], v[1] + delta_v_km_s[1], v[2] + delta_v_km_s[2]]
    
    # Create new orbit
    return Orbit.from_vectors(Earth, r * u.km, v_new * u.km / u.s)
