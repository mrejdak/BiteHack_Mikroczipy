import random
from orbital_physics import create_orbit
from mocat_bridge import get_mean_debris_count
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u

def generate_debris_orbits(num_debris=None, altitude_km=800):
    """
    Generates `num_debris` random orbits at a specific altitude shell.
    If `num_debris` is None, it queries the MOCAT simulation.
    """
    if num_debris is None:
        try:
            print("Querying MOCAT for debris count...")
            num_debris = get_mean_debris_count()
            print(f"MOCAT returned: {num_debris} objects")
        except Exception as e:
            print(f"MOCAT failed: {e}, defaulting to 500")
            num_debris = 500

    orbits = []
    
    # 2. Debris Cloud with Variable Altitudes
    # 400km to 1200km is typical LEO range for debris
    MIN_ALT = 400
    MAX_ALT = 1200

    for _ in range(num_debris):
        # Random Altitude Shell
        alt = random.uniform(MIN_ALT, MAX_ALT)

        # Randomize Keplerian elements
        # Inclination: 0 to 180 degrees (Retrograde allowed)
        inc = random.uniform(0, 180)
        
        # RAAN (Right Ascension of Ascending Node): 0 to 360 deg
        raan = random.uniform(0, 360)
        
        # True Anomaly (Where it is along the circle): 0 to 360 deg
        nu = random.uniform(0, 360)
        
        # Create full orbit definition
        # specific_energy calculation is handled internally by Poliastro's constructors
        # We use .from_classical method for full control, or our helper if it supports it.
        # Our helper `create_orbit` is simple, let's make a more advanced one here or map to it.
        
        # Using Poliastro directly for full parameter control
        orb = Orbit.from_classical(
            Earth,
            (Earth.R.to(u.km).value + alt) * u.km, # Semi-major axis using random alt
            0 * u.one,                              # Eccentricity (Circular for now)
            inc * u.deg,                            # Inclination
            raan * u.deg,                           # RAAN
            0 * u.deg,                              # Arg of Perigee (doesn't matter for circ)
            nu * u.deg                              # True Anomaly
        )
        orbits.append(orb)
        
    return orbits
