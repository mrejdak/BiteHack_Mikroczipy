
# Headless Physics Verification
import numpy as np
from src.simulation.physics import OrbitalPhysics
from astropy import units as u

def test_lambert():
    print("Initializing Physics...")
    physics = OrbitalPhysics()
    
    # Create an initial orbit (LEO)
    r_start = np.array([7000000.0, 0.0, 0.0]) # 7000 km
    v_start = np.array([0.0, 7500.0, 0.0]) # ~7.5 km/s
    orbit_start = physics.create_orbit(r_start, v_start)
    
    # Target: Same orbit but 90 degrees phase ahead?
    # Let's just pick a random point
    target_pos = np.array([0.0, 7000000.0, 0.0]) # 90 deg roughly
    
    dt = 1000.0
    
    print("Solving Lambert...")
    req_v, cost = physics.solve_lambert(orbit_start, target_pos, dt)
    
    if req_v is not None:
        print(f"Lambert Success! Cost: {cost:.2f} m/s")
        print(f"Required V: {req_v}")
    else:
        print("Lambert Failed.")

if __name__ == "__main__":
    try:
        test_lambert()
    except Exception as e:
        print(f"Test Crashed: {e}")
