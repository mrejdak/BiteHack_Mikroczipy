import numpy as np
from src.core.physics import OrbitalPhysics
from src.core.entities import Mothership, Debris

def test_core():
    print("Initializing Physics...")
    physics = OrbitalPhysics()
    
    # 1. Setup Orbits
    # LEO approx
    r_ms = np.array([7000000.0, 0.0, 0.0])
    v_ms = np.array([0.0, 7546.0, 0.0]) # Circular speed ~7.5km/s
    orbit_ms = physics.create_orbit_from_vectors(r_ms, v_ms)
    
    ms = Mothership(0, orbit_ms, physics)
    
    # Target: Closer and simpler
    # MS is at 0 degrees. Debris at 10 degrees?
    # R = 7000km
    angle = np.deg2rad(10)
    x = 7000000.0 * np.cos(angle)
    y = 7000000.0 * np.sin(angle)
    r_deb = np.array([x, y, 0.0])
    v_deb = np.array([-7546.0 * np.sin(angle), 7546.0 * np.cos(angle), 0.0]) 
    orbit_deb = physics.create_orbit_from_vectors(r_deb, v_deb)
    
    debris = Debris(1, orbit_deb, physics)
    
    print(f"Mothership Pos: {ms.position}")
    print(f"Debris Pos: {debris.position}")
    
    # 2. Plan Transfer
    dt = 1000.0 # seconds
    
    # Needs Future Position of Debris
    future_orbit = physics.propagate(debris.orbit, dt)
    future_pos = physics.get_state_vector(future_orbit)[:3]
    
    print(f"Debris Future Pos (t+{dt}): {future_pos}")
    
    req_v, cost = ms.plan_transfer(future_pos, dt)
    
    if req_v is not None:
        print(f"Transfer Plan Found! Cost: {cost:.2f} m/s")
        
        # 3. Execute
        ms.execute_transfer(req_v, cost, debris.id, dt)
        print("Transfer Executed. Mothership is on intercept course.")
        
        # 4. Propagate Both
        print("Propagating simulation...")
        ms.update(dt)
        debris.update(dt)
        
        # 5. Check Error
        dist = np.linalg.norm(ms.position - debris.position)
        print(f"Final Distance after {dt}s: {dist:.2f} meters")
        
        if dist < 1000.0: # 1km tolerance
            print("SUCCESS: Intercept successful!")
        else:
            print("WARNING: Intercept missed (large error). Check math.")
            
    else:
        print("FAILED: No Lambert solution found.")
        # Debug why
        future_orbit = physics.propagate(debris.orbit, dt)
        future_pos = physics.get_state_vector(future_orbit)[:3]
        req_v, cost = physics.solve_lambert(ms.orbit, future_pos, dt)
        print(f"Debug: Solver returned cost {cost}")

if __name__ == "__main__":
    test_core()
