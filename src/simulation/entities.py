import numpy as np
from .physics import OrbitalPhysics
import time
from astropy import units as u

class Entity:
    def __init__(self, id, r_vec, v_vec, physics: OrbitalPhysics):
        self.id = id
        self.physics = physics
        # Create initial orbit
        self.orbit = self.physics.create_orbit(r_vec, v_vec)
        self.active = True
        self.trace = [] # For drawing trails

    @property
    def position(self):
        # Return position in meters
        return self.orbit.r.to(u.m).value 
        
    @property
    def velocity(self):
         # Return velocity in m/s
        return self.orbit.v.to(u.m / u.s).value

    def update(self, dt):
        if self.active:
            # Poliastro propagate returns a NEW orbit object at t_current + dt
            # We want to step forward.
            self.orbit = self.physics.propagate(self.orbit, dt)
            
            # Store trace for visualization (every N steps?)
            pos = self.position
            self.trace.append(pos)
            if len(self.trace) > 50:
                self.trace.pop(0)

class Mothership(Entity):
    def __init__(self, id, r_vec, v_vec, physics, fuel=5000.0):
        super().__init__(id, r_vec, v_vec, physics)
        self.fuel = fuel
        self.state_status = "IDLE" 
        self.target_debris_id = None
        self.transfer_end_time = 0

    def execute_maneuver(self, delta_v_vec):
        cost = np.linalg.norm(delta_v_vec)
        if self.fuel >= cost:
            self.fuel -= cost
            # Apply impulsive maneuver involves recreating the orbit with new velocity vector at current position
            from astropy import units as u
            
            curr_r = self.orbit.r
            curr_v = self.orbit.v.to(u.m/u.s).value
            
            new_v = curr_v + delta_v_vec
            
            self.orbit = self.physics.create_orbit(curr_r.value, new_v)
            return True
        return False

class Debris(Entity):
    def __init__(self, id, r_vec, v_vec, physics):
        super().__init__(id, r_vec, v_vec, physics)
