import numpy as np
from .physics import OrbitalPhysics

class Entity:
    def __init__(self, uid, orbit, physics: OrbitalPhysics):
        self.id = uid
        self.orbit = orbit
        self.physics = physics
        self.active = True
        
    @property
    def position(self):
        # Helper for clean access
        return self.physics.get_state_vector(self.orbit)[:3]
        
    @property
    def velocity(self):
        return self.physics.get_state_vector(self.orbit)[3:]

    def update(self, dt):
        if self.active:
            self.orbit = self.physics.propagate(self.orbit, dt)

class Mothership(Entity):
    def __init__(self, uid, orbit, physics, fuel=5000.0):
        super().__init__(uid, orbit, physics)
        self.fuel = fuel
        self.state = "IDLE" # IDLE, TRANSFER
        self.target_id = None
        self.transfer_timer = 0.0
        self.transfer_duration = 0.0
        
    def plan_transfer(self, target_pos_future, dt_transfer, max_dv=1.5):
        """
        Calculates if transfer is possible.
        Does NOT execute it.
        Returns cost if possible, else Inf.
        """
        req_v, cost = self.physics.solve_lambert(self.orbit, target_pos_future, dt_transfer)
        
        if req_v is not None:
             # Check Constraints
             if cost > self.fuel:
                 return None, float('inf')
             if cost > max_dv:
                 return None, float('inf')
                 
             # Check Safety
             # Create hypothetical orbit to check perigee
             curr_r = self.position
             transfer_orbit = self.physics.create_orbit_from_vectors(curr_r, req_v)
             
             if not self.physics.check_orbit_safety(transfer_orbit):
                 return None, float('inf')
                 
             return req_v, cost
             
        return None, float('inf')
        
    def execute_transfer(self, req_v, cost, target_id, duration):
        if self.fuel >= cost:
            self.fuel -= cost
            # Apply impulsive maneuver: Change velocity instantly
            # We create a new orbit at CURRENT position but with NEW velocity
            curr_r = self.position
            self.orbit = self.physics.create_orbit_from_vectors(curr_r, req_v)
            
            self.state = "TRANSFER"
            self.target_id = target_id
            self.transfer_duration = duration
            self.transfer_timer = 0.0
            return True
        return False
        
    def update(self, dt):
        super().update(dt) # Propagates physical orbit
        
        if self.state == "TRANSFER":
            self.transfer_timer += dt
            # Check if arrived? 
            # Logic handled by Environment (checking proximity)
            
    def complete_transfer(self):
        self.state = "IDLE"
        self.target_id = None
        self.transfer_timer = 0.0

class Debris(Entity):
    pass
