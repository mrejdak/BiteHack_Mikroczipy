import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Constants
G = 6.67430e-11
M_EARTH = 5.972e24
R_EARTH = 6371000.0  # meters
MU = G * M_EARTH  # Standard gravitational parameter

@dataclass
class OrbitalBody:
    id: str
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    mass: float  # kg
    radius: float # collision radius in meters
    type: str  # 'satellite', 'debris', 'janitor'
    fuel: float = 0.0 # Only for active satellites
    active: bool = True 

class PhysicsEngine:
    def __init__(self, dt: float = 1.0):
        self.dt = dt  # Time step in seconds
        self.bodies: List[OrbitalBody] = []
        self.time_elapsed = 0.0

    def add_body(self, body: OrbitalBody):
        self.bodies.append(body)

    def remove_body(self, body_id: str):
        self.bodies = [b for b in self.bodies if b.id != body_id]

    def get_body(self, body_id: str) -> Optional[OrbitalBody]:
        for b in self.bodies:
            if b.id == body_id:
                return b
        return None

    def calculate_gravity(self, position: np.ndarray) -> np.ndarray:
        # F = G*M*m / r^2
        # a = G*M / r^2 * (unit vector)
        # a = -mu * r_vec / |r|^3
        r_mag = np.linalg.norm(position)
        if r_mag == 0:
            return np.zeros(3)
        acceleration = -MU * position / (r_mag**3)
        return acceleration

    def step(self, thrust_vectors: dict[str, np.ndarray] = None):
        """
        thrust_vectors: dict of body_id -> force_vector [fx, fy, fz] (Newtons)
        """
        if thrust_vectors is None:
            thrust_vectors = {}

        for body in self.bodies:
            if not body.active:
                continue

            # 1. Gravity
            acc_gravity = self.calculate_gravity(body.position)

            # 2. Thrust (if applicable)
            acc_thrust = np.zeros(3)
            if body.id in thrust_vectors and body.fuel > 0:
                thrust = thrust_vectors[body.id]
                # Consume fuel (simplified: 1 unit per Newton-second or similar logic)
                # Here we assume fuel is "seconds of thrust" or similar abstract unit for Hackathon, 
                # but let's make it proportional to impulse.
                thrust_mag = np.linalg.norm(thrust)
                if thrust_mag > 0:
                    # Specific Impulse logic (simplified)
                    # Fuel consumption = Force * dt * constant
                    fuel_cost = thrust_mag * self.dt * 0.001 
                    if body.fuel >= fuel_cost:
                        body.fuel -= fuel_cost
                        acc_thrust = thrust / body.mass
                    else:
                        # Partial thrust or engine cutoff
                        body.fuel = 0

            # 3. Integration (Symplectic Euler or RK4 is better, but Euler is fine for short steps/hackathon)
            # v = v + a * dt
            # x = x + v * dt
            total_acc = acc_gravity + acc_thrust
            
            body.velocity += total_acc * self.dt
            body.position += body.velocity * self.dt

            # Check for crash into Earth
            if np.linalg.norm(body.position) < R_EARTH:
                body.active = False
                # In simulation we might want to just mark it as 'crashed'
        
        self.time_elapsed += self.dt

    def check_collisions(self) -> List[Tuple[str, str]]:
        collisions = []
        # Naive O(N^2) check - fine for < 100 objects
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                b1 = self.bodies[i]
                b2 = self.bodies[j]
                if not b1.active or not b2.active:
                    continue
                
                dist = np.linalg.norm(b1.position - b2.position)
                if dist < (b1.radius + b2.radius):
                    collisions.append((b1.id, b2.id))
        return collisions

    def get_relative_state_lvlh(self, origin_id: str, target_id: str) -> dict:
        """
        Returns target's position and velocity in the Local-Vertical/Local-Horizontal (LVLH) frame
        centered on the origin body (agent).
        
        Frame definition (Hill's Frame):
        - X (Radial): Along direction from Earth Center -> Satellite
        - Y (Transverse/Along-Track): Perpendicular to X, in the orbital plane (roughly velocity direction)
        - Z (Normal/Cross-Track): Perpendicular to orbital plane (Momentum vector)
        """
        origin = self.get_body(origin_id)
        target = self.get_body(target_id)
        
        if not origin or not target:
            return None

        r_vec = origin.position
        v_vec = origin.velocity
        
        # 1. Define Basis Vectors for LVLH Frame
        # Radial (X)
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0: return None
        i_unit = r_vec / r_mag
        
        # Cross-Track / Normal (Z)
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)
        if h_mag == 0: return None # No orbit?
        k_unit = h_vec / h_mag
        
        # Along-Track / Transverse (Y)
        j_unit = np.cross(k_unit, i_unit)
        
        # Rotation Matrix (Inertial -> LVLH)
        # Rows are the projected basis vectors
        Q = np.array([i_unit, j_unit, k_unit])
        
        # 2. Relative Position (Inertial)
        delta_r_inertial = target.position - origin.position
        
        # 3. Relative Position (LVLH)
        dr_lvlh = Q @ delta_r_inertial
        
        # 4. Relative Velocity (LVLH)
        # NOTE: Simple relative velocity Q @ (v_t - v_o) is OK for small distances,
        # but for rotating frames, we strictly need kinematic terms (Omega x r).
        # For Hackathon RL inputs, simple relative velocity rotated to frame is often "enough" 
        # for neutral AI to figure it out, but implementing the transport theorem term helps.
        # Transport Theorem: v_rel_inertial = v_rel_rotating + omega x r_rel
        # So v_rel_rotating = v_rel_inertial - omega x r_rel
        
        # Orbital Angular Velocity Vector (Omega)
        # omega = (r x v) / r^2 ... wait, h = r x v. omega vector is roughly h / r^2
        omega_vec = h_vec / (r_mag**2)
        
        delta_v_inertial = target.velocity - origin.velocity
        # Correct for frame rotation
        cross_term = np.cross(omega_vec, delta_r_inertial)
        dv_rotating_inertial_frame = delta_v_inertial - cross_term
        
        dv_lvlh = Q @ dv_rotating_inertial_frame
        
        return {
            'dr': dr_lvlh, # [x, y, z] relative pos
            'dv': dv_lvlh, # [vx, vy, vz] relative vel
            'dist': np.linalg.norm(delta_r_inertial)
        }

    def get_state_for_agent(self, agent_id: str, k_nearest_debris: int = 1) -> dict:
        agent = self.get_body(agent_id)
        if not agent:
            return {}
        
        # Relative vectors to all debris
        debris_list = [b for b in self.bodies if b.type == 'debris' and b.active]
        
        # Sort by distance
        debris_info = []
        for d in debris_list:
            rel_pos = d.position - agent.position
            rel_vel = d.velocity - agent.velocity
            dist = np.linalg.norm(rel_pos)
            debris_info.append({
                'id': d.id,
                'dist': dist,
                'rel_pos': rel_pos,
                'rel_vel': rel_vel,
                'obj': d
            })
        
        debris_info.sort(key=lambda x: x['dist'])
        
        # Return structured data for Gym Env
        return {
            'agent_pos': agent.position,
            'agent_vel': agent.velocity,
            'fuel': agent.fuel,
            'targets': debris_info[:k_nearest_debris]
        }
