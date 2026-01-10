import numpy as np
from stable_baselines3 import PPO
from backend.schemas import InferenceRequest, ThrustCommand, Vector3
from backend.simulation.physics_engine import PhysicsEngine, R_EARTH 

class JanitorActor:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        # We need a mini-physics engine helper just for coordinate transforms
        self.engine = PhysicsEngine() 

    def predict(self, request: InferenceRequest) -> ThrustCommand:
        # 1. Select best target (Closest logic or Strategy logic)
        # For now: Closest target that is ACTIVE
        best_target = None
        min_dist = float('inf')
        
        agent_pos = np.array(request.agent.position.to_list())
        agent_vel = np.array(request.agent.velocity.to_list())
        
        for t in request.targets:
            t_pos = np.array(t.position.to_list())
            dist = np.linalg.norm(t_pos - agent_pos)
            if dist < min_dist:
                min_dist = dist
                best_target = t
                
        if not best_target:
            # No targets? Idling.
            return ThrustCommand(force_x=0, force_y=0, force_z=0, thrust_percentage=0)

        # 2. Prepare Observation Vector
        # We need to replicate the LVLH logic from gym_env.py EXACTLY.
        # This means we need to "fake" the PhysicsEngine body objects to use its math helper
        # Or just reimplement the math here. Let's reuse the helper for DRY.
        
        # Inject mock bodies into engine
        self.engine.bodies = []
        from backend.simulation.physics_engine import OrbitalBody
        
        mock_agent = OrbitalBody(id="me", position=agent_pos, velocity=agent_vel, mass=100, radius=1, type="janitor")
        mock_target = OrbitalBody(id="tgt", position=np.array(best_target.position.to_list()), 
                                        velocity=np.array(best_target.velocity.to_list()), mass=10, radius=2, type="debris")
        
        self.engine.add_body(mock_agent)
        self.engine.add_body(mock_target)
        
        # Get Relative State (LVLH)
        rel_state = self.engine.get_relative_state_lvlh("me", "tgt")
        if not rel_state:
             return ThrustCommand(force_x=0, force_y=0, force_z=0, thrust_percentage=0)

        # Normalize (Must match gym_env.py EXACTLY!)
        dr = rel_state['dr'] / 1000.0  # 1km = 1.0
        dv = rel_state['dv'] / 10.0    # 10m/s = 1.0
        fuel = request.agent.fuel / 1000.0

        obs = np.concatenate([dr, dv, [fuel]]).astype(np.float32)

        # 3. Model Inference
        print("DEBUG: Calling model.predict...")
        action, _states = self.model.predict(obs, deterministic=True)
        print(f"DEBUG: Model prediction done: {action}")
        
        # 4. Action -> Thrust Vector
        # Action is [-1, 1] in LVLH frame (if we assumed that in Env) or Body frame.
        # In Env step(), we rotated action to inertial. We must do the same here 
        # to tell the API "Here is the global thrust vector request".
        
        max_thrust = 1000.0
        thrust_lvlh = action * max_thrust
        
        # Rotate LVLH -> Inertial
        r_vec = agent_pos
        v_vec = agent_vel
        r_mag = np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        
        if r_mag > 0 and np.linalg.norm(h_vec) > 0:
            i_unit = r_vec / r_mag
            k_unit = h_vec / np.linalg.norm(h_vec)
            j_unit = np.cross(k_unit, i_unit)
            Q_T = np.column_stack([i_unit, j_unit, k_unit])
            
            thrust_inertial = Q_T @ thrust_lvlh
        else:
            thrust_inertial = thrust_lvlh
            
        return ThrustCommand(
            force_x=float(thrust_inertial[0]),
            force_y=float(thrust_inertial[1]),
            force_z=float(thrust_inertial[2]),
            thrust_percentage=float(np.linalg.norm(action)) # Approximate magnitude
        )
