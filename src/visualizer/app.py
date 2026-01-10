from ursina import *
from src.gym_env.space_env import SpaceJanitorEnv
import numpy as np
import os

SCALE_KM = 1000.0 # 1 Unit = 1000 km

class JanitorApp:
    def __init__(self, env: SpaceJanitorEnv):
        self.app = Ursina()
        self.env = env
        
        # Camera
        self.camera = EditorCamera()
        self.camera.position = (0, 0, -40)
        
        # Lights
        PointLight(parent=camera, position=(0,0,-20))
        AmbientLight(color=color.rgba(100, 100, 100, 0.1))
        
        # Earth
        tex_path = os.path.join(os.getcwd(), 'earth_texture.png')
        tex = load_texture(tex_path) if os.path.exists(tex_path) else None
        
        self.earth = Entity(
            model='sphere',
            texture=tex,
            scale=12.742, # 12,742 km diameter
            rotation=(0,0,0)
        )
        
        # Background
        Sky(texture='sky_default')

        # Entities
        self.ms_entities = {}
        self.debris_entities = {}
        self.trails = {} # id -> list of points
        
        self.step_text = Text(text="Step: 0", position=(-0.85, 0.45), scale=1.5)
        self.reward_text = Text(text="Total Reward: 0", position=(-0.85, 0.40), scale=1.5)
        
        self.total_reward = 0.0
        self.sim_step_count = 0
        
        # Bind update
        self.manager = Entity(update=self.update_loop)
        
    def run(self):
        self.app.run()
        
    def update_loop(self):
        # 1. Step Logic (Every few frames? Or every frame?)
        # Let's do 1 step per frame for smoothness if dt is small
        # Or slow it down.
        if self.sim_step_count % 2 == 0:
             # Random Agent for now (Or placeholder for RL)
             # actions = self.env.action_space.sample()
             
             # Smarter heuristic for demo:
             # If IDLE, pick a random target
             actions = []
             for ms in self.env.motherships:
                 if ms.state == "IDLE":
                     # Pick random debris
                     if len(self.env.debris_list) > 0:
                         actions.append(np.random.randint(0, self.env.num_debris))
                     else:
                         actions.append(self.env.num_debris)
                 else:
                     actions.append(self.env.num_debris)
             
             obs, reward, terminated, truncated, _ = self.env.step(np.array(actions))
             self.total_reward += reward
             
             self.step_text.text = f"Step: {self.env.time_elapsed:.0f}s"
             self.reward_text.text = f"Total Reward: {self.total_reward:.2f}"
             
             if terminated:
                 print("Mission Complete!")
                 self.env.reset()
                 self.total_reward = 0
                 
        self.sim_step_count += 1
        
        # 2. Update Visuals
        self.earth.rotation_y -= 0.05
        self.update_entities()

    def update_entities(self):
        # Update Motherships
        for ms in self.env.motherships:
            pos = ms.position / SCALE_KM
            # y-up mapping? 
            # Physics: x,y,z. Usually Z is North.
            # Ursina: Y is Up.
            # So (x, z, y) usually works best if Z is North.
            x, y, z = pos
            u_pos = (x, z, y)
             
            if ms.id not in self.ms_entities:
                top = Entity(
                    model='cube', color=color.green, scale=0.4, position=u_pos, unlit=True
                )
                self.ms_entities[ms.id] = top
                self.trails[ms.id] = []
            else:
                self.ms_entities[ms.id].position = u_pos
                
            # Trail
            # Efficient trail: simple Entity dots or Mesh?
            # Creating entities is expensive. Let's just create dot every N steps
            if self.sim_step_count % 10 == 0:
                Entity(model='sphere', color=color.green, scale=0.05, position=u_pos, unlit=True)

        # Update Debris
        existing_ids = set()
        for d in self.env.debris_list:
            if d.active:
                existing_ids.add(d.id)
                pos = d.position / SCALE_KM
                x, y, z = pos
                u_pos = (x, z, y)
                
                if d.id not in self.debris_entities:
                    self.debris_entities[d.id] = Entity(
                        model='sphere', color=color.red, scale=0.2, position=u_pos, unlit=True
                    )
                else:
                    self.debris_entities[d.id].position = u_pos
            else:
                # Remove if exists
                if d.id in self.debris_entities:
                    destroy(self.debris_entities[d.id])
                    del self.debris_entities[d.id]
