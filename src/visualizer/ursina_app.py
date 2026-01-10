from ursina import *
from simulation.env import SpaceJanitorEnv
import numpy as np
import os

# Scale Factor 
SCALE_KM = 1000.0 

class SimApp:
    def __init__(self, env: SpaceJanitorEnv):
        # Composition, not Inheritance
        self.app = Ursina()
        self.env = env
        
        # Camera Setup
        window.title = "Space Janitor Swarm 3D"
        window.borderless = False
        window.fullscreen = False
        window.exit_button.visible = False
        window.fps_counter.enabled = True
        
        # Editor Camera
        self.camera = EditorCamera()
        self.camera.position = (0, 0, -30) 
        
        # Lighting
        PointLight(parent=camera, position=(0,0,-20))
        AmbientLight(color=color.rgba(100, 100, 100, 0.1))

        # --- Earth ---
        # --- Earth ---
        # Try loading texture explicitly
        tex_path = os.path.join(os.getcwd(), 'earth_texture.png')
        if not os.path.exists(tex_path):
            print(f"WARNING: Texture file not found at {tex_path}")
            tex = None
        else:
            print(f"Loading texture from {tex_path}")
            tex = load_texture(tex_path)

        self.earth = Entity(
            model='sphere',
            texture=tex,
            scale=12.742, 
            rotation=(0,0,0) 
        )
        # Background
        Sky(texture='sky_default') 

        # --- Entities ---
        self.ms_entities = {}
        self.debris_entities = {}
        
        # Initialize Entities
        self.update_entities()
        
        # Step counter
        self.sim_step = 0

    def run(self):
        # Register update function
        # Ursina calls the 'update' function in global scope or entity scope automagically if defined.
        # But we are in a class. We need to assign it.
        # Check Ursina docs pattern: 
        # def update(): ...
        # app.run()
        
        # Or, we can create an Entity that handles update.
        # Or just assign to the magic global update if possible? 
        # Better: create a main 'Manager' entity.
        
        self.manager = Entity(update=self.update_loop)
        self.app.run()

    def update_loop(self):
        # Game Loop
        
        # 1. Step Physics
        if self.sim_step % 1 == 0: 
             # Simple random actions for demo
             if self.sim_step % 60 == 0:
                 actions = [self.env.num_debris] * self.env.num_motherships 
                 self.env.step(np.array(actions))
             else:
                 actions = [self.env.num_debris] * self.env.num_motherships
                 self.env.step(np.array(actions))
                 
        self.sim_step += 1
        
        # 2. Update Visuals
        self.update_earth_rotation()
        self.update_entity_positions()
        
    def update_entities(self):
         # Initial population
         self.update_entity_positions()
        
    def update_earth_rotation(self):
        self.earth.rotation_y -= 0.05 
        
    def update_entity_positions(self):
        # Update Motherships
        for ms in self.env.motherships:
            pos_km = ms.position / SCALE_KM
            # Physics (x, y, z) -> Ursina (x, z, y) usually for Z-up to Y-up
            # But let's stick to direct mapping first to see axis.
            x, y, z = pos_km
            
            # Switch Y/Z for cleaner view if needed, but physics seems to be XYZ
            
            if ms.id not in self.ms_entities:
                print(f"Creating Mothership {ms.id} at Ursina Pos: {(x, z, y)}")
                self.ms_entities[ms.id] = Entity(
                    model='cube',
                    color=color.green,
                    scale=0.5, # Increased scale for visibility
                    unlit=True, # Make it glow
                    position=(x, z, y) 
                )
            else:
                self.ms_entities[ms.id].position = (x, z, y)
                
        # Update Debris
        for d in self.env.debris_list:
            if not d.active:
                if d.id in self.debris_entities:
                    destroy(self.debris_entities[d.id])
                    del self.debris_entities[d.id]
                continue
                
            pos_km = d.position / SCALE_KM
            x, y, z = pos_km
            
            if d.id not in self.debris_entities:
                self.debris_entities[d.id] = Entity(
                    model='sphere',
                    color=color.red,
                    scale=0.3, # Increased scale
                    unlit=True,
                    position=(x, z, y)
                )
            else:
                self.debris_entities[d.id].position = (x, z, y)
