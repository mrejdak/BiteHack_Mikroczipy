from ursina import *
from ursina.prefabs.trail_renderer import TrailRenderer
from orbital_physics import create_orbit, get_position, SCALE_FACTOR
from debris_generator import generate_debris_orbits

app = Ursina()

# --- Camera Setup ---
# OrbitCamera lets you rotate around the target (Earth) by right-clicking and dragging
camera = EditorCamera() 
camera.position = (0, 0, -50) 
camera.look_at((0, 0, 0))

# --- Visuals ---
# Earth (Radius ~6371 km)
earth_radius_sim = 6371 * SCALE_FACTOR
earth = Entity(
    model='sphere', 
    scale=earth_radius_sim * 2,
    color=color.white, # Color must be white to show texture clearly
    texture='earth_texture.png'
)

Text(text='Earth', position=(0, 0.1), origin=(0,0), scale=1)

# Satellites & Debris
objects = []
object_orbits = []

# 1. Active Satellites (Colored)
configs = [
    (1000, 45, color.red),
    (2000, 90, color.green),       
    (500, 0, color.yellow),        
    (1500, 30, color.cyan),
    (1200, 135, color.magenta),
    (800, 10, color.orange)
]

for alt, inc, col in configs:
    orb = create_orbit(alt_km=alt, inc_deg=inc)
    object_orbits.append(orb)
    sat = Entity(model='sphere', scale=0.3, color=col)
    TrailRenderer(parent=sat, thickness=2, color=col, length=50)
    objects.append(sat)

# 2. Debris Cloud (Grey)
# Generate debris using MOCAT derived count (defaults to triggering MOCAT if no arg)
debris_orbits = generate_debris_orbits() 
for orb in debris_orbits:
    object_orbits.append(orb)
    deb = Entity(model='sphere', scale=0.1, color=color.gray)
    objects.append(deb)

# --- Simulation Logic ---
time_accel = 100.0
sim_time = 0.0

def update():
    global sim_time
    sim_time += time.dt * time_accel
    
    # Update all objects (Satellites + Debris)
    for i, obj in enumerate(objects):
        # Get position from Poliastro
        pos_km = get_position(object_orbits[i], sim_time)
        
        # Coordinate Mapping
        x = pos_km[0] * SCALE_FACTOR
        y = pos_km[2] * SCALE_FACTOR # Z -> Y (Up)
        z = pos_km[1] * SCALE_FACTOR # Y -> Z (Depth)
        
        obj.position = (x, y, z)
    
    earth.rotation_y += 10 * time.dt

if __name__ == '__main__':
    app.run()
