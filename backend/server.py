from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import threading
import sys
import os
import numpy as np

# Ensure local libs are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orbital_physics import create_orbit, get_position, apply_impulse, get_state_vector
from debris_generator import generate_debris_orbits
from astropy import units as u

app = Flask(__name__)
CORS(app)

# --- Simulation State ---
objects = []
motherships = [] # Agents
start_time = time.time()
sim_speed = 10.0  # Speed multiplier (10x realtime for visible but realistic motion)

# --- Helpers ---
def get_current_sim_time():
    return (time.time() - start_time) * sim_speed

# --- Initialization ---
def init_simulation():
    global objects, motherships
    objects = []
    motherships = []
    
    # 1. Active Satellites (Static for now)
    # 1. Active Satellites (Removed for clean debris simulation)
    # sat_configs = [] 
    # for cfg in sat_configs: ...
        
    # 2. Debris Cloud
    print("Generating debris...")
    try:
        debris_orbits = generate_debris_orbits(num_debris=None) # Defaults to MOCAT query
    except Exception as e:
        print(f"MOCAT Error: {e}")
        debris_orbits = []
        
    print(f"Generated {len(debris_orbits)} debris objects.")
    
    for i, orb in enumerate(debris_orbits):
        objects.append({
            "id": f"deb_{i}",
            "type": "debris",
            "orbit": orb,
            "color": "#808080",
            "pos": [0,0,0]
        })
    
    print("Simulation initialized.")

# Initialize on startup
init_simulation()

# --- Public API (Frontend) ---

@app.route('/api/init', methods=['GET'])
def get_init():
    """Returns static object metadata (color, type, id)"""
    all_objs = objects + motherships
    data = [{
        "id": obj["id"],
        "type": obj["type"],
        "color": obj["color"]
    } for obj in all_objs]
    return jsonify({"objects": data})

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns current positions of all objects"""
    elapsed = get_current_sim_time()
    state = []
    
    # 1. Passive Objects
    for obj in objects:
        pos_km = get_position(obj["orbit"], elapsed)
        state.append({
            "id": obj["id"],
            "pos": [pos_km[0], pos_km[1], pos_km[2]]
        })
        
    # 2. Active Motherships
    for ms in motherships:
         # Propagate relative to their last maneuver epoch
        dt = elapsed - ms["last_epoch"]
        prop_orbit = ms["orbit"].propagate(dt * u.s)
        r_km, v_kms = get_state_vector(prop_orbit)
        state.append({
            "id": ms["id"],
            "type": "mothership",
            "pos": [r_km[0], r_km[1], r_km[2]],
            "vel": [v_kms[0], v_kms[1], v_kms[2]],  # Velocity for orientation
            "telemetry": ms.get("telemetry", {})
        })
        
    return jsonify({
        "time": elapsed,
        "objects": state
    })

# --- Network Interface (Mothership Agents) ---

import random # Ensure random is imported at top or here

@app.route('/api/mothership/register', methods=['POST'])
def register_mothership():
    """Spawns a new mothership agent."""
    ms_id = f"mothership_{len(motherships) + 1}"
    
    # Spawn at Random Orbit (Same range as debris for better interaction)
    # Altitude: 400 - 1200 km
    alt = random.uniform(400, 1200)
    inc = random.uniform(0, 100) # Varied inclination
    
    orb = create_orbit(alt_km=alt, inc_deg=inc)
    
    motherships.append({
        "id": ms_id,
        "type": "mothership",
        "orbit": orb,
        "last_epoch": get_current_sim_time(),
        "color": "#00FFFF", # Cyan / Hero Color
        "pos": [0,0,0],
        # Telemetry Store
        "telemetry": {
            "fuel": 1000.0,
            "thrust": [0,0,0],
            "target_dist": 0.0
        }
    })
    
    print(f"Registered Agent: {ms_id} at {alt:.1f}km, {inc:.1f}deg")
    return jsonify({"id": ms_id, "status": "registered"})

@app.route('/api/mothership/sync', methods=['POST'])
def sync_mothership():
    """
    Simulates the Network Interface.
    Input: { "id": str, "action": { "delta_v": [x, y, z] }, "sensor_radius": 1000 }
    Output: { "nearby_objects": [ ... ] }
    """
    data = request.json
    ms_id = data.get("id")
    action = data.get("action", {})
    # Default sensor radius
    radius = data.get("sensor_radius", 1000) 
    
    # Find the ship
    ship = next((m for m in motherships if m["id"] == ms_id), None)
    if not ship:
        return jsonify({"error": "Mothership not found"}), 404
        
    current_time = get_current_sim_time()
    
    # 1. Apply Action (Maneuver)
    # 1. Apply Action (Maneuver)
    dv = action.get("delta_v")
    
    # Update Telemetry (Fuel/Thrust)
    ship["telemetry"]["thrust"] = dv if dv else [0,0,0]
    # Simple fuel burn logic for viz
    if dv:
        burn = np.linalg.norm(dv) * 10.0 
        ship["telemetry"]["fuel"] = max(0, ship["telemetry"]["fuel"] - burn)

    # Check if dv is non-zero
    if dv and len(dv) == 3 and np.linalg.norm(dv) > 1e-6:
        # Clamp excessive delta-v to prevent numerical instability
        dv_mag = np.linalg.norm(dv)
        MAX_DV = 0.5  # km/s per maneuver (500 m/s)
        if dv_mag > MAX_DV:
            scale = MAX_DV / dv_mag
            dv = [d * scale for d in dv]
            print(f"Agent {ms_id}: Clamped dv from {dv_mag:.3f} to {MAX_DV} km/s")
        
        # Time since last epoch
        dt = current_time - ship["last_epoch"]
        
        # Propagate to current time to get state
        current_orbit_obj = ship["orbit"].propagate((dt * u.s)) 
        
        # Apply Impulse
        new_orbit = apply_impulse(current_orbit_obj, dv)
        
        # PHYSICS VALIDATION: Check if the new orbit is safe
        EARTH_RADIUS_KM = 6371.0
        MIN_ALTITUDE_KM = 100.0  # Can't be below 100km (atmosphere)
        
        try:
            # Check perigee (closest approach to Earth)
            perigee_km = float(new_orbit.a.to(u.km).value * (1 - new_orbit.ecc.value))
            
            # Check eccentricity (must be < 1 for bound orbit)
            ecc = float(new_orbit.ecc.value)
            
            if ecc >= 1.0:
                # Hyperbolic escape trajectory - reject
                print(f"Agent {ms_id}: REJECTED maneuver - would escape orbit (e={ecc:.2f})")
            elif perigee_km < (EARTH_RADIUS_KM + MIN_ALTITUDE_KM):
                # Would crash into Earth - reject
                print(f"Agent {ms_id}: REJECTED maneuver - would crash (perigee={perigee_km:.0f}km)")
            else:
                # Safe maneuver - apply it
                ship["orbit"] = new_orbit
                ship["last_epoch"] = current_time
                print(f"Agent {ms_id} maneuvered: dv={[round(d,4) for d in dv]} at t={current_time:.1f}s, new perigee={perigee_km:.0f}km")
        except Exception as e:
            print(f"Agent {ms_id}: Orbit validation error: {e}")

    # 2. Sensor Step (Spatial Query)
    dt_sensor = current_time - ship["last_epoch"]
    
    # Get current state (Pos + Vel) to help the agent calculate LVLH
    # get_position only returns pos, we need full state vector
    prop_orbit = ship["orbit"].propagate(dt_sensor * u.s)
    r_ship, v_ship = get_state_vector(prop_orbit)
    
    nearby = []
    
    # Check passive objects
    for obj in objects:
        # Get obj position at current time
        # We need efficient propagation here. For now do full prop per object.
        prop_obj = obj["orbit"].propagate(current_time * u.s)
        r_obj, v_obj = get_state_vector(prop_obj)
        
        dist = np.linalg.norm(np.array(r_ship) - np.array(r_obj))
        
        if dist <= radius:
            nearby.append({
                "id": obj["id"],
                "type": obj["type"],
                "pos": [r_obj[0], r_obj[1], r_obj[2]],
                "vel": [v_obj[0], v_obj[1], v_obj[2]], # ADDED VELOCITY
                "distance": dist
            })
            
            # REMOVAL LOGIC (Capture)
            # If agent is very close (e.g., < 1.0 km = 1000 meters)
            CAPTURE_RADIUS_KM = 1.0
            if dist < CAPTURE_RADIUS_KM:
                print(f"[CAPTURE] Agent {ms_id} captured {obj['id']} at dist {dist:.3f} km!")
                objects.remove(obj) # Remove from global simulation list
    
    # Find and store closest target for visualization
    debris_nearby = [n for n in nearby if n["type"] == "debris"]
    if debris_nearby:
        closest = min(debris_nearby, key=lambda x: x["distance"])
        ship["telemetry"]["target_id"] = closest["id"]
        ship["telemetry"]["target_pos"] = closest["pos"]
        ship["telemetry"]["target_dist"] = closest["distance"]
    else:
        ship["telemetry"]["target_id"] = None
        ship["telemetry"]["target_pos"] = None
        ship["telemetry"]["target_dist"] = None
            
    return jsonify({
        "timestamp": current_time,
        "self_state": {
            "pos": [r_ship[0], r_ship[1], r_ship[2]],
            "vel": [v_ship[0], v_ship[1], v_ship[2]] # ADDED VELOCITY
        },
        "nearby_objects": nearby
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
