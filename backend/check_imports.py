import sys
import os
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    from schemas import InferenceRequest
    print("Schemas imported.")
except ImportError as e:
    print(f"Schemas FAIL: {e}")

try:
    from simulation.physics_engine import PhysicsEngine
    print("Physics Engine imported.")
except ImportError as e:
    print(f"Physics Engine FAIL: {e}")

try:
    from simulation.gym_env import OrbitalJanitorEnv
    print("Gym Env imported.")
except ImportError as e:
    print(f"Gym Env FAIL: {e}")

try:
    from actor import JanitorActor
    print("Actor imported.")
except ImportError as e:
    print(f"Actor FAIL: {e}")
