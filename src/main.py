import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gym_env.space_env import SpaceJanitorEnv
from src.visualizer.app import JanitorApp

def main():
    print("Initializing Space Janitor Environment (Rebooted)...")
    # 20 Debris, 3 Motherships
    env = SpaceJanitorEnv(num_motherships=3, num_debris=20, dt=10.0)
    
    print("Starting Ursina 3D Visualization...")
    app = JanitorApp(env)
    app.run()

if __name__ == "__main__":
    main()
