import numpy as np
from src.gym_env.space_env import SpaceJanitorEnv

def verify():
    print("Creating Env...")
    env = SpaceJanitorEnv(num_motherships=2, num_debris=3)
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, _ = env.reset()
    print(f"Initial Obs shape: {obs.shape}")
    print(f"Initial Obs sample: {obs[:10]}")
    
    done = False
    step = 0
    while not done and step < 50:
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step}: Reward {reward}")
            
        if terminated or truncated:
            done = True
            
        step += 1
        
    print("Environment Verification Complete.")

if __name__ == "__main__":
    verify()
