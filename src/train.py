import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.gym_env.space_env import SpaceJanitorEnv

def train():
    # Ensure logs folder exists
    msg = "Starting Training..."
    print(msg)
    
    # Create Env
    env = SpaceJanitorEnv(num_motherships=3, num_debris=10, dt=10.0)
    
    # Check Env
    check_env(env)
    print("Environment Verified.")
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
    )
    
    # Train
    # Short training for demonstration: 10,000 steps
    # In reality, this needs 100k+ to learn complex orbital mechanics
    model.learn(total_timesteps=10000)
    
    # Save
    path = os.path.join(os.getcwd(), "ppo_space_janitor")
    model.save(path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    train()
