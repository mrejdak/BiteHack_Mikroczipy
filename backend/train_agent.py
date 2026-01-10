import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from simulation.gym_env import OrbitalJanitorEnv

def train():
    print("Initializing Environment...")
    env = OrbitalJanitorEnv(num_debris=5, max_steps=500)
    
    # Check if the environment follows Gym API
    print("Checking Environment consistency...")
    check_env(env)
    
    print("Creating PPO Model...")
    # MLP Policy (Multi-Layer Perceptron) because inputs are vectors, not images
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99, # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Entropy coefficient to encourage exploration
        n_steps=2048,
        batch_size=64
    )
    
    # Save a checkpoint every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/logs/',
        name_prefix='janitor_model'
    )
    
    print("Starting Training (Target: 20,000 steps)...")
    # In a real Hackathon you'd run this for much longer or parallelize
    model.learn(total_timesteps=20_000, callback=checkpoint_callback)
    
    print("Training Complete. Saving Final Model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/janitor_final")
    
    # Test Run
    print("Running Test Episode...")
    obs, _ = env.reset()
    total_reward = 0
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished after {i} steps. Total Reward: {total_reward}")
            break

if __name__ == "__main__":
    train()
