import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.train import train

def main():
    model_file = "model.pth"
    
    # 1. Clear old model to force training
    if os.path.exists(model_file):
        print(f"Removing existing {model_file} to force fresh training...")
        os.remove(model_file)
        
    # 2. Run Training
    print("Starting standalone training (400 Epochs)...")
    agent, gnn, history = train(num_episodes=10)
    
    # 3. Validation
    if os.path.exists(model_file):
        print(f"\nSUCCESS: Model trained and saved to {os.path.abspath(model_file)}")
        print("You can now restart the main app/backend, and it will load this model automatically.")
    else:
        print("\nERROR: Model file was not created!")

if __name__ == "__main__":
    main()
