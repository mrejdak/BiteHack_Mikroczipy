import sys
import os
import traceback

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from src.train import train
    print("Import successful. Starting training smoke test...")
    
    def callback(ep, reward, eps):
        print(f"Callback: Ep={ep}, R={reward}, Eps={eps}")

    train(num_episodes=1, progress_callback=callback)
    print("Training smoke test PASSED.")

except Exception:
    print("Training smoke test FAILED.")
    traceback.print_exc()
