from backend.agent import MothershipAgent
import time

def main():
    agents = []
    print("Spawning 3 Mothership Agents...")
    
    for _ in range(3):
        a = MothershipAgent(sensor_radius=1500)
        a.start()
        agents.append(a)
        time.sleep(0.5) # Stagger start
        
    try:
        print("Agents running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping agents...")
        for a in agents:
            a.stop()
            
if __name__ == "__main__":
    main()
