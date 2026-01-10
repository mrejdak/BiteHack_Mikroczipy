import sys
import os

# Ensure backend directory is in path if running from there
sys.path.append(os.getcwd())

try:
    from actor import JanitorActor
    
    print("Attempting to load model...")
    actor = JanitorActor("../models/janitor_final.zip")
    print("SUCCESS: Model loaded.")

    print("Attempting prediction...")
    from schemas import InferenceRequest, Vector3, AgentState, TargetObject
    
    req = InferenceRequest(
        agent=AgentState(position=Vector3(x=7000000, y=0, z=0), velocity=Vector3(x=0, y=7500, z=0), fuel=100),
        targets=[TargetObject(id="tgt1", position=Vector3(x=7000100, y=0, z=0), velocity=Vector3(x=0, y=7505, z=0))]
    )
    cmd = actor.predict(req)
    print(f"SUCCESS: Prediction result: {cmd}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
