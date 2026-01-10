from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import random
import os

from backend.schemas import InferenceRequest, ThrustCommand
from backend.actor import JanitorActor

app = FastAPI()

# Global Actor Instance
actor = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global actor
    model_path = "models/janitor_final.zip"
    if os.path.exists(model_path):
        try:
            actor = JanitorActor(model_path)
            print("ACE: Janitor Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
    else:
        print(f"WARNING: Model not found at {model_path}. Inference endpoints will fail.")

@app.get("/")
async def root():
    return {"message": "Orbital Janitor Backend is running"}

@app.post("/predict", response_model=ThrustCommand)
def predict_thrust(request: InferenceRequest):
    if not actor:
        raise HTTPException(status_code=503, detail="Model not initialized. Check server logs.")
    
    try:
        command = actor.predict(request)
        return command
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock simulation step
            # In real implementation this will come from the Simulation Engine
            data = {
                "satellites": [
                    {"id": i, "x": random.uniform(-10, 10), "y": random.uniform(-10, 10), "z": random.uniform(-10, 10)}
                    for i in range(10)
                ],
                "debris": [
                    {"id": i, "x": random.uniform(-10, 10), "y": random.uniform(-10, 10), "z": random.uniform(-10, 10)}
                    for i in range(5)
                ]
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.1) # 10 FPS
    except Exception as e:
        print(f"Connection closed: {e}")
