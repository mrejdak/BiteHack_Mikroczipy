from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Fire Detection API",
    description="API for satellite fire detection and infrastructure alerts",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import detect

app.include_router(detect.router, prefix="/api/v1", tags=["detection"])

@app.get("/")
async def root():
    return {"message": "Fire Detection API is running"}
