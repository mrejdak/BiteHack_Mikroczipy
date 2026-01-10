from pydantic import BaseModel
from typing import List, Tuple

class Vector3(BaseModel):
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

class AgentState(BaseModel):
    position: Vector3 # Inertial relative to Earth Center (m)
    velocity: Vector3 # Inertial (m/s)
    fuel: float       # Seconds of thrust remaining

class TargetObject(BaseModel):
    id: str
    position: Vector3
    velocity: Vector3

class InferenceRequest(BaseModel):
    agent: AgentState
    targets: List[TargetObject] # All known targets within range

class ThrustCommand(BaseModel):
    force_x: float
    force_y: float
    force_z: float
    thrust_percentage: float # 0.0 to 1.0 (magnitude)
