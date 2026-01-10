"""
Comprehensive test suite for Orbital Janitor API endpoints
Tests agent loading, prediction endpoint, and decision quality
"""
import requests
import json
import numpy as np
import time
from typing import Dict, List
from backend.schemas import InferenceRequest, AgentState, TargetObject, Vector3, ThrustCommand

BASE_URL = "http://localhost:8001"

def test_root_endpoint():
    """Test basic root endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Root Endpoint")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        print(f"[OK] Root endpoint OK: {data}")
        return True
    except Exception as e:
        print(f"[FAIL] Root endpoint failed: {e}")
        return False

def test_agent_loading():
    """Test if agent is loaded (by checking if predict endpoint works)"""
    print("\n" + "="*60)
    print("TEST 2: Agent Loading Check")
    print("="*60)
    
    # Create a minimal valid request
    test_request = InferenceRequest(
        agent=AgentState(
            position=Vector3(x=7000000.0, y=0.0, z=0.0),  # ~7000km orbit
            velocity=Vector3(x=0.0, y=7546.0, z=0.0),  # ~7.5 km/s orbital velocity
            fuel=1000.0
        ),
        targets=[
            TargetObject(
                id="debris_1",
                position=Vector3(x=7001000.0, y=0.0, z=0.0),  # 1km ahead
                velocity=Vector3(x=0.0, y=7546.0, z=0.0)
            )
        ]
    )
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 503:
            print("[FAIL] Agent not loaded (503 Service Unavailable)")
            print(f"  Response: {response.json()}")
            return False
        elif response.status_code == 200:
            result = response.json()
            print(f"[OK] Agent loaded successfully!")
            print(f"  Response: {result}")
            return True
        else:
            print(f"[FAIL] Unexpected status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Agent loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_distance(pos1: Vector3, pos2: Vector3) -> float:
    """Calculate Euclidean distance between two positions"""
    return np.sqrt(
        (pos1.x - pos2.x)**2 + 
        (pos1.y - pos2.y)**2 + 
        (pos1.z - pos2.z)**2
    )

def calculate_thrust_direction_quality(
    agent_pos: Vector3,
    target_pos: Vector3,
    thrust: ThrustCommand
) -> Dict:
    """
    Analyze if thrust direction makes sense relative to target position.
    Returns quality metrics.
    """
    # Vector from agent to target
    to_target = Vector3(
        x=target_pos.x - agent_pos.x,
        y=target_pos.y - agent_pos.y,
        z=target_pos.z - agent_pos.z
    )
    
    # Normalize
    dist_to_target = calculate_distance(agent_pos, target_pos)
    if dist_to_target == 0:
        return {"error": "Agent and target at same position"}
    
    to_target_normalized = Vector3(
        x=to_target.x / dist_to_target,
        y=to_target.y / dist_to_target,
        z=to_target.z / dist_to_target
    )
    
    # Thrust vector
    thrust_mag = np.sqrt(
        thrust.force_x**2 + 
        thrust.force_y**2 + 
        thrust.force_z**2
    )
    
    if thrust_mag == 0:
        return {
            "thrust_magnitude": 0,
            "alignment": None,
            "note": "No thrust applied"
        }
    
    thrust_normalized = Vector3(
        x=thrust.force_x / thrust_mag,
        y=thrust.force_y / thrust_mag,
        z=thrust.force_z / thrust_mag
    )
    
    # Dot product = cosine of angle (1.0 = perfectly aligned, 0 = perpendicular, -1 = opposite)
    alignment = (
        to_target_normalized.x * thrust_normalized.x +
        to_target_normalized.y * thrust_normalized.y +
        to_target_normalized.z * thrust_normalized.z
    )
    
    return {
        "thrust_magnitude": thrust_mag,
        "distance_to_target": dist_to_target,
        "alignment": alignment,
        "alignment_angle_deg": np.arccos(np.clip(alignment, -1, 1)) * 180 / np.pi,
        "is_towards_target": alignment > 0.5,  # More than 60 degrees towards target
        "thrust_percentage": thrust.thrust_percentage
    }

def test_prediction_scenarios():
    """Test prediction endpoint with various realistic scenarios"""
    print("\n" + "="*60)
    print("TEST 3: Prediction Scenarios & Decision Quality")
    print("="*60)
    
    scenarios = [
        {
            "name": "Close Target Ahead",
            "agent": AgentState(
                position=Vector3(x=7000000.0, y=0.0, z=0.0),
                velocity=Vector3(x=0.0, y=7546.0, z=0.0),
                fuel=1000.0
            ),
            "targets": [
                TargetObject(
                    id="close_ahead",
                    position=Vector3(x=7001000.0, y=0.0, z=0.0),  # 1km ahead
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                )
            ]
        },
        {
            "name": "Target Above",
            "agent": AgentState(
                position=Vector3(x=7000000.0, y=0.0, z=0.0),
                velocity=Vector3(x=0.0, y=7546.0, z=0.0),
                fuel=1000.0
            ),
            "targets": [
                TargetObject(
                    id="above",
                    position=Vector3(x=7000000.0, y=0.0, z=5000.0),  # 5km above
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                )
            ]
        },
        {
            "name": "Multiple Targets - Closest Should Be Selected",
            "agent": AgentState(
                position=Vector3(x=7000000.0, y=0.0, z=0.0),
                velocity=Vector3(x=0.0, y=7546.0, z=0.0),
                fuel=1000.0
            ),
            "targets": [
                TargetObject(
                    id="far",
                    position=Vector3(x=7010000.0, y=0.0, z=0.0),  # 10km away
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                ),
                TargetObject(
                    id="close",
                    position=Vector3(x=7000500.0, y=0.0, z=0.0),  # 500m away (closest)
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                ),
                TargetObject(
                    id="medium",
                    position=Vector3(x=7002000.0, y=0.0, z=0.0),  # 2km away
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                )
            ]
        },
        {
            "name": "No Targets - Should Idle",
            "agent": AgentState(
                position=Vector3(x=7000000.0, y=0.0, z=0.0),
                velocity=Vector3(x=0.0, y=7546.0, z=0.0),
                fuel=1000.0
            ),
            "targets": []
        },
        {
            "name": "Low Fuel - Should Conserve",
            "agent": AgentState(
                position=Vector3(x=7000000.0, y=0.0, z=0.0),
                velocity=Vector3(x=0.0, y=7546.0, z=0.0),
                fuel=10.0  # Very low fuel
            ),
            "targets": [
                TargetObject(
                    id="distant",
                    position=Vector3(x=7020000.0, y=0.0, z=0.0),  # 20km away
                    velocity=Vector3(x=0.0, y=7546.0, z=0.0)
                )
            ]
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        
        request = InferenceRequest(
            agent=scenario["agent"],
            targets=scenario["targets"]
        )
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=request.model_dump(),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"  [FAIL] Failed with status {response.status_code}: {response.text}")
                results.append({"scenario": scenario["name"], "success": False})
                continue
            
            result = ThrustCommand(**response.json())
            print(f"  [OK] Prediction received:")
            print(f"    Force: ({result.force_x:.2f}, {result.force_y:.2f}, {result.force_z:.2f}) N")
            print(f"    Thrust %: {result.thrust_percentage:.3f}")
            
            # Analyze decision quality if we have targets
            if scenario["targets"]:
                # Find closest target (what agent should be targeting)
                closest = min(
                    scenario["targets"],
                    key=lambda t: calculate_distance(scenario["agent"].position, t.position)
                )
                
                quality = calculate_thrust_direction_quality(
                    scenario["agent"].position,
                    closest.position,
                    result
                )
                
                print(f"  Decision Quality Analysis:")
                print(f"    Distance to target: {quality['distance_to_target']:.2f} m")
                print(f"    Thrust magnitude: {quality['thrust_magnitude']:.2f} N")
                print(f"    Alignment with target: {quality['alignment']:.3f}")
                print(f"    Angle to target: {quality['alignment_angle_deg']:.1f}°")
                print(f"    Towards target: {'✓ YES' if quality['is_towards_target'] else '✗ NO'}")
                
                results.append({
                    "scenario": scenario["name"],
                    "success": True,
                    "quality": quality
                })
            else:
                # No targets - should be zero thrust
                thrust_mag = np.sqrt(
                    result.force_x**2 + result.force_y**2 + result.force_z**2
                )
                if thrust_mag < 0.1:
                    print(f"  [OK] Correctly idling (no targets)")
                else:
                    print(f"  [WARN] Warning: Non-zero thrust with no targets")
                
                results.append({
                    "scenario": scenario["name"],
                    "success": True,
                    "idling": thrust_mag < 0.1
                })
                
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"scenario": scenario["name"], "success": False})
    
    return results

def test_websocket():
    """Test WebSocket endpoint"""
    print("\n" + "="*60)
    print("TEST 4: WebSocket Endpoint")
    print("="*60)
    
    try:
        import websocket
        import json
        
        messages_received = []
        
        def on_message(ws, message):
            data = json.loads(message)
            messages_received.append(data)
            print(f"  Received message {len(messages_received)}: {len(data.get('satellites', []))} satellites, {len(data.get('debris', []))} debris")
        
        def on_error(ws, error):
            print(f"  ✗ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"  WebSocket closed")
        
        def on_open(ws):
            print(f"  [OK] WebSocket connected")
        
        ws_url = BASE_URL.replace("http://", "ws://") + "/ws"
        print(f"  Connecting to {ws_url}...")
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run for 2 seconds to receive a few messages
        ws.run_forever()
        
        if len(messages_received) > 0:
            print(f"  [OK] Received {len(messages_received)} messages")
            return True
        else:
            print(f"  [FAIL] No messages received")
            return False
            
    except ImportError:
        print("  [SKIP] websocket-client not installed, skipping WebSocket test")
        print("  Install with: pip install websocket-client")
        return None
    except Exception as e:
        print(f"  [FAIL] WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def wait_for_server(max_retries=15, delay=2):
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            if response.status_code == 200:
                print("[OK] Server is ready!")
                return True
        except Exception as e:
            if i == max_retries - 1:
                print(f"  Last error: {e}")
        time.sleep(delay)
        print(f"  Retry {i+1}/{max_retries}...")
    return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ORBITAL JANITOR API - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Wait for server
    if not wait_for_server():
        print("\n[FAIL] Server is not responding. Make sure it's running on port 8000")
        return
    
    # Run tests
    test_results = {
        "root": test_root_endpoint(),
        "agent_loading": test_agent_loading(),
        "predictions": test_prediction_scenarios(),
        "websocket": test_websocket()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Root endpoint: {'[PASS]' if test_results['root'] else '[FAIL]'}")
    print(f"Agent loading: {'[PASS]' if test_results['agent_loading'] else '[FAIL]'}")
    
    if isinstance(test_results['predictions'], list):
        passed = sum(1 for r in test_results['predictions'] if r.get('success', False))
        total = len(test_results['predictions'])
        print(f"Prediction scenarios: {passed}/{total} passed")
        
        # Quality summary
        quality_tests = [r for r in test_results['predictions'] if 'quality' in r]
        if quality_tests:
            towards_target = sum(1 for r in quality_tests if r['quality'].get('is_towards_target', False))
            print(f"Decision quality: {towards_target}/{len(quality_tests)} decisions point towards target")
    
    if test_results['websocket'] is not None:
        print(f"WebSocket: {'[PASS]' if test_results['websocket'] else '[FAIL]'}")
    else:
        print(f"WebSocket: [SKIPPED] (websocket-client not installed)")

if __name__ == "__main__":
    main()
