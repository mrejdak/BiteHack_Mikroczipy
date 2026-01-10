import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Line } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';

const SCALE_FACTOR = 0.001; // Scale km to units (Earth R=6371km -> 6.3 units)

// --- COMPONENTS ---

function Earth() {
  const texture = useMemo(() => new THREE.TextureLoader().load('/earth_texture.png'), []);

  return (
    <mesh rotation={[0, 0, 0]}>
      <sphereGeometry args={[6.371, 64, 64]} />
      <meshStandardMaterial map={texture} emissiveMap={texture} emissive={0x222222} emissiveIntensity={0.2} />
    </mesh>
  );
}

function ClickableObject({ obj, onSelect, isSelected }) {
  const meshRef = useRef();
  const groupRef = useRef();
  const pos = [obj.pos[0] * SCALE_FACTOR, obj.pos[2] * SCALE_FACTOR, obj.pos[1] * SCALE_FACTOR];

  // For motherships: orient towards velocity direction
  useFrame(() => {
    if (obj.type === 'mothership' && obj.vel && groupRef.current) {
      const vel = new THREE.Vector3(obj.vel[0], obj.vel[2], obj.vel[1]);
      if (vel.length() > 0.001) {
        const targetPos = new THREE.Vector3(pos[0], pos[1], pos[2]).add(vel.normalize());
        groupRef.current.lookAt(targetPos);
      }
    }
  });

  if (obj.type === 'mothership') {
    return (
      <group ref={groupRef} position={pos}>
        <mesh
          ref={meshRef}
          rotation={[Math.PI / 2, 0, 0]}
          onClick={(e) => { e.stopPropagation(); onSelect(obj); }}
        >
          <coneGeometry args={[0.12, 0.35, 8]} />
          <meshStandardMaterial
            color={isSelected ? '#FFFF00' : '#00FFFF'}
            emissive={isSelected ? '#FFFF00' : '#00AAAA'}
            emissiveIntensity={1.5}
            metalness={0.7}
            roughness={0.3}
          />
        </mesh>
        <pointLight position={[0, -0.2, 0]} color="#00FFFF" intensity={0.5} distance={1} />
      </group>
    );
  }

  return (
    <mesh
      ref={meshRef}
      position={pos}
      onClick={(e) => { e.stopPropagation(); onSelect(obj); }}
    >
      <sphereGeometry args={[obj.type === 'debris' ? 0.02 : 0.15, 6, 6]} />
      <meshStandardMaterial
        color={isSelected ? '#FFFF00' : (obj.color || '#888888')}
        emissive={isSelected ? '#FFFF00' : 0x000000}
        emissiveIntensity={isSelected ? 2.0 : 0.5}
      />
    </mesh>
  );
}

function CameraController({ targetId, objects, controlsRef }) {
  useFrame(() => {
    if (!targetId || !controlsRef.current) return;

    const target = objects.find(o => o.id === targetId);
    if (!target || !target.pos) return;

    const pos = new THREE.Vector3(
      target.pos[0] * SCALE_FACTOR,
      target.pos[2] * SCALE_FACTOR,
      target.pos[1] * SCALE_FACTOR
    );

    controlsRef.current.target.lerp(pos, 0.05);
    controlsRef.current.update();
  });

  return null;
}

function TargetingLine({ from, to }) {
  if (!from || !to) return null;

  const points = [
    [from[0] * SCALE_FACTOR, from[2] * SCALE_FACTOR, from[1] * SCALE_FACTOR],
    [to[0] * SCALE_FACTOR, to[2] * SCALE_FACTOR, to[1] * SCALE_FACTOR]
  ];

  return (
    <Line
      points={points}
      color="#FF6600"
      lineWidth={1}
      dashed={true}
      dashScale={50}
      dashSize={0.1}
      dashOffset={0}
    />
  );
}

function Scene({ objects, onSelect, selectedId, controlsRef }) {
  const motherships = objects.filter(o => o.type === 'mothership');

  return (
    <>
      <Earth />

      {/* All Objects */}
      {objects.map((obj) => (
        <ClickableObject
          key={obj.id}
          obj={obj}
          onSelect={onSelect}
          isSelected={selectedId === obj.id}
        />
      ))}

      {/* Targeting Lines */}
      {motherships.map((ms) => {
        const tel = ms.telemetry || {};
        if (tel.target_pos && ms.pos) {
          return <TargetingLine key={`line-${ms.id}`} from={ms.pos} to={tel.target_pos} />;
        }
        return null;
      })}

      {/* Camera Tracking */}
      <CameraController targetId={selectedId} objects={objects} controlsRef={controlsRef} />
    </>
  );
}

function CaptureNotifications({ captures }) {
  return (
    <div style={{
      position: 'absolute',
      bottom: 80,
      right: 20,
      width: '300px',
      pointerEvents: 'none'
    }}>
      {captures.map((c, i) => (
        <div
          key={i}
          style={{
            background: 'rgba(0, 255, 100, 0.9)',
            color: '#000',
            padding: '10px 15px',
            borderRadius: '6px',
            marginBottom: '8px',
            fontFamily: 'monospace',
            fontWeight: 'bold',
            fontSize: '14px',
            animation: 'fadeIn 0.3s ease-out'
          }}
        >
          ✓ CAPTURED: {c.debrisId}
          <div style={{ fontSize: '11px', fontWeight: 'normal' }}>by {c.agentId}</div>
        </div>
      ))}
    </div>
  );
}

function TelemetryHUD({ objects, selectedId, onClearSelection, debrisCount }) {
  const agents = objects.filter(o => o.type === 'mothership');
  const selectedObject = objects.find(o => o.id === selectedId);

  return (
    <div style={{
      position: 'absolute',
      top: 20,
      right: 20,
      width: '320px',
      background: 'rgba(0, 20, 40, 0.9)',
      border: '1px solid #00FFFF',
      borderRadius: '8px',
      padding: '15px',
      color: '#00FFFF',
      fontFamily: 'monospace',
      maxHeight: '80vh',
      overflowY: 'auto'
    }}>
      {/* Stats */}
      <div style={{ marginBottom: '10px', fontSize: '12px', color: '#888' }}>
        Debris: {debrisCount} | Agents: {agents.length}
      </div>

      {/* Selected Object Info */}
      {selectedObject && (
        <div style={{ marginBottom: '15px', borderBottom: '1px solid #00FFFF', paddingBottom: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, color: '#FFFF00' }}>🎯 TRACKING: {selectedObject.id}</h4>
            <button
              onClick={onClearSelection}
              style={{ background: 'none', border: '1px solid #FF5555', color: '#FF5555', cursor: 'pointer', padding: '2px 8px', borderRadius: '3px' }}
            >
              ✕
            </button>
          </div>
          <div style={{ fontSize: '12px', marginTop: '5px' }}>
            Type: {selectedObject.type?.toUpperCase() || 'UNKNOWN'}
          </div>
          {selectedObject.pos && (
            <div style={{ fontSize: '11px', color: '#888' }}>
              Pos: [{selectedObject.pos.map(p => p.toFixed(0)).join(', ')}] km
            </div>
          )}
        </div>
      )}

      {/* Agent Telemetry */}
      <h3 style={{ margin: '0 0 10px 0', borderBottom: '1px solid #00FFFF', paddingBottom: '5px' }}>AGENT TELEMETRY</h3>
      {agents.length === 0 ? (
        <div style={{ color: '#666', fontSize: '12px' }}>No agents spawned. Click button below.</div>
      ) : (
        agents.map(agent => {
          const tel = agent.telemetry || {};
          const thrust = tel.thrust || [0, 0, 0];
          const thrustMag = Math.sqrt(thrust[0] ** 2 + thrust[1] ** 2 + thrust[2] ** 2);
          const isThrusting = thrustMag > 0.0001;

          return (
            <div
              key={agent.id}
              style={{
                marginBottom: '12px',
                padding: '8px',
                background: selectedId === agent.id ? 'rgba(255,255,0,0.2)' : 'rgba(0,255,255,0.1)',
                borderRadius: '4px',
                cursor: 'pointer',
                border: selectedId === agent.id ? '1px solid #FFFF00' : '1px solid transparent'
              }}
            >
              <div style={{ fontWeight: 'bold', color: 'white' }}>{agent.id.toUpperCase()}</div>
              <div style={{ fontSize: '12px', color: isThrusting ? '#0f0' : '#888' }}>
                {isThrusting ? '🚀 THRUSTING' : '○ IDLE'}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginTop: '4px' }}>
                <span>FUEL:</span>
                <span>{Math.round(tel.fuel || 0)} / 1000</span>
              </div>
              <div style={{ width: '100%', height: '4px', background: '#222', marginTop: '2px', borderRadius: '2px' }}>
                <div style={{
                  width: `${Math.min(100, (tel.fuel || 0) / 10)}%`,
                  height: '100%',
                  background: (tel.fuel || 0) > 200 ? '#00FFFF' : '#FF5555',
                  borderRadius: '2px'
                }} />
              </div>

              {tel.target_id ? (
                <div style={{ fontSize: '11px', marginTop: '4px', color: '#FF6600' }}>
                  🎯 {tel.target_id} ({tel.target_dist?.toFixed(0) || '?'} km)
                </div>
              ) : (
                <div style={{ fontSize: '11px', marginTop: '4px', color: '#555' }}>
                  Scanning...
                </div>
              )}
            </div>
          );
        })
      )}

      <div style={{ fontSize: '10px', color: '#555', marginTop: '10px', textAlign: 'center' }}>
        Click any object to track camera
      </div>
    </div>
  );
}

export default function App() {
  const [objects, setObjects] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [captures, setCaptures] = useState([]);
  const [prevDebrisCount, setPrevDebrisCount] = useState(0);
  const controlsRef = useRef();

  // Polling loop for state
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('http://localhost:5000/api/state')
        .then(res => {
          const newObjects = res.data.objects || [];
          setObjects(newObjects);

          // Detect captures (debris count decreased)
          const newDebrisCount = newObjects.filter(o => o.type === 'debris').length;
          if (prevDebrisCount > 0 && newDebrisCount < prevDebrisCount) {
            const diff = prevDebrisCount - newDebrisCount;
            // Add capture notification
            setCaptures(prev => [...prev.slice(-4), {
              debrisId: `debris (${diff} captured)`,
              agentId: 'Mothership',
              time: Date.now()
            }]);
          }
          setPrevDebrisCount(newDebrisCount);
        })
        .catch(() => { });
    }, 150);
    return () => clearInterval(interval);
  }, [prevDebrisCount]);

  // Auto-remove old notifications
  useEffect(() => {
    const interval = setInterval(() => {
      setCaptures(prev => prev.filter(c => Date.now() - c.time < 5000));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleSelect = useCallback((obj) => {
    setSelectedId(obj.id);
    console.log("Tracking:", obj.id);
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedId(null);
  }, []);

  const spawnAgent = () => {
    axios.post('http://localhost:5000/api/mothership/register')
      .then(res => console.log("Spawned:", res.data))
      .catch(err => console.error(err));
  };

  const debrisCount = objects.filter(o => o.type === 'debris').length;

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000' }}>
      <Canvas camera={{ position: [0, 0, 25], fov: 45 }}>
        <ambientLight intensity={1.5} />
        <pointLight position={[10, 10, 10]} intensity={2.0} />
        <directionalLight position={[-10, 5, 5]} intensity={1.0} />

        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

        <Scene
          objects={objects}
          onSelect={handleSelect}
          selectedId={selectedId}
          controlsRef={controlsRef}
        />

        <OrbitControls ref={controlsRef} enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>

      {/* Title */}
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', fontFamily: 'monospace', pointerEvents: 'none' }}>
        <h1 style={{ margin: 0, fontSize: '28px' }}>Orbital Janitor</h1>
        <p style={{ margin: '5px 0 0 0', color: '#00FFFF' }}>BiteHack 2026</p>
      </div>

      {/* HUD */}
      <TelemetryHUD
        objects={objects}
        selectedId={selectedId}
        onClearSelection={handleClearSelection}
        debrisCount={debrisCount}
      />

      {/* Capture Notifications */}
      <CaptureNotifications captures={captures} />

      {/* Controls */}
      <div style={{ position: 'absolute', bottom: 20, left: 20, pointerEvents: 'auto' }}>
        <button
          onClick={spawnAgent}
          style={{
            padding: '12px 24px',
            background: 'rgba(0, 255, 255, 0.15)',
            color: '#00FFFF',
            border: '2px solid #00FFFF',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            fontFamily: 'monospace',
            transition: 'all 0.2s'
          }}
          onMouseOver={(e) => e.target.style.background = 'rgba(0, 255, 255, 0.3)'}
          onMouseOut={(e) => e.target.style.background = 'rgba(0, 255, 255, 0.15)'}
        >
          + SPAWN AGENT
        </button>
      </div>

      {/* CSS Animation */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}
