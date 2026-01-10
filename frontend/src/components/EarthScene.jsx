import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';

// --- Constants ---
const EARTH_RADIUS = 6.371; // Scale down: 1 unit = 1000km approx
const SAT_SIZE = 0.07;

function Earth() {
  const texture = useLoader(THREE.TextureLoader, '/global_texture.jpg');
  return (
    <mesh rotation={[Math.PI / 2, 0, 0]}>
      <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  );
}

function Satellite({ id, position, isActive, isPath, isSrc, isDst, onClick }) {
  const pos = useMemo(() => new THREE.Vector3(...position).divideScalar(1000), [position]);

  const color = useMemo(() => {
    if (isSrc) return '#00ff00'; // Pure Green for Source
    if (isDst) return '#ff0000'; // Pure Red for Dest
    if (!isActive) return '#ffff00'; // Yellow for Turned Off
    if (isPath) return '#00ff00'; // Path Green (maybe lighter?) or keep uniform
    // Let's differentiate Path: Lime, Source: Green, Dst: Red
    if (isPath) return '#adff2f'; // GreenYellow
    return '#1e90ff'; // Dodger blue (Default)
  }, [isActive, isPath, isSrc, isDst]);

  // Scale up Src/Dst
  const scale = (isSrc || isDst) ? 1.5 : 1;

  return (
    <mesh position={pos} scale={[scale, scale, scale]} onClick={(e) => { e.stopPropagation(); onClick(id); }}>
      <sphereGeometry args={[SAT_SIZE, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} />
    </mesh>
  );
}

function Connections({ edges, satellites, activePath, aiPath, brokenNodes, isPaused }) {
  if (!satellites || satellites.length === 0) return null;

  const satMap = useMemo(() => {
    const map = {};
    satellites.forEach(s => map[s.id] = new THREE.Vector3(...s.position).divideScalar(1000));
    return map;
  }, [satellites]);

  const { cleanEdges, pathEdges, aiPathEdges } = useMemo(() => {
    const clean = [];
    const path = [];
    const ai = [];
    const processedPairs = new Set(); // For labeling

    // Optimal Path (Green)
    if (activePath && activePath.length > 1) {
      for (let i = 0; i < activePath.length - 1; i++) {
        const u = activePath[i];
        const v = activePath[i + 1];
        if (satMap[u] && satMap[v]) {
          path.push([satMap[u], satMap[v]]);
        }
      }
    }

    // AI Path (Magenta)
    if (aiPath && aiPath.length > 1) {
      for (let i = 0; i < aiPath.length - 1; i++) {
        const u = aiPath[i];
        const v = aiPath[i + 1];
        if (satMap[u] && satMap[v]) {
          ai.push([satMap[u], satMap[v]]);
        }
      }
    }

    // Grid Edges
    edges.forEach(edge => {
      const u = edge.u !== undefined ? edge.u : edge[0];
      const v = edge.v !== undefined ? edge.v : edge[1];
      const load = edge.load || 0;

      if (brokenNodes.has(u) || brokenNodes.has(v)) return;

      if (satMap[u] && satMap[v]) {
        let color = 'cyan';
        if (load > 0.8) color = '#ff3333';
        else if (load > 0.5) color = '#ffaa33';

        // Label Logic: Only add label for u < v to avoid duplicate text at same spot
        let labelPos = null;
        if (u < v) {
          // True Midpoint
          labelPos = new THREE.Vector3().addVectors(satMap[u], satMap[v]).multiplyScalar(0.5);
        }

        clean.push({ points: [satMap[u], satMap[v]], color, load, labelPos });
      }
    });

    return { cleanEdges: clean, pathEdges: path, aiPathEdges: ai };
  }, [edges, satellites, activePath, aiPath, brokenNodes, satMap]);

  return (
    <group>
      {/* Background Grid */}
      {cleanEdges.map((item, i) => (
        <React.Fragment key={`edge-${i}`}>
          <Line
            points={item.points}
            color={item.color}
            opacity={0.3 + (item.load * 0.5)}
            transparent
            lineWidth={0.5 + (item.load * 1)}
          />
          {isPaused && item.labelPos && (
            <group position={item.labelPos} ref={(ref) => ref && ref.lookAt(new THREE.Vector3().copy(item.labelPos).multiplyScalar(2))}>
              <Text
                color={item.color}
                fontSize={0.2}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
              >
                {Math.round(item.load * 100)}%
              </Text>
            </group>
          )}
        </React.Fragment>
      ))}

      {/* AI Path - Magenta Neon (under optimal) */}
      {aiPathEdges.map((points, i) => (
        <Line
          key={`aipath-${i}`}
          points={points}
          color="#ff00ff"
          lineWidth={4}
          depthTest={false} // Ensure it draws ON TOP of grid
          renderOrder={1}
        />
      ))}

      {/* Optimal Path - Thick Lime Neon */}
      {pathEdges.map((points, i) => (
        <Line
          key={`path-${i}`}
          points={points}
          color="#00ff00"
          lineWidth={5}
          depthTest={false} // Draw on top
          renderOrder={2}
        />
      ))}
    </group>
  );
}

export default function EarthScene() {
  const [data, setData] = useState({ satellites: [], path: [], ai_path: [], time: 0, is_paused: false, orbits: [] });
  const [edges, setEdges] = useState([]);
  const [brokenNodes, setBrokenNodes] = useState(new Set());

  // Polling Loop
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('http://localhost:8000/simulation/state')
        .then(res => {
          setData(res.data);
          // Update broken nodes set
          const broken = new Set(res.data.satellites.filter(s => !s.is_active).map(s => s.id));
          setBrokenNodes(broken);

          // Edges are now dynamic and come from backend state!
          if (res.data.edges) {
            setEdges(res.data.edges);
          }
        })
        .catch(err => console.error("Polling error", err));
    }, 100); // 10 FPS poll
    return () => clearInterval(interval);
  }, []);

  const togglePause = () => {
    const endpoint = data.is_paused ? 'resume' : 'pause';
    axios.post(`http://localhost:8000/simulation/${endpoint}`)
      .then(res => console.log(res.data))
      .catch(console.error);
  };

  const handleToggle = async (id) => {
    console.log("Toggling", id);
    try {
      await axios.post(`http://localhost:8000/simulation/toggle/${id}`);
      // State update will happen on next poll
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: 'black' }}>
      <Canvas camera={{ position: [0, 20, 0], up: [0, 0, 1], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[50, 50, 50]} intensity={1} />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

        <Earth />

        {data.satellites.map(sat => (
          <Satellite
            key={sat.id}
            {...sat}
            isPath={data.path.includes(sat.id)}
            isActive={!brokenNodes.has(sat.id)}
            isSrc={sat.id === data.src_id}
            isDst={sat.id === data.dst_id}
            onClick={handleToggle}
          />
        ))}

        {data.orbits && data.orbits.map((orbitPoints, i) => (
          <Line
            key={`orbit-${i}`}
            points={orbitPoints.map(p => new THREE.Vector3(...p).divideScalar(1000))}
            color="#ffffff"
            opacity={0.1}
            transparent
            lineWidth={0.5}
            depthWrite={false}
          />
        ))}

        <Connections
          edges={edges}
          satellites={data.satellites}
          activePath={data.path}
          aiPath={data.ai_path}
          brokenNodes={brokenNodes}
          isPaused={data.is_paused}
        />

        <OrbitControls minDistance={2} maxDistance={100} enableDamping dampingFactor={0.1} rotateSpeed={0.5} />
      </Canvas>

      {/* HUD */}
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', fontFamily: 'monospace', pointerEvents: 'none', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '8px' }}>
        <h2>Satellite Network (GRouting)</h2>
        <p>Time: {data.time.toFixed(1)}s {data.is_paused ? "(PAUSED)" : ""}</p>
        <p>Active Sats: {data.satellites.length - brokenNodes.size}/{data.satellites.length}</p>

        <div style={{ pointerEvents: 'auto', marginTop: '10px', marginBottom: '10px' }}>
          <button
            onClick={togglePause}
            style={{
              background: data.is_paused ? 'lime' : 'orange',
              color: 'black', border: 'none', padding: '5px 10px',
              fontWeight: 'bold', cursor: 'pointer'
            }}
          >
            {data.is_paused ? "RESUME SIMULATION" : "PAUSE & INSPECT"}
          </button>
        </div>

        <div style={{ marginTop: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: 20, height: 4, background: '#00ff00' }}></div> Optimal Path (Dijkstra)</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: 20, height: 4, background: '#ff00ff' }}></div> AI Agent Path (Learning)</div>
        </div>

        <div style={{ marginTop: '10px' }}>
          <p>Link Traffic:</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: 10, height: 10, background: 'cyan' }}></div> Low</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: 10, height: 10, background: '#ffaa33' }}></div> Medium</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: 10, height: 10, background: '#ff3333' }}></div> High (Congestion)</div>
        </div>
      </div>
    </div>
  );
}
