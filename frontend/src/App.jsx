import React, { useState, useEffect } from 'react';
import EarthScene from './components/EarthScene';
import axios from 'axios';

function TrainingPanel() {
  const [status, setStatus] = useState({ is_training: false, episode: 0, reward: 0, history: [] });

  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('http://localhost:8000/training/status')
        .then(res => setStatus(res.data))
        .catch(console.error);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const startTraining = () => {
    axios.post('http://localhost:8000/training/start')
      .then(res => console.log(res.data))
      .catch(console.error);
  };

  return (
    <div style={{
      position: 'absolute', bottom: 20, right: 20,
      background: 'rgba(0, 20, 40, 0.8)', padding: '20px',
      border: '1px solid cyan', borderRadius: '10px', color: 'cyan',
      fontFamily: 'monospace', width: '300px'
    }}>
      <h3>AI Training Control</h3>
      <div style={{ marginBottom: '10px' }}>
        Episode: {status.episode} <br />
        Last Reward: {status.reward.toFixed(2)} <br />
        Epsilon: {status.epsilon ? status.epsilon.toFixed(3) : 'N/A'}
      </div>

      <button
        onClick={startTraining}
        disabled={status.is_training}
        style={{
          background: status.is_training ? 'gray' : 'cyan',
          color: 'black', border: 'none', padding: '10px 20px',
          fontWeight: 'bold', cursor: status.is_training ? 'default' : 'pointer',
          width: '100%'
        }}
      >
        {status.is_training ? 'TRAINING IN PROGRESS...' : 'START TRAINING'}
      </button>

      {/* Simple Graph */}
      <div style={{ marginTop: '15px', height: '60px', borderBottom: '1px solid white', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
        {status.history.map((pt, i) => {
          // Normalize height. Abs max reward approx 20? Min around -10?
          const h = Math.max(0, Math.min(100, (pt.reward + 10) * 3));
          return (
            <div key={i} style={{ width: '4px', height: `${h}%`, background: pt.reward > 0 ? 'lime' : 'red' }} />
          )
        })}
      </div>
    </div>
  );
}

function App() {
  return (
    <>
      <EarthScene />
      <TrainingPanel />
    </>
  );
}

export default App;
