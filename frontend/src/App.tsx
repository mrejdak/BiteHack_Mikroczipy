import { useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import FireMap from './components/Map/FireMap';
import AlertList from './components/Sidebar/AlertList';
import { detectFire } from './services/api';
import { useAlertStore } from './store/useAlertStore';
import './index.css';

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { setLoading, addAlerts, clearAlerts, loading } = useAlertStore();

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const response = await detectFire(file);
      if (response.fire_detected) {
        addAlerts(response.alerts);
        toast.error(`🔥 ${response.alerts.length} fire(s) detected!`, {
          position: 'top-right',
          theme: 'dark',
        });
      } else {
        toast.success('✅ No fires detected in this image.', {
          position: 'top-right',
          theme: 'dark',
        });
      }
    } catch (error) {
      console.error('Detection failed:', error);
      toast.error('❌ Failed to process image. Is the backend running?', {
        position: 'top-right',
        theme: 'dark',
      });
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="dashboard">
      {/* Sidebar */}
      <aside className="sidebar">
        <header className="sidebar-header">
          <h1>🛰️ Fire Detection System</h1>
          <p>Satellite Image Analysis</p>
        </header>

        {/* Upload Button */}
        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleUpload}
            style={{ display: 'none' }}
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className={`upload-label ${loading ? 'loading' : ''}`}
          >
            {loading ? '⏳ Analyzing...' : '📤 Upload Satellite Image'}
          </label>
          <button onClick={clearAlerts} className="clear-btn">
            🗑️ Clear Alerts
          </button>
        </div>

        {/* Alert List */}
        <div className="alert-list">
          <AlertList />
        </div>
      </aside>

      {/* Map */}
      <main className="map-container">
        <FireMap />
      </main>

      <ToastContainer aria-label="Notifications" />
    </div>
  );
}

export default App;
