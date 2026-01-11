import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import FireMap from './components/Map/FireMap';
import AlertList from './components/Sidebar/AlertList';
import { detectMockFire } from './services/api';
import { useAlertStore } from './store/useAlertStore';
import './index.css';

function App() {
  const { setLoading, addAlerts, clearAlerts, loading } = useAlertStore();

  const handleMockData = async () => {
    setLoading(true);
    try {
      // Use mock image 1
      const response = await detectMockFire(1);
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

        {/* Mock Data Button */}
        <div className="upload-section">
          <button
            onClick={handleMockData}
            disabled={loading}
            className={`upload-label ${loading ? 'loading' : ''}`}
            style={{ border: 'none', width: '100%' }}
          >
            {loading ? '⏳ Analyzing...' : '🛰️ Mock Satellite Data'}
          </button>
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

