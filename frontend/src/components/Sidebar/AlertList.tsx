import { useAlertStore } from '../../store/useAlertStore';

const AlertList = () => {
    const { alerts, loading } = useAlertStore();

    return (
        <>
            <h2>
                🚨 Fire Alerts
                {loading && <span className="loading-text">(Loading...)</span>}
            </h2>

            {alerts.length === 0 && !loading && (
                <p className="no-alerts">No alerts yet. Upload an image to detect fires.</p>
            )}

            {alerts.map((alert, index) => (
                <div key={index} className="alert-card">
                    <div className="alert-card-header">
                        <span className="alert-title">🔥 Alert #{index + 1}</span>
                        <span className={`severity-badge ${alert.severity}`}>
                            {alert.severity.toUpperCase()}
                        </span>
                    </div>

                    <div className="alert-info">
                        <p>📍 Lat: {alert.location.lat.toFixed(4)}, Lon: {alert.location.lon.toFixed(4)}</p>
                        <p>🕐 {new Date(alert.timestamp).toLocaleString()}</p>
                    </div>

                    {alert.infrastructure.length > 0 && (
                        <div className="infrastructure-section">
                            <h4>⚠️ Nearby Infrastructure:</h4>
                            {alert.infrastructure.map((infra, i) => (
                                <div key={i} className="infrastructure-item">
                                    <span style={{ fontWeight: 500 }}>{infra.name || 'Unnamed Facility'}</span>
                                    {infra.type && <span style={{ color: '#9ca3af' }}> · {infra.type}</span>}
                                    {infra.operator && <span style={{ color: '#9ca3af' }}> · {infra.operator}</span>}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ))}
        </>
    );
};

export default AlertList;
