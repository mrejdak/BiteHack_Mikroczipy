import { Fragment } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import { useAlertStore } from '../../store/useAlertStore';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icon issue
delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const fireIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
});

const FireMap = () => {
    const { alerts } = useAlertStore();

    return (
        <MapContainer
            center={[52.0, 19.0]}
            zoom={6}
            style={{ height: '100%', width: '100%' }}
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {alerts.map((alert, index) => (
                <Fragment key={index}>
                    <Circle
                        center={[alert.location.lat, alert.location.lon]}
                        radius={1000}
                        pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: 0.3 }}
                    />
                    <Marker position={[alert.location.lat, alert.location.lon]} icon={fireIcon}>
                        <Popup>
                            <div>
                                <h3>🔥 Fire Detected</h3>
                                <p>Severity: {alert.severity}</p>
                                <p>Time: {new Date(alert.timestamp).toLocaleString()}</p>
                                {alert.infrastructure.length > 0 && (
                                    <div>
                                        <h4>Nearby Infrastructure:</h4>
                                        <ul>
                                            {alert.infrastructure.map((infra, i) => (
                                                <li key={i}>{infra.name || 'Unnamed'} ({infra.type || 'Unknown'})</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </Popup>
                    </Marker>
                </Fragment>
            ))}
        </MapContainer>
    );
};

export default FireMap;
