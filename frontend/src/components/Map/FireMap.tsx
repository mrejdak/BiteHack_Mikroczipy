import { Fragment, useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polygon } from 'react-leaflet';
import { useAlertStore } from '../../store/useAlertStore';
import { simulateFireSpread } from '../../services/api';
import type { FireSpreadPoint, WeatherInfo } from '../../types';
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

interface SpreadData {
    alertIndex: number;
    points: FireSpreadPoint[];
    weather: WeatherInfo | null;
    loading: boolean;
    origin: { lat: number; lon: number } | null;
    maxRadiusMeters: number;
    totalHours: number;
}

// Calculate distance between two lat/lon points in meters (Haversine formula)
const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371000; // Earth's radius in meters
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
};

const FireMap = () => {
    const { alerts } = useAlertStore();
    const [spreadData, setSpreadData] = useState<SpreadData | null>(null);

    const handleSimulateSpread = useCallback(async (alertIndex: number, lat: number, lon: number, hours: number = 1) => {
        // Show loading state
        setSpreadData({ alertIndex, points: [], weather: null, loading: true, origin: { lat, lon }, maxRadiusMeters: 0, totalHours: hours });

        try {
            const result = await simulateFireSpread(lat, lon, hours);
            if (result.success && result.fire_origin) {
                // Calculate maximum distance from origin
                let maxRadius = 0;
                for (const point of result.affected_areas) {
                    const distance = calculateDistance(
                        result.fire_origin.lat,
                        result.fire_origin.lon,
                        point.lat,
                        point.lon
                    );
                    if (distance > maxRadius) {
                        maxRadius = distance;
                    }
                }

                setSpreadData({
                    alertIndex,
                    points: result.affected_areas,
                    weather: result.weather,
                    loading: false,
                    origin: result.fire_origin,
                    maxRadiusMeters: maxRadius,
                    totalHours: result.simulation_duration_hours,
                });
            } else {
                console.error('Fire simulation failed:', result.error);
                setSpreadData(null);
            }
        } catch (error) {
            console.error('Error simulating fire spread:', error);
            setSpreadData(null);
        }
    }, []);

    const clearSpread = useCallback(() => {
        setSpreadData(null);
    }, []);

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

            {/* Render fire spread area as a comet shape based on wind direction */}
            {spreadData && !spreadData.loading && spreadData.origin && spreadData.maxRadiusMeters > 0 && spreadData.weather && (() => {
                // Generate comet-shaped polygon elongated in wind direction
                const origin = spreadData.origin;
                const radius = spreadData.maxRadiusMeters;
                // Wind direction is where wind comes FROM, fire spreads TOWARDS (opposite)
                const windFromDeg = spreadData.weather.wind_direction_deg;
                const spreadDirRad = ((windFromDeg + 180) % 360) * Math.PI / 180;

                // Create comet shape: elongated ellipse in wind direction
                const points: [number, number][] = [];
                const numPoints = 36;

                // Meters to degrees (approximate)
                const metersToDegLat = 1 / 111320;
                const metersToDegLon = 1 / (111320 * Math.cos(origin.lat * Math.PI / 180));

                for (let i = 0; i < numPoints; i++) {
                    const angle = (i / numPoints) * 2 * Math.PI;

                    // Calculate distance from center - elongate in spread direction
                    const angleDiff = Math.abs(angle - spreadDirRad);
                    const normalizedDiff = Math.min(angleDiff, 2 * Math.PI - angleDiff);

                    // Comet shape: long in spread direction, shorter opposite
                    let distMultiplier = 0.5; // base size
                    if (normalizedDiff < Math.PI / 2) {
                        // In spread direction - elongate more
                        distMultiplier = 0.5 + 1.5 * (1 - normalizedDiff / (Math.PI / 2));
                    }

                    const dist = radius * distMultiplier;

                    const dx = dist * Math.sin(angle);
                    const dy = dist * Math.cos(angle);

                    points.push([
                        origin.lat + dy * metersToDegLat,
                        origin.lon + dx * metersToDegLon
                    ]);
                }

                return (
                    <Polygon
                        positions={points}
                        pathOptions={{
                            color: '#ff4400',
                            fillColor: '#ff6600',
                            fillOpacity: 0.35,
                            weight: 2,
                        }}
                    />
                );
            })()}

            {/* Render fire alert markers - NO default circle */}
            {alerts.map((alert, index) => (
                <Fragment key={index}>
                    <Marker position={[alert.location.lat, alert.location.lon]} icon={fireIcon}>
                        <Popup>
                            <div style={{ minWidth: '200px' }}>
                                <h3>🔥 Fire Detected</h3>
                                <p>Severity: {alert.severity}</p>
                                <p>Time: {new Date(alert.timestamp).toLocaleString()}</p>

                                {/* Simulate button */}
                                {spreadData?.alertIndex === index && spreadData.loading ? (
                                    <p style={{ color: '#ff6600', fontWeight: 'bold' }}>
                                        ⏳ Simulating fire spread...
                                    </p>
                                ) : spreadData?.alertIndex === index ? (
                                    <div>
                                        <p style={{ color: '#ff4400', fontWeight: 'bold' }}>
                                            🔥 Spread after {spreadData.totalHours}hr ({(spreadData.maxRadiusMeters / 1000).toFixed(2)} km)
                                        </p>
                                        {spreadData.weather && (
                                            <p style={{ fontSize: '0.9em' }}>
                                                Wind: {spreadData.weather.wind_speed_mph.toFixed(1)} mph @ {spreadData.weather.wind_direction_deg}°
                                            </p>
                                        )}
                                        <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                                            <button
                                                onClick={() => handleSimulateSpread(index, alert.location.lat, alert.location.lon, spreadData.totalHours + 1)}
                                                style={{
                                                    padding: '6px 12px',
                                                    backgroundColor: '#ff6600',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontWeight: 'bold',
                                                }}
                                            >
                                                +1hr More
                                            </button>
                                            <button
                                                onClick={clearSpread}
                                                style={{
                                                    padding: '6px 12px',
                                                    backgroundColor: '#666',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                }}
                                            >
                                                Clear
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <button
                                        onClick={() => handleSimulateSpread(index, alert.location.lat, alert.location.lon)}
                                        style={{
                                            marginTop: '8px',
                                            padding: '8px 16px',
                                            backgroundColor: '#ff4400',
                                            color: 'white',
                                            border: 'none',
                                            borderRadius: '4px',
                                            cursor: 'pointer',
                                            fontWeight: 'bold',
                                        }}
                                    >
                                        📊 Simulate 1hr Spread
                                    </button>
                                )}

                                {/* Show annotated satellite image if available, otherwise show infrastructure */}
                                {alert.annotated_image_base64 ? (
                                    <div style={{ marginTop: '10px' }}>
                                        <h4>🛰️ Detected Fire:</h4>
                                        <img
                                            src={`data:image/png;base64,${alert.annotated_image_base64}`}
                                            alt="Fire detection"
                                            style={{
                                                width: '100%',
                                                maxWidth: '300px',
                                                borderRadius: '4px',
                                                border: '2px solid #ff4400'
                                            }}
                                        />
                                    </div>
                                ) : alert.infrastructure.length > 0 && (
                                    <div style={{ marginTop: '10px' }}>
                                        <h4>Nearby Infrastructure:</h4>
                                        <ul style={{ paddingLeft: '16px', margin: 0 }}>
                                            {alert.infrastructure.map((infra, i) => (
                                                <li key={i}>{infra.name || 'Unnamed'} ({infra.type || 'Unknown'})</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </Popup>
                    </Marker>
                </Fragment >
            ))}
        </MapContainer >
    );
};

export default FireMap;
