export interface GeoLocation {
    lat: number;
    lon: number;
}

export interface InfrastructureData {
    name?: string;
    type?: string;
    operator?: string;
    address?: string;
}

export interface FireAlert {
    location: GeoLocation;
    infrastructure: InfrastructureData[];
    timestamp: string;
    severity: string;
}

export interface DetectionResponse {
    fire_detected: boolean;
    alerts: FireAlert[];
}

// Fire Spread Simulation Types
export interface FireSpreadPoint {
    lat: number;
    lon: number;
    status: 'burning' | 'burned';
}

export interface WeatherInfo {
    wind_speed_mph: number;
    wind_direction_deg: number;
    humidity: number;
}

export interface FireSpreadResponse {
    success: boolean;
    fire_origin: GeoLocation | null;
    affected_areas: FireSpreadPoint[];
    simulation_duration_hours: number;
    weather: WeatherInfo | null;
    error: string | null;
}
