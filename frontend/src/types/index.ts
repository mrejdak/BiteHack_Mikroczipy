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
