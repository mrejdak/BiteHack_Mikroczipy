import axios from 'axios';
import type { DetectionResponse } from '../types';

const API_URL = 'http://localhost:8000/api/v1';

export const api = axios.create({
    baseURL: API_URL,
});

export const detectFire = async (file: File): Promise<DetectionResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<DetectionResponse>('/detect', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};
