import { create } from 'zustand';
import { FireAlert } from '../types';

interface AlertStore {
    alerts: FireAlert[];
    loading: boolean;
    setAlerts: (alerts: FireAlert[]) => void;
    setLoading: (loading: boolean) => void;
    addAlerts: (newAlerts: FireAlert[]) => void;
    clearAlerts: () => void;
}

export const useAlertStore = create<AlertStore>((set) => ({
    alerts: [],
    loading: false,
    setAlerts: (alerts) => set({ alerts }),
    setLoading: (loading) => set({ loading }),
    addAlerts: (newAlerts) => set((state) => ({ alerts: [...state.alerts, ...newAlerts] })),
    clearAlerts: () => set({ alerts: [] }),
}));
