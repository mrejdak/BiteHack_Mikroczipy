# BiteHack Project 2025

This repository contains the source code for the BiteHack project, featuring two advanced modules for space and earth monitoring: **Fire Detection** and **GNN Satellite Networking**.

## 🌍 Modules Overview

### 1. Fire Detection System (`fire_detection/`)
A comprehensive system for detecting wildfires using satellite imagery.
- **Frontend**: Interactive React application with Leaflet maps for visualizing fire hotspots.
- **Backend**: Python FastAPI service utilizing Computer Vision (OpenCV, YOLO/Ultralytics) and geospatial libraries (Rasterio, Shapely) to process satellite data.

### 2. GNN Satellite Routing (`gnn_network/`)
A simulation and visualization of satellite network routing using Graph Neural Networks and Deep Q Reinforcement Learning.
- **Frontend**: High-performance React + Vite application for visualizing satellite constellations and routing paths.
- **Backend**: FastAPI Application with PyTorch and NetworkX, implementing intelligent routing algorithms that optimize for latency and throughput in dynamic space topologies.

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+ recommended)
- **Python** (v3.10+ recommended)

---

### 🔥 Fire Detection System

#### Backend
1. Navigate to the backend directory:
   ```bash
   cd fire_detection/backend
   ```
2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   # or venv\Scripts\activate # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

#### Frontend
1. Navigate to the fire detection root (where `package.json` is located):
   ```bash
   cd fire_detection
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

---

### 🛰️ GNN Satellite Network

#### Backend
1. Navigate to the module root:
   ```bash
   cd gnn_network
   ```
2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the simulation server:
   ```bash
   # Run from the gnn_network root to ensure 'src' imports work correctly
   uvicorn backend.main:app --reload
   ```

#### Frontend
1. Navigate to the frontend directory:
   ```bash
   cd gnn_network/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the visualization interface:
   ```bash
   npm run dev
   ```

## 📄 License
See the [LICENSE](LICENSE) file for details.
