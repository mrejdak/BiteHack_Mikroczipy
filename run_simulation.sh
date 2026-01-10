#!/bin/bash

# Kill any existing processes on ports 5000 (Flask) and 5173 (Vite)
fuser -k 5000/tcp 2>/dev/null
fuser -k 5173/tcp 2>/dev/null

# 1. Start Backend (sim + server)
echo "🚀 Starting Backend Server..."
./.venv/bin/python backend/server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
sleep 5

# 2. Start Frontend (visualization)
echo "🌍 Starting Frontend..."
cd frontend
npm run dev -- --host > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ System is UP!"
echo "   - Visualization: http://localhost:5173"
echo "   - Backend API:   http://localhost:5000"
echo ""
echo "🤖 Spawning Agents in 5 seconds..."
sleep 5

# 3. Launch Agents
./.venv/bin/python launch_swarm.py

# Cleanup on exit
kill $BACKEND_PID
kill $FRONTEND_PID
