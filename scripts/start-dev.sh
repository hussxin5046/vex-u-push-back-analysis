#!/bin/bash

# VEX U Analysis Platform Development Server Startup
echo "🚀 Starting VEX U Analysis Platform in development mode..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping development servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap CTRL+C
trap cleanup SIGINT

# Start backend server
echo "🐍 Starting Flask backend server..."
cd apps/backend
source venv/bin/activate
python3 app.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID) - http://localhost:8000"
cd ../..

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "⚛️ Starting React frontend server..."
cd apps/frontend
npm start &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID) - http://localhost:3000"
cd ../..

echo ""
echo "🎉 Development servers are running:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for processes
wait