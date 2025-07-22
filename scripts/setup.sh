#!/bin/bash

# VEX U Analysis Platform Setup Script
echo "🚀 Setting up VEX U Analysis Platform..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please review and update .env file with your configuration"
fi

# Setup Python virtual environment for backend
echo "🐍 Setting up Python backend environment..."
cd apps/backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
cd ../..

# Setup Python environment for analysis package
echo "📊 Setting up VEX analysis package..."
cd packages/vex-analysis
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
cd ../..

# Setup Node.js frontend
echo "⚛️ Setting up React frontend..."
cd apps/frontend
npm install
cd ../..

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review .env file and update configuration"
echo "2. Run 'npm run dev' to start development servers"
echo "3. Open http://localhost:3000 for frontend"
echo "4. Backend API available at http://localhost:8000"