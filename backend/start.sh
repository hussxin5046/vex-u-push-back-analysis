#!/bin/bash

# VEX U Scoring Analysis Platform - Backend Startup Script

echo "🚀 Starting VEX U Analysis API Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env .env.example 2>/dev/null || echo "# Copy and configure .env file"
fi

# Create upload directory
mkdir -p uploads

# Start the server
echo "🌟 Starting Flask server..."
export FLASK_APP=app.py
export FLASK_ENV=development

python app.py