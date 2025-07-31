#!/bin/bash

# Docker Image Generation Service Start Script

# Read MODEL_TYPE from .env file if it exists
if [ -f .env ]; then
    export $(grep -E '^MODEL_TYPE=' .env | xargs)
fi

# Use MODEL_TYPE from .env or default to flux
MODEL_TYPE=${MODEL_TYPE:-flux}

# Show message about model selection
if [ ! -f .env ] || ! grep -q '^MODEL_TYPE=' .env; then
    echo "ℹ️  No MODEL_TYPE found in .env file. Using default model: FLUX"
    echo "   To use SD3.5, add MODEL_TYPE=sd3 to your .env file"
    echo ""
fi

# Validate model type
if [[ "$MODEL_TYPE" != "flux" && "$MODEL_TYPE" != "sd3" ]]; then
    echo "⚠️  Invalid MODEL_TYPE in .env: $MODEL_TYPE"
    echo "   Valid options: flux, sd3"
    echo "   Defaulting to: FLUX"
    MODEL_TYPE="flux"
fi

echo "Starting Image Generation service with ${MODEL_TYPE^^} model..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if service is already running
if docker-compose ps | grep -q "Up"; then
    echo "Image service is already running!"
    echo "Use ./docker-stop.sh to stop it first."
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs
mkdir -p models

# Export MODEL_TYPE for docker-compose
export MODEL_TYPE

# Build and start the service
echo "Building Docker image..."
docker-compose build

echo "Starting image service container..."
docker-compose up -d

# Wait for service to be ready
echo "Waiting for service to be ready..."
echo "Showing container logs (will stop when service is ready):"
echo ""

# Start showing logs in the background
docker-compose logs -f &
LOGS_PID=$!

# Wait for service to be ready
for i in {1..60}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        # Kill the logs process
        kill $LOGS_PID 2>/dev/null
        
        # Get model info from health check
        HEALTH_INFO=$(curl -s http://localhost:5000/health | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model_type', 'unknown'))")
        
        echo "* =============================== *"
        echo "✅ Image service is ready!"
        echo "🎨 Model: ${HEALTH_INFO^^}"
        echo ""
        echo "🚀 Service URLs:"
        echo "   Health check: http://localhost:5000/health"
        echo "   Generate image: POST http://localhost:5000/generate"
        echo "   Generate file: POST http://localhost:5000/generate_file"
        
        if [[ "$MODEL_TYPE" == "sd3" ]]; then
            echo "   Generate with reference: POST http://localhost:5000/generate_with_reference"
        fi
        
        echo ""
        echo "📁 Generated images will be saved to: ./outputs/"
        echo "📊 View container logs: docker-compose logs -f"
        echo "🛑 Stop service: ./docker-stop.sh"
        echo ""
        echo "💡 To switch models:"
        echo "   1. Stop the service: ./docker-stop.sh"
        echo "   2. Set MODEL_TYPE=sd3 (or flux) in .env file"
        echo "   3. Start the service: ./docker-start.sh"
        exit 0
    fi
    sleep 5
done

# Kill the logs process if we timeout
kill $LOGS_PID 2>/dev/null

echo ""
echo "⚠️  Service started but health check failed."
echo "Check logs with: docker-compose logs"
echo "Container status: docker-compose ps"