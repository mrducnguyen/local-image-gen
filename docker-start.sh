#!/bin/bash

# Docker Flux Service Start Script

echo "Starting Flux service with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if service is already running
if docker-compose ps | grep -q "Up"; then
    echo "Flux service is already running!"
    echo "Use ./docker-stop.sh to stop it first."
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs
mkdir -p models

# Build and start the service
echo "Building Docker image..."
docker-compose build

echo "Starting Flux service container..."
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
        echo "* =============================== *"
        echo "✅ Flux service is ready!"
        echo ""
        echo "🚀 Service URLs:"
        echo "   Health check: http://localhost:5000/health"
        echo "   Generate image: POST http://localhost:5000/generate"
        echo "   Generate file: POST http://localhost:5000/generate_file"
        echo ""
        echo "📁 Generated images will be saved to: ./outputs/"
        echo "📊 View container logs: docker-compose logs -f"
        echo "🛑 Stop service: ./docker-stop.sh"
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