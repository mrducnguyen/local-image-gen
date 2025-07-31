#!/bin/bash

# Docker SD3 Service Start Script

echo "Starting SD3.5-Large-IP-Adapter service with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if service is already running
if docker ps | grep -q "sd3-service"; then
    echo "SD3 service is already running!"
    echo "Use ./docker-stop-sd3.sh to stop it first."
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs
mkdir -p models

# Build and start the service
echo "Building Docker image..."
docker-compose -f docker-compose.sd3.yml build

echo "Starting SD3 service container..."
docker-compose -f docker-compose.sd3.yml up -d

# Wait for service to be ready
echo "Waiting for service to be ready..."
echo "Showing container logs (will stop when service is ready):"
echo ""

# Start showing logs in the background
docker-compose -f docker-compose.sd3.yml logs -f &
LOGS_PID=$!

# Wait for service to be ready
for i in {1..60}; do
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        # Kill the logs process
        kill $LOGS_PID 2>/dev/null
        echo "* =============================== *"
        echo "✅ SD3.5-Large-IP-Adapter service is ready!"
        echo ""
        echo "🚀 Service URLs:"
        echo "   Health check: http://localhost:5001/health"
        echo "   Generate image: POST http://localhost:5001/generate"
        echo "   Generate file: POST http://localhost:5001/generate_file"
        echo "   Generate with reference: POST http://localhost:5001/generate_with_reference"
        echo ""
        echo "📁 Generated images will be saved to: ./outputs/"
        echo "📊 View container logs: docker-compose -f docker-compose.sd3.yml logs -f"
        echo "🛑 Stop service: ./docker-stop-sd3.sh"
        exit 0
    fi
    sleep 5
done

# Kill the logs process if we timeout
kill $LOGS_PID 2>/dev/null

echo ""
echo "⚠️  Service started but health check failed."
echo "Check logs with: docker-compose -f docker-compose.sd3.yml logs"
echo "Container status: docker-compose -f docker-compose.sd3.yml ps"