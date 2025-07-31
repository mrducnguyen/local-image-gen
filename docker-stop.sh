#!/bin/bash

# Docker Image Generation Service Stop Script

echo "Stopping Image Generation service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running."
    exit 1
fi

# Stop the service
if docker-compose ps | grep -q "Up"; then
    echo "Stopping containers..."
    docker-compose down
    echo "✅ Image service stopped successfully!"
else
    echo "ℹ️  No running image service containers found."
fi

# Optional: Clean up images (uncomment if you want to clean up)
# echo "Cleaning up Docker images..."
# docker-compose down --rmi all --volumes --remove-orphans

echo "Done."