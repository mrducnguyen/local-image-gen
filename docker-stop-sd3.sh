#!/bin/bash

# Docker SD3 Service Stop Script

echo "Stopping SD3.5-Large-IP-Adapter service..."

# Stop and remove containers
echo "Stopping containers..."
docker-compose -f docker-compose.sd3.yml down

echo "✅ SD3 service stopped successfully!"
echo "Done."