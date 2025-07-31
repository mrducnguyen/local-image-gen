#!/bin/bash

# Flux.1.schnell Server Startup Script

# Check if server is already running
if pgrep -f "flux_server.py" > /dev/null; then
    echo "Flux server is already running!"
    echo "Use ./stop.sh to stop it first."
    exit 1
fi

echo "Starting Flux.1.schnell server..."

# Set environment variables for ROCm optimization
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HSA_ENABLE_SDMA=0

# Start the server in background
nohup python flux_server.py > flux_server.log 2>&1 &

# Get the PID
PID=$!
echo $PID > flux_server.pid

echo "Server starting with PID: $PID"
echo "Logs are being written to flux_server.log"

# Wait a bit and check if server started successfully
sleep 5

if ps -p $PID > /dev/null; then
    echo "Server started successfully!"
    echo "Waiting for model to load..."
    
    # Wait for server to be ready (max 2 minutes)
    for i in {1..24}; do
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            echo "Server is ready!"
            echo "Access the server at http://localhost:5000"
            echo ""
            echo "Endpoints:"
            echo "  - Health check: http://localhost:5000/health"
            echo "  - Generate image: POST http://localhost:5000/generate"
            echo "  - Generate file: POST http://localhost:5000/generate_file"
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    
    echo ""
    echo "Server started but health check failed. Check flux_server.log for details."
else
    echo "Failed to start server. Check flux_server.log for errors."
    exit 1
fi