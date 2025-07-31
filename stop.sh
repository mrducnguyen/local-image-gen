#!/bin/bash

# Flux.1.schnell Server Stop Script

echo "Stopping Flux server..."

# Check if PID file exists
if [ -f flux_server.pid ]; then
    PID=$(cat flux_server.pid)
    
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "Sent stop signal to server (PID: $PID)"
        
        # Wait for process to stop (max 30 seconds)
        for i in {1..30}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "Server stopped successfully!"
                rm -f flux_server.pid
                exit 0
            fi
            echo -n "."
            sleep 1
        done
        
        # Force kill if still running
        echo ""
        echo "Server didn't stop gracefully. Force killing..."
        kill -9 $PID 2>/dev/null
        rm -f flux_server.pid
        echo "Server force stopped."
    else
        echo "Server process (PID: $PID) not found. Cleaning up PID file."
        rm -f flux_server.pid
    fi
else
    # Try to find and kill by process name
    PIDS=$(pgrep -f "flux_server.py")
    if [ -n "$PIDS" ]; then
        echo "Found server process(es): $PIDS"
        kill $PIDS
        echo "Sent stop signal to process(es)"
        
        # Wait a bit
        sleep 3
        
        # Check if any still running
        REMAINING=$(pgrep -f "flux_server.py")
        if [ -n "$REMAINING" ]; then
            echo "Force killing remaining processes: $REMAINING"
            kill -9 $REMAINING 2>/dev/null
        fi
        echo "Server stopped."
    else
        echo "No Flux server process found."
    fi
fi

echo "Done."