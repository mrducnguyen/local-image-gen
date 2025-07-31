FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY unified_server.py .
COPY .env* ./

# Create outputs directory
RUN mkdir -p outputs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default to FLUX model (can be overridden with MODEL_TYPE env var)
ENV MODEL_TYPE=flux

# Run the application
CMD ["python", "unified_server.py"]