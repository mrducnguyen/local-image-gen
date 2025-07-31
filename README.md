# FLUX.1.schnell Docker Service

A containerized Flask API service for generating images using the FLUX.1.schnell model with AMD GPU acceleration via ROCm on WSL2.

## ⚠️ Important: WSL2 Only

This service **requires WSL2** and cannot run on native Windows Docker Desktop due to lack of AMD GPU support. Choose your installation method below.

## Installation Methods

### Option 1: Docker (Recommended)

**Minimal Setup** - Docker handles all dependencies inside the container.

#### Prerequisites for Docker:
1. **Install AMD Radeon Software for Windows** (latest version)
2. **Enable WSL2** with Ubuntu 22.04 or 24.04
3. **Install Docker Desktop** with WSL2 backend enabled

That's it! The Docker container includes all ROCm, PyTorch, and Python dependencies.

#### Quick Start:
```bash
git clone <repository>
cd flux-service
./docker-start.sh
```

### Option 2: Native WSL2 Installation

**Full Control** - Install everything directly in WSL2 for development or customization.

#### Prerequisites for Native WSL2:

##### 1. WSL2 ROCm Installation

**Essential:** Follow the official AMD ROCm WSL2 installation guide:  
📋 https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html

**Key Steps:**
1. **Install AMD Radeon Software for Windows** (latest version)
2. **Enable WSL2** with Ubuntu 22.04 or 24.04
3. **Install ROCm packages in WSL2:**
   ```bash
   # Add ROCm repository
   wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/jammy/amdgpu-install_6.4.50402-1_all.deb
   sudo dpkg -i amdgpu-install_6.4.50402-1_all.deb
   sudo apt update
   
   # Install ROCm
   sudo amdgpu-install --usecase=wsl,rocm --no-dkms

   # Verify installation
   rocminfo
   ```
4. **Add user to render group:**
   ```bash
   sudo usermod -a -G render,video $USER
   ```
5. **Reboot WSL2** and verify GPU detection:
   ```bash
   wsl --shutdown
   # Restart WSL2
   rocm-smi
   ```

##### 2. Python Version Management

**Important:** This service requires **Python 3.12.4** for compatibility with the ROCm PyTorch build.

**Using pyenv (Recommended):**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install and set Python 3.12.4
pyenv install 3.12.4
pyenv global 3.12.4  # or pyenv local 3.12.4 for project-specific

# Verify version
python --version  # Should show Python 3.12.4
```

##### 3. PyTorch ROCm Installation

**Essential:** Follow the PyTorch ROCm installation guide:  
📋 https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html

**Key Steps:**
1. **Ensure you're using Python 3.12.4:**
   ```bash
   python --version  # Must show Python 3.12.4
   ```
2. **Install PyTorch with ROCm support:**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```
3. **Verify PyTorch can detect your GPU:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

##### 4. Install Project Dependencies

```bash
git clone <repository>
cd flux-service
pip install -r requirements.txt
```

#### Quick Start (Native WSL2):
```bash
./start.sh  # For native installation
```

## FLUX.1.schnell Memory Requirements

### Model Components and Memory Breakdown

FLUX.1-schnell consists of several large components that contribute to its memory footprint:

1. **DiT (Diffusion Transformer)**: ~12 billion parameters
2. **T5-XXL Text Encoder**: ~4.5 billion parameters (largest memory consumer)
3. **CLIP Text Encoder**: Additional text understanding
4. **VAE (Variational Autoencoder)**: For latent space operations

### Total Memory Requirements

**System RAM Requirements:**
- **Minimum**: 32GB RAM (with potential swapping)
- **Recommended**: 48GB RAM (observed typical usage)
- **Optimal**: 64GB RAM (for smooth loading without swapping)

**Why 48-64GB RAM?**
- Model size on disk: ~24GB
- Loading overhead: PyTorch requires 2-2.5x the model size during loading
- Components loaded: T5 encoder, CLIP encoder, VAE, and the main transformer
- The T5-XXL encoder alone can consume 10-15GB during loading

**GPU VRAM Requirements:**
- **After loading**: 12-16GB VRAM for inference
- **During generation**: Additional 2-4GB depending on image resolution

### AMD ROCm-Specific Loading Strategy

For AMD GPUs on ROCm/WSL2, we use a two-step loading process:

1. **Load to CPU first**: More reliable than device_map options on ROCm
2. **Move to GPU**: Ensures all components are on GPU for fast generation

```python
# ROCm-optimized approach:
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16  # Better than float16 for ROCm
)
pipe = pipe.to("cuda")  # Move entire pipeline to GPU
```

**Note on device_map with ROCm:**
- `device_map="auto"` is not supported on ROCm
- `device_map="balanced"` causes partial CPU offloading, severely impacting performance
- Direct `.to("cuda")` after CPU loading is the most reliable method

## Usage Examples

**⚠️ Always run from WSL2 terminal!**

### For Docker Installation:

1. **Wait for model loading** (5-10 minutes on first run):
   ```bash
   # Monitor progress
   docker-compose logs -f
   ```

2. **Test the service:**
   ```bash
   curl http://localhost:5000/health
   ```

3. **Generate an image:**
   ```bash
   curl -X POST http://localhost:5000/generate_file \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "a cute golden retriever dog playing in a sunny park",
       "width": 512,
       "height": 512,
       "num_inference_steps": 20
     }' \
     --output generated_image.png
   ```

4. **Stop the service:**
   ```bash
   ./docker-stop.sh
   ```

### For Native WSL2 Installation:

Same API usage as above, but the service runs directly in your WSL2 environment with your installed dependencies.

## Expected Results

### Successful Setup Verification

**ROCm Installation Check:**
```bash
$ rocm-smi
======================= ROCm System Management Interface =======================
========================= Concise Info =========================
GPU[0]		: card0
	Name: AMD Radeon RX 7900 XTX
	Temperature: 35.0°C
	Power: 15.0W
	GPU Utilization: 0%
	Memory Utilization: 0%
```

**Python Version Check:**
```bash
$ python --version
Python 3.12.4
```

**PyTorch GPU Detection:**
```bash
$ python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
True AMD Radeon RX 7900 XTX
```

### Service Startup Logs

**Expected docker-compose logs output:**
```
flux-service-1  | Starting Flux.1.schnell server...
flux-service-1  | Loading Flux.1.schnell model...
flux-service-1  | Fetching 23 files: 100%|██████████| 23/23 [07:11<00:00, 18.75s/it]
flux-service-1  | Loading pipeline components: 100%|██████████| 7/7 [00:07<00:00,  1.06s/it]
flux-service-1  | Pipeline loaded successfully on CPU
flux-service-1  | Using sequential offload for GPU acceleration
flux-service-1  | Model loaded successfully!
flux-service-1  |  * Running on http://127.0.0.1:5000
```

### Health Check Response

**Successful health endpoint:**
```bash
$ curl http://localhost:5000/health
{"model_loaded":true,"status":"healthy"}
```

### Image Generation Response

**Successful generation (base64 endpoint):**
```bash
$ curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompt": "a cute dog", "width": 512, "height": 512}'
{"image_base64":"iVBORw0KGgoAAAANSUhEUgAAA...","generation_time":18.42}
```

**Successful generation (file endpoint):**
```bash
$ curl -X POST http://localhost:5000/generate_file -H "Content-Type: application/json" -d '{"prompt": "a cute dog", "width": 512, "height": 512}' --output dog.png
# Creates dog.png file in current directory
```

### Performance Expectations

**Typical generation times on AMD RX 7900 XTX:**
- **512x512, 20 steps**: ~18-25 seconds
- **1024x1024, 20 steps**: ~45-60 seconds  
- **First run**: Additional 5-10 minutes for model download

**Memory usage:**
- **System RAM**: ~8-12GB during generation
- **GPU VRAM**: ~6-8GB for 512x512, ~10-12GB for 1024x1024

### Using Local Installation

1. **Start the service:**
   ```bash
   ./start.sh
   ```

2. **Stop the service:**
   ```bash
   ./stop.sh
   ```

## API Endpoints

### Health Check
- **GET** `/health`
- Returns service status and model loading state

### Generate Image (Base64)
- **POST** `/generate`
- **Request Body:**
  ```json
  {
    "prompt": "Your image description",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 42
  }
  ```
- **Response:** JSON with base64 encoded image

### Generate Image (File)
- **POST** `/generate_file`
- **Request Body:** Same as above
- **Response:** PNG file download

## Configuration

### Environment Variables

- `HSA_ENABLE_SDMA=0` - Disables SDMA for WSL compatibility
- `PYTORCH_HIP_ALLOC_CONF` - Memory allocation settings
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` - Enables experimental features
- `HF_TOKEN` - Hugging Face token (optional, in .env file)

### Docker Volumes

- `./outputs:/app/outputs` - Generated images storage
- `./models:/app/models` - Model cache (optional)

## Performance

Typical generation times on AMD Radeon RX 9070 XT:
- **512x512**: ~18 seconds (4 steps)
- **1024x1024**: ~43 seconds (4 steps)

## Hardware Requirements

- **GPU:** AMD Radeon RX 6000 series or newer (16GB+ VRAM recommended)
  - Minimum: 12GB VRAM (will work but may be tight)
  - Recommended: 16GB+ VRAM for comfortable operation
- **RAM:** 48GB+ system memory
  - Minimum: 32GB (will use swap, slower loading)
  - Recommended: 48GB (typical usage during model loading)
  - Optimal: 64GB+ (smooth operation without swapping)
- **Storage:** 30GB+ free space
  - Model download: ~24GB
  - Additional space for outputs and cache
- **OS:** Windows 11 with WSL2 enabled
- **Supported GPU architectures:** RDNA2, RDNA3, CDNA2, CDNA3

## Container Configuration

The docker-compose.yml includes essential WSL2 configurations:

- **Device mapping:** `/dev/dxg` for DirectX GPU access
- **Library mounts:** WSL DirectX and ROCm runtime libraries
- **Environment variables:** ROCm optimization settings
- **Resource limits:** 8GB shared memory for large model loading

## Troubleshooting

### "No HIP GPUs are available"
1. **Verify ROCm installation:**
   ```bash
   rocm-smi
   ```
2. **Check PyTorch GPU detection:**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```
3. **Ensure running in WSL2**, not native Windows

### Container startup takes 5+ minutes
- **This is normal** - FLUX.1.schnell downloads ~23 model files on first run
- Subsequent starts are faster as models are cached
- Monitor progress: `docker-compose logs -f`

### Memory issues
- **Insufficient RAM error during loading:**
  - FLUX.1-schnell requires 48GB+ RAM for comfortable loading
  - With 32GB RAM, enable swap space (at least 32GB)
  - Monitor memory usage during loading: `free -h`
- **GPU out of memory during generation:**
  - Requires 12-16GB VRAM after model is loaded
  - Use smaller image sizes (512x512 instead of 1024x1024)
  - Enable CPU offloading if GPU memory is limited

### Sequential offload errors
- Always load pipeline on CPU first
- Use `enable_sequential_cpu_offload()` instead of `.to('cuda')`
- Never mix sequential offload with manual device placement

### Check Container Logs
```bash
docker-compose logs -f
```

### Check Container Status
```bash
docker-compose ps
```

### Rebuild Container
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Development

### Project Structure
```
flux-service/
├── flux_server.py          # Main Flask application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Docker Compose configuration
├── start.sh / stop.sh     # Local development scripts
├── docker-start.sh        # Docker deployment scripts
├── docker-stop.sh
└── outputs/               # Generated images directory
```

### Building Custom Image
```bash
docker build -t flux-service:custom .
```

## License

This project uses the Flux.1-schnell model which has its own licensing terms. Please review the model's license before commercial use.