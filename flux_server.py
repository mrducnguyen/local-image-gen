from flask import Flask, request, jsonify, send_file
from diffusers import FluxPipeline
import torch
from PIL import Image
import io
import base64
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variable to store the pipeline
pipeline = None

def load_model():
    """Load the Flux.1.schnell model - optimized for AMD ROCm GPUs
    
    This implementation loads the model to CPU first, then uses sequential CPU
    offload to move components to GPU as needed during inference. This approach
    is more reliable for ROCm/AMD GPUs and memory-constrained systems.
    
    IMPORTANT: Memory requirements for FLUX.1-schnell:
    - Minimum: 32GB RAM (will use swap, slower)
    - Recommended: 48GB RAM (typical usage observed)
    - Optimal: 64GB+ RAM (smooth loading without swapping)
    
    The model is ~24GB on disk but requires 2-2.5x during loading due to:
    - T5-XXL encoder (~10-15GB)
    - CLIP encoder, VAE, and transformer components
    - PyTorch loading overhead
    """
    global pipeline
    print("Loading Flux.1.schnell model...")
    print("Note: Requires 48GB+ RAM for model loading (64GB recommended)")
    
    # Get token from environment variable (.env file)
    hf_token = os.getenv('HF_TOKEN')
    
    # Load the pipeline on CPU first to avoid GPU issues
    print("Loading pipeline components...")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,  # Better than float16 for Flux and ROCm
        token=hf_token  # Use token if provided
    )
    
    print("Pipeline loaded successfully on CPU")
    
    # Enable sequential CPU offload for memory efficiency
    # This will automatically move components to GPU as needed  
    try:
        pipeline.enable_sequential_cpu_offload()
        print("Sequential CPU offload enabled - components will move to GPU as needed")
    except Exception as e:
        print(f"Could not enable sequential offload: {e}")
        # Fallback: try to move the entire pipeline to GPU
        try:
            print("Attempting to move pipeline to GPU...")
            pipeline = pipeline.to("cuda")
            print("Pipeline moved to GPU successfully!")
        except Exception as e:
            print(f"Error moving to GPU: {e}")
            print("Will use CPU inference")
    
    # PyTorch's native SDPA is automatically used and works well on WSL/ROCm
    print("Using PyTorch native scaled dot product attention")
    print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": pipeline is not None})

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from text prompt"""
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Optional parameters - default to Full HD (1920x1080)
        width = data.get('width', 1920)
        height = data.get('height', 1080)
        num_inference_steps = data.get('steps', 4)  # Schnell is optimized for 4 steps
        guidance_scale = data.get('guidance_scale', 0.0)  # Schnell doesn't use guidance
        seed = data.get('seed', None)
        
        print(f"Generating image for prompt: {prompt}")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=512
                ).images[0]
        
        # Convert image to base64 for JSON response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Optionally save image locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_output_{timestamp}.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        return jsonify({
            "success": True,
            "image": img_base64,
            "filename": filename,
            "prompt": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "seed": seed
            }
        })
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_file', methods=['POST'])
def generate_image_file():
    """Generate image and return as file"""
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Parameters - default to Full HD (1920x1080)
        width = data.get('width', 1920)
        height = data.get('height', 1080)
        num_inference_steps = data.get('steps', 4)
        seed = data.get('seed', None)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,
                    max_sequence_length=512
                ).images[0]
        
        # Return image as file
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'flux_generated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flux.1.schnell server...")
    
    # Load model on startup
    load_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)