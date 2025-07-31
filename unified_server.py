from flask import Flask, request, jsonify, send_file
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from transformers import SiglipVisionModel, SiglipImageProcessor
import torch
from PIL import Image
import io
import base64
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variables
pipeline = None
image_encoder = None
image_processor = None
model_type = None  # 'flux' or 'sd3'

# Global default parameters
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 576
DEFAULT_STEPS = 7

def load_flux_model():
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
        print("Will use CPU inference")
    
    # PyTorch's native SDPA is automatically used and works well on WSL/ROCm
    print("Using PyTorch native scaled dot product attention")
    print("FLUX model loaded successfully!")

def load_sd3_model():
    """Load the SD3.5-Large model - optimized for AMD ROCm GPUs
    
    This implementation loads the model to CPU first, then uses sequential CPU
    offload to move components to GPU as needed during inference. This approach
    is more reliable for ROCm/AMD GPUs and memory-constrained systems.
    
    Based on FLUX experience: Direct .to("cuda") doesn't work on ROCm/WSL.
    """
    global pipeline, image_encoder, image_processor
    print("Loading SD3.5-Large model...")
    print("Using CPU loading with sequential offload (proven approach)")
    
    # Get token from environment variable (.env file)
    hf_token = os.getenv('HF_TOKEN')
    
    # Load the pipeline on CPU first to avoid GPU issues
    print("Loading pipeline components...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,  # Better than float16 for ROCm
        token=hf_token
    )
    
    print("Pipeline loaded successfully on CPU")
    
    # Enable sequential CPU offload for memory efficiency
    # This will automatically move components to GPU as needed  
    try:
        pipeline.enable_sequential_cpu_offload()
        print("✅ Sequential CPU offload enabled - components will move to GPU as needed")
    except Exception as e:
        print(f"❌ Could not enable sequential offload: {e}")
        print("Will use CPU inference")
    
    print("SD3.5 model loaded successfully!")

def load_model():
    """Load the specified model based on environment variable or default to FLUX"""
    global model_type
    
    # Get model type from environment variable or command line argument
    model_type = os.getenv('MODEL_TYPE', 'flux').lower()
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    
    # Validate model type
    if model_type not in ['flux', 'sd3']:
        print(f"❌ Invalid model type: {model_type}")
        print("   Valid options: flux, sd3")
        print("   Defaulting to FLUX")
        model_type = 'flux'
    
    print(f"🚀 Selected model: {model_type.upper()}")
    
    if model_type == 'flux':
        load_flux_model()
    else:
        load_sd3_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": pipeline is not None,
        "model_type": model_type
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from text prompt with optional image prompt"""
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Model-specific defaults
        if model_type == 'flux':
            # FLUX defaults
            width = data.get('width', DEFAULT_WIDTH)
            height = data.get('height', DEFAULT_HEIGHT)
            num_inference_steps = data.get('steps', DEFAULT_STEPS)
            guidance_scale = data.get('guidance_scale', 0.0)  # Schnell doesn't use guidance
            negative_prompt = None  # FLUX doesn't use negative prompts
        else:
            # SD3.5 defaults
            width = data.get('width', DEFAULT_WIDTH)
            height = data.get('height', DEFAULT_HEIGHT)
            num_inference_steps = data.get('steps', DEFAULT_STEPS)
            guidance_scale = data.get('guidance_scale', 5.0)
            negative_prompt = data.get('negative_prompt', 'lowres, low quality, worst quality')
        
        seed = data.get('seed', None)
        
        # IP-Adapter parameters (SD3 only)
        reference_image_base64 = data.get('reference_image', None)
        ipadapter_scale = data.get('ipadapter_scale', 0.8)
        
        print(f"Generating image for prompt: {prompt}")
        print(f"Model: {model_type.upper()}, Size: {width}x{height}, Steps: {num_inference_steps}")
        
        if reference_image_base64 and model_type == 'sd3':
            print(f"Using reference image with IP-Adapter scale: {ipadapter_scale}")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            if model_type == 'flux':
                torch.manual_seed(seed)
            else:
                generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add model-specific parameters
        if model_type == 'flux':
            generation_kwargs["max_sequence_length"] = 512
        else:
            generation_kwargs["negative_prompt"] = negative_prompt
            generation_kwargs["generator"] = generator
        
        # Process reference image if provided (SD3 only)
        if reference_image_base64 and model_type == 'sd3' and image_encoder is not None:
            try:
                # Decode base64 image
                img_data = base64.b64decode(reference_image_base64)
                reference_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Process image for CLIP
                clip_image = image_processor(images=reference_image, return_tensors="pt").pixel_values
                clip_image = clip_image.to(torch.float16).to(image_encoder.device)
                
                # Extract image features
                clip_image_embeds = image_encoder(pixel_values=clip_image, output_hidden_states=True).hidden_states[-2]
                
                # Add IP-Adapter parameters
                generation_kwargs["clip_image"] = clip_image_embeds
                generation_kwargs["ipadapter_scale"] = ipadapter_scale
                
            except Exception as e:
                print(f"Error processing reference image: {e}")
                # Continue without reference image
        
        # Generate image
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipeline(**generation_kwargs).images[0]
        
        # Convert image to base64 for JSON response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Save image locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_output_{timestamp}.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        response_data = {
            "success": True,
            "image": img_base64,
            "filename": filename,
            "prompt": prompt,
            "model": model_type,
            "parameters": {
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
        
        if model_type == 'sd3' and negative_prompt:
            response_data["parameters"]["negative_prompt"] = negative_prompt
        
        if reference_image_base64 and model_type == 'sd3':
            response_data["parameters"]["ipadapter_scale"] = ipadapter_scale
            response_data["parameters"]["used_reference_image"] = True
        
        return jsonify(response_data)
        
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
        
        # Model-specific defaults
        if model_type == 'flux':
            width = data.get('width', DEFAULT_WIDTH)
            height = data.get('height', DEFAULT_HEIGHT)
            num_inference_steps = data.get('steps', DEFAULT_STEPS)
            guidance_scale = data.get('guidance_scale', 0.0)
            negative_prompt = None
        else:
            width = data.get('width', DEFAULT_WIDTH)
            height = data.get('height', DEFAULT_HEIGHT)
            num_inference_steps = data.get('steps', DEFAULT_STEPS)
            guidance_scale = data.get('guidance_scale', 5.0)
            negative_prompt = data.get('negative_prompt', 'lowres, low quality, worst quality')
        
        seed = data.get('seed', None)
        
        # IP-Adapter parameters (SD3 only)
        reference_image_base64 = data.get('reference_image', None)
        ipadapter_scale = data.get('ipadapter_scale', 0.8)
        
        # Set seed
        generator = None
        if seed is not None:
            if model_type == 'flux':
                torch.manual_seed(seed)
            else:
                generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add model-specific parameters
        if model_type == 'flux':
            generation_kwargs["max_sequence_length"] = 512
        else:
            generation_kwargs["negative_prompt"] = negative_prompt
            generation_kwargs["generator"] = generator
        
        # Process reference image if provided (SD3 only)
        if reference_image_base64 and model_type == 'sd3' and image_encoder is not None:
            try:
                img_data = base64.b64decode(reference_image_base64)
                reference_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                clip_image = image_processor(images=reference_image, return_tensors="pt").pixel_values
                clip_image = clip_image.to(torch.float16).to(image_encoder.device)
                
                clip_image_embeds = image_encoder(pixel_values=clip_image, output_hidden_states=True).hidden_states[-2]
                
                generation_kwargs["clip_image"] = clip_image_embeds
                generation_kwargs["ipadapter_scale"] = ipadapter_scale
                
            except Exception as e:
                print(f"Error processing reference image: {e}")
        
        # Generate image
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipeline(**generation_kwargs).images[0]
        
        # Return image as file
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{model_type}_generated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_with_reference', methods=['POST'])
def generate_with_reference():
    """Dedicated endpoint for image-to-image generation with reference (SD3 only)"""
    try:
        if model_type != 'sd3':
            return jsonify({"error": "Reference image generation is only supported for SD3 model"}), 400
            
        if pipeline is None or image_encoder is None:
            return jsonify({"error": "Model or image encoder not loaded"}), 500
        
        # Check if request has files
        if 'reference_image' not in request.files:
            return jsonify({"error": "No reference image provided"}), 400
        
        reference_file = request.files['reference_image']
        prompt = request.form.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Parameters from form data
        width = int(request.form.get('width', DEFAULT_WIDTH))
        height = int(request.form.get('height', DEFAULT_HEIGHT))
        num_inference_steps = int(request.form.get('steps', DEFAULT_STEPS))
        guidance_scale = float(request.form.get('guidance_scale', 5.0))
        ipadapter_scale = float(request.form.get('ipadapter_scale', 0.8))
        seed = request.form.get('seed', None)
        if seed:
            seed = int(seed)
        negative_prompt = request.form.get('negative_prompt', 'lowres, low quality, worst quality')
        
        # Load reference image
        reference_image = Image.open(reference_file.stream).convert("RGB")
        
        print(f"Generating with reference image, prompt: {prompt}")
        
        # Process reference image
        clip_image = image_processor(images=reference_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(torch.float16).to(image_encoder.device)
        
        # Extract image features
        clip_image_embeds = image_encoder(pixel_values=clip_image, output_hidden_states=True).hidden_states[-2]
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate image
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    clip_image=clip_image_embeds,
                    ipadapter_scale=ipadapter_scale,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
        
        # Convert to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Save locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd3_ref_{timestamp}.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        return jsonify({
            "success": True,
            "image": img_base64,
            "filename": filename,
            "prompt": prompt,
            "model": "sd3",
            "parameters": {
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "ipadapter_scale": ipadapter_scale,
                "seed": seed,
                "negative_prompt": negative_prompt,
                "used_reference_image": True
            }
        })
        
    except Exception as e:
        print(f"Error generating with reference: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎨 Unified Image Generation Server")
    print("=" * 60)
    
    # Show usage
    print("\nUsage:")
    print("  python unified_server.py [model_type]")
    print("  MODEL_TYPE=sd3 python unified_server.py")
    print("\nModel types:")
    print("  flux - FLUX.1-schnell (default)")
    print("  sd3  - Stable Diffusion 3.5 Large")
    print("=" * 60)
    
    # Load model on startup
    load_model()
    
    # Start Flask server
    print(f"\n✅ Starting server with {model_type.upper()} model...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)