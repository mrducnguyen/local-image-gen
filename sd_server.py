from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusion3Pipeline
from transformers import SiglipVisionModel, SiglipImageProcessor
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

# Global variables to store the pipeline and image encoder
pipeline = None
image_encoder = None
image_processor = None

def load_model():
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
    
    print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": pipeline is not None
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
        
        # Optional parameters
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        num_inference_steps = data.get('steps', 24)
        guidance_scale = data.get('guidance_scale', 5.0)
        seed = data.get('seed', None)
        negative_prompt = data.get('negative_prompt', 'lowres, low quality, worst quality')
        
        # IP-Adapter parameters
        reference_image_base64 = data.get('reference_image', None)
        ipadapter_scale = data.get('ipadapter_scale', 0.8)
        
        print(f"Generating image for prompt: {prompt}")
        if reference_image_base64:
            print(f"Using reference image with IP-Adapter scale: {ipadapter_scale}")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # Process reference image if provided
        if reference_image_base64 and image_encoder is not None:
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
        filename = f"sd35_output_{timestamp}.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        response_data = {
            "success": True,
            "image": img_base64,
            "filename": filename,
            "prompt": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "negative_prompt": negative_prompt
            }
        }
        
        if reference_image_base64:
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
        
        # Parameters
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        num_inference_steps = data.get('steps', 24)
        guidance_scale = data.get('guidance_scale', 5.0)
        seed = data.get('seed', None)
        negative_prompt = data.get('negative_prompt', 'lowres, low quality, worst quality')
        
        # IP-Adapter parameters
        reference_image_base64 = data.get('reference_image', None)
        ipadapter_scale = data.get('ipadapter_scale', 0.8)
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare generation kwargs
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # Process reference image if provided
        if reference_image_base64 and image_encoder is not None:
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
            download_name=f'sd35_generated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_with_reference', methods=['POST'])
def generate_with_reference():
    """Dedicated endpoint for image-to-image generation with reference"""
    try:
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
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 1024))
        num_inference_steps = int(request.form.get('steps', 24))
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
        filename = f"sd35_ref_{timestamp}.png"
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
    print("Starting SD3.5-Large server...")
    print("Port: 5001 (to run alongside FLUX server on 5000)")
    
    # Load model on startup
    load_model()
    
    # Start Flask server on different port
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)