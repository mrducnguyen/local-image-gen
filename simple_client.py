import requests
import base64
import json
import sys
from PIL import Image
import io
import time

API_URL = "http://localhost:5000"

def check_service():
    """Check if the service is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is healthy")
            print(f"   Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Service returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to service: {e}")
        return False

def generate_image(prompt, steps=4, width=1024, height=1024, seed=None):
    """Generate image from prompt"""
    data = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "seed": seed
    }
    
    print(f"📝 Sending prompt: '{prompt}'")
    print(f"⚙️  Settings: {steps} steps, {width}x{height}")
    
    try:
        response = requests.post(f"{API_URL}/generate", json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Decode base64 image
            img_data = base64.b64decode(result['image'])
            image = Image.open(io.BytesIO(img_data))
            
            # Save image
            filename = f"generated_{int(time.time())}.png"
            image.save(filename)
            
            print(f"✅ Image saved: {filename}")
            print(f"📁 Saved as: {result['filename']}")
            
            return filename
        else:
            error = response.json().get('error', 'Unknown error')
            print(f"❌ Error: {error}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def generate_image_file(prompt, steps=4, width=1024, height=1024, seed=None):
    """Generate image and download as file"""
    data = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "seed": seed
    }
    
    print(f"📝 Generating file for: '{prompt}'")
    
    try:
        response = requests.post(f"{API_URL}/generate_file", json=data, timeout=60)
        
        if response.status_code == 200:
            filename = f"flux_{prompt[:20].replace(' ', '_')}_{int(time.time())}.png"
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Image downloaded: {filename}")
            return filename
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    print("🎨 FLUX Image Generator Client")
    print("=" * 40)
    
    # Check service first
    if not check_service():
        print("\n💡 Start the Docker service first:")
        print("   docker run -p 5000:5000 --device=/dev/kfd --device=/dev/dri ...")
        return
    
    if len(sys.argv) > 1:
        # Command line mode
        prompt = " ".join(sys.argv[1:])
        generate_image(prompt)
    else:
        # Interactive mode
        print("\n🚀 Interactive mode (type 'quit' to exit)")
        while True:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if prompt:
                generate_image(prompt)
            else:
                print("Please enter a prompt!")
    
    print("\n👋 Happy generating!")

if __name__ == "__main__":
    main()