import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import requests
import base64
import json
from PIL import Image, ImageTk
import io
import threading

class FluxGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FLUX Image Generator")
        self.root.geometry("800x900")
        
        self.api_url = "http://localhost:5001"
        self.current_image = None
        
        self.setup_ui()
        self.check_service()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="FLUX Image Generator", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Checking service...", foreground="orange")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Prompt input
        ttk.Label(main_frame, text="Prompt:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.prompt_var = tk.StringVar()
        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var, width=60)
        prompt_entry.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Steps
        ttk.Label(settings_frame, text="Steps:").grid(row=0, column=0, sticky=tk.W)
        self.steps_var = tk.IntVar(value=4)
        steps_spin = ttk.Spinbox(settings_frame, from_=1, to=20, textvariable=self.steps_var, width=10)
        steps_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Size
        ttk.Label(settings_frame, text="Size:").grid(row=0, column=2, sticky=tk.W)
        self.size_var = tk.StringVar(value="1024x1024")
        size_combo = ttk.Combobox(settings_frame, textvariable=self.size_var, 
                                  values=["512x512", "768x768", "1024x1024"], width=10)
        size_combo.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="🎨 Generate Image", 
                                       command=self.generate_image, state="disabled")
        self.generate_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image display
        self.image_frame = ttk.LabelFrame(main_frame, text="Generated Image", padding="5")
        self.image_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame, text="No image generated yet")
        self.image_label.grid(row=0, column=0)
        
        # Save button
        self.save_btn = ttk.Button(main_frame, text="💾 Save Image", 
                                   command=self.save_image, state="disabled")
        self.save_btn.grid(row=8, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
    
    def check_service(self):
        """Check if the API service is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['model_loaded']:
                    self.status_label.config(text="✅ Service ready", foreground="green")
                    self.generate_btn.config(state="normal")
                else:
                    self.status_label.config(text="⏳ Model loading...", foreground="orange")
                    self.root.after(3000, self.check_service)  # Check again in 3s
            else:
                self.status_label.config(text="❌ Service error", foreground="red")
        except:
            self.status_label.config(text="❌ Service not running", foreground="red")
    
    def generate_image(self):
        """Generate image in background thread"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt!")
            return
        
        # Disable button and start progress
        self.generate_btn.config(state="disabled")
        self.progress.start()
        
        # Run in background thread
        thread = threading.Thread(target=self._generate_worker, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def _generate_worker(self, prompt):
        """Background worker for generation"""
        try:
            # Parse settings
            steps = self.steps_var.get()
            size = self.size_var.get()
            width, height = map(int, size.split('x'))
            
            data = {
                "prompt": prompt,
                "steps": steps,
                "width": width,
                "height": height
            }
            
            response = requests.post(f"{self.api_url}/generate", json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                # Decode image
                img_data = base64.b64decode(result['image'])
                image = Image.open(io.BytesIO(img_data))
                self.current_image = image
                
                # Update UI in main thread
                self.root.after(0, self._update_image, image, result.get('filename', 'generated'))
            else:
                error = response.json().get('error', 'Unknown error')
                self.root.after(0, self._show_error, f"Generation failed: {error}")
                
        except Exception as e:
            self.root.after(0, self._show_error, f"Error: {str(e)}")
    
    def _update_image(self, image, filename):
        """Update UI with generated image"""
        # Resize for display
        display_image = image.copy()
        display_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter format
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep reference
        
        # Update UI
        self.progress.stop()
        self.generate_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.status_label.config(text=f"✅ Generated: {filename}", foreground="green")
    
    def _show_error(self, error_msg):
        """Show error and reset UI"""
        self.progress.stop()
        self.generate_btn.config(state="normal")
        messagebox.showerror("Error", error_msg)
    
    def save_image(self):
        """Save the current image"""
        if self.current_image:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if filename:
                self.current_image.save(filename)
                messagebox.showinfo("Success", f"Image saved: {filename}")

def main():
    root = tk.Tk()
    app = FluxGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()