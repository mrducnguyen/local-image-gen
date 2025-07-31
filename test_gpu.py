import torch
import subprocess
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Test GPU memory allocation
print("\nTesting GPU memory allocation...")
x = torch.randn(1000, 1000).cuda()
print(f"Allocated tensor on GPU: {x.device}")

# Monitor GPU usage
print("\nGPU Memory Usage:")
print(torch.cuda.memory_summary())