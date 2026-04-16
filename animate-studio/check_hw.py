import torch
import psutil

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    print("GPU: N/A")

mem = psutil.virtual_memory()
print(f"RAM Total: {mem.total / 1e9:.1f} GB")
print(f"RAM Available: {mem.available / 1e9:.1f} GB")
print(f"RAM Used: {mem.percent}%")
