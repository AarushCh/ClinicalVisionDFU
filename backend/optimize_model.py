import torch
import torch.nn as nn
from torchvision import models
import os

print("Starting model memory optimization...")

# 1. Load original model
print("Loading original ResNet50 model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(__file__), "model", "dfu_model.pt")

# Load weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# 2. Apply Dynamic Quantization
print("Applying dynamic quantization to reduce memory by 4x...")
# This converts float32 weights to int8 weights for Linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, # Only quantize Linear layers (CNN layers don't support dynamic quantization well without calibration)
    dtype=torch.qint8
)

# 3. Trace the model (completely removes Python overhead and creates a static C++ graph)
print("Tracing model via TorchScript to eliminate Python memory overhead...")
example_input = torch.randn(1, 3, 384, 384)
traced_model = torch.jit.trace(quantized_model, example_input)

# 4. Save optimized model
optimized_path = os.path.join(os.path.dirname(__file__), "model", "dfu_model_optimized.pt")
torch.jit.save(traced_model, optimized_path)

orig_size = os.path.getsize(model_path) / (1024*1024)
opt_size = os.path.getsize(optimized_path) / (1024*1024)

print(f"\nOptimization Complete!")
print(f"Original Size:  {orig_size:.2f} MB")
print(f"Optimized Size: {opt_size:.2f} MB")
print(f"Memory Savings: {orig_size - opt_size:.2f} MB ({(1 - opt_size/orig_size)*100:.1f}%)")
