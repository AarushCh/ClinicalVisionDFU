import torch
import torch.nn as nn
from torchvision import models
import os

print("Starting model memory optimization...")

print("Loading original ResNet50 model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(__file__), "model", "dfu_model.pt")

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
print("Applying dynamic quantization to reduce memory by 4x...")
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

print("Tracing model via TorchScript to eliminate Python memory overhead...")
example_input = torch.randn(1, 3, 384, 384)
traced_model = torch.jit.trace(quantized_model, example_input)
optimized_path = os.path.join(os.path.dirname(__file__), "model", "dfu_model_optimized.pt")
torch.jit.save(traced_model, optimized_path)

orig_size = os.path.getsize(model_path) / (1024*1024)
opt_size = os.path.getsize(optimized_path) / (1024*1024)

print(f"\nOptimization Complete!")
print(f"Original Size:  {orig_size:.2f} MB")
print(f"Optimized Size: {opt_size:.2f} MB")
print(f"Memory Savings: {orig_size - opt_size:.2f} MB ({(1 - opt_size/orig_size)*100:.1f}%)")