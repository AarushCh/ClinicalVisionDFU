import cv2
import numpy as np
import torch
import torch.nn.functional as F
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activations(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(save_activations)
        self.target_layer.register_full_backward_hook(save_gradients)

    def generate(self, input_tensor):
        self.model.eval() # Ensure model is in eval mode
        output = self.model(input_tensor)
        
        # Index 0 is the Ulcer/Disease class. 
        ulcer_confidence = F.softmax(output, dim=1)[0, 0].item() 
        
        self.model.zero_grad()
        
        # THE FIX: ALWAYS force the AI to look for the Ulcer (Class 0).
        # We backward pass on the raw logit, not softmax.
        target = output[0, 0]
        target.backward()

        # Global average pooling on gradients
        gradients = self.gradients.detach().cpu().numpy()[0] # Shape: (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # Shape: (C, H, W)
        
        weights = np.mean(gradients, axis=(1, 2)) # Shape: (C,)
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        
        return cam, ulcer_confidence

def generate_gradcam_heatmap(model, img_pil, input_tensor):
    # Find the last convolutional layer (for ResNet/EfficientNet style)
    # Target layer layer4 is generic, let's make sure it's the actual module
    target_layer = None
    for name, module in model.named_modules():
        if 'layer4' in name and isinstance(module, torch.nn.Conv2d):
             target_layer = module
    
    if target_layer is None:
        # Fallback if specific layer naming differs
        target_layer = list(model.modules())[-2] if len(list(model.modules())) > 1 else list(model.modules())[-1]
        
    cam_extractor = GradCAM(model, target_layer)
    
    heatmap_raw, img_confidence = cam_extractor.generate(input_tensor)
    
    # Original image dimensions for overlay to PREVENT DISTORTION
    orig_w, orig_h = img_pil.size
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Resize heatmap to match ORIGINAL image dimensions
    heatmap = cv2.resize(heatmap_raw, (orig_w, orig_h))
    
    # 1. Base Normalization
    if np.max(heatmap) != np.min(heatmap):
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    else:
        heatmap = np.zeros_like(heatmap)
        
    # 2. Peak Amplification
    heatmap = heatmap ** 1.5 # Softer peak amplification
    
    # 3. Smooth out blockiness scaled to original image size
    kernel_size = max(15, min(orig_w, orig_h) // 20)
    if kernel_size % 2 == 0: kernel_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    # 4. Strict Noise Floor to erase the healthy background
    noise_floor = 0.2 # lowered slightly to catch edge boundaries
    heatmap = np.clip(heatmap - noise_floor, 0, 1)
    
    # Re-normalize
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # 5. Dynamic Opacity
    max_opacity = 0.8 if img_confidence > 0.6 else 0.55
    alpha = heatmap[..., np.newaxis] * max_opacity
    
    # Safe Alpha Blending
    overlayed = (img_bgr * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    _, buffer = cv2.imencode('.jpg', overlayed)
    return base64.b64encode(buffer).decode('utf-8'), img_confidence