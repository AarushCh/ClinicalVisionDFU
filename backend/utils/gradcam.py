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
        with torch.inference_mode(mode=False), torch.enable_grad():
            input_tensor.requires_grad_(True)
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
    
    # 1. Base Normalization to [0, 1]
    if np.max(heatmap) != np.min(heatmap):
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    else:
        heatmap = np.zeros_like(heatmap)
        
    # 2. Smooth Interpolation to remove 12x12 grid blockiness
    kernel_size = max(25, min(orig_w, orig_h) // 10)
    if kernel_size % 2 == 0: kernel_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    # 3. Boost mid-tier activations (Square root scaling expands the mid-range)
    # This prevents the heatmap from collapsing into a tiny intense dot
    heatmap = heatmap ** 0.5 
    
    # Re-normalize just to be safe
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # 4. Color Mapping
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # 5. Dynamic Alpha Blending
    # Make the heatmap highly transparent where the AI isn't looking (heatmap ~ 0)
    # Make it semi-opaque where the AI is looking (heatmap ~ 1)
    base_alpha = 0.55 if img_confidence > 0.6 else 0.4
    
    # The alpha channel is scaled directly by the numerical intensity of the heatmap itself
    # So healthy skin remains fully visible, diseased skin gets colored.
    alpha = (heatmap[..., np.newaxis] * base_alpha)
    
    # Safe Alpha Blending Formula
    overlayed = (img_bgr * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    _, buffer = cv2.imencode('.jpg', overlayed)
    return base64.b64encode(buffer).decode('utf-8'), img_confidence