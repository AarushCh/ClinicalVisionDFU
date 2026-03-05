import cv2
import numpy as np
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from PIL import Image

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

        target_layer.register_forward_hook(save_activations)
        target_layer.register_full_backward_hook(save_gradients)

    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.softmax(dim=1)[0, 1].item()

def overlay_heatmap(img_pil, heatmap):
    img = np.array(img_pil.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    
    _, buffer = cv2.imencode('.jpg', overlayed)
    return base64.b64encode(buffer).decode('utf-8')