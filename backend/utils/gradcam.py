import cv2
import numpy as np
import torch
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

        target_layer.register_forward_hook(save_activations)
        target_layer.register_full_backward_hook(save_gradients)

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        
        # THE FIX: Apply a 40% threshold filter to kill the background noise (toes)
        cam = np.where(cam > np.max(cam) * 0.4, cam, 0)
        
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        confidence = output.softmax(dim=1)[0, 0].item() 
        return cam, confidence

def generate_gradcam_heatmap(model, img_pil, input_tensor):
    target_layer = model.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)
    
    heatmap, img_confidence = cam_extractor.generate(input_tensor)
    
    img = np.array(img_pil.resize((224, 224)))
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    overlayed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
    
    _, buffer = cv2.imencode('.jpg', overlayed)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return heatmap_base64, img_confidence