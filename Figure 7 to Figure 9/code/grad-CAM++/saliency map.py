import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

# ==================== Grad-CAM++ implementation ====================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output)

        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + gradients.mul(activations).sum(dim=[2,3], keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8)

        weights = (alpha * F.relu(gradients)).sum(dim=[2,3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

# ==================== Device and model ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = densenet121(num_classes=6).to(device)
model_weight_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\densenet121\densenet121_best_acc.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
model.eval()

# ==================== Paths ====================
actual_image_folder_absolute_path = r"F:\IndustrialInspectionCode\adversarial_samples\Li=4\PGD\targeted\attack_on_densenet121_trained_on_NEU-DET"
output_folder_absolute_path_saliency_map = r"F:\IndustrialInspectionCode\explanation_maps\grad-CAM++\Li=4\PGD\targeted\attack_on_densenet121_trained_on_NEU-DET"
os.makedirs(output_folder_absolute_path_saliency_map, exist_ok=True)

# ==================== Preprocessing ====================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

image_files = sorted(os.listdir(actual_image_folder_absolute_path), key=sort_func)

# ==================== Batch processing ====================
for file_name in image_files:
    print(f"Processing: {file_name}")
    img_path = os.path.join(actual_image_folder_absolute_path, file_name)

    # Read the image
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    input_tensor = data_transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
    target_class = torch.argmax(output).item()

    # Grad-CAM++
    grad_cam_pp = GradCAMPlusPlus(model, model.features.denseblock4.denselayer16.conv2)
    cam_mask = grad_cam_pp(input_tensor, class_idx=target_class)

    # Generate the heatmap (using PIL to ensure reliable saving)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0

    # Overlay
    result = (heatmap * 0.5 + img_np * 0.5) * 255
    result = result.clip(0, 255).astype(np.uint8)

    #  Save with PIL to ensure the file is written.
    result_img = Image.fromarray(result)
    save_path = os.path.join(output_folder_absolute_path_saliency_map, file_name)
    result_img.save(save_path)

print(" All Grad-CAM++ outputs have been saved!")