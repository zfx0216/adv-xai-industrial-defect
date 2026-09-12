import os
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torchvision.models import densenet121
from pytorch_grad_cam import EigenCAM


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    # Normalize
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    # Generate a color heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    # Overlay: original image + heatmap
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = np.clip(cam, 0, 1)
    return np.uint8(255 * cam)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== Your paths (no changes required)======================
actual_image_folder_absolute_path = r"F:\IndustrialInspectionCode\adversarial_samples\Li=4\MI-FGSM\untargeted\attack_on_densenet121_trained_on_NEU-DET"
output_folder_absolute_path_saliency_map = r"F:\IndustrialInspectionCode\explanation_maps\eign-CAM\Li=4\MI-FGSM\untargeted\attack_on_densenet121_trained_on_NEU-DET"
json_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\densenet121\class_indices.json"
weights_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\densenet121\densenet121_best_acc.pth"
# ==============================================================

starting_image_number = 1

if not os.path.exists(output_folder_absolute_path_saliency_map):
    os.makedirs(output_folder_absolute_path_saliency_map)

file_list = os.listdir(actual_image_folder_absolute_path)
file_list = sorted(file_list, key=sort_func)

# Load the model
with open(json_path, "r") as f:
    class_indict = json.load(f)

model = densenet121(num_classes=6).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

# Process each image
for file_name in file_list:
    image_number = int(''.join(filter(str.isdigit, file_name)))
    if image_number < starting_image_number:
        continue

    print(f"Generating: {file_name}")
    img_path = os.path.join(actual_image_folder_absolute_path, file_name)
    save_path = os.path.join(output_folder_absolute_path_saliency_map, file_name)

    # Image preprocessing
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = data_transform(img_pil).unsqueeze(0).to(device)

    # Predict the class
    with torch.no_grad():
        output = model(img_tensor)
        class_index = torch.argmax(output, dim=1).item()

    # CAM core
    target_layers = [model.features.denseblock4.denselayer16.conv2]
    cam = EigenCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[class_index])[0, :]

    # Generate the color overlay
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam)

    # Save
    Image.fromarray(visualization).save(save_path)

print(" All color heatmaps have been saved!")