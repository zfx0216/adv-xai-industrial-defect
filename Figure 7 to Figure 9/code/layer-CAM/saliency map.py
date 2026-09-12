"""
The main function of this code file is to generate a salient diagram of the current explanatory method
Step one, you need to set the 'actual_image_folder.absolutepath' to the absolute path of the folder where the original images are stored
Step two, you need to set the absolute path of the folder where the generated saliency map is saved to 'export_folder_absolute_path_Saliency_map'
Step three,you need to set "weights_path" as the absolute path of the weight file
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import LayerCAM
import os
from pytorch_grad_cam.utils.image import show_cam_on_image


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import actual image
actual_image_folder_absolute_path = r"F:\IndustrialInspectionCode\adversarial_samples\original_images\densenet121_trained_on_NEU-DET"

# Specify the saliency map save address
output_folder_absolute_path_saliency_map = r"F:\IndustrialInspectionCode\explanation_maps\layer-CAM\original_images\densenet121_trained_on_NEU-DET"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


starting_image_number = 1

if not os.path.exists(output_folder_absolute_path_saliency_map):
    os.makedirs(output_folder_absolute_path_saliency_map)

file_list = os.listdir(actual_image_folder_absolute_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    image_number = int(''.join(filter(str.isdigit, file_name)))

    if image_number < starting_image_number:
        continue

    print(file_name)
    input_file_path = os.path.join(actual_image_folder_absolute_path, file_name)
    output_file_path = os.path.join(output_folder_absolute_path_saliency_map, file_name)

    model = models.densenet121(num_classes=6)
    # Load model weights
    model_weight_absolute_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\densenet121\densenet121_best_acc.pth"
    model.load_state_dict(torch.load(model_weight_absolute_path, map_location=device))
    model.eval()

    image_path = input_file_path
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    target_layer = model.features.denseblock4.denselayer16.conv2
    cam = LayerCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)

    grayscale_cam = grayscale_cam[0, :]

    input_image = np.array(image) / 255.0

    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    image = Image.fromarray((visualization).astype(np.uint8))
    image.save(output_file_path)