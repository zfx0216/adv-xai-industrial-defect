"""
The main function of this folder is to generate experimental evaluation images using the method.
First, you need to set the parameter "model_weight_absolute_path" to the absolute path of the model's weight file. For example, if you are using the AlexNet model, you need to set this to the absolute path of the AlexNet weight file.
Second, you need to set the parameter "actual_image_folder_absolute_path" to the absolute path of the folder storing the original experimental images.
Third, you need to set "output_folder_path_increase" to the absolute path of the folder for storing experimental images used to evaluate the increase in classification value, and set "output_folder_path_decrease" to the absolute path of the folder for storing experimental images used to evaluate the decrease in classification value.
Finally, execute this file to generate the experimental evaluation images using the method.
"""


import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import LayerCAM
import math
from torchvision.models import alexnet

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def save_cam_to_txt(file_path, cam):
    np.savetxt(file_path, cam, fmt='%f')

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = alexnet(num_classes=1000)

    # Load model weights
    model_weight_absolute_path = "..."
    model.load_state_dict(torch.load(model_weight_absolute_path, map_location=device))
    model.eval()
    target_layers = [model.features[-1]]

    # Perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Import actual image
    actual_image_folder_absolute_path = "..."

    # Specify the tampering images save address
    output_folder_path_increase = "..."
    output_folder_path_decrease = "..."

    ratio_increase = 1.01
    ratio_decrease = 0.99

    starting_image_number = 1

    if not os.path.exists(output_folder_path_increase):
        os.makedirs(output_folder_path_increase)
    if not os.path.exists(output_folder_path_decrease):
        os.makedirs(output_folder_path_decrease)

    file_list = os.listdir(actual_image_folder_absolute_path)
    file_list = sorted(file_list, key=sort_func)

    for file_name in file_list:
        image_number = int(''.join(filter(str.isdigit, file_name)))

        if image_number < starting_image_number:
            continue

        print(file_name)
        input_file_path = os.path.join(actual_image_folder_absolute_path, file_name)
        output_file_path_increase = os.path.join(output_folder_path_increase, file_name)
        output_file_path_decrease = os.path.join(output_folder_path_decrease, file_name)

        image = Image.open(input_file_path)
        image = image.resize((224, 224))
        actual_image_matrix = np.array(image)

        # The R, G, B three channel matrix of the actual image
        actual_image_channel_R = actual_image_matrix[:, :, 0]
        actual_image_channel_R = actual_image_channel_R.astype(np.float64)
        actual_image_channel_G = actual_image_matrix[:, :, 1]
        actual_image_channel_G = actual_image_channel_G.astype(np.float64)
        actual_image_channel_B = actual_image_matrix[:, :, 2]
        actual_image_channel_B = actual_image_channel_B.astype(np.float64)

        actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
        actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
        actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225

        actual_image_R_channel_increase = actual_image_matrix[:, :, 0].copy()
        actual_image_R_channel_increase = actual_image_R_channel_increase.astype(np.float64)
        actual_image_G_channel_increase = actual_image_matrix[:, :, 1].copy()
        actual_image_G_channel_increase = actual_image_G_channel_increase.astype(np.float64)
        actual_image_B_channel_increase = actual_image_matrix[:, :, 2].copy()
        actual_image_B_channel_increase = actual_image_B_channel_increase.astype(np.float64)

        actual_image_R_channel_decrease = actual_image_matrix[:, :, 0].copy()
        actual_image_R_channel_decrease = actual_image_R_channel_decrease.astype(np.float64)
        actual_image_G_channel_decrease = actual_image_matrix[:, :, 1].copy()
        actual_image_G_channel_decrease = actual_image_G_channel_decrease.astype(np.float64)
        actual_image_B_channel_decrease = actual_image_matrix[:, :, 2].copy()
        actual_image_B_channel_decrease = actual_image_B_channel_decrease.astype(np.float64)

        img = Image.open(input_file_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].numpy()

        target_category = [np.argmax(probabilities)]
        print(target_category)

        cam = LayerCAM(model=model, target_layers=target_layers)

        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]

        for i in range(224):
            for j in range(224):
                if grayscale_cam[i][j] >= 0.5:
                    if actual_image_transform_matrix_R[i][j] > 0:
                        actual_image_R_channel_increase[i][j] = actual_image_channel_R[i][j] * ratio_increase
                        actual_image_R_channel_decrease[i][j] = actual_image_channel_R[i][j] * ratio_decrease
                    if actual_image_transform_matrix_R[i][j] < 0:
                        actual_image_R_channel_increase[i][j] = actual_image_channel_R[i][j] * ratio_increase
                        actual_image_R_channel_decrease[i][j] = actual_image_channel_R[i][j] * ratio_decrease

                    if actual_image_transform_matrix_G[i][j] > 0:
                        actual_image_G_channel_increase[i][j] = actual_image_channel_G[i][j] * ratio_increase
                        actual_image_G_channel_decrease[i][j] = actual_image_channel_G[i][j] * ratio_decrease
                    if actual_image_transform_matrix_G[i][j] < 0:
                        actual_image_G_channel_increase[i][j] = actual_image_channel_G[i][j] * ratio_increase
                        actual_image_G_channel_decrease[i][j] = actual_image_channel_G[i][j] * ratio_decrease

                    if actual_image_transform_matrix_B[i][j] > 0:
                        actual_image_B_channel_increase[i][j] = actual_image_channel_B[i][j] * ratio_increase
                        actual_image_B_channel_decrease[i][j] = actual_image_channel_B[i][j] * ratio_decrease
                    if actual_image_transform_matrix_B[i][j] < 0:
                        actual_image_B_channel_increase[i][j] = actual_image_channel_B[i][j] * ratio_increase
                        actual_image_B_channel_decrease[i][j] = actual_image_channel_B[i][j] * ratio_decrease

        actual_image_R_channel_increase = np.clip(actual_image_R_channel_increase, 0, 255)
        actual_image_G_channel_increase = np.clip(actual_image_G_channel_increase, 0, 255)
        actual_image_B_channel_increase = np.clip(actual_image_B_channel_increase, 0, 255)

        actual_image_R_channel_decrease = np.clip(actual_image_R_channel_decrease, 0, 255)
        actual_image_G_channel_decrease = np.clip(actual_image_G_channel_decrease, 0, 255)
        actual_image_B_channel_decrease = np.clip(actual_image_B_channel_decrease, 0, 255)


        image_rgb = np.stack([actual_image_R_channel_increase, actual_image_G_channel_increase, actual_image_B_channel_increase], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save(output_file_path_increase)

        image_rgb = np.stack([actual_image_R_channel_decrease, actual_image_G_channel_decrease, actual_image_B_channel_decrease], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save(output_file_path_decrease)


if __name__ == '__main__':
    main()
