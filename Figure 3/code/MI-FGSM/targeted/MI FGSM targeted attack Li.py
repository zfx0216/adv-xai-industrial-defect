import torch
from torchvision import transforms
from torchvision.models import densenet121
import os
from PIL import Image
import numpy as np
from MIFGSM import MIFGSM
import json
import matplotlib.pyplot as plt
import time

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_current = densenet121

# Create and load model
model = model_current(num_classes=6).to(device)

# Absolute path to import AlexNet weight file
weights_path = r"F:\IndustrialInspectionCode\weights\MTSD\densenet121\densenet121_best_acc.pth"

assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Iterative step size
iteration_step_size = 1 / 255

# Calculate the maximum number of iterations
max_num_iterative = 20

# The absolute path to store the original image folder
folder_path = r"F:\IndustrialInspectionCode\adversarial_samples\original_images\densenet121_trained_on_MTSD"

# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = r"F:\IndustrialInspectionCode\adversarial_samples\Li=8\MI-FGSM\targeted\attack_on_densenet121_trained_on_MTSD"

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

attack_start_time = time.time()

for file_name in file_list:
    print(file_name)
    image_path = os.path.join(folder_path, file_name)

    image = Image.open(image_path).convert('RGB')

    # actual image
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
    actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
    actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

    image = data_transform(image).unsqueeze(0).cuda()

    # Load image
    img_absolute_path = image_path
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_absolute_path = r"F:\IndustrialInspectionCode\weights\MTSD\densenet121\class_indices.json"
    assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

    with open(json_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_current(num_classes=6).to(device)

    # Load model weights
    weights_absolute_path = weights_path
    assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
    model.load_state_dict(torch.load(weights_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 6)

    # When generating adversarial samples, it is a no target attack, so the true label of the image needs to be set in "[]" here
    label = torch.tensor([top_indices[3]]).cuda()

    atk = MIFGSM(model, decay=1, steps=max_num_iterative, alpha=iteration_step_size)
    # Set to targeted attack mode
    atk.set_mode_targeted_by_label()
    adv_image = atk(image, label, actual_image_transform_matrix, actual_image_matrix)


    # Combine three channels into an RGB image
    image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)

attack_time = time.time() - attack_start_time
print("The attack time is: ", attack_time/10, "s")