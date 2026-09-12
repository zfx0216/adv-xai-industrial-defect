import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import numpy as np
import os

# ====================== Configuration ======================
PIXEL_EPS = 8
EPS = PIXEL_EPS / 255.0
# ====================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Paths
weights_path = r"F:\IndustrialInspectionCode\weights\MTSD\densenet121\densenet121_best_acc.pth"
folder_path = r"F:\IndustrialInspectionCode\adversarial_samples\original_images\densenet121_trained_on_MTSD"
save_path = r"F:\IndustrialInspectionCode\adversarial_samples\Li=8\FGSM\targeted\attack_on_densenet121_trained_on_MTSD"

# Model
model = densenet121(num_classes=6).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

os.makedirs(save_path, exist_ok=True)


def sort_func(f):
    return int(''.join(filter(str.isdigit, f)))


file_list = sorted(os.listdir(folder_path), key=sort_func)


# ====================== FGSM targeted attack ======================
def fgsm_attack_targeted(model, images_norm, target_label):
    # Compute gradients directly on the normalized image
    images_norm.requires_grad = True

    # Feed directly to the model (already normalized)
    outputs = model(images_norm)
    loss = nn.CrossEntropyLoss()(outputs, target_label)

    model.zero_grad()
    loss.backward()
    grad = images_norm.grad.sign()

    # Targeted attack: move in the opposite direction of the gradient
    adv_images = images_norm - EPS * grad
    return adv_images


# ====================== Run ======================
for file_name in file_list:
    print("Processing: ", file_name)

    img = Image.open(os.path.join(folder_path, file_name)).convert("RGB")
    original = np.array(img.resize((224, 224)))
    img_tensor = data_transform(img).unsqueeze(0).to(device)

    # Get predictions and use the third-ranked class as the target label
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor)).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 6)
        target_label = top_indices[3]
        target_label = target_label.unsqueeze(0).to(device)

    # Targeted attack
    adv_tensor = fgsm_attack_targeted(model, img_tensor, target_label)

    # Convert back to an image
    adv = adv_tensor.squeeze().detach().cpu().numpy()
    mean_np = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std_np = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    adv = adv * std_np + mean_np
    adv = adv * 255

    adv = np.transpose(adv, (1, 2, 0))
    adv = np.clip(adv, original - PIXEL_EPS, original + PIXEL_EPS)
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    adv_img = Image.fromarray(adv)
    adv_img.save(os.path.join(save_path, file_name))