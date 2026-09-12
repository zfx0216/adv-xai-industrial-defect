import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import LayerCAM, GradCAMPlusPlus, EigenCAM
from skimage.metrics import structural_similarity as ssim

# --- 1. Configuration paths ---
MODEL_PATH = r"F:\IndustrialInspectionCode\weights\PGD_adversarially_trained_models\NEU-DET\densenet121\densenet121_best_acc.pth"
ORIG_IMG_DIR = r"F:\IndustrialInspectionCode\adversarial_samples\original_images\densenet121_trained_on_NEU-DET"
# Paths to the two adversarial sample groups
ADV_GROUPS = [
    {
        "name": "Untargeted FGSM",
        "dir": r"F:\IndustrialInspectionCode\adversarial_samples_from_adversarially_trained_models\Li=8\FGSM\untargeted\attack_on_densenet121_trained_on_NEU-DET"
    },
    {
        "name": "Targeted FGSM",
        "dir": r"F:\IndustrialInspectionCode\adversarial_samples_from_adversarially_trained_models\Li=8\FGSM\targeted\attack_on_densenet121_trained_on_NEU-DET"
    }
]

# --- 2. Core metric functions ---
def calculate_metrics(map1, map2):
    # Normalize to [0, 1]
    map1 = (map1 - map1.min()) / (map1.max() - map1.min())
    map2 = (map2 - map2.min()) / (map2.max() - map2.min())

    # 1. IoU: use the 80th percentile as the threshold
    threshold = np.percentile(map1, 80)
    bin1 = (map1 > threshold).astype(int)
    bin2 = (map2 > threshold).astype(int)
    inter = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()
    iou = inter / union if union != 0 else 0.0

    # 2. SSIM
    ssim_val = ssim(map1, map2, data_range=1.0)

    # 3. ASD: centroid drift distance
    def get_centroid(h):
        y, x = np.indices(h.shape)
        total = h.sum()
        if total < 1e-8:
            return np.array([0.0, 0.0])
        return np.array([(y * h).sum() / total, (x * h).sum() / total])

    asd = np.linalg.norm(get_centroid(map1) - get_centroid(map2))
    return iou, ssim_val, asd

# --- 3. Main program ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Instantiate the ResNet50 model and adapt it for 6-class classification
    model = models.densenet121(weights=None)

    # model.fc = nn.Linear(model.fc.in_features, 6)

    in_channels = model.classifier.in_features
    model.classifier = nn.Linear(in_channels, 6)

    # B. Load weights
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device).eval()

    # C. CAM configuration
    # target_layers = [model.layer4[-1]]
    target_layers = [model.features.denseblock4.denselayer16.conv2]
    # target_layers = [model.blocks[-1].norm1]

    explainers = {
        'LayerCAM': LayerCAM(model=model, target_layers=target_layers),
        'GradCAM++': GradCAMPlusPlus(model=model, target_layers=target_layers),
        'EigenCAM': EigenCAM(model=model, target_layers=target_layers)
    }

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Store results for both adversarial groups: key = group name
    all_group_results = {}

    # Iterate over the two adversarial sample groups
    for group_info in ADV_GROUPS:
        group_name = group_info["name"]
        adv_dir = group_info["dir"]
        print(f"\n==================== Computing: {group_name} ====================")
        # Initialize the metric cache for the current group
        group_results = {name: {'iou': [], 'ssim': [], 'asd': []} for name in explainers}

        print(f"{'Filename':<20} | {'Method':<10} | {'IoU':<8} | {'SSIM':<8} | {'ASD':<8}")
        print("-" * 65)

        # Iterate over the original images
        for filename in os.listdir(ORIG_IMG_DIR):
            orig_path = os.path.join(ORIG_IMG_DIR, filename)
            adv_path = os.path.join(adv_dir, filename)
            if not os.path.exists(adv_path):
                continue

            img_orig = preprocess(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
            img_adv = preprocess(Image.open(adv_path).convert('RGB')).unsqueeze(0).to(device)

            for cam_name, explainer in explainers.items():
                cam_orig = explainer(input_tensor=img_orig, targets=None)[0]
                cam_adv = explainer(input_tensor=img_adv, targets=None)[0]

                iou, ssim_v, asd = calculate_metrics(cam_orig, cam_adv)

                group_results[cam_name]['iou'].append(iou)
                group_results[cam_name]['ssim'].append(ssim_v)
                group_results[cam_name]['asd'].append(asd)

                print(f"{filename[:18]:<20} | {cam_name:<10} | {iou:.4f}   | {ssim_v:.4f}   | {asd:.4f}")

        # Save the current group results
        all_group_results[group_name] = group_results

        # Print the mean for this group
        print("-" * 65)
        print(f"[{group_name}] Mean metrics for each explanation method:")
        for cam_name, data in group_results.items():
            if len(data['iou']) > 0:
                avg_iou = np.mean(data['iou'])
                avg_ssim = np.mean(data['ssim'])
                avg_asd = np.mean(data['asd'])
                print(f"{cam_name:<10}: IoU={avg_iou:.5f}, SSIM={avg_ssim:.5f}, ASD={avg_asd:.5f}")

    # Summarize and compare after all runs
    print("\n\n==================== Summary comparison of the two adversarial sample groups ====================")
    for group_name, group_data in all_group_results.items():
        print(f"\n{group_name}:")
        for cam_name, data in group_data.items():
            if len(data['iou']) > 0:
                print(f"  {cam_name:<10} IoU={np.mean(data['iou']):.5f}, SSIM={np.mean(data['ssim']):.5f}, ASD={np.mean(data['asd']):.5f}")

if __name__ == "__main__":
    main()