import torch
from torchvision.models import vit_b_16
from torchvision import transforms
import os
import json
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    original = []
    modification = []
    original_img_names = []
    modify_img_names = []

    # Image folder path
    img_folder_path = r"F:\IndustrialInspectionCode\adversarial_samples\original_images\vitb16_trained_on_NEU-DET"
    assert os.path.exists(img_folder_path), "Folder '{}' does not exist".format(img_folder_path)
    img_modify_folder_path = r"F:\IndustrialInspectionCode\adversarial_samples\Li=2\MI-FGSM\untargeted\attack_on_vitb16_trained_on_NEU-DET"
    assert os.path.exists(img_modify_folder_path), "Folder '{}' does not exist".format(img_modify_folder_path)

    # Data preprocessing
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Read the class index file
    json_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\vitb16\class_indices.json"
    assert os.path.exists(json_path), "File '{}' does not exist".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create and load the model
    model = vit_b_16(num_classes=6).to(device)
    weights_path = r"F:\IndustrialInspectionCode\weights\NEU-DET\vitb16\vitb16_best_acc.pth"
    assert os.path.exists(weights_path), "File '{}' does not exist".format(weights_path)
    # model.load_state_dict(torch.load(weights_path))
    state_dict = torch.load(weights_path, weights_only=True)

    # Automatically adapt the classifier head key
    if "heads.head.1.weight" in state_dict:
        state_dict["heads.head.weight"] = state_dict.pop("heads.head.1.weight")
        state_dict["heads.head.bias"] = state_dict.pop("heads.head.1.bias")

    model.load_state_dict(state_dict)
    model.eval()

    # Iterate over all images in the image folder
    for img_name in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_name)

        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)

        # Get the prediction for the original label
        o_prediction = model(img.to(device))
        o_label_index = torch.argmax(o_prediction, dim=1).item()
        print(f'{img_name}{"original:"}{o_label_index}')
        original.append(o_label_index)
    # Iterate over all images in the image folder
    for img_name in os.listdir(img_modify_folder_path):
        img_path = os.path.join(img_modify_folder_path, img_name)

        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)

        # Get the prediction for the modified label
        m_prediction = model(img.to(device))
        m_label_index = torch.argmax(m_prediction, dim=1).item()
        print(f'{img_name}{"modification:"}{m_label_index}')
        modification.append(m_label_index)
    s = 0
    for i in range(len(original)):
        if original[i] != modification[i]:
            s += 1
    rate = (f'{s / len(original) * 100}{"%"}')
    print(rate)

if __name__ == "__main__":
    main()