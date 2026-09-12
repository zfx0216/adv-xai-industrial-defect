import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import csv

def evaluate_model(model, dataloader, device):
    model.eval()
    total_samples, correct_predictions = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    return accuracy, correct_predictions, total_samples

def main():
    loss_sequence = []
    acc_sequence = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train":transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    model_name = 'vitb16'
    save_dir = r'F:\IndustrialInspectionCode\weights\PGD_adversarially_trained_models\NEU-DET\vitb16'

    train_dataset = datasets.ImageFolder(
        root=r"F:\IndustrialInspectionCode\adversarial_training_data\vitb16\NEU-DET\train",
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    print(cla_dict)
    with open(os.path.join(save_dir, 'class_indices.json'), 'w') as json_file:
        json_file.write(json.dumps(cla_dict, indent=4))

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    test_dataset = datasets.ImageFolder(
        root=r"F:\IndustrialInspectionCode\original_data\NEU-DET\test",
        transform=data_transform["test"]
    )
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if str(device) == 'cuda:0' else False
    )

    test_classes_list = test_dataset.class_to_idx
    if test_classes_list != classes_list:
        print("Warning: the training and test sets have different classes!")

    # ===================== ViT model =====================
    net = vit_b_16(num_classes=1000)
    pretrained_weights_path = r"E:\Desktop\weight_1000\vit_b_16-c867db91.pth"
    state_dict = torch.load(pretrained_weights_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)

    num_features = net.heads.head.in_features
    net.heads.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 6)
    )

    for param in net.parameters():
        param.requires_grad = False
    for param in net.heads.head.parameters():
        param.requires_grad = True

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.heads.head.parameters(), lr=1e-4)

    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    epochs = 30
    best_loss = float('inf')
    best_acc = 0.0
    best_loss_model_path = os.path.join(save_dir, f'{model_name}_best_loss.pth')
    best_acc_model_path = os.path.join(save_dir, f'{model_name}_best_acc.pth')

    # ===================== [NEW] Create log file =====================
    log_file = os.path.join(save_dir, "train_log.csv")
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Test_Accuracy"])  # header
    # ===============================================================

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} lr:{:.6f}".format(
                epoch + 1, epochs, loss, optimizer.param_groups[0]['lr']
            )

        epoch_loss = running_loss / len(train_loader)
        loss_sequence.append(epoch_loss)

        test_accuracy, test_correct, test_total = evaluate_model(net, test_loader, device)
        acc_sequence.append(test_accuracy)

        print('[epoch %d] train_loss: %.3f  lr: %.6f' %
              (epoch + 1, epoch_loss, optimizer.param_groups[0]['lr']))
        print(f'[epoch {epoch + 1}] test_accuracy: {test_accuracy:.2f}%')

        scheduler_plateau.step(epoch_loss)

        # Save the best model
        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), best_loss_model_path)
            print(f"New best loss saved: {best_loss:.4f}")

        if test_accuracy >= best_acc:
            best_acc = test_accuracy
            torch.save(net.state_dict(), best_acc_model_path)
            print(f"New best acc saved: {best_acc:.2f}%")

        # ===================== [NEW] Save data to a file each epoch =====================
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, round(epoch_loss, 4), round(test_accuracy, 2)])
        # ====================================================================

        # Plot
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', color='tab:blue', fontsize=12)
        ax1.plot(range(1, epoch+2), loss_sequence, color='tab:blue', linewidth=2, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Test Accuracy (%)', color='tab:orange', fontsize=12)
        ax2.plot(range(1, epoch+2), acc_sequence, color='tab:orange', linewidth=2, label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax1.legend(loc='best')
        plt.title('Training Loss and Test Accuracy', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_acc_plot.png'), dpi=300)
        plt.close()

    print('Finished Training! Best Acc: ', best_acc)

if __name__ == '__main__':
    main()