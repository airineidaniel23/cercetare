import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class FrameDataset(Dataset):
    def __init__(self, frames_folder, target_folder, transform=None):
        """
        Initialize dataset with frames and target folders, and optionally apply transforms.
        Ensures frames and targets are isolated within their respective subfolders.
        """
        self.frames_folder = frames_folder
        self.target_folder = target_folder
        self.transform = transform
        self.frame_gap = 29

        self.data_groups, self.folder_ids = self._collect_data()
        self.data_indices = self._create_indices()
        print(f"Initialized dataset with {sum(len(g) for g in self.data_groups)} frame-target pairs.")

    def _collect_data(self):
        """
        Collect all frame and target file paths, grouped by subfolders.
        """
        data_groups = []
        folder_ids = []

        for folder_idx, parent_folder in enumerate(sorted(os.listdir(self.frames_folder))):
            parent_folder_path = os.path.join(self.frames_folder, parent_folder)
            target_parent_folder_path = os.path.join(self.target_folder, parent_folder)
            
            if os.path.isdir(parent_folder_path):
                for subfolder in sorted(os.listdir(parent_folder_path)):
                    frame_subfolder = os.path.join(parent_folder_path, subfolder)
                    target_subfolder = os.path.join(target_parent_folder_path, subfolder)
                    
                    if os.path.isdir(frame_subfolder) and os.path.isdir(target_subfolder):
                        frames = sorted(os.listdir(frame_subfolder))
                        targets = sorted(os.listdir(target_subfolder))
                        
                        if len(frames) != len(targets):
                            raise ValueError(f"Mismatched files in {frame_subfolder} and {target_subfolder}.")
                        
                        data_groups.append([
                            (os.path.join(frame_subfolder, frame), os.path.join(target_subfolder, target))
                            for frame, target in zip(frames, targets)
                        ])
                        folder_ids.append(folder_idx)
        return data_groups, folder_ids

    def _create_indices(self):
        """
        Create a flat list of valid indices, ensuring frame sequences stay within the same folder.
        """
        indices = []
        for group_idx, group in enumerate(self.data_groups):
            indices.extend([(group_idx, idx) for idx in range(len(group) - 2 * self.frame_gap)])
        return indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, flat_idx):
        group_idx, local_idx = self.data_indices[flat_idx]
        group = self.data_groups[group_idx]
        folder_id = self.folder_ids[group_idx]
        
        frame_paths = [
            group[local_idx + i * self.frame_gap][0] for i in range(3)
        ]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        
        if self.transform:
            images = [self.transform(img) for img in images]
        image = torch.cat(images, dim=0)
        
        target_path = group[local_idx + 2 * self.frame_gap][1]
        with open(target_path, 'r') as file:
            target = np.array([int(x) for x in file.readline().strip().split()])
        target = np.argmax(target)
        
        return image, target, folder_id

class CNNModel(nn.Module):
    def __init__(self, num_folders = 2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256 + num_folders, 256)
        self.fc3 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)
        
        self.folder_embedding = nn.Embedding(num_folders, num_folders)

    def forward(self, x, folder_id):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        folder_embed = self.folder_embedding(folder_id)
        combined = torch.cat((x, folder_embed), dim=1)
        combined = self.dropout(torch.relu(self.fc2(combined)))
        output = self.fc3(combined)
        return output

def train_and_evaluate(model, train_loader, criterion, optimizer, epochs=8, num_classes=8):
    best_acc = 0.0
    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}...")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for batch_idx, (images, labels, folder_ids) in enumerate(train_loader):
            images, labels, folder_ids = images.to(device), labels.to(device), folder_ids.to(device)
            optimizer.zero_grad()
            outputs = model(images, folder_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class-wise accuracy computation
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

        # End of epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Compute and print class-wise accuracy
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                class_accuracies.append(acc)
                print(f"Class {i} Accuracy: {acc:.4f}")
            else:
                class_accuracies.append(0.0)
                print(f"Class {i} Accuracy: N/A (No samples)")

        avg_class_accuracy = sum(class_accuracies) / num_classes
        print(f"Average Class Accuracy: {avg_class_accuracy:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f'New best accuracy: {best_acc:.4f}. Saving model...')
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, train_accuracies


def plot_metrics(train_losses, train_accuracies):
    print("Plotting metrics...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    print("Metrics plotted.")

def plot_confusion_matrix(model, data_loader, classes):
    print("Generating confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            print(f"Evaluating batch {batch_idx + 1}/{len(data_loader)}...")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.show()
    print("Confusion matrix plotted.")

if __name__ == "__main__":
    print("Starting script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    print("Preparing dataset...")
    train_dataset = FrameDataset(frames_folder='frames', target_folder='target', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"DataLoader created with {len(train_loader)} batches.")
    model = CNNModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting training...")
    train_losses, train_accuracies = train_and_evaluate(model, train_loader, criterion, optimizer, epochs=10)
    print("Training completed. Plotting results...")
    plot_metrics(train_losses, train_accuracies)
    classes = [f"Class {i}" for i in range(8)]
    plot_confusion_matrix(model, train_loader, classes)
    print("Script finished.")
