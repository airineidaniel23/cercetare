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
        self.frame_gap = 29  # Set the gap to 29 frames for 1-second intervals
        
        # Collect all frames and corresponding target paths grouped by subfolder
        self.data_groups = self._collect_data()
        self.data_indices = self._create_indices()
        print(f"Initialized dataset with {sum(len(g) for g in self.data_groups)} frame-target pairs.")

    def _collect_data(self):
        """
        Collect all frame and target file paths, grouped by subfolders.
        """
        data_groups = []
        for subfolder in sorted(os.listdir(self.frames_folder)):
            frame_subfolder = os.path.join(self.frames_folder, subfolder)
            target_subfolder = os.path.join(self.target_folder, subfolder)
            
            if os.path.isdir(frame_subfolder) and os.path.isdir(target_subfolder):
                frames = sorted(os.listdir(frame_subfolder))
                targets = sorted(os.listdir(target_subfolder))
                
                # Ensure matching number of frames and targets
                if len(frames) != len(targets):
                    raise ValueError(f"Mismatched files in {frame_subfolder} and {target_subfolder}.")
                
                # Store paired paths as a group
                data_groups.append([
                    (os.path.join(frame_subfolder, frame), os.path.join(target_subfolder, target))
                    for frame, target in zip(frames, targets)
                ])
        return data_groups

    def _create_indices(self):
        """
        Create a flat list of valid indices, ensuring frame sequences stay within the same folder.
        """
        indices = []
        for group_idx, group in enumerate(self.data_groups):
            # Valid indices for this group, considering the frame gap
            indices.extend([(group_idx, idx) for idx in range(len(group) - 2 * self.frame_gap)])
        return indices

    def __len__(self):
        # Total number of valid samples across all groups
        return len(self.data_indices)

    def __getitem__(self, flat_idx):
        # Map flat index to group and local index
        group_idx, local_idx = self.data_indices[flat_idx]
        group = self.data_groups[group_idx]
        
        # Load three frames with a gap of `self.frame_gap` frames between them
        frame_paths = [
            group[local_idx + i * self.frame_gap][0] for i in range(3)
        ]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        
        # Apply transformations and concatenate along the channel dimension
        if self.transform:
            images = [self.transform(img) for img in images]
        image = torch.cat(images, dim=0)  # Concatenate to form a 9-channel input
        
        # Load the target label for the middle frame
        target_path = group[local_idx + 2 * self.frame_gap][1]
        with open(target_path, 'r') as file:
            target = np.array([int(x) for x in file.readline().strip().split()])
        target = np.argmax(target)
        
        return image, target

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Adapt first convolutional layer to accept 9 input channels
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 filters
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



def train_and_evaluate(model, train_loader, criterion, optimizer, epochs=8):
    best_acc = 0.0
    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}...")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

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
    train_losses, train_accuracies = train_and_evaluate(model, train_loader, criterion, optimizer, epochs=5)
    print("Training completed. Plotting results...")
    plot_metrics(train_losses, train_accuracies)
    classes = [f"Class {i}" for i in range(8)]
    plot_confusion_matrix(model, train_loader, classes)
    print("Script finished.")
