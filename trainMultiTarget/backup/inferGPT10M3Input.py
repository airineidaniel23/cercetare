import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np

class FrameInferenceDataset(Dataset):
    def __init__(self, frames_folder, folder_id, transform=None):
        """
        Args:
            frames_folder (str): Path to the folder containing frames.
            folder_id (int): ID of the parent folder for all frames in this dataset.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.frames_folder = frames_folder
        self.folder_id = folder_id  # Add folder ID as a constant input
        self.transform = transform
        self.frames = sorted(os.listdir(frames_folder))
        self.frame_gap = 29  # Set frame gap to 29 frames for 1-second intervals
        
    def __len__(self):
        # Adjust length to ensure we have 3 frames spaced by self.frame_gap for each sample
        return len(self.frames) - 2 * self.frame_gap
    
    def __getitem__(self, idx):
        # Load three frames spaced by `self.frame_gap`
        frame_paths = [
            os.path.join(self.frames_folder, self.frames[idx + i * self.frame_gap]) 
            for i in range(3)
        ]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        
        # Apply transformations and concatenate along the channel dimension
        if self.transform:
            images = [self.transform(img) for img in images]
        image = torch.cat(images, dim=0)  # Concatenate to create a 9-channel input
        
        return image, self.folder_id, self.frames[idx + 2 * self.frame_gap]  # Return folder ID and middle frame as filename


class CNNModel(nn.Module):
    def __init__(self, num_folders=2):
        super(CNNModel, self).__init__()
        # Adapt first convolutional layer to accept 9 input channels
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 filters
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256 + num_folders, 256)  # Include folder ID embedding size
        self.fc3 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)
        
        # Embedding for folder IDs
        self.folder_embedding = nn.Embedding(num_folders, num_folders)

    def forward(self, x, folder_id):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        
        # Get folder ID embedding
        folder_embed = self.folder_embedding(folder_id)
        
        # Concatenate image features with folder embedding
        combined = torch.cat((x, folder_embed), dim=1)
        combined = self.dropout(torch.relu(self.fc2(combined)))
        output = self.fc3(combined)
        return output


def load_model(model_path, num_folders):
    model = CNNModel(num_folders=num_folders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_one_hot(output_folder, filename, class_idx, num_classes=8):
    one_hot = np.zeros(num_classes, dtype=int)
    one_hot[class_idx] = 1
    with open(os.path.join(output_folder, filename.replace(".jpg", ".txt")), "w") as f:
        f.write(" ".join(map(str, one_hot)) + "\n")


def infer(model, data_loader, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with torch.no_grad():
        for images, folder_ids, filenames in data_loader:
            images = images.to(device)
            folder_ids = folder_ids.to(device)
            outputs = model(images, folder_ids)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(filenames)):
                save_one_hot(output_folder, filenames[i], predicted[i].item())


if __name__ == "__main__":
    frames_folder = 'testFrames/1'
    output_folder = 'testInferred/1'
    model_path = 'best_model_baseline.pth'
    folder_id = 0  # Assign the folder ID for this test folder
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = FrameInferenceDataset(frames_folder, folder_id, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    num_folders = 2  # Total number of folder IDs used during training
    model = load_model(model_path, num_folders=num_folders)
    infer(model, data_loader, output_folder)

    # Check if CUDA is being used
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("CUDA not available, using CPU")
    
    print("Inference completed. One-hot encoded files are saved in 'testInferred' folder.")
