import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np


class FrameInferenceDataset(Dataset):
    def __init__(self, frames_folder, target_image_path, transform=None):
        """
        Args:
            frames_folder (str): Path to the folder containing frames.
            target_image_path (str): Path to the target image for this folder.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.frames_folder = frames_folder
        self.target_image_path = target_image_path
        self.transform = transform
        self.frames = sorted(os.listdir(frames_folder))
        self.frame_gap = 29  # Set frame gap to 29 frames for 1-second intervals

        # Load the target image once
        self.target_image = Image.open(self.target_image_path).convert("L")  # Grayscale image

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
            target_image = self.transform(self.target_image)
        else:
            target_image = transforms.ToTensor()(self.target_image)

        # Concatenate the three RGB images and the target image
        target_image = target_image.expand(1, -1, -1)  # Ensure it's a single-channel tensor
        image = torch.cat(images + [target_image], dim=0)  # Concatenate to create a 10-channel input

        return image, self.frames[idx + 2 * self.frame_gap]  # Return middle frame as filename


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Adapt first convolutional layer to accept 10 input channels
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 filters
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


def load_model(model_path):
    model = CNNModel()
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
        for images, filenames in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(filenames)):
                save_one_hot(output_folder, filenames[i], predicted[i].item())


if __name__ == "__main__":
    frames_folder = 'testFrames/1'
    frames_folder_parent = 'testFrames/'
    output_folder = 'testInferred/1'
    model_path = 'best_model.pth'
    target_image_path = os.path.join(os.path.dirname(frames_folder_parent), "target.jpg")

    # Ensure the target image exists
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = FrameInferenceDataset(frames_folder, target_image_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(model_path)
    infer(model, data_loader, output_folder)

    # Check if CUDA is being used
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("CUDA not available, using CPU")

    print("Inference completed. One-hot encoded files are saved in 'testInferred' folder.")
