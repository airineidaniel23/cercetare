import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class FrameInferenceDataset(Dataset):
    def __init__(self, frames_folder, transform=None):
        self.frames_folder = frames_folder
        self.transform = transform
        self.frames = sorted(os.listdir(frames_folder))
        self.frame_gap = 29  # Set frame gap to 50 frames
        
    def __len__(self):
        return len(self.frames) - 2 * self.frame_gap
    
    def __getitem__(self, idx):
        frame_paths = [
            os.path.join(self.frames_folder, self.frames[idx + i * self.frame_gap]) 
            for i in range(3)
        ]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        
        if self.transform:
            images = [self.transform(img) for img in images]
        image = torch.cat(images, dim=0)  # Concatenate to create a 9-channel input
        
        return image, self.frames[idx + 2 * self.frame_gap]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(guided_relu(self.conv1(x)))
        x = self.pool(guided_relu(self.conv2(x)))
        x = self.pool(guided_relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(guided_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define a custom guided ReLU
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        grad_input[grad_output <= 0] = 0
        return grad_input

def guided_relu(x):
    return GuidedReLU.apply(x)

def load_model(model_path):
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_guided_backprop(image, gradients, filename):
    # Confirm the shape of gradients; gradients[:3] should have shape [3, H, W]
    gradients = gradients.cpu().numpy()  # Convert to numpy array
    if gradients.shape[0] == 1:  # If batch dimension exists, squeeze it
        gradients = gradients.squeeze(0)
    
    # Compute heatmap as the mean of the absolute gradients along the color channels
    heatmap = np.mean(np.abs(gradients), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Detach and convert the original image to numpy format
    image = image.permute(1, 2, 0).detach().cpu().numpy()  # Convert image to H x W x 3 format
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Blend with heatmap
    plt.axis('off')
    plt.savefig(filename)
    plt.close()



def guided_backprop_infer(model, data_loader, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for images, filenames in data_loader:
        images = images.to(device)
        images.requires_grad = True  # Enable gradients on input
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Get the gradients for the predicted class
        for i in range(len(filenames)):
            class_idx = predicted[i].item()
            model.zero_grad()
            outputs[i, class_idx].backward(retain_graph=True)
            gradients = images.grad.data[i]  # Guided gradients
            
            # Save heatmap overlay of the middle frame
            middle_frame = images[i, :3]  # Extract first 3 channels for RGB
            output_filename = os.path.join(output_folder, f"heatmap_{filenames[i]}")
            visualize_guided_backprop(middle_frame, gradients[:3], output_filename)  # Use RGB gradients

if __name__ == "__main__":
    frames_folder = 'testFrames/1'
    output_folder = 'gbt'
    model_path = 'best_model.pth'
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = FrameInferenceDataset(frames_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = load_model(model_path)
    guided_backprop_infer(model, data_loader, output_folder)
    
    print("Guided backpropagation completed. Heatmap images are saved in 'testInferred' folder.")
