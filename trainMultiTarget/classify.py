import os
import numpy as np
from math import radians, sin, cos, pi

# Paths to the folders (Update the paths)
movement_folder = 'testTarget/1'  # Ensure this path is correct

globalw = 0

def smooth_values(values, window_size=90):
    smoothed_values = []
    for i in range(len(values)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(values), i + window_size // 2 + 1)
        smoothed_values.append(np.mean(values[start_index:end_index]))
    return smoothed_values

# Get all movement data files
movement_files = sorted([f for f in os.listdir(movement_folder) if f.endswith('.txt')])

# Check if the folder contains any txt files
if len(movement_files) == 0:
    print(f"No movement files found in the directory: {movement_folder}. Cannot process sectors.")

# Function to compute sector index from angle
def calculate_sector_index(angle):
    sector_offset = 22.5  # Offset to rotate the sectors correctly
    num_sectors = 8  # Number of compass directions (N, NE, E, SE, S, SW, W, NW)
    sector_angle = 360 / num_sectors  # Each sector spans 45 degrees
    angle = -angle * 2100 + pi / 2  # Map the angle based on wy
    sector_index = int(((2 * pi - angle - pi / 8) % (2 * pi)) // (pi / 4))  # Determine the index of the highlighted sector
    return sector_index

# Load all wy values from movement files
wy_values = []
for i, movement_file in enumerate(movement_files):
    with open(os.path.join(movement_folder, movement_file), 'r') as f:
        movement_data = f.readline().strip().split()
        wy = float(movement_data[4])  # Assuming wy is the 5th value
        wy_values.append(wy)

# Smooth the wy values
smoothed_wy_values = smooth_values(wy_values)

# Adjust the window size for summing future values
window_size = 60

# Loop through the movement files and calculate sector index
for i, movement_file in enumerate(movement_files):
    if i < len(movement_files):
        # Sum the next 60 smoothed wy values or as many as available
        #sum_wy = sum(smoothed_wy_values[i:min(i + window_size, len(smoothed_wy_values))])
        sum_wy = smoothed_wy_values[i]
        wy = sum_wy  # Use the summed wy
    else:
        # No more movement files, use the last available wy sum
        wy = sum_wy
    
    # Determine the angle for sector selection (adjust according to your data scale)
    angle_in_degrees = wy
    
    # Calculate the sector index
    sector_index = calculate_sector_index(angle_in_degrees)
    
    # Create one-hot encoding for the sector index (8 classes)
    one_hot_encoding = [0] * 8
    one_hot_encoding[sector_index] = 1
    
    # Overwrite the movement file with the one-hot encoded sector index
    with open(os.path.join(movement_folder, movement_file), 'w') as f:
        f.write(' '.join(map(str, one_hot_encoding)) + '\n')

print("Sector indices logged in one-hot encoding for each movement file.")
