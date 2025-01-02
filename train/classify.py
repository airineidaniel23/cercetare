import os
import numpy as np
from math import pi

# Set the parent folder that contains subfolders (1, 2, etc.)
movement_parent_folder = 'testTarget'

globalw = 0

def smooth_values(values, window_size=120):
    """
    Smooth the values by taking a moving average with the given window size.
    """
    smoothed_values = []
    for i in range(len(values)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(values), i + window_size // 2 + 1)
        smoothed_values.append(np.mean(values[start_index:end_index]))
    return smoothed_values

def calculate_sector_index(angle):
    """
    Compute the index (0-7) for the sector based on the input angle.
    Adjust logic depending on how you want to map the angles to sectors.
    """
    # Number of compass directions (N, NE, E, SE, S, SW, W, NW)
    num_sectors = 8
    # Convert the angle from your data scale to a rotation in radians
    angle = -angle * 2100 + pi / 2
    # Determine which sector to place the angle in (0 to 7).
    sector_index = int(((2 * pi - angle - pi / 8) % (2 * pi)) // (pi / 4))
    return sector_index

# List all subfolders in the parent folder
subfolders = [
    os.path.join(movement_parent_folder, d)
    for d in os.listdir(movement_parent_folder)
    if os.path.isdir(os.path.join(movement_parent_folder, d))
]

# Process each subfolder
for subfolder in subfolders:
    # Get all movement data files in the current subfolder
    movement_files = sorted([
        f for f in os.listdir(subfolder) if f.endswith('.txt')
    ])
    
    # Check if there are any txt files
    if len(movement_files) == 0:
        print(f"No movement files found in {subfolder}. Skipping...")
        continue
    
    # Load all wy values from the movement files
    wy_values = []
    for movement_file in movement_files:
        with open(os.path.join(subfolder, movement_file), 'r') as f:
            movement_data = f.readline().strip().split()
            # Change the index below if wy is at a different position
            wy = float(movement_data[4])  
            wy_values.append(wy)
    
    # Smooth the wy values
    smoothed_wy_values = smooth_values(wy_values)

    # Adjust the window size if you want to sum future values
    window_size = 90

    # Loop through the movement files and calculate the sector index
    for i, movement_file in enumerate(movement_files):
        # Either sum future values here or just take the current smoothed value
        # sum_wy = sum(smoothed_wy_values[i : min(i + window_size, len(smoothed_wy_values))])
        wy = smoothed_wy_values[i]
        
        # Compute sector index
        sector_index = calculate_sector_index(wy)
        
        # Create one-hot encoding for the sector index (8 classes)
        one_hot_encoding = [0] * 8
        one_hot_encoding[sector_index] = 1
        
        # Overwrite the movement file
        with open(os.path.join(subfolder, movement_file), 'w') as f:
            f.write(' '.join(map(str, one_hot_encoding)) + '\n')
    
    print(f"Classification applied for all files in: {subfolder}")

print("Done processing all subfolders in the 'movement' directory.")
