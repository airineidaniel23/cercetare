import cv2
import numpy as np
import os
from math import radians, sin, cos, pi

# Paths to the folders (Update the paths)
frames_folder = 'testFrames/intors_4'  # Ensure this path is correct
movement_folder = 'testInferred/intors_4'
output_video = 'sters.mp4'

# Get all frame files and corresponding movement data
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])  # Assuming frames are JPG
movement_files = sorted([f for f in os.listdir(movement_folder) if f.endswith('.txt')])

# Check if the folder contains images
if len(frame_files) == 0:
    raise ValueError(f"No image files found in the directory: {frames_folder}")

# Check if the folder contains any txt files
if len(movement_files) == 0:
    print(f"No movement files found in the directory: {movement_folder}. All frames will use default rotation.")

# Load the first frame to get dimensions
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
if first_frame is None:
    raise ValueError(f"Unable to load the first image: {frame_files[0]}")

height, width, layers = first_frame.shape

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Function to draw the compass and highlight the correct sector based on one-hot encoding
def draw_compass_with_highlight(img, center, size, sector_index, highlight_color=(0, 0, 255), normal_color=(200, 200, 200)):
    num_sectors = 8  # Number of compass directions (N, NE, E, SE, S, SW, W, NW)
    sector_angle = 360 / num_sectors  # Each sector spans 45 degrees
    sector_offset = 22.5  # Offset to rotate the sectors correctly

    # Calculate triangle points for each sector
    directions = []
    for i in range(num_sectors):
        # The angle for this sector in radians
        theta = radians(i * sector_angle + 2 * sector_offset)
        # The direction points are drawn relative to the center
        direction = [
            (int(center[0] + size * cos(theta)), int(center[1] + size * sin(theta))),  # Large outer point (points outward)
            (int(center[0] + size * 0.6 * cos(theta + pi / 8)), int(center[1] + size * 0.6 * sin(theta + pi / 8))),  # Small inner point
            (int(center[0] + size * 0.6 * cos(theta - pi / 8)), int(center[1] + size * 0.6 * sin(theta - pi / 8)))  # Small inner point
        ]
        directions.append(direction)

    # Draw each triangle for the compass
    for i, points in enumerate(directions):
        color = highlight_color if i == sector_index else normal_color
        cv2.fillPoly(img, [np.array(points)], color)

# Loop through frames and movement files
for i, frame_file in enumerate(frame_files):
    # Load frame
    frame = cv2.imread(os.path.join(frames_folder, frame_file))

    if i < len(movement_files):
        # Read the movement file and extract the one-hot encoding
        with open(os.path.join(movement_folder, movement_files[i]), 'r') as f:
            movement_data = f.readline().strip().split()
            one_hot_encoding = list(map(int, movement_data[:8]))  # Assuming one-hot encoding is the first 8 values
        
        # Find the index of the sector with a 1 (it should only have one '1' in the list)
        sector_index = one_hot_encoding.index(1)
    else:
        # If no more movement files, use a default sector index (for example, 0)
        sector_index = 0

    # Set the location for the circle in the lower right corner
    circle_center = (width - 30, height - 30)  # Adjust these coordinates if needed
    circle_radius = 25  # Adjust the radius size if needed

    # Draw the compass with the highlighted sector based on the one-hot encoding
    draw_compass_with_highlight(frame, circle_center, circle_radius, sector_index)

    # Write the frame with the highlighted sector to the video
    video_writer.write(frame)

# Release video writer
video_writer.release()

print("Video compilation with one-hot encoded sector highlighting completed.")
