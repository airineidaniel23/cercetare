import cv2
import os

# Set paths
video_folder = 'videosRaw'  # Folder where the video is located
video_filename = '2.mp4'  # Replace with your video file name
output_folder = 'videosRawData/2/frames'  # Folder where frames will be saved

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the video
video_path = os.path.join(video_folder, video_filename)
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame index counter
frame_index = 1

# Process the video frame by frame
while True:
    ret, frame = cap.read()  # Read the next frame from the video

    if not ret:
        break  # If there are no more frames, stop the loop

    # Resize the frame to 256x256
    resized_frame = cv2.resize(frame, (216, 384))

    # Save the frame with a name like frame_000001, frame_000002, etc.
    frame_filename = f"frame_{frame_index:06d}.jpg"  # 6-digit frame number with leading zeros
    frame_path = os.path.join(output_folder, frame_filename)

    # Write the resized frame to the output folder
    cv2.imwrite(frame_path, resized_frame)

    # Increment the frame index
    frame_index += 1

# Release the video capture object
cap.release()

print(f"Frames saved in {output_folder}.")
