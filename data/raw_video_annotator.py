import cv2
import os
import glob
import time
from collections import deque

# Settings
FRAME_RATE = 25  # frames per second
SKIP_BIG = 150
SKIP_SMALL = 30
FRAME_FORMAT = "frame_{:06d}.jpg"
TEXT_FILE = "videosGraphPositions/1.txt"

# Initialize variables
paused = False
current_frame_index = 0
annotation_numbers = None
annotation_start = None

# Load frames
frame_files = sorted(glob.glob("videosRawData/1/frames/frame_*.jpg"))
if not frame_files:
    print("No frames found. Please ensure frames are named as 'frame_000001.jpg' and located in the same folder.")
    exit()

# Function to write to the annotation file
def write_annotation(numbers, start_or_end, frame_index):
    with open(TEXT_FILE, "a") as file:
        file.write(f"{' '.join(map(str, numbers))} {start_or_end} {frame_index}\n")

# Function to prompt for 3 numbers
def prompt_numbers():
    while True:
        try:
            numbers = input("Enter 3 consecutive numbers separated by spaces: ").strip().split()
            if len(numbers) != 3 or not all(num.isdigit() for num in numbers):
                raise ValueError("Invalid input. Please enter exactly 3 numbers.")
            return list(map(int, numbers))
        except ValueError as e:
            print(e)

# Function to display frames
def display_frame(frame_path):
    frame = cv2.imread(frame_path)
    if frame is not None:
        cv2.imshow("Video Player", frame)
    else:
        print(f"Could not load frame: {frame_path}")

# Main loop
while True:
    if not paused:
        frame_path = frame_files[current_frame_index]
        display_frame(frame_path)
        time.sleep(1 / FRAME_RATE)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord(' '):  # Pause/Play
        paused = not paused
    elif key == ord('a'):  # Skip backwards big
        current_frame_index = max(0, current_frame_index - SKIP_BIG)
    elif key == ord('d'):  # Skip forward big
        current_frame_index = min(len(frame_files) - 1, current_frame_index + SKIP_BIG)
    elif key == ord('z'):  # Skip backwards small
        current_frame_index = max(0, current_frame_index - SKIP_SMALL)
    elif key == ord('c'):  # Skip forward small
        current_frame_index = min(len(frame_files) - 1, current_frame_index + SKIP_SMALL)
    elif key == ord('k'):  # Annotate
        if annotation_numbers is None:  # Prompt for numbers
            annotation_numbers = prompt_numbers()
            annotation_start = current_frame_index
            write_annotation(annotation_numbers, "start", annotation_start)
            print(f"Annotation started at frame {annotation_start}")
        else:  # Write end annotation
            write_annotation(annotation_numbers, "end", current_frame_index)
            print(f"Annotation ended at frame {current_frame_index}")
            annotation_numbers = None
            annotation_start = None

    # Automatically advance frames if not paused
    if not paused:
        current_frame_index += 1
        if current_frame_index >= len(frame_files):
            current_frame_index = len(frame_files) - 1

cv2.destroyAllWindows()
