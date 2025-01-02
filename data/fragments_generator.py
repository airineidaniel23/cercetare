import os
import shutil
import numpy as np


def process_video(video_number):
    # Define paths
    annotations_path = f"videosGraphPositions/{video_number}.txt"
    raw_data_path = f"videosRawData/{video_number}"
    fragments_path = "fragments"

    # Check if annotation file exists
    if not os.path.exists(annotations_path):
        print(f"Annotation file {annotations_path} not found!")
        return

    # Read the annotation file
    with open(annotations_path, "r") as file:
        annotations = file.readlines()

    # Dictionary to track the occurrence of each class in this video
    class_occurrence = {}

    # Process each annotation line
    for line in annotations:
        parts = line.strip().split()
        class_key = "_".join(parts[:3])  # e.g., "1_2_3"
        interval_type = parts[3]  # "start" or "end"
        frame = int(parts[4])  # Frame number

        if interval_type == "start":
            start_frame = frame
        elif interval_type == "end":
            end_frame = frame
            if class_key not in class_occurrence:
                class_occurrence[class_key] = 0
            class_occurrence[class_key] += 1

            # Create the output folder structure
            fragment_folder = f"{fragments_path}/{class_key}/{video_number}_{class_occurrence[class_key]}_{class_key}"
            os.makedirs(fragment_folder, exist_ok=True)

            # Create subfolders for data categories
            for subfolder in ["depth", "flow", "frames", "output"]:
                os.makedirs(f"{fragment_folder}/{subfolder}", exist_ok=True)

            # Copy files for the interval
            for i in range(start_frame, end_frame):
                frame_id = f"frame_{i:06d}"
                for subfolder in ["depth", "flow", "frames", "output"]:
                    src_file = f"{raw_data_path}/{subfolder}/{frame_id}"
                    if subfolder == "depth":
                        src_file += "_pred.npy"
                    elif subfolder == "flow":
                        src_file += ".flo"
                    elif subfolder == "frames":
                        src_file += ".jpg"
                    elif subfolder == "output":
                        src_file += ".txt"

                    if os.path.exists(src_file):
                        shutil.copy(src_file, f"{fragment_folder}/{subfolder}/")
                    else:
                        print(f"File {src_file} not found, skipping.")

    print(f"Processing of video {video_number} completed.")


# Call the function with the video number as input
video_number = input("Enter the video number to process: ")
process_video(video_number)
