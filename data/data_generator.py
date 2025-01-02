import os
import shutil

def process_segments(video_number, segments, folder_prefix="occ_"):
    """
    Process segments for a given video_number and copy the corresponding frames and
    output data into a single input folder and a single output folder. Each occurrence
    has its own subfolder inside input/ and output/, named with the given prefix.

    :param video_number: The number of the video (int or str).
    :param segments: List of segment strings (e.g., ['1_2_3', '2_3_14', '3_14_13']).
    :param folder_prefix: A prefix for the occurrence folders (default "occ_").
    """

    fragments_path = "fragments"
    generated_data_path = "generatedData"

    # Paths for the single input and output folders
    input_folder = os.path.join(generated_data_path, "input")
    output_folder = os.path.join(generated_data_path, "output")

    # Ensure top-level generatedData, input, and output directories exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    occurrence_index = 1

    while True:
        found_all_segments = True

        # Name for the subfolder of this occurrence
        occurrence_subfolder_name = f"{folder_prefix}{occurrence_index}"
        occurrence_input_folder = os.path.join(input_folder, occurrence_subfolder_name)
        occurrence_output_folder = os.path.join(output_folder, occurrence_subfolder_name)

        # Create the occurrence subfolders if we're going to copy any files
        # (We do it here; if we don't find any segments, we can remove these or just leave them empty)
        os.makedirs(occurrence_input_folder, exist_ok=True)
        os.makedirs(occurrence_output_folder, exist_ok=True)

        for segment in segments:
            segment_folder = f"{fragments_path}/{segment}/{video_number}_{occurrence_index}_{segment}"

            # If the required segment folder for this occurrence does not exist, stop.
            if not os.path.exists(segment_folder):
                found_all_segments = False
                break

            # Copy frames from this segment occurrence to the occurrence_input_folder
            frames_path = os.path.join(segment_folder, "frames")
            if os.path.exists(frames_path):
                for file_name in os.listdir(frames_path):
                    src_file = os.path.join(frames_path, file_name)
                    # Optional: rename the file with the video_number prefix, if desired
                    dest_file = os.path.join(occurrence_input_folder, f"{video_number}_{file_name}")
                    shutil.copy(src_file, dest_file)

            # Copy output data from this segment occurrence to the occurrence_output_folder
            output_path = os.path.join(segment_folder, "output")
            if os.path.exists(output_path):
                for file_name in os.listdir(output_path):
                    src_file = os.path.join(output_path, file_name)
                    # Optional: rename the file with the video_number prefix, if desired
                    dest_file = os.path.join(occurrence_output_folder, f"{video_number}_{file_name}")
                    shutil.copy(src_file, dest_file)

        if not found_all_segments:
            # Remove the empty occurrence folders if they were just created (optional cleanup)
            if not os.listdir(occurrence_input_folder):
                os.rmdir(occurrence_input_folder)
            if not os.listdir(occurrence_output_folder):
                os.rmdir(occurrence_output_folder)
            break

        print(f"Processed occurrence {occurrence_index} for segments {segments} in video {video_number}.")

        occurrence_index += 1

    print(f"All occurrences processed for video {video_number} with segments {segments}.")

# Example usage:
if __name__ == "__main__":
    video_number = 1
    segments = ['9_8_7', '8_7_10', '7_10_13']
    prefix = "intors_"  # Choose any prefix you like

    process_segments(video_number, [segment.strip() for segment in segments], folder_prefix=prefix)
