import json
import os

# Function to extract fall and stroke intervals from a video folder
# This is for the labeled videos provided on google drive, link is in the ReadMe


def extract_event_intervals(video_folder):
    obj_train_data_path = os.path.join(video_folder, 'obj_train_data')
    frame_files = sorted([f for f in os.listdir(
        obj_train_data_path) if f.endswith('.txt')])

    fall_intervals = []
    stroke_intervals = []
    current_fall_start = None
    current_stroke_start = None

    for frame_file in frame_files:
        frame_number = int(frame_file.replace(
            'frame_', '').replace('.txt', ''))
        with open(os.path.join(obj_train_data_path, frame_file), 'r') as f:
            labels = f.readlines()

        # Check if there is a label for a fall (1) or stroke (2)
        detected_fall = any(label.startswith('1 ') for label in labels)
        detected_stroke = any(label.startswith('2 ') for label in labels)

        # Handle falls
        if detected_fall:
            if current_fall_start is None:
                current_fall_start = frame_number
        elif current_fall_start is not None:
            fall_intervals.append(
                (current_fall_start, frame_number - 1))
            current_fall_start = None

        # Handle strokes
        if detected_stroke:
            if current_stroke_start is None:
                current_stroke_start = frame_number
        elif current_stroke_start is not None:
            stroke_intervals.append(
                (current_stroke_start, frame_number - 1))
            current_stroke_start = None

    return fall_intervals, stroke_intervals


def process_all_videos(labeled_videos_folder):
    video_folders = [os.path.join(labeled_videos_folder, folder) for folder in os.listdir(
        labeled_videos_folder) if os.path.isdir(os.path.join(labeled_videos_folder, folder))]

    all_video_stats = {}

    for video_folder in video_folders:
        video_name = os.path.basename(video_folder)
        fall_intervals, stroke_intervals = extract_event_intervals(
            video_folder)
        all_video_stats[video_name] = {
            "fall_intervals": fall_intervals,
            "stroke_intervals": stroke_intervals
        }
        print(f"Processed {video_name}:")
        print(f"  Fall Intervals: {fall_intervals}")
        print(f"  Stroke Intervals: {stroke_intervals}")

    return all_video_stats


labeled_videos_folder = 'labeledMinIOVideos'
all_stats = process_all_videos(labeled_videos_folder)

with open('video_event_intervals.json', 'w') as f:
    json.dump(all_stats, f, indent=4)
