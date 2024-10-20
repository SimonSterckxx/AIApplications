import os
import tkinter as tk
from tkinter import messagebox, PhotoImage
from PIL import Image, ImageTk
from minio import Minio
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import threading
import subprocess
import webview
import json

# MinIO Configuration
minio_url = "193.191.177.33:22555"
minio_user = "academic-weapons"
minio_passwd = "academic-weapons-peace"
bucket_name = "eyes4rescue-academic-weapons"

minio_client = Minio(
    minio_url,
    access_key=minio_user,
    secret_key=minio_passwd,
    secure=False
)

bucket_name = "eyes4rescue-academic-weapons"
folder_prefix_neg = "movies/negative/"
folder_prefix_pos = "movies/positive/"


# List videos from MinIO
def list_minio_videos():
    video_files = {"Negative": [], "Positive": []}
    objects_neg = minio_client.list_objects(
        bucket_name, prefix=folder_prefix_neg, recursive=True)
    for obj in objects_neg:
        if obj.object_name.endswith(('.mp4', '.avi', '.mkv')):
            video_files["Negative"].append(obj.object_name)

    objects_pos = minio_client.list_objects(
        bucket_name, prefix=folder_prefix_pos, recursive=True)
    for obj in objects_pos:
        if obj.object_name.endswith(('.mp4', '.avi', '.mkv')):
            video_files["Positive"].append(obj.object_name)

    return video_files


# Generate thumbnails for videos
def generate_thumbnail(video_url):
    cap = cv2.VideoCapture(video_url)
    ret, frame = cap.read()

    if ret:
        thumbnail = cv2.resize(frame, (150, 100))
        thumbnail_image = Image.fromarray(
            cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(thumbnail_image)
    else:
        print(f"Failed to generate thumbnail for {video_url}")
        return None


# Video processing class
class VideoProcessor(threading.Thread):
    def __init__(self, video_url, canvas, status_label, timer_label):
        super().__init__()
        self.video_url = video_url
        self.canvas = canvas
        self.status_label = status_label
        self.timer_label = timer_label
        self.running = True
        self.cap = None
        self.model = YOLO('yolov8s.pt')  # Load YOLOv8 model
        self.classnames = []
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        with open('classes.txt', 'r') as f:
            self.classnames = f.read().splitlines()

        # Custom statistics for the streamlit page
        self.total_falls_detected = 0
        self.total_false_alarms = 0
        self.total_missed_falls = 0
        self.lying_down_start_time = None
        self.stroke_detected = False
        self.lying_down_duration_threshold = 15  # seconds

    def run(self):
        self.cap = cv2.VideoCapture(self.video_url)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_time = total_frames / fps
        lying_down_start_time = None
        stroke_detected = False
        lying_down_duration_threshold = 15
        frame_skip = 5  # Process every 5th frame
        frame_counter = 0  # Initialize a frame counter

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from stream.")
                self.stop()
                break

            # Increment the frame counter
            frame_counter += 1

            # Update timer label
            elapsed_time = frame_counter / fps
            remaining_time = total_time - elapsed_time
            self.timer_label.config(
                text=f"Time Remaining: {max(0, int(remaining_time))} seconds")

            # Resize frame while maintaining aspect ratio
            height, width, _ = frame.shape
            aspect_ratio = width / height
            new_width = 980
            new_height = int(new_width / aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Skip processing if we are not on the correct frame
            if frame_counter % frame_skip != 0:
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
                time.sleep(0.01)  # Small sleep for responsiveness
                continue

            # Process the frame
            results = self.model(resized_frame)
            detected_fall = False

            # Draw bounding boxes and labels on the resized frame
            for info in results:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil(box.conf[0] * 100)
                    class_detect = self.classnames[int(box.cls[0])]

                    if confidence < 80 or class_detect != 'person':
                        continue

                    width_box, height_box = x2 - x1, y2 - y1
                    cvzone.cornerRect(
                        resized_frame, [x1, y1, width_box, height_box], l=30, rt=6)
                    cvzone.putTextRect(resized_frame, f'{class_detect}', [
                                       x1 + 8, y1 - 12], thickness=2, scale=2)

                    if height_box < width_box:
                        cvzone.putTextRect(resized_frame, 'Fall Detected', [
                                           x1 + 8, y2 + 30], thickness=2, scale=2)
                        self.total_falls_detected += 1

                        if lying_down_start_time is None:
                            lying_down_start_time = time.time()

                        elif time.time() - lying_down_start_time > lying_down_duration_threshold:
                            if not stroke_detected:  # Update status only once
                                stroke_detected = True
                                # Update label to show stroke detected
                                self.status_label.config(
                                    text="Stroke Detected!", fg="red")
                            cvzone.putTextRect(resized_frame, 'Stroke Detected', [
                                               x1 + 8, y1 - 50], thickness=2, scale=2)

                    else:
                        lying_down_start_time = None
                        stroke_detected = False

            # Convert processed frame to PhotoImage
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
            time.sleep(0.01)  # Small sleep for responsiveness

            # False Positive (fall detected but it wasn't an actual fall)
            if detected_fall and not self.stroke_detected:
                self.false_positives += 1
                self.total_false_alarms += 1

            # False Negative (fall missed)
            if not detected_fall and self.stroke_detected:
                self.false_negatives += 1
                self.total_missed_falls += 1

        self.cap.release()  # Release the video capture when done

    def save_statistics(self):
        stats = {
            "total_falls_detected": self.total_falls_detected,
            "total_false_alarms": self.total_false_alarms,
            "total_missed_falls": self.total_missed_falls,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives
        }
        print("Saving statistics:", stats)  # Debugging line
        with open("stats.json", "w") as f:
            json.dump(stats, f)

    def stop(self):
        print("Stopping video processor...")
        self.save_statistics()
        self.running = False
        if self.cap:
            self.cap.release()


# Function to run Streamlit dashboard in a separate thread
def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit.py"])


# Function to launch the Streamlit app in PyWebView inside Tkinter
def open_streamlit_dashboard():
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    # Wait a moment to allow Streamlit to start
    time.sleep(3)

    # Create PyWebView window inside Tkinter
    window = webview.create_window(
        "Fall Detection Dashboard", "http://localhost:8501")
    webview.start()


def run_detection(selected_video, canvas, status_label, timer_label, root):
    if not selected_video:
        messagebox.showerror("Error", "Please select a video.")
        return

    # Stop any currently running video processor
    if hasattr(run_detection, 'video_processor') and run_detection.video_processor.is_alive():
        # Debugging line
        print("Stopping the currently running video processor...")
        run_detection.video_processor.stop()  # Call stop to end the video
        run_detection.video_processor.join()  # Wait for the thread to finish

    # Start the new video processor
    video_url = minio_client.presigned_get_object(bucket_name, selected_video)
    run_detection.video_processor = VideoProcessor(
        video_url, canvas, status_label, timer_label)
    run_detection.video_processor.start()
    print("Started new video processor.")  # Log the new video start


# Tkinter window creation and video selection
def create_tkinter_window(video_files):
    root = tk.Tk()
    root.title("Select Video for Fall Detection")

    # Create a Canvas for video playback
    canvas = tk.Canvas(root, width=980, height=740)
    canvas.grid(row=0, column=1, padx=5, pady=5)

    # Status label for stroke detection
    status_label = tk.Label(root, text="", font=("Helvetica", 16))
    status_label.grid(row=1, column=1, padx=5, pady=5)

    # Timer label for showing remaining video time
    timer_label = tk.Label(
        root, text="Time Remaining: 0 seconds", font=("Helvetica", 14))
    timer_label.grid(row=2, column=1, padx=5, pady=5)

    # Frames for Negative and Positive video thumbnails
    frame_neg = tk.Frame(root)
    frame_neg.grid(row=0, column=0, padx=5, pady=5)

    frame_pos = tk.Frame(root)
    frame_pos.grid(row=0, column=2, padx=5, pady=5)

    tk.Label(frame_neg, text="Negative Videos").pack()
    tk.Label(frame_pos, text="Positive Videos").pack()

    # Dictionary to hold thumbnail images (to prevent garbage collection)
    thumbnails_neg = []
    thumbnails_pos = []

    # Display thumbnails for negative videos
    for video in video_files["Negative"]:
        video_url = minio_client.presigned_get_object(bucket_name, video)
        thumbnail = generate_thumbnail(video_url)
        if thumbnail:
            label = tk.Label(frame_neg, image=thumbnail)
            label.pack()
            thumbnails_neg.append(thumbnail)
            title_label = tk.Label(frame_neg, text=os.path.basename(video))
            title_label.pack()
            label.bind("<Button-1>", lambda e, v=video: run_detection(v,
                       canvas, status_label, timer_label, root))

    # Display thumbnails for positive videos
    for video in video_files["Positive"]:
        video_url = minio_client.presigned_get_object(bucket_name, video)
        thumbnail = generate_thumbnail(video_url)
        if thumbnail:
            label = tk.Label(frame_pos, image=thumbnail)
            label.pack()
            thumbnails_pos.append(thumbnail)
            title_label = tk.Label(frame_pos, text=os.path.basename(video))
            title_label.pack()
            label.bind("<Button-1>", lambda e, v=video: run_detection(v,
                       canvas, status_label, timer_label, root))

    # Add the button for statistics
    stats_button = tk.Button(root, text="Show Statistics",
                             command=open_streamlit_dashboard)
    stats_button.grid(row=3, column=1, pady=5)

    root.mainloop()


if __name__ == "__main__":
    video_files = list_minio_videos()
    if video_files["Negative"] or video_files["Positive"]:
        create_tkinter_window(video_files)
    else:
        print("No video files found in the Minio bucket.")
