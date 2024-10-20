import os
import tkinter as tk
from tkinter import Listbox, Scrollbar, messagebox, PhotoImage
from PIL import Image, ImageTk
from minio import Minio
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import tempfile
import threading

#In this file we test the AI on our MINIO data to see if it really works, here we found out overfitting was an issue

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


class VideoProcessor(threading.Thread):
    def __init__(self, video_url, canvas, status_label, timer_label):
        super().__init__()
        self.video_url = video_url
        self.canvas = canvas
        self.status_label = status_label
        self.timer_label = timer_label
        self.running = True
        self.cap = None

        # Load your custom fall detection model
        self.model = YOLO('yolov11_fall_detection.pt')
        self.classnames = []
        with open('classes.txt', 'r') as f:
            self.classnames = f.read().splitlines()

    def run(self):
        self.cap = cv2.VideoCapture(self.video_url)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_time = total_frames / fps

        frame_counter = 0
        frame_skip = 5
        fall_detected = False

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from stream.")
                break

            # Increment the frame counter
            frame_counter += 1

            # Update timer label
            elapsed_time = frame_counter / fps
            remaining_time = total_time - elapsed_time
            self.timer_label.config(text=f"Time Remaining: {max(0, int(remaining_time))} seconds")

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

            # Process the frame using your custom fall detection model
            results = self.model(resized_frame)

            # Draw bounding boxes and labels on the resized frame
            for info in results:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                    confidence = math.ceil(box.conf[0] * 100)
                    class_detect = self.classnames[int(box.cls[0])]  # Get the class name from your model

                    if confidence < 80:
                        continue  # Skip low-confidence detections

                    # Draw the bounding box and class label on the frame
                    width_box, height_box = x2 - x1, y2 - y1
                    cvzone.cornerRect(resized_frame, [x1, y1, width_box, height_box], l=30, rt=6)
                    cvzone.putTextRect(resized_frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                    # If the model detects a fall (assuming your model classifies a fall as 'fall')
                    if class_detect == 'fall':
                        if not fall_detected:
                            fall_detected = True
                            self.status_label.config(text="Fall Detected!", fg="red")
                        cvzone.putTextRect(resized_frame, 'Fall Detected', [x1 + 8, y2 + 30], thickness=2, scale=2)

            # Convert processed frame to PhotoImage
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
            time.sleep(0.01)  # Small sleep for responsiveness

        self.cap.release()  # Release the video capture when done

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def run_detection(selected_video, canvas, status_label, timer_label, root):
    if not selected_video:
        messagebox.showerror("Error", "Please select a video.")
        return

    # Stop any currently running video processor
    if hasattr(run_detection, 'video_processor') and run_detection.video_processor.is_alive():
        run_detection.video_processor.stop()
        run_detection.video_processor.join()

    video_url = minio_client.presigned_get_object(bucket_name, selected_video)
    run_detection.video_processor = VideoProcessor(video_url, canvas, status_label, timer_label)
    run_detection.video_processor.start()

def create_tkinter_window(video_files):
    root = tk.Tk()
    root.title("Select Video for Fall Detection")

    # Create a Canvas for video playback
    canvas = tk.Canvas(root, width=980, height=740)
    canvas.grid(row=0, column=1, padx=10, pady=10)  # Center the canvas between columns

    # Status label for stroke detection
    status_label = tk.Label(root, text="", font=("Helvetica", 16))
    status_label.grid(row=1, column=1, padx=10, pady=10)  # Place status label below the canvas

    # Timer label for showing remaining video time
    timer_label = tk.Label(root, text="Time Remaining: 0 seconds", font=("Helvetica", 14))
    timer_label.grid(row=2, column=1, padx=10, pady=5)  # Place timer label below the status label

    # Frames for Negative and Positive video thumbnails
    frame_neg = tk.Frame(root)
    frame_neg.grid(row=0, column=0, padx=10, pady=10)

    frame_pos = tk.Frame(root)
    frame_pos.grid(row=0, column=2, padx=10, pady=10)

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
            label.bind("<Button-1>", lambda e, v=video: run_detection(v, canvas, status_label, timer_label, root))

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
            label.bind("<Button-1>", lambda e, v=video: run_detection(v, canvas, status_label, timer_label, root))

    root.mainloop()

# Main Program
if __name__ == "__main__":
    video_files = list_minio_videos()
    if video_files["Negative"] or video_files["Positive"]:
        create_tkinter_window(video_files)
    else:
        print("No video files found in the Minio bucket.")
