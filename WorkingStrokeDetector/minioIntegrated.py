import os
import tkinter as tk
from tkinter import Listbox, Scrollbar, messagebox
from minio import Minio
import cv2
import cvzone
import math
import time
from ultralytics import YOLO

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


def run_detection(selected_video):
    if not selected_video:
        messagebox.showerror("Error", "Please select a video.")
        return

    video_url = minio_client.presigned_get_object(bucket_name, selected_video)

    cap = cv2.VideoCapture(video_url)
    model = YOLO('yolov8s.pt')

    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    lying_down_start_time = None
    stroke_detected = False
    lying_down_duration_threshold = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from stream.")
            break

        frame = cv2.resize(frame, (980, 740))
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100)
                class_detect = classnames[int(box.cls[0])]

                if confidence < 80 or class_detect != 'person':
                    continue

                width, height = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [
                                   x1 + 8, y1 - 12], thickness=2, scale=2)

                if height < width:
                    cvzone.putTextRect(frame, 'Fall Detected', [
                                       x1 + 8, y2 + 30], thickness=2, scale=2)

                    if lying_down_start_time is None:
                        lying_down_start_time = time.time()

                    elif time.time() - lying_down_start_time > lying_down_duration_threshold:
                        stroke_detected = True
                        cvzone.putTextRect(frame, 'Stroke Detected', [
                                           x1 + 8, y1 - 50], thickness=2, scale=2)

                else:
                    lying_down_start_time = None
                    stroke_detected = False

        cv2.imshow('Video Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_tkinter_window(video_files):
    root = tk.Tk()
    root.title("Select Video for Fall Detection")

    frame_neg = tk.Frame(root)
    frame_neg.pack(side=tk.LEFT, padx=10, pady=10)

    frame_pos = tk.Frame(root)
    frame_pos.pack(side=tk.RIGHT, padx=10, pady=10)

    tk.Label(frame_neg, text="Negative Videos").pack()
    tk.Label(frame_pos, text="Positive Videos").pack()

    listbox_neg = Listbox(frame_neg)
    listbox_neg.pack()

    for video in video_files["Negative"]:
        listbox_neg.insert(tk.END, video)

    listbox_pos = Listbox(frame_pos)
    listbox_pos.pack()

    for video in video_files["Positive"]:
        listbox_pos.insert(tk.END, video)

    def on_select():
        selected_video_neg = listbox_neg.curselection()
        selected_video_pos = listbox_pos.curselection()

        if selected_video_neg:
            video = listbox_neg.get(selected_video_neg)
            run_detection(video)
        elif selected_video_pos:
            video = listbox_pos.get(selected_video_pos)
            run_detection(video)
        else:
            messagebox.showerror("Error", "Please select a video.")

    run_button = tk.Button(root, text="Run Detection", command=on_select)
    run_button.pack()

    root.mainloop()


if __name__ == "__main__":
    video_files = list_minio_videos()
    if video_files["Negative"] or video_files["Positive"]:
        create_tkinter_window(video_files)
    else:
        print("No video files found in the Minio bucket.")
