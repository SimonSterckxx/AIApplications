import cv2
import cvzone
import math
import time
import subprocess
from ultralytics import YOLO

#Hier is een functie die de twitch of youtube link pakt en daar een streamlink command op doet
def get_stream_url(twitch_url):
    try:
        # Use subprocess to get the direct stream URL using streamlink
        result = subprocess.run(['streamlink', twitch_url, 'best', '--stream-url'], capture_output=True, text=True)
        stream_url = result.stdout.strip()
        return stream_url
    except Exception as e:
        print(f"Error getting stream URL: {e}")
        return None


twitch_url = 'https://www.youtube.com/watch?v=ctxi_0Lz9uU'

stream_url = get_stream_url(twitch_url)

#debug
if not stream_url:
    print("Could not retrieve the stream URL.")
    exit()

# Open the live stream using OpenCV
cap = cv2.VideoCapture(stream_url)

#debug
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


model = YOLO('yolov8s.pt')


classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Variable
lying_down_start_time = None
stroke_detected = False
lying_down_duration_threshold = 15  # Time threshold for stroke detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from stream.")
        break

    # Resize the frame if needed
    frame = cv2.resize(frame, (980, 740))

    results = model(frame)

    # Process each detection
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = math.ceil(box.conf[0] * 100)
            class_detect = classnames[int(box.cls[0])]

            # Confidence check en kijkt of het object wel een persoon is zo niet continue
            if confidence < 80 or class_detect != 'person':
                continue

            # Vierkant maken
            width, height = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)


            if height < width:
                # Show 'Fall Detected' message
                cvzone.putTextRect(frame, 'Fall Detected', [x1 + 8, y2 + 30], thickness=2, scale=2)

                # If lying down and not already tracked, start the timer
                if lying_down_start_time is None:
                    lying_down_start_time = time.time()

                elif time.time() - lying_down_start_time > lying_down_duration_threshold:
                    stroke_detected = True
                    cvzone.putTextRect(frame, 'Stroke Detected', [x1 + 8, y1 - 50], thickness=2, scale=2)

            else:
                lying_down_start_time = None
                stroke_detected = False


    cv2.imshow('Live Stream Frame', frame)

#stop de loop door t te duwen
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
