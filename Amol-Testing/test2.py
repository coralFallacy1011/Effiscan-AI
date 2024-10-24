from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file
input_video = 'cctv_footages/footage2.mp4'
cap = cv2.VideoCapture(input_video)

# IoU calculation function
def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection box
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    return inter_area / float(box1_area + box2_area - inter_area)

# Function to check overlapping anchors based on IoU
def anchors_overlap(anchors, iou_threshold=0.5):
    n = len(anchors)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(anchors[i], anchors[j])
            if iou > iou_threshold:
                overlaps.append((i, j, iou))  # Log overlapping anchors
    return overlaps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the YOLOv8 model to predict on the frame
    result = model.predict(frame, stream=True)

    # Store detections as (x, y, w, h)
    new_detections = []
    for r in result:
        for box in r.boxes:
            # Get bounding box coordinates (x, y, width, height)
            x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
            w = x2 - x1
            h = y2 - y1
            new_detections.append((x1, y1, w, h))

    # Check for overlapping anchors based on IoU
    overlaps = anchors_overlap(new_detections, iou_threshold=0.5)

    if overlaps:
        for i, j, iou in overlaps:
            print(f"Anchors {i} and {j} are overlapping with IoU = {iou:.2f}")

    # Draw bounding boxes and visualize the result
    for (x, y, w, h) in new_detections:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()