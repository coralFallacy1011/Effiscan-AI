from ultralytics import YOLO
import cv2
import numpy as np
#import time

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

url = "http://192.168.71.153:8080/video"

# Open the video file
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object to save the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Efficiency tracking variables
efficiency_start_time_person1 = None
efficiency_start_time_person2 = None
total_efficiency_time_person1 = 0
total_efficiency_time_person2 = 0
# Threshold to consider "near" the table (in pixels, adjust as needed)
efficiency_threshold = 100
# Placeholder for table detection (set these bounding boxes manually or use detection logic)
# Assume table1 is on the left side and table2 on the right
table1_bbox = (100, 400, 200, 100)  # (x, y, width, height)
table2_bbox = (400, 400, 200, 100)  # (x, y, width, height)

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

def draw_workspace(x, y, width, height, canvas):
    # Draw tables
    cv2.rectangle(canvas, (x, y),
                  (x + width, y + height),
                  (255, 0, 0), 2)

def add_detection(frame_model, output):
    for r in frame_model:
        for box in r.boxes:
            # Get bounding box coordinates (x, y, width, height)
            x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            output.append((x1, y1, w, h, cx, cy))

'''def check_overlap(anchors, workspace_center, threshold=100):
    distance = np.linalg.norm(np.array(anchors[4:5]) - np.array(workspace_center))
    if distance > threshold:
        return True
    return False'''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #resize frame for better processing speed
    #frame = cv2.resize(frame, (640, 480))

    # Use the YOLOv8 model to predict on the frame
    #result = model.predict(frame, stream=True)

    result = model(frame)
    filtered_results = [r for r in result[0].boxes.data if int(r[-1]) == 0]

    '''# Store detections as (x, y, w, h)
    new_detections = []
    add_detection(result, new_detections)'''

    '''for r in result:
        for box in r.boxes:
            # Get bounding box coordinates (x, y, width, height)
            x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
            w = x2 - x1
            h = y2 - y1
            new_detections.append((x1, y1, w, h))'''

    '''# Check for overlapping anchors based on IoU
    overlaps = anchors_overlap(new_detections, iou_threshold=0.5)

    if overlaps:
        for i, j, iou in overlaps:
            print(f"Anchors {i} and {j} are overlapping with IoU = {iou:.2f}")

    # Draw bounding boxes and visualize the result
    #for (x, y, w, h) in new_detections:
    #    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)'''

    # Convert results to OpenCV format
    try:
        annotated_frame = filtered_results[0].plot()
    except IndexError:
        annotated_frame = None
        print("Code Terminated.")

    #creating workspaces
    '''work1 = [100, 40, 200, 100]
    c1 = [work1[0]+work1[2]//2, work1[1]+work1[3]//2]
    work2 = [400, 40, 200, 100]
    draw_workspace(work1[0], work1[1], work1[2], work1[3], annotated_frame)
    draw_workspace(work2[0], work2[1], work2[2], work2[3], annotated_frame)'''

    #Display efficiency
    #print(check_overlap(new_detections, c1))
    '''cv2.putText(annotated_frame, f"Person 1 Efficiency: {total_efficiency_time_person1:.2f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Person 2 Efficiency: {total_efficiency_time_person2:.2f}s",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)'''

    # Display the frame
    cv2.imshow('Phone Camera Stream - YOLOv8 Detection', annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()