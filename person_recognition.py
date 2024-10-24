import cv2  
import time  
import numpy as np  

# Load YOLO model  
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  
layer_names = net.getLayerNames()  
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]  

# Load COCO class names  
with open("coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]  

# Focus only on detecting "person" class  
person_class_id = classes.index("person")  

# Initialize trackers dictionary and sitting time  
trackers = {}  
sitting_times = {}  
tracker_id_counter = 0  

# Open the video file  
cap = cv2.VideoCapture('cctv_footages/footage2.mp4')  

# Set a starting timestamp  
start_time = time.time()  

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

    height, width, _ = frame.shape  

    # Prepare the frame for object detection  
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
    net.setInput(blob)  
    detections = net.forward(output_layers)  

    current_time = time.time()  

    # New detections  
    new_detections = []  

    # Process detections  
    for detection in detections:  
        for object_data in detection:  
            scores = object_data[5:]  
            class_id = np.argmax(scores)  
            confidence = scores[class_id]  

            # Detect person with a confidence threshold  
            if class_id == person_class_id and confidence > 0.5:  
                # Object detected  
                center_x = int(object_data[0] * width)  
                center_y = int(object_data[1] * height)  
                w = int(object_data[2] * width)  
                h = int(object_data[3] * height)  

                # Calculate coordinates  
                x = int(center_x - w / 2)  
                y = int(center_y - h / 2)  

                # Append new detection  
                new_detections.append((x, y, w, h, current_time))  

    # Check for overlapping anchors based on IoU
    overlaps = anchors_overlap([(x, y, w, h) for x, y, w, h, _ in new_detections], iou_threshold=0.5)

    if overlaps:
        for i, j, iou in overlaps:
            print(f"Anchors {i} and {j} are overlapping with IoU = {iou:.2f}")

    # Update trackers  
    for tracker_id, tracker_data in list(trackers.items()):  
        tracker = tracker_data["tracker"]  
        success, bbox = tracker.update(frame)  

        if success:  
            x, y, w, h = map(int, bbox)  
            is_sitting = h < (height / 2)  

            # Update tracker state for sitting  
            if is_sitting and not tracker_data["sitting"]:  
                tracker_data["start_time"] = current_time  
                tracker_data["sitting"] = True  
            elif not is_sitting and tracker_data["sitting"]:  
                sitting_duration = current_time - tracker_data["start_time"]  
                print(f"Person {tracker_id} was sitting for {sitting_duration:.2f} seconds.")  
                tracker_data["start_time"] = None  
                tracker_data["sitting"] = False  

            # Draw bounding box and label  
            color = (0, 255, 0) if is_sitting else (255, 0, 0)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  
            cv2.putText(frame, f"Sitting: {is_sitting}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
        else:  
            # If the tracker lost track, you may want to remove it  
            del trackers[tracker_id]  

    # Associate new detections with existing trackers  
    for (x, y, w, h, detected_time) in new_detections:  
        detection_box = (x, y, w, h)  
        
        # Check against existing trackers to find overlaps  
        found = False  
        for tracker_id, tracker_data in trackers.items():  
            tracker_box = tracker_data["bbox"]  

            # Calculate IoU to check if this detection matches an existing tracker  
            iou = calculate_iou(tracker_box, detection_box)  
            if iou > 0.3:  # IoU threshold to consider as the same person  
                found = True  
                # Update the tracker box  
                tracker_data["bbox"] = detection_box  # Update bbox  
                break  # No need to reinitialize the tracker here  
        
        # If not found, create a new tracker  
        if not found:  
            new_tracker = cv2.TrackerKCF_create()  # Change based on your OpenCV version   
            new_tracker.init(frame, detection_box)  

            tracker_id = tracker_id_counter  
            tracker_id_counter += 1  

            # Store new tracker  
            trackers[tracker_id] = {  
                "tracker": new_tracker,  
                "start_time": detected_time if h < (height / 2) else None,  
                "bbox": detection_box,  
                "sitting": h < (height / 2)  
            }  

    # Display the frame  
    cv2.imshow('Frame', frame)  

    # Break loop on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# Release resources  
cap.release()  
cv2.destroyAllWindows()