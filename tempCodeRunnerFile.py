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
cap = cv2.VideoCapture('cctv_footage.mp4')

# Set a starting timestamp
start_time = time.time()

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

    # New frame - reset trackers if needed
    new_trackers = {}

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

                # Determine if the person is likely sitting
                is_sitting = h < (height / 2)  # Example assumption

                # Create a tracker for new person detected
                new_tracker = cv2.TrackerKCF_create()
                bbox = (x, y, w, h)
                new_tracker.init(frame, bbox)
                
                tracker_id = tracker_id_counter
                tracker_id_counter += 1
                
                # Store tracker and sitting start time if sitting
                new_trackers[tracker_id] = {
                    "tracker": new_tracker,
                    "start_time": current_time if is_sitting else None,
                    "bbox": bbox,
                    "sitting": is_sitting
                }

    # Update current frame trackers
    trackers = new_trackers

    # Update tracker state and calculate sitting times
    for tracker_id, tracker_data in trackers.items():
        tracker = tracker_data["tracker"]
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            is_sitting = h < (height / 2)

            # Check if the person has started/stopped sitting
            if is_sitting and not tracker_data["sitting"]:
                # Person has started sitting
                tracker_data["start_time"] = current_time
                tracker_data["sitting"] = True
            elif not is_sitting and tracker_data["sitting"]:
                # Person has stopped sitting, calculate time
                sitting_duration = current_time - tracker_data["start_time"]
                print(f"Person {tracker_id} was sitting for {sitting_duration:.2f} seconds.")
                tracker_data["start_time"] = None
                tracker_data["sitting"] = False

            # Draw bounding box and label
            color = (0, 255, 0) if is_sitting else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Sitting: {is_sitting}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
