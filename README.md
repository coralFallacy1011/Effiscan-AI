## Initialisation
Download the Prototype Final Version Folder and download all dependencies.
Modify the last lines of the script ```Recording = EffiSense("cctv_footage2.mp4")``` with 0 or 1 to use live cameras or a path for a recording.
use ```q``` to change cameras and ```e``` to exit.

## Tech Stack & Tools
Python 3 for Core programming language for building the entire logic and integration.

OpenCV (cv2) used for video capture, frame manipulation, drawing bounding boxes, and displaying output in real-time.

Ultralytics YOLOv8 Deep learning model for object detection — specifically detects persons (ID: 0) and laptops (ID: 63).

math library for calculating Euclidean distances between bounding boxes.

time library For tracking working durations and session times.

## Visuals

Sample 1: 

![image](https://github.com/user-attachments/assets/2a715ae5-59ea-4764-b7db-e45b1cb83e58)

Sample 2:

![image](https://github.com/user-attachments/assets/fb9a2512-17a8-44ab-969d-94c558b9be2c)
