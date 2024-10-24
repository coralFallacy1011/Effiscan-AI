import time
import math
import cv2
from ultralytics import YOLO

# Global variable to track total time since the session started
global_start_time = time.time()  # Start time of the entire session

def draw_workspace(x, y, width, height, canvas):
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 0, 0), 2)

def calculate_center(box):
    """Calculates the center of a bounding box"""
    x1, y1, x2, y2 = box[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class EffiSense:
    def __init__(self, camera):
        self.model = YOLO("yolov8s.pt")
        self.class_names = {0: 'person', 63: 'laptop'}
        self.cap = cv2.VideoCapture(camera)
        self.working_timers = {}  # Dictionary to track working time for each person
        self.start_times = {}     # Store start time for each person
        self.active_states = {}   # Track if person is within threshold

    def process(self, ret, frame, object):
        if not ret:
            return []
        # Perform inference
        results = self.model(frame)
        filtered_results = [r for r in results[0].boxes.data if int(r[-1]) == object]
        return filtered_results

    def draw(self, results, color):
        for box in results:
            x1, y1, x2, y2, score, class_id = box
            label = self.class_names[int(class_id)]
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(self.frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def update_timer(self, person_id, within_threshold):
        """Update the working timer for a person only when within threshold distance"""
        if person_id not in self.working_timers:
            self.working_timers[person_id] = 0  # Initialize timer
            self.start_times[person_id] = None  # Start time not set yet
            self.active_states[person_id] = False

        # Start timer if within threshold
        if within_threshold:
            if not self.active_states[person_id]:
                # Entering threshold: Start the timer
                self.start_times[person_id] = time.time()
                self.active_states[person_id] = True
            else:
                # Accumulate time while still in threshold
                if self.start_times[person_id] is not None:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_times[person_id]
                    self.working_timers[person_id] += elapsed_time
                    self.start_times[person_id] = current_time
        else:
            # Person has left threshold, stop updating the time
            self.active_states[person_id] = False

        # Calculate total session time
        total_time = time.time() - global_start_time

        # Calculate efficiency for this person
        working_time = self.working_timers[person_id]
        efficiency = (working_time / total_time) * 100 if total_time > 0 else 0

        # Display the working time and efficiency for the person
        cv2.putText(self.frame, f'Person {person_id + 1} Working Time: {working_time:.2f} s', 
                    (50, 50 + 30 * person_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(self.frame, f'Efficiency: {efficiency:.2f}%', 
                    (50, 70 + 30 * person_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def display(self):
        threshold_distance = 200  # Set the threshold distance to 100 pixels

        while self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break

            persons = self.process(self.ret, self.frame, 0)
            laptops = self.process(self.ret, self.frame, 63)

            self.draw(persons, (0, 255, 0))
            self.draw(laptops, (0, 0, 255))

            # Process each detected person
            for person_idx, person_box in enumerate(persons):
                person_center = calculate_center(person_box)

                # Find the closest laptop to this person
                closest_laptop_distance = float('inf')
                closest_laptop_center = None

                for laptop_box in laptops:
                    laptop_center = calculate_center(laptop_box)
                    distance = calculate_distance(person_center, laptop_center)

                    # Check if this laptop is closer than the previous one
                    if distance < closest_laptop_distance:
                        closest_laptop_distance = distance
                        closest_laptop_center = laptop_center

                # Check if the person is within the threshold distance
                within_threshold = closest_laptop_distance < threshold_distance

                self.update_timer(person_idx, within_threshold)  # Update timer for this person

            # Show the frame with the detections and timer info
            cv2.imshow('Person and Laptop Detection with Timer', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('e'):
                exit()

    def terminate(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Code terminated.")

if __name__ == '__main__':
    Cam1 = EffiSense(0)
    #LapCam = EffiSense(1) #Use this line if a webcam is connected to your PC
    Recording = EffiSense("cctv_footage2.mp4") #video name

    while True:
        #LapCam.display() #Use this line if a webcam is connected to your PC
        Cam1.display()
        Recording.display()
