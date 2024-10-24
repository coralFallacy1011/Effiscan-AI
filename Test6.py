from fontTools.ttx import process
from ultralytics import YOLO
import cv2
import keyboard

def draw_workspace(x, y, width, height, canvas):
    # Draw tables
    cv2.rectangle(canvas, (x, y),
                  (x + width, y + height),
                  (255, 0, 0), 2)

class EffiSense:
    def __init__(self, camera):
        self.model = YOLO("yolov8s.pt")
        self.class_names = ['person']
        self.cap = cv2.VideoCapture(camera)


    def process(self, ret, frame):
        if not ret:
            return

        # Perform inference
        results = self.model(frame)
        filtered_results = [r for r in results[0].boxes.data if int(r[-1]) == 0]
        return filtered_results

    def draw(self, results):
        work1 = [100, 40, 200, 100]
        work2 = [400, 40, 200, 100]
        draw_workspace(work1[0], work1[1], work1[2], work1[3], self.frame)
        draw_workspace(work2[0], work2[1], work2[2], work2[3], self.frame)

        for box in results:
            x1, y1, x2, y2, score, class_id = box
            label = self.class_names[int(class_id)]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(self.frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def display(self):
        while self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break

            filtered = self.process(self.ret, self.frame)
            self.draw(filtered)

            cv2.imshow('Person Detection', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('e'):
                exit()

    def terminate(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Code terminated.")

if __name__ == '__main__':
    WebCam = EffiSense(0)
    LapCam = EffiSense(1)

    while True:
        LapCam.display()
        WebCam.display()

    LapCam.terminate()
    WebCam.terminate()