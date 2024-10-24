from utils import read_frames, save_video
from ultralytics import YOLO
def main():
    video_frames = read_frames('cctv_footages/footage2.mp4')
    model = YOLO('yolov8s.pt')
    res = model.predict(video_frames[130], conf = 0.1)
    names = res[0].names
    print(res)
    print('==+===================================================')
    for box in res[0].boxes:
        print(box.xyxy, names[int(box.cls[0])])

if __name__ == '__main__':
    main()