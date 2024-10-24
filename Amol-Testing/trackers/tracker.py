from ultralytics import YOLO

class Tracker():
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 0
        detetections = []
        for i in range(0, len(frames), batch_size):
            detetections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detetections += detetections_batch

        return detetections

    def get_object_tracks(self, frames):

        detetections = self.detect_frames(frames)