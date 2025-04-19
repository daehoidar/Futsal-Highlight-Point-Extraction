import cv2
from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path: str = 'yolov8_ball.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        # choose ball class id = 0 or as defined
        for res in results:
            boxes = res.boxes
n            for box, cls in zip(boxes.xyxy, boxes.cls):
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    return ((x1+x2)//2, (y1+y2)//2)
        return None