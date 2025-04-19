# person_detection.py

from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.4):

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        
        results = self.model.predict(source=frame, conf=self.conf_threshold, classes=[0], verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy().astype(int)
        detections = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cropped = frame[y1:y2, x1:x2]
            detections.append(((x1, y1, x2, y2), cropped))

        return detections
