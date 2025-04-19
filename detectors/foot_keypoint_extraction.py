from ultralytics import YOLO

class FootKeypointExtractor:
    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        self.model = YOLO(model_path)

    def extract(self, frame):
        results = self.model(frame)
        for res in results:
            # each res.keypoints: [num_people,17,3]
            if res.keypoints is not None:
                pts = res.keypoints[0]
                # foot idx: left ankle=11, right ankle=14
                left = tuple(pts[11][:2].astype(int))
                right = tuple(pts[14][:2].astype(int))
                return (left, right)
        return None