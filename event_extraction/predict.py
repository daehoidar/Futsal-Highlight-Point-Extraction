import torch
import argparse
from utils import sample_clips
from detectors.ball_detection import BallDetector
from detectors.number_extraction import NumberExtractor
from detectors.foot_keypoint_extraction import FootKeypointExtractor
from detectors.person_detection import PersonDetector
from event_extraction.train import EventModel
import os


def predict(args):
    # load event model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"[INFO] 모델 경로 {model_path}가 존재하지 않습니다. pretrained 모델을 사용합니다.")
        model_path = os.path.join('pretrained', 'event_model_soccernet.pt')

    model = EventModel(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device).eval()
    video = args.video_path

    # slide through video in 5s windows
    events = []
    for t in range(0, int(args.video_length), 5):
        clips = sample_clips(video, t, 5)
        logits = model(torch.stack(clips).unsqueeze(0).to(args.device))
        cls = torch.argmax(logits, dim=1).item()
        if cls != 2:  # not 'none'
            events.append((t, cls))

    # for each event, find subject
    ball = BallDetector(args.ball_model)
    number = NumberExtractor()
    pose = FootKeypointExtractor()
    person_detector = PersonDetector()

    results = []
    for t, cls in events:
        frames = sample_clips(video, max(0, t-3), 3)  # 직전 3초
        avg_distances = []

        for f in frames:
            balls = ball.detect(f)  # 하나의 위치 반환 가정 (x, y)
            persons = person_detector.detect(f)  # 각 사람 객체에 대해 (bbox, cropped image)
            for i, (bbox, cropped_img) in enumerate(persons):
                number_id = number.extract(cropped_img)
                foot_kp = pose.extract_from_bbox(f, bbox)  # 해당 bbox 안의 keypoint
                if balls and foot_kp:
                    bx, by = balls[0]  # 단일 볼 기준
                    fx, fy = foot_kp
                    dist = ((bx - fx)**2 + (by - fy)**2)**0.5
                    avg_distances.append((number_id, dist))

        # 평균 거리 계산
        from collections import defaultdict
        dist_dict = defaultdict(list)
        for pid, d in avg_distances:
            if pid is not None:
                dist_dict[pid].append(d)

        avg_dist_by_player = {pid: sum(ds)/len(ds) for pid, ds in dist_dict.items()}
        best_player = min(avg_dist_by_player.items(), key=lambda x: x[1])[0] if avg_dist_by_player else None
        results.append({'time': t, 'event': cls, 'player': best_player})

    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, default='pretrained/event_model_soccernet.pt')
    parser.add_argument('--ball-model', type=str, default='yolov8_ball.pt')
    parser.add_argument('--video-length', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    predict(args)